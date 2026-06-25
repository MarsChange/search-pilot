from __future__ import annotations

import json
import re
from typing import Any

from deep_research.compressor.compressor import ContextCompressor
from deep_research.llm import LLMClient, extract_json_object
from deep_research.prompts import SYNTHESIZER_SYSTEM_PROMPT
from deep_research.schemas import ResearchPlan, ResearchReport, WorkerResult, confidence_to_float


class DeepResearchSynthesizer:
    def __init__(self, llm: LLMClient, *, compressor: ContextCompressor | None = None) -> None:
        self.llm = llm
        self.compressor = compressor or ContextCompressor(llm=None)

    async def synthesize(
        self,
        *,
        query: str,
        plan: ResearchPlan,
        results: list[WorkerResult],
        state_history: list[dict[str, Any]],
        states: list[dict[str, Any]],
        coverage: dict[str, Any],
        critique_history: list[dict[str, Any]] | None = None,
    ) -> ResearchReport:
        sources = self._collect_sources(results)
        evidence_context = self._build_evidence_context(results)
        compressed = await self.compressor.compress([evidence_context], query=query)
        prompt = f"""研究问题：{query}

研究计划：
{json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)}

覆盖检查：
{json.dumps(coverage, ensure_ascii=False, indent=2)}

证据：
{"\n\n".join(compressed)}

请直接输出严格 JSON，不要输出 Markdown，不要输出解释性前后缀。"""
        try:
            content = await self.llm.complete(
                [
                    {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
        except Exception:
            content = self._fallback_report(query, results, coverage)
        if not content.strip():
            content = self._fallback_report(query, results, coverage)
        content = self._ensure_answer_json(content, results, coverage, query=query)
        return ResearchReport(
            query=query,
            content=content,
            sources=sources,
            confidence=self._confidence(results, coverage),
            state_history=state_history,
            states=states,
            coverage=coverage,
            critique_history=critique_history or [],
        )

    @staticmethod
    def _build_evidence_context(results: list[WorkerResult]) -> str:
        blocks = []
        for result in results:
            blocks.append(f"## {result.state_id} [{result.status}]\n{result.summary}")
            if result.candidate_answer:
                blocks.append(f"候选答案: {result.candidate_answer}")
            if result.canonical_names:
                blocks.append("标准名/别名: " + "；".join(result.canonical_names))
            if result.answer_form_hint:
                blocks.append(f"答案格式提示: {result.answer_form_hint}")
            for evidence in result.evidence:
                blocks.append(
                    "- {claim}\n  来源: {source} {url} | 日期: {date} | 置信度: {confidence} | 类型: {evidence_type}".format(
                        claim=evidence.get("claim", ""),
                        source=evidence.get("source", ""),
                        url=evidence.get("url", ""),
                        date=evidence.get("date", ""),
                        confidence=evidence.get("confidence", ""),
                        evidence_type=evidence.get("evidence_type", ""),
                    )
                )
            if result.conflicts:
                blocks.append("冲突：" + "；".join(result.conflicts))
            if result.open_questions:
                blocks.append("未解决问题：" + "；".join(result.open_questions))
        return "\n".join(blocks)

    @staticmethod
    def _collect_sources(results: list[WorkerResult]) -> list[dict[str, Any]]:
        sources = []
        seen = set()
        for result in results:
            for evidence in result.evidence:
                url = evidence.get("url", "")
                key = url or evidence.get("source") or evidence.get("claim")
                if not key or key in seen:
                    continue
                seen.add(key)
                sources.append(
                    {
                        "title": evidence.get("source", ""),
                        "url": url,
                        "snippet": evidence.get("claim", ""),
                        "date": evidence.get("date", ""),
                        "state_id": result.state_id,
                    }
                )
        return sources

    @staticmethod
    def _confidence(results: list[WorkerResult], coverage: dict[str, Any]) -> float:
        if not results:
            return 0.0
        evidence_scores = [
            confidence_to_float(evidence.get("confidence"))
            for result in results
            for evidence in result.evidence
        ]
        evidence_score = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.35
        resolved = sum(1 for result in results if result.normalized_status() == "resolved")
        success_score = resolved / len(results)
        coverage_score = 1.0 if coverage.get("complete") else 0.7
        return round(max(0.0, min(1.0, evidence_score * 0.5 + success_score * 0.3 + coverage_score * 0.2)), 3)

    @staticmethod
    def _fallback_report(query: str, results: list[WorkerResult], coverage: dict[str, Any]) -> str:
        candidate = DeepResearchSynthesizer._best_fallback_candidate(query, results)
        supporting_facts = []
        sources = []
        for result in results:
            for evidence in result.evidence:
                claim = evidence.get("claim", "")
                if claim:
                    supporting_facts.append(str(claim))
                url = evidence.get("url") or evidence.get("source")
                if url:
                    sources.append(str(url))
        payload = {
            "answer": candidate,
            "confidence": "low" if coverage.get("missing") else "medium",
            "supporting_facts": supporting_facts[:5],
            "sources": list(dict.fromkeys(sources))[:5],
            "uncertainty": "LLM synthesis unavailable; answer was inferred from collected evidence.",
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _ensure_answer_json(
        content: str,
        results: list[WorkerResult],
        coverage: dict[str, Any],
        *,
        query: str = "",
    ) -> str:
        data = extract_json_object(content)
        if not data or "answer" not in data:
            return DeepResearchSynthesizer._fallback_report(query, results, coverage)
        answer = str(data.get("answer", "")).strip()
        if DeepResearchSynthesizer._is_invalid_answer(answer, query=query):
            return DeepResearchSynthesizer._fallback_report(query, results, coverage)
        payload = {
            "answer": answer,
            "confidence": str(data.get("confidence", "medium")),
            "supporting_facts": list(data.get("supporting_facts", []))[:8]
            if isinstance(data.get("supporting_facts", []), list)
            else [str(data.get("supporting_facts", ""))],
            "sources": list(data.get("sources", []))[:8]
            if isinstance(data.get("sources", []), list)
            else [str(data.get("sources", ""))],
            "uncertainty": str(data.get("uncertainty", "")),
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _best_fallback_candidate(query: str, results: list[WorkerResult]) -> str:
        candidates: list[tuple[float, str]] = []
        for result in results:
            base_score = confidence_to_float("high" if result.normalized_status() == "resolved" else "medium")
            values = [result.candidate_answer]
            values.extend(result.canonical_names)
            for value in values:
                candidate = str(value or "").strip()
                if DeepResearchSynthesizer._is_invalid_answer(candidate, query=query):
                    continue
                candidates.append((base_score, candidate))
            for finding in result.key_findings[:2]:
                candidate = DeepResearchSynthesizer._extract_candidate_from_finding(str(finding), query=query)
                if candidate and not DeepResearchSynthesizer._is_invalid_answer(candidate, query=query):
                    candidates.append((base_score * 0.8, candidate))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: (item[0], len(item[1]) <= 120), reverse=True)
        return candidates[0][1]

    @staticmethod
    def _extract_candidate_from_finding(finding: str, *, query: str) -> str:
        if DeepResearchSynthesizer._expects_numeric_answer(query):
            match = re.search(r"\b\d{1,5}\b", finding)
            return match.group(0) if match else ""
        return ""

    @staticmethod
    def _is_invalid_answer(answer: str, *, query: str = "") -> bool:
        normalized = re.sub(r"[\s。.!?]+", " ", (answer or "").strip().lower()).strip()
        if not normalized:
            return True
        if normalized in {"unknown", "unk", "n/a", "na", "none", "null", "not found", "not available", "无法确定", "不确定", "未知"}:
            return True
        if DeepResearchSynthesizer._expects_numeric_answer(query) and not re.fullmatch(r"\d+(?:\.\d+)?", answer.strip()):
            return True
        return False

    @staticmethod
    def _expects_numeric_answer(query: str) -> bool:
        lowered = (query or "").lower()
        numeric_markers = (
            "number of pages",
            "total number of pages",
            "how many pages",
            "多少页",
            "页数",
            "总页数",
        )
        return any(marker in lowered for marker in numeric_markers)


def extract_final_answer(content: str) -> str:
    data = extract_json_object(content)
    if isinstance(data, dict) and "answer" in data:
        return str(data.get("answer", "")).strip()
    return (content or "").strip()


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)>\]]+", text or "")
