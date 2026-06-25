from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from typing import Any, Callable

from deep_research.adversarial.blue_agent import BlueAgent
from deep_research.adversarial.loop import AdversarialLoop
from deep_research.adversarial.red_agent import RedAgent
from deep_research.compressor.compressor import ContextCompressor
from deep_research.llm import DashScopeLLM, LLMClient
from deep_research.memory.long_term import MemoryEntry
from deep_research.memory.memory_store import SharedMemoryStore
from deep_research.planner import DeepResearchPlanner
from deep_research.schemas import (
    CoverageReport,
    DeepResearchResult,
    FSMState,
    ResearchPlan,
    ResearchReport,
    ResearchState,
    StateHistoryEvent,
    WorkerResult,
    confidence_to_float,
)
from deep_research.state_graph import ResearchStateGraph
from deep_research.tool_adapter import FunctionToolExecutor
from deep_research.worker import DeepResearchWorker
from deep_research.synthesizer import DeepResearchSynthesizer, extract_final_answer


class DeepResearchRunner:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        tool_executor: Any | None = None,
        embedder: Any | None = None,
        memory_db_path: str | None = None,
        session_id: str | None = None,
        max_concurrent: int = 3,
        max_replans: int = 2,
        max_adversarial_rounds: int = 2,
        state_timeout_seconds: int = 120,
        event_sink: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.llm = llm or DashScopeLLM()
        self.session_id = session_id or str(uuid.uuid4())
        self.max_concurrent = max(1, max_concurrent)
        self.max_replans = max(0, max_replans)
        self.max_adversarial_rounds = max(0, max_adversarial_rounds)
        self.state_timeout_seconds = state_timeout_seconds
        self.memory_store = SharedMemoryStore(
            memory_db_path or os.getenv("DEEP_RESEARCH_MEMORY_DB", "data/deep_research_memory.db"),
            session_id=self.session_id,
            embedder=embedder,
        )
        self.compressor = ContextCompressor(llm=self.llm)
        self.planner = DeepResearchPlanner(self.llm)
        self.tool_executor = tool_executor or FunctionToolExecutor()
        self.worker = DeepResearchWorker(
            self.llm,
            tool_executor=self.tool_executor,
            compressor=self.compressor,
        )
        self.synthesizer = DeepResearchSynthesizer(self.llm, compressor=self.compressor)
        self.adversarial = AdversarialLoop(
            red_agent=RedAgent(self.llm),
            blue_agent=BlueAgent(self.llm),
            max_rounds=max_adversarial_rounds,
        )
        self.state_history: list[dict[str, Any]] = []
        self.num_replans = 0
        self.num_searches = 0
        self.event_sink = event_sink

    async def run(self, query: str) -> DeepResearchResult:
        start = time.monotonic()
        graph = ResearchStateGraph()
        results: list[WorkerResult] = []
        plan: ResearchPlan | None = None
        coverage_report = CoverageReport(complete=False)
        final_report: ResearchReport | None = None

        try:
            self._transition(FSMState.PLANNING)
            memory_context = self.memory_store.get_context_for_query(query, max_tokens=2500)
            plan = await self.planner.create_plan(query, memory_context)
            for state in plan.states:
                if not state.timeout_seconds:
                    state.timeout_seconds = self.state_timeout_seconds
                graph.add_state(state)
            self._emit(
                {
                    "type": "plan",
                    "plan": plan.to_dict(),
                    "states": [state.to_dict() for state in graph.states()],
                    "coverage_checklist": plan.coverage_checklist,
                }
            )

            while True:
                self._transition(FSMState.DISPATCHING)
                ready = graph.ready_states(limit=self.max_concurrent)
                if ready:
                    self._emit(
                        {
                            "type": "dispatch",
                            "ready_states": [state.to_dict() for state in ready],
                            "max_concurrent": self.max_concurrent,
                        }
                    )
                    batch = await self._run_ready_states(query, graph, ready, results)
                    results.extend(batch)
                elif graph.has_pending():
                    for state in graph.states():
                        if state.status.value == "pending":
                            graph.mark_skipped(state.state_id, "dependencies unavailable")
                    break

                self._transition(FSMState.COLLECTING)
                self._collect_to_memory(query, results)

                self._transition(FSMState.COVERAGE_CHECK)
                coverage_report = self._check_coverage(plan, graph, results)
                self._emit({"type": "coverage", "coverage": coverage_report.to_dict()})
                if coverage_report.complete and graph.all_terminal():
                    break
                if self._has_sufficient_final_answer(plan, graph, results, coverage_report):
                    for state in graph.states():
                        if state.status.value == "pending":
                            graph.mark_skipped(state.state_id, "final answer already resolved")
                    self._emit(
                        {
                            "type": "early_stop",
                            "reason": "final_answer_resolved",
                            "coverage": coverage_report.to_dict(),
                        }
                    )
                    break

                if graph.ready_states():
                    continue

                should_replan = (
                    (graph.should_replan() or not coverage_report.complete)
                    and self.num_replans < self.max_replans
                )
                if not should_replan:
                    if not graph.ready_states() and graph.all_terminal():
                        break
                    if self.num_replans >= self.max_replans:
                        break
                    if not graph.ready_states():
                        break

                self._transition(FSMState.REPLANNING)
                self.num_replans += 1
                self._emit(
                    {
                        "type": "replan_start",
                        "round": self.num_replans,
                        "coverage": coverage_report.to_dict(),
                    }
                )
                await self._replan(query, plan, graph, results, coverage_report)
                self._emit(
                    {
                        "type": "replan_result",
                        "round": self.num_replans,
                        "states": [state.to_dict() for state in graph.states()],
                    }
                )
                if not graph.ready_states() and not graph.has_pending():
                    break

            self._transition(FSMState.SYNTHESIZING)
            final_report = await self.synthesizer.synthesize(
                query=query,
                plan=plan,
                results=results,
                state_history=self.state_history,
                states=[state.to_dict() for state in graph.states()],
                coverage=coverage_report.to_dict(),
                critique_history=[],
            )

            self._transition(FSMState.ADVERSARIAL_REVIEW)
            if self.max_adversarial_rounds > 0:
                final_report, critique = await self.adversarial.run(final_report)
                final_report.critique_history = critique

            self._transition(FSMState.FINALIZING)
            elapsed = time.monotonic() - start
            self._transition(FSMState.DONE)
            final_report.state_history = self.state_history
            final_report.states = [state.to_dict() for state in graph.states()]
            metadata = self._metadata(
                elapsed_seconds=elapsed,
                num_states=len(graph.states()),
                adversarial_rounds=len(final_report.critique_history),
            )
            answer = extract_final_answer(final_report.content)
            self._emit(
                {
                    "type": "final",
                    "answer": answer,
                    "metadata": metadata,
                    "confidence": final_report.confidence,
                }
            )
            return DeepResearchResult(answer=answer, report=final_report, metadata=metadata)
        except Exception as exc:
            self._transition(FSMState.FAILED, {"error": str(exc)})
            elapsed = time.monotonic() - start
            content = self._failed_report(query, exc, results)
            final_report = ResearchReport(
                query=query,
                content=content,
                sources=[],
                confidence=0.0,
                state_history=self.state_history,
                states=[state.to_dict() for state in graph.states()],
                coverage=coverage_report.to_dict(),
                critique_history=[],
            )
            return DeepResearchResult(
                answer=content,
                report=final_report,
                metadata=self._metadata(
                    elapsed_seconds=elapsed,
                    num_states=len(graph.states()),
                    adversarial_rounds=0,
                    failed=True,
                ),
            )

    async def _run_ready_states(
        self,
        query: str,
        graph: ResearchStateGraph,
        ready: list[ResearchState],
        previous_results: list[WorkerResult],
    ) -> list[WorkerResult]:
        for state in ready:
            graph.mark_running(state.state_id)
            self._emit({"type": "state_start", "state": state.to_dict()})

        async def run_one(state: ResearchState) -> WorkerResult:
            try:
                memory_context = self.memory_store.get_context_for_query(state.description, max_tokens=1500)
                result = await asyncio.wait_for(
                    self.worker.run(
                        query=query,
                        state=state,
                        memory_context=memory_context,
                        prior_results=[item.to_dict() for item in previous_results],
                    ),
                    timeout=state.timeout_seconds or self.state_timeout_seconds,
                )
                self.num_searches += len(result.tool_calls)
                if result.normalized_status() == "failed":
                    graph.mark_failed(state.state_id, result.error or result.summary)
                else:
                    graph.mark_success(state.state_id)
                self._emit(
                    {
                        "type": "state_result",
                        "state_id": state.state_id,
                        "status": graph.get_state(state.state_id).status.value,
                        "result": result.to_dict(),
                    }
                )
                return result
            except asyncio.TimeoutError:
                graph.mark_failed(state.state_id, "state timeout")
                result = WorkerResult(
                    status="failed",
                    state_id=state.state_id,
                    summary="State execution timed out.",
                    error="timeout",
                    open_questions=["State timed out."],
                )
                self._emit(
                    {
                        "type": "state_result",
                        "state_id": state.state_id,
                        "status": "failed",
                        "result": result.to_dict(),
                    }
                )
                return result
            except Exception as exc:
                graph.mark_failed(state.state_id, str(exc))
                result = WorkerResult(
                    status="failed",
                    state_id=state.state_id,
                    summary="State execution failed.",
                    error=str(exc),
                    open_questions=[str(exc)],
                )
                self._emit(
                    {
                        "type": "state_result",
                        "state_id": state.state_id,
                        "status": "failed",
                        "result": result.to_dict(),
                    }
                )
                return result

        return await asyncio.gather(*(run_one(state) for state in ready))

    def _collect_to_memory(self, query: str, results: list[WorkerResult]) -> None:
        for result in results:
            for idx, evidence in enumerate(result.evidence):
                claim = str(evidence.get("claim", "")).strip()
                if not claim:
                    continue
                entry = MemoryEntry(
                    entry_id=f"{result.state_id}:{idx}:{abs(hash(claim))}",
                    session_id=self.session_id,
                    claim=claim,
                    source=str(evidence.get("source", "")),
                    url=str(evidence.get("url", "")),
                    confidence=confidence_to_float(evidence.get("confidence")),
                    agent_id=result.state_id,
                    timestamp=time.time(),
                    evidence_type=str(evidence.get("evidence_type", "secondary")),
                    topic=query[:80],
                    embedding=[],
                    metadata={"date": evidence.get("date", ""), "state_id": result.state_id},
                )
                self.memory_store.put(entry)

    def _check_coverage(
        self,
        plan: ResearchPlan,
        graph: ResearchStateGraph,
        results: list[WorkerResult],
    ) -> CoverageReport:
        coverage = graph.coverage_status(plan.coverage_checklist)
        failed_states = [state.state_id for state in graph.failed_states()]
        open_questions = [
            question
            for result in results
            for question in result.open_questions
            if question
        ]
        raw_conflicts = [
            conflict
            for result in results
            for conflict in result.conflicts
            if conflict
        ]
        raw_conflicts.extend([record.claim_1 + " <-> " + record.claim_2 for record in self.memory_store.get_conflicts("open")])
        conflicts = self._answer_relevant_conflicts(results, raw_conflicts)
        has_primary = any(
            str(evidence.get("evidence_type", "")).lower() == "primary"
            for result in results
            for evidence in result.evidence
        )
        needs_recency = "recency" in plan.coverage_checklist or any("最新" in flag or "时效" in flag for flag in plan.risk_flags)
        has_date = any(evidence.get("date") for result in results for evidence in result.evidence)
        complete = (
            bool(coverage["complete"])
            and not failed_states
            and not conflicts
            and not open_questions
            and (has_primary or bool(results))
            and (not needs_recency or has_date)
        )
        notes = []
        if not has_primary:
            notes.append("缺少一手来源，已降低置信度。")
        if needs_recency and not has_date:
            notes.append("缺少发布日期或数据周期。")
        return CoverageReport(
            complete=complete,
            covered=list(coverage["covered"]),
            missing=list(coverage["missing"]),
            failed_states=failed_states,
            open_questions=open_questions,
            conflicts=conflicts,
            missing_authoritative_sources=not has_primary,
            missing_recency=needs_recency and not has_date,
            notes=notes,
        )

    def _has_sufficient_final_answer(
        self,
        plan: ResearchPlan,
        graph: ResearchStateGraph,
        results: list[WorkerResult],
        coverage: CoverageReport,
    ) -> bool:
        allowed_missing = {"verification", "official", "recency", "source", "sources"}
        if any(item not in allowed_missing for item in coverage.missing):
            return False
        states_by_id = {state.state_id: state for state in graph.states()}
        for result in results:
            if result.normalized_status() != "resolved":
                continue
            candidate = (result.candidate_answer or "").strip()
            if not candidate or candidate.lower() in {"unknown", "n/a", "none", "null"}:
                continue
            if self._answer_relevant_conflicts([result], result.conflicts):
                continue
            state = states_by_id.get(result.state_id)
            state_tags = set(state.coverage_tags if state else [])
            expected_output = (state.expected_output if state else "").lower()
            answer_state = (
                "final_answer" in state_tags
                or "answer_format" in state_tags
                or "candidate_answer" in expected_output
                or "final" in expected_output
                or bool(result.answer_form_hint)
            )
            if not answer_state:
                continue
            if self._max_evidence_confidence(result) < 0.75:
                continue
            return True
        return False

    def _answer_relevant_conflicts(
        self,
        results: list[WorkerResult],
        conflicts: list[str],
    ) -> list[str]:
        if not conflicts:
            return []
        candidates = self._resolved_candidate_terms(results)
        if not candidates:
            return conflicts
        return [
            conflict
            for conflict in conflicts
            if any(self._mentions_term(conflict, candidate) for candidate in candidates)
        ]

    def _resolved_candidate_terms(self, results: list[WorkerResult]) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for result in results:
            if result.normalized_status() != "resolved":
                continue
            if self._max_evidence_confidence(result) < 0.75:
                continue
            raw_terms = [result.candidate_answer, *result.canonical_names]
            for term in raw_terms:
                normalized = (term or "").strip()
                key = normalized.lower()
                if len(normalized) < 3 or key in {"unknown", "n/a", "none", "null"} or key in seen:
                    continue
                terms.append(normalized)
                seen.add(key)
        return terms

    @staticmethod
    def _mentions_term(text: str, term: str) -> bool:
        lowered = text.lower()
        needle = term.lower().strip()
        if not needle:
            return False
        if re.search(r"[\u4e00-\u9fff]", needle):
            return needle in lowered
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", lowered) is not None

    @staticmethod
    def _max_evidence_confidence(result: WorkerResult) -> float:
        if not result.evidence:
            return 0.0
        return max(confidence_to_float(evidence.get("confidence")) for evidence in result.evidence)

    async def _replan(
        self,
        query: str,
        plan: ResearchPlan,
        graph: ResearchStateGraph,
        results: list[WorkerResult],
        coverage: CoverageReport,
    ) -> None:
        successful = [result.to_dict() for result in results if result.normalized_status() == "resolved"]
        failed_states = []
        for state in graph.failed_states():
            payload = state.to_dict()
            payload["failure_reason"] = state.metadata.get("failure_reason", "")
            failed_states.append(payload)
        memory_context = self.memory_store.get_context_for_query(query, max_tokens=2500)
        new_states = await self.planner.replan(
            query=query,
            plan=plan,
            successful_results=successful,
            failed_states=failed_states,
            coverage_gaps=coverage.missing,
            conflicts=coverage.conflicts,
            memory_context=memory_context,
        )
        existing = {state.state_id for state in graph.states()}
        for state in new_states:
            if state.state_id in existing:
                state.state_id = f"{state.state_id}_replan_{self.num_replans}"
            graph.add_state(state)

    def _transition(self, state: FSMState, detail: dict[str, Any] | None = None) -> None:
        event = StateHistoryEvent(state=state.name, detail=detail or {}).to_dict()
        self.state_history.append(event)
        self._emit({"type": "state", **event})

    def _emit(self, event: dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event)
        except Exception:
            pass

    def _metadata(
        self,
        *,
        elapsed_seconds: float,
        num_states: int,
        adversarial_rounds: int,
        failed: bool = False,
    ) -> dict[str, Any]:
        return {
            "mode": "deep_research",
            "session_id": self.session_id,
            "num_states": num_states,
            "num_searches": self.num_searches,
            "num_replans": self.num_replans,
            "adversarial_rounds": adversarial_rounds,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "failed": failed,
            "llm_usage": self._llm_usage(),
            "compression": self.compressor.get_stats(),
        }

    def _llm_usage(self) -> dict[str, Any]:
        getter = getattr(self.llm, "get_usage", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter())
        except Exception:
            return {}

    @staticmethod
    def _failed_report(query: str, exc: Exception, results: list[WorkerResult]) -> str:
        lines = [
            f"# 研究报告：{query}",
            "",
            "## 执行摘要",
            "研究流程未能完整完成，以下为错误说明和已获得的有限证据。",
            "",
            "## 证据质量与不确定性",
            f"- 错误：{type(exc).__name__}: {exc}",
        ]
        for result in results:
            if result.summary:
                lines.append(f"- {result.state_id}: {result.summary}")
        return "\n".join(lines)
