from __future__ import annotations

import json
import re
from typing import Any

from deep_research.llm import LLMClient, extract_json_object
from deep_research.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    REPLANNER_SYSTEM_PROMPT,
)
from deep_research.schemas import ResearchPlan, ResearchState, ResearchStateType


class DeepResearchPlanner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def create_plan(self, query: str, memory_context: str = "") -> ResearchPlan:
        prompt = PLANNER_USER_TEMPLATE.format(
            query=query,
            memory_context=memory_context or "无",
        )
        try:
            raw = await self.llm.complete(
                [
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            data = extract_json_object(raw)
            if data:
                return self._parse_plan(query, data)
        except Exception:
            pass
        return self._fallback_plan(query)

    async def replan(
        self,
        *,
        query: str,
        plan: ResearchPlan,
        successful_results: list[dict[str, Any]],
        failed_states: list[dict[str, Any]],
        coverage_gaps: list[str],
        conflicts: list[str],
        memory_context: str,
    ) -> list[ResearchState]:
        payload = {
            "original_query": query,
            "coverage_checklist": plan.coverage_checklist,
            "successful_results": successful_results,
            "failed_states": failed_states,
            "coverage_gaps": coverage_gaps,
            "conflicts": conflicts,
            "memory_context": memory_context,
        }
        try:
            raw = await self.llm.complete(
                [
                    {"role": "system", "content": REPLANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "请基于以下状态输出新增或替换 states：\n"
                        + json.dumps(payload, ensure_ascii=False, indent=2),
                    },
                ],
                temperature=0.1,
            )
            data = extract_json_object(raw)
            if data:
                states_raw = data.get("states") or data.get("new_states") or data.get("replacements") or []
                states = self._parse_states(states_raw, prefix="replan")
                if states:
                    return states
        except Exception:
            pass
        return self._fallback_replan_states(
            query,
            coverage_gaps,
            conflicts,
            failed_states,
            successful_results=successful_results,
        )

    def _parse_plan(self, query: str, data: dict[str, Any]) -> ResearchPlan:
        states = self._parse_states(data.get("states", []), prefix="state")
        if not any(state.state_type == ResearchStateType.VERIFY for state in states):
            states.append(
                ResearchState(
                    state_id="state_verify",
                    state_type=ResearchStateType.VERIFY,
                    description="确认最终候选答案的名称和答案格式",
                    dependencies=[states[0].state_id] if states else [],
                    search_queries=["final answer candidate official name"],
                    expected_output="facts",
                    coverage_tags=["verification"],
                    priority=99,
                )
            )
        return ResearchPlan(
            query=query,
            research_intent=str(data.get("research_intent", query)),
            scope=dict(data.get("scope", {})),
            success_criteria=list(data.get("success_criteria", [])),
            coverage_checklist=list(data.get("coverage_checklist", [])) or self._infer_coverage(query),
            risk_flags=list(data.get("risk_flags", [])),
            states=states,
            raw_plan=data,
        )

    def _parse_states(self, states_raw: Any, prefix: str) -> list[ResearchState]:
        if not isinstance(states_raw, list):
            return []
        states = []
        seen: set[str] = set()
        for idx, item in enumerate(states_raw, 1):
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item.setdefault("state_id", f"{prefix}_{idx}")
            if item["state_id"] in seen:
                item["state_id"] = f"{item['state_id']}_{idx}"
            seen.add(item["state_id"])
            try:
                state = ResearchState.from_dict(item)
            except Exception:
                continue
            if not state.description:
                continue
            if not state.search_queries:
                state.search_queries = [state.description]
            states.append(state)
        return states

    def _fallback_plan(self, query: str) -> ResearchPlan:
        states = [
            ResearchState(
                state_id="state_1",
                state_type=ResearchStateType.SEARCH,
                description=f"检索与研究问题直接相关的权威来源：{query}",
                dependencies=[],
                search_queries=[query],
                expected_output="facts",
                coverage_tags=["official", "background"],
                priority=1,
            ),
            ResearchState(
                state_id="state_2",
                state_type=ResearchStateType.VERIFY,
                description="确认最终候选答案的名称和答案格式",
                dependencies=["state_1"],
                search_queries=["final answer candidate official name"],
                expected_output="facts",
                coverage_tags=["verification"],
                priority=2,
            ),
        ]
        return ResearchPlan(
            query=query,
            research_intent=query,
            scope={"time_range": "未指定", "region": "未指定", "entities": [], "decision_context": "业务研究"},
            success_criteria=["直接回答研究问题", "关键事实有来源"],
            coverage_checklist=self._infer_coverage(query),
            risk_flags=["planner_fallback"],
            states=states,
            raw_plan={"fallback": True},
        )

    def _fallback_replan_states(
        self,
        query: str,
        coverage_gaps: list[str],
        conflicts: list[str],
        failed_states: list[dict[str, Any]],
        successful_results: list[dict[str, Any]] | None = None,
    ) -> list[ResearchState]:
        states: list[ResearchState] = []
        for idx, gap in enumerate(coverage_gaps, 1):
            states.append(
                ResearchState(
                    state_id=f"replan_gap_{idx}",
                    state_type=ResearchStateType.SEARCH,
                    description=f"补充覆盖缺口：{gap}",
                    dependencies=[],
                    search_queries=[f"{gap} official evidence"],
                    coverage_tags=[gap],
                    priority=1,
                    parent_state_id=None,
                    backtrack_reason=f"coverage gap: {gap}",
                )
            )
        relevant_conflicts = self._filter_conflicts_by_successful_candidates(conflicts, successful_results or [])
        for idx, conflict in enumerate(relevant_conflicts, 1):
            states.append(
                ResearchState(
                    state_id=f"verify_conflict_{idx}",
                    state_type=ResearchStateType.VERIFY,
                    description=f"核查矛盾事实：{conflict}",
                    dependencies=[],
                    search_queries=["conflict verification authoritative source"],
                    coverage_tags=["verification"],
                    priority=0,
                    backtrack_reason=f"conflict: {conflict}",
                )
            )
        for idx, failed in enumerate(failed_states, 1):
            states.append(
                ResearchState(
                    state_id=f"retry_failed_{idx}",
                    state_type=ResearchStateType.SEARCH,
                    description=f"重试失败任务：{failed.get('description', query)}",
                    dependencies=[],
                    search_queries=[str(failed.get("description") or query)],
                    coverage_tags=list(failed.get("coverage_tags", [])),
                    priority=2,
                    backtrack_reason=str(failed.get("failure_reason", "failed state")),
                )
            )
        return states

    @classmethod
    def _filter_conflicts_by_successful_candidates(
        cls,
        conflicts: list[str],
        successful_results: list[dict[str, Any]],
    ) -> list[str]:
        candidates = cls._successful_candidate_terms(successful_results)
        if not candidates:
            return conflicts
        return [
            conflict
            for conflict in conflicts
            if any(cls._mentions_term(conflict, candidate) for candidate in candidates)
        ]

    @staticmethod
    def _successful_candidate_terms(successful_results: list[dict[str, Any]]) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for result in successful_results:
            if str(result.get("status", "")).lower() not in {"resolved", "success"}:
                continue
            raw_terms = [str(result.get("candidate_answer") or "")]
            raw_terms.extend(str(item) for item in result.get("canonical_names", []) or [])
            for term in raw_terms:
                normalized = term.strip()
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
    def _infer_coverage(query: str) -> list[str]:
        lowered = query.lower()
        checklist = ["official", "verification"]
        if any(word in query for word in ["风险", "合规", "监管"]) or "risk" in lowered:
            checklist.append("risk")
        if any(word in query for word in ["竞品", "竞争", "对比"]) or "competitor" in lowered:
            checklist.append("competitor")
        if any(word in query for word in ["最新", "当前", "近一年", "today", "current"]):
            checklist.append("recency")
        return checklist
