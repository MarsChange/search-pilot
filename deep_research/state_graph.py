from __future__ import annotations

import itertools
import time
from typing import Any

from deep_research.schemas import (
    ResearchState,
    ResearchStateStatus,
    ResearchStateType,
)


class ResearchStateGraph:
    """Dynamic research state graph with dependency-aware scheduling."""

    def __init__(self) -> None:
        self._states: dict[str, ResearchState] = {}
        self.state_history: list[dict[str, Any]] = []
        self._generated_counter = itertools.count(1)

    def add_state(self, state: ResearchState) -> None:
        if not state.state_id:
            state.state_id = self._next_id(state.state_type.value)
        self._states[state.state_id] = state
        self._record("add_state", state.state_id, {"type": state.state_type.value})

    def add_states(self, states: list[ResearchState]) -> None:
        for state in states:
            self.add_state(state)

    def get_state(self, state_id: str) -> ResearchState:
        return self._states[state_id]

    def states(self) -> list[ResearchState]:
        return list(self._states.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [state.to_dict() for state in self.states()],
            "state_history": list(self.state_history),
        }

    def ready_states(self, limit: int | None = None) -> list[ResearchState]:
        ready = []
        for state in self._states.values():
            if state.status != ResearchStateStatus.PENDING:
                continue
            if all(self._dependency_succeeded(dep) for dep in state.dependencies):
                ready.append(state)
        ready.sort(key=lambda item: (item.priority, item.state_id))
        return ready[:limit] if limit is not None else ready

    def has_pending(self) -> bool:
        return any(state.status == ResearchStateStatus.PENDING for state in self._states.values())

    def all_terminal(self) -> bool:
        return all(
            state.status in {
                ResearchStateStatus.SUCCESS,
                ResearchStateStatus.FAILED,
                ResearchStateStatus.SKIPPED,
            }
            for state in self._states.values()
        )

    def mark_running(self, state_id: str) -> None:
        self._set_status(state_id, ResearchStateStatus.RUNNING)

    def mark_success(self, state_id: str) -> None:
        self._set_status(state_id, ResearchStateStatus.SUCCESS)

    def mark_failed(self, state_id: str, reason: str = "") -> None:
        state = self._states[state_id]
        state.retry_count += 1
        state.metadata["failure_reason"] = reason
        self._set_status(state_id, ResearchStateStatus.FAILED, {"reason": reason})

    def mark_skipped(self, state_id: str, reason: str = "") -> None:
        self._states[state_id].metadata["skip_reason"] = reason
        self._set_status(state_id, ResearchStateStatus.SKIPPED, {"reason": reason})

    def failure_rate(self) -> float:
        terminal = [
            state for state in self._states.values()
            if state.status in {
                ResearchStateStatus.SUCCESS,
                ResearchStateStatus.FAILED,
                ResearchStateStatus.SKIPPED,
            }
        ]
        if not terminal:
            return 0.0
        failed = sum(1 for state in terminal if state.status == ResearchStateStatus.FAILED)
        return failed / len(terminal)

    def should_replan(self, failure_threshold: float = 0.5) -> bool:
        return self.failure_rate() > failure_threshold

    def coverage_status(self, checklist: list[str]) -> dict[str, list[str] | bool]:
        expected = [item for item in checklist if item]
        covered_set = set()
        for state in self._states.values():
            if state.status == ResearchStateStatus.SUCCESS:
                covered_set.update(state.coverage_tags)
        covered = [item for item in expected if item in covered_set]
        missing = [item for item in expected if item not in covered_set]
        return {"complete": not missing, "covered": covered, "missing": missing}

    def failed_states(self) -> list[ResearchState]:
        return [state for state in self._states.values() if state.status == ResearchStateStatus.FAILED]

    def add_verify_state(
        self,
        *,
        description: str,
        search_queries: list[str],
        coverage_tags: list[str],
        parent_state_id: str | None,
        reason: str,
    ) -> ResearchState:
        state = ResearchState(
            state_id=self._next_id("verify"),
            state_type=ResearchStateType.VERIFY,
            description=description,
            dependencies=[],
            search_queries=search_queries,
            expected_output="facts",
            coverage_tags=coverage_tags,
            priority=1,
            parent_state_id=parent_state_id,
            backtrack_reason=reason,
        )
        self.add_state(state)
        return state

    def add_backtrack_state(
        self,
        *,
        description: str,
        search_queries: list[str],
        coverage_tags: list[str],
        parent_state_id: str | None,
        reason: str,
    ) -> ResearchState:
        state = ResearchState(
            state_id=self._next_id("backtrack"),
            state_type=ResearchStateType.BACKTRACK,
            description=description,
            dependencies=[],
            search_queries=search_queries,
            expected_output="facts",
            coverage_tags=coverage_tags,
            priority=0,
            parent_state_id=parent_state_id,
            backtrack_reason=reason,
        )
        self.add_state(state)
        return state

    def _dependency_succeeded(self, state_id: str) -> bool:
        state = self._states.get(state_id)
        return bool(state and state.status == ResearchStateStatus.SUCCESS)

    def _set_status(
        self,
        state_id: str,
        status: ResearchStateStatus,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._states[state_id].status = status
        self._record("status", state_id, {"status": status.value, **(detail or {})})

    def _record(self, event: str, state_id: str, detail: dict[str, Any] | None = None) -> None:
        self.state_history.append(
            {
                "event": event,
                "state_id": state_id,
                "timestamp": time.time(),
                "detail": detail or {},
            }
        )

    def _next_id(self, prefix: str) -> str:
        candidate = f"{prefix}_{next(self._generated_counter)}"
        while candidate in self._states:
            candidate = f"{prefix}_{next(self._generated_counter)}"
        return candidate
