from __future__ import annotations

import time
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class FSMState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    COLLECTING = "collecting"
    COVERAGE_CHECK = "coverage_check"
    REPLANNING = "replanning"
    SYNTHESIZING = "synthesizing"
    ADVERSARIAL_REVIEW = "adversarial_review"
    FINALIZING = "finalizing"
    DONE = "done"
    FAILED = "failed"


class ResearchStateType(Enum):
    SEARCH = "search"
    ANALYZE = "analyze"
    VERIFY = "verify"
    BACKTRACK = "backtrack"
    SYNTHESIZE = "synthesize"


class ResearchStateStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


CONFIDENCE_TO_FLOAT = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.35,
}

EVIDENCE_TYPE_WEIGHT = {
    "primary": 1.0,
    "secondary": 0.8,
    "inference": 0.55,
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: jsonable(val) for key, val in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def confidence_to_float(value: str | float | int | None) -> float:
    if isinstance(value, (float, int)):
        return clamp01(float(value))
    return CONFIDENCE_TO_FLOAT.get(str(value or "medium").lower(), 0.6)


@dataclass
class ResearchState:
    state_id: str
    state_type: ResearchStateType | str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: ResearchStateStatus | str = ResearchStateStatus.PENDING
    search_queries: list[str] = field(default_factory=list)
    expected_output: str = "facts"
    coverage_tags: list[str] = field(default_factory=list)
    priority: int = 1
    retry_count: int = 0
    parent_state_id: str | None = None
    backtrack_reason: str | None = None
    timeout_seconds: int = 90
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state_type, ResearchStateType):
            self.state_type = ResearchStateType(str(self.state_type))
        if not isinstance(self.status, ResearchStateStatus):
            self.status = ResearchStateStatus(str(self.status))

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchState":
        return cls(
            state_id=str(data.get("state_id") or data.get("task_id") or ""),
            state_type=data.get("state_type", "search"),
            description=str(data.get("description", "")),
            dependencies=list(data.get("dependencies", [])),
            status=data.get("status", "pending"),
            search_queries=list(data.get("search_queries") or data.get("search_hints") or []),
            expected_output=str(data.get("expected_output", "facts")),
            coverage_tags=list(data.get("coverage_tags", [])),
            priority=int(data.get("priority", 1)),
            retry_count=int(data.get("retry_count", 0)),
            parent_state_id=data.get("parent_state_id"),
            backtrack_reason=data.get("backtrack_reason"),
            timeout_seconds=int(data.get("timeout_seconds", 90)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ResearchPlan:
    query: str
    research_intent: str
    scope: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    coverage_checklist: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    states: list[ResearchState] = field(default_factory=list)
    raw_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class WorkerResult:
    status: str
    state_id: str
    summary: str = ""
    candidate_answer: str = ""
    key_findings: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recommended_followups: list[str] = field(default_factory=list)
    canonical_names: list[str] = field(default_factory=list)
    answer_form_hint: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def normalized_status(self) -> str:
        status = str(self.status or "partial").lower()
        if status in {"resolved", "partial", "failed"}:
            return status
        return "partial"

    def to_dict(self) -> dict[str, Any]:
        data = jsonable(self)
        data["status"] = self.normalized_status()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerResult":
        return cls(
            status=str(data.get("status", "partial")),
            state_id=str(data.get("state_id", "")),
            summary=str(data.get("summary", "")),
            candidate_answer=str(data.get("candidate_answer") or data.get("subtask_answer") or ""),
            key_findings=list(data.get("key_findings", [])),
            evidence=list(data.get("evidence", [])),
            conflicts=list(data.get("conflicts", [])),
            open_questions=list(data.get("open_questions", [])),
            recommended_followups=list(data.get("recommended_followups", [])),
            canonical_names=list(data.get("canonical_names", [])),
            answer_form_hint=str(data.get("answer_form_hint", "")),
            tool_calls=list(data.get("tool_calls", [])),
            error=data.get("error"),
        )


@dataclass
class CoverageReport:
    complete: bool
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed_states: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing_authoritative_sources: bool = False
    missing_recency: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class ResearchReport:
    query: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    state_history: list[dict[str, Any]] = field(default_factory=list)
    states: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    critique_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = jsonable(self)
        data["confidence"] = round(clamp01(self.confidence), 3)
        return data


@dataclass
class DeepResearchResult:
    answer: str
    report: ResearchReport | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "report": self.report.to_dict() if self.report else None,
            "metadata": jsonable(self.metadata),
        }


@dataclass
class StateHistoryEvent:
    state: str
    timestamp: float = field(default_factory=time.time)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)
