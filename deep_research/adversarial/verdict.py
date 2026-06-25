from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DIMENSION_KEYS = [
    "factual_accuracy",
    "hallucination_risk",
    "citation_quality",
    "logical_consistency",
    "coverage",
    "recency",
    "business_usefulness",
]


@dataclass(frozen=True)
class RedIssue:
    severity: str
    dimension: str
    location: str
    problem: str
    required_fix: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedIssue":
        return cls(
            severity=str(data.get("severity", "minor")),
            dimension=str(data.get("dimension", "")),
            location=str(data.get("location", "")),
            problem=str(data.get("problem", "")),
            required_fix=str(data.get("required_fix", "clarify_uncertainty")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "dimension": self.dimension,
            "location": self.location,
            "problem": self.problem,
            "required_fix": self.required_fix,
        }


@dataclass
class RedVerdict:
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    issues: list[RedIssue] = field(default_factory=list)
    passed: bool = False
    raw_feedback: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], raw_feedback: str = "") -> "RedVerdict":
        scores = {
            key: float(data.get("dimension_scores", {}).get(key, 0.0))
            for key in DIMENSION_KEYS
        }
        overall = float(data.get("overall_score", 0.0))
        if not overall and scores:
            overall = sum(scores.values()) / len(scores)
        return cls(
            overall_score=max(0.0, min(10.0, overall)),
            dimension_scores=scores,
            issues=[
                RedIssue.from_dict(item)
                for item in data.get("issues", [])
                if isinstance(item, dict)
            ],
            passed=bool(data.get("pass", False)),
            raw_feedback=raw_feedback,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "dimension_scores": self.dimension_scores,
            "issues": [issue.to_dict() for issue in self.issues],
            "pass": self.passed,
            "raw_feedback": self.raw_feedback,
        }
