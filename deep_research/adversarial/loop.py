from __future__ import annotations

from copy import deepcopy
from typing import Any

from deep_research.adversarial.blue_agent import BlueAgent
from deep_research.adversarial.red_agent import RedAgent
from deep_research.adversarial.verdict import RedIssue
from deep_research.schemas import ResearchReport


class AdversarialLoop:
    def __init__(
        self,
        red_agent: RedAgent,
        blue_agent: BlueAgent,
        *,
        max_rounds: int = 2,
        score_threshold: float = 8.0,
        delta_threshold: float = 0.3,
    ) -> None:
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.max_rounds = max(1, max_rounds)
        self.score_threshold = score_threshold
        self.delta_threshold = delta_threshold

    async def run(self, report: ResearchReport) -> tuple[ResearchReport, list[dict[str, Any]]]:
        current = deepcopy(report)
        history: list[dict[str, Any]] = []
        previous_score: float | None = None
        seen_issues: set[RedIssue] = set()

        for round_index in range(1, self.max_rounds + 1):
            verdict = await self.red_agent.attack(current)
            repeated = bool(seen_issues.intersection(set(verdict.issues)))
            record = {
                "round": round_index,
                "overall_score": round(verdict.overall_score, 3),
                "dimension_scores": verdict.dimension_scores,
                "issues": [issue.to_dict() for issue in verdict.issues],
                "oscillation_detected": repeated,
                "stop_reason": "",
                "changes": [],
                "remaining_risks": [],
                "raw_feedback": verdict.raw_feedback,
            }
            if repeated:
                record["stop_reason"] = "oscillation_detected"
                history.append(record)
                break
            if verdict.overall_score >= self.score_threshold or verdict.passed:
                record["stop_reason"] = "score_threshold_met"
                history.append(record)
                break
            if previous_score is not None and abs(verdict.overall_score - previous_score) < self.delta_threshold:
                record["stop_reason"] = "delta_converged"
                history.append(record)
                break

            fixed_report, fix_info = await self.blue_agent.defend(current, verdict)
            current = fixed_report
            seen_issues.update(verdict.issues)
            previous_score = verdict.overall_score
            record["changes"] = fix_info.get("changes", [])
            record["remaining_risks"] = fix_info.get("remaining_risks", [])
            if round_index >= self.max_rounds:
                record["stop_reason"] = "max_rounds_reached"
            history.append(record)

        current.critique_history = history
        if history:
            current.confidence = max(current.confidence, min(1.0, history[-1]["overall_score"] / 10.0))
        return current, history
