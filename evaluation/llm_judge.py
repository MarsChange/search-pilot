from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deep_research.llm import DashScopeLLM, extract_json_object
from deep_research.runner import DeepResearchRunner


JUDGE_SYSTEM_PROMPT = """你是多跳问答评测裁判。你的任务是判断模型预测答案和标准答案是否指向同一个对象。

重要原则：
1. 不要使用严格字符串匹配，也不要要求字符级正则化一致。
2. 如果预测答案是标准答案的别名、常见缩写、翻译名、姓名称呼变体、顺序变体，且在题目语境下明确指向同一个对象，应判为 correct。
3. 如果预测答案是相关对象、上位概念、下位对象、线索中的中间实体、同名但不同对象、只回答了部分名称，应判为 incorrect。
4. 如果预测为空、明显不确定、给出多个互斥候选且未选择，应判为 incorrect。
5. 标准答案是最终裁判锚点；你只需要判断“是否同一个对象”，不是评价推理过程。
6. 不要输出 chain-of-thought。输出严格 JSON。

输出格式：
{
  "is_same_object": true,
  "judgement": "correct|incorrect|uncertain",
  "canonical_gold": "...",
  "canonical_prediction": "...",
  "reason": "一句话说明"
}"""


class JudgeLLM(Protocol):
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        ...


async def judge_one(
    *,
    llm: JudgeLLM,
    question: str,
    gold_answer: str,
    predicted_answer: str,
) -> dict[str, Any]:
    prompt = f"""题目：
{question}

标准答案：
{gold_answer}

模型预测答案：
{predicted_answer}

请判断标准答案和预测答案是否是同一个对象。"""
    raw = await llm.complete(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    data = extract_json_object(raw) or {}
    is_same = bool(data.get("is_same_object", False))
    judgement = str(data.get("judgement") or ("correct" if is_same else "incorrect")).lower()
    if judgement not in {"correct", "incorrect", "uncertain"}:
        judgement = "correct" if is_same else "incorrect"
    return {
        "is_same_object": is_same,
        "judgement": judgement,
        "canonical_gold": str(data.get("canonical_gold", gold_answer)),
        "canonical_prediction": str(data.get("canonical_prediction", predicted_answer)),
        "reason": str(data.get("reason", "")),
        "raw_judge": raw,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_text(row: dict[str, Any]) -> str:
    for key in ("predicted_answer", "prediction", "answer", "model_answer"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    response = row.get("response")
    if isinstance(response, dict):
        value = response.get("answer")
        if isinstance(value, str):
            return value
    return ""


def index_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for idx, row in enumerate(rows):
        key = str(row.get("id", idx))
        indexed[key] = row
    return indexed


async def run_agent_for_gold(
    rows: list[dict[str, Any]],
    *,
    max_concurrent: int,
    max_replans: int,
    max_adversarial_rounds: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            runner = DeepResearchRunner(
                session_id=f"eval-{row.get('id', '')}-{int(time.time())}",
                max_concurrent=max_concurrent,
                max_replans=max_replans,
                max_adversarial_rounds=max_adversarial_rounds,
            )
            result = await runner.run(str(row["question"]))
            payload = result.to_dict()
            return {
                "id": row.get("id"),
                "question": row.get("question"),
                "gold_answer": row.get("answer"),
                "predicted_answer": payload["answer"],
                "response": payload,
            }

    return await asyncio.gather(*(run_one(row) for row in rows))


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gold_rows = load_jsonl(args.gold)
    if args.start:
        gold_rows = gold_rows[args.start :]
    if args.limit is not None:
        gold_rows = gold_rows[: args.limit]

    if args.run_agent:
        pred_rows = await run_agent_for_gold(
            gold_rows,
            max_concurrent=args.agent_concurrency,
            max_replans=args.max_replans,
            max_adversarial_rounds=args.max_adversarial_rounds,
        )
        if args.predictions_out:
            write_jsonl(args.predictions_out, pred_rows)
    else:
        if not args.predictions:
            raise ValueError("--predictions is required unless --run-agent is set")
        pred_rows = load_jsonl(args.predictions)

    pred_by_id = index_predictions(pred_rows)
    llm = DashScopeLLM(model=args.judge_model or os.getenv("JUDGE_MODEL") or os.getenv("QWEN_MODEL", "qwen-max"))
    semaphore = asyncio.Semaphore(args.judge_concurrency)

    async def judge_row(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
        row_id = str(row.get("id", ordinal))
        pred_row = pred_by_id.get(row_id, {})
        pred = prediction_text(pred_row)
        async with semaphore:
            verdict = await judge_one(
                llm=llm,
                question=str(row.get("question", "")),
                gold_answer=str(row.get("answer", "")),
                predicted_answer=pred,
            )
        return {
            "id": row.get("id", ordinal),
            "question": row.get("question", ""),
            "gold_answer": row.get("answer", ""),
            "predicted_answer": pred,
            **verdict,
        }

    judged = await asyncio.gather(*(judge_row(row, idx) for idx, row in enumerate(gold_rows)))
    correct = sum(1 for row in judged if row["judgement"] == "correct" and row["is_same_object"])
    total = len(judged)
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "output": args.output,
    }
    write_jsonl(args.output, judged)
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluator for test/data_with_answer.jsonl multi-hop QA.")
    parser.add_argument("--gold", default="test/data_with_answer.jsonl", help="Gold JSONL with id/question/answer.")
    parser.add_argument("--predictions", help="Prediction JSONL with id and predicted_answer/prediction/answer.")
    parser.add_argument("--output", default="outputs/judge_results.jsonl", help="Output JSONL for per-item judge results.")
    parser.add_argument("--predictions-out", help="Where to save predictions when --run-agent is used.")
    parser.add_argument("--run-agent", action="store_true", help="Run the current DeepResearchRunner before judging.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows after --start.")
    parser.add_argument("--start", type=int, default=0, help="Start offset in gold file.")
    parser.add_argument("--judge-model", help="Judge model name. Defaults to JUDGE_MODEL or QWEN_MODEL.")
    parser.add_argument("--judge-concurrency", type=int, default=2)
    parser.add_argument("--agent-concurrency", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--max-adversarial-rounds", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(evaluate(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
