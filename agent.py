from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from deep_research.llm import DashScopeLLM
from deep_research.runtime_logging import emit_runtime_log
from deep_research.runner import DeepResearchRunner
from evaluation.llm_judge import judge_one


app = FastAPI(title="Tianchi Deep Research Agent")
PROJECT_ROOT = Path(__file__).resolve().parent
VISUALIZER_HTML = PROJECT_ROOT / "web" / "research_visualizer.html"


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "question": "请研究某行业近一年市场变化、竞品和风险",
                "deep_research": True,
                "session_id": "demo-session",
                "max_concurrent": 3,
                "max_replans": 2,
                "max_adversarial_rounds": 2,
                "return_markdown_only": False,
            }
        },
    )

    question: str
    deep_research: bool = True
    session_id: str | None = None
    max_concurrent: int = 3
    max_replans: int = 2
    max_adversarial_rounds: int = 2
    return_markdown_only: bool = False
    memory_db_path: str | None = None


class QueryResponse(BaseModel):
    answer: str
    report: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class EvaluationRequest(BaseModel):
    dataset_path: str = str(PROJECT_ROOT / "test" / "data_with_answer.jsonl")
    start: int = 0
    limit: int | None = 200
    max_concurrent: int = 3
    max_replans: int = 2
    max_adversarial_rounds: int = 0
    judge_model: str | None = None
    checkpoint_path: str | None = None
    resume_from_checkpoint: bool = True
    reset_checkpoint: bool = False
    rerun_incorrect_checkpoint_items: bool = False


def _build_runner(req: QueryRequest, event_sink=None) -> DeepResearchRunner:
    return DeepResearchRunner(
        session_id=req.session_id,
        max_concurrent=req.max_concurrent,
        max_replans=req.max_replans,
        max_adversarial_rounds=req.max_adversarial_rounds,
        memory_db_path=req.memory_db_path,
        event_sink=event_sink,
    )


@app.post("/", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse | StreamingResponse:
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(_stream_result(req), media_type="text/event-stream")

    result = await _run_request(req)
    return _response_from_result(req, result)


@app.post("/stream")
async def stream(req: QueryRequest) -> StreamingResponse:
    return StreamingResponse(_stream_result(req), media_type="text/event-stream")


@app.get("/visualize", response_class=HTMLResponse)
async def visualize() -> HTMLResponse:
    return HTMLResponse(VISUALIZER_HTML.read_text(encoding="utf-8"))


@app.get("/env-status")
async def env_status() -> dict[str, Any]:
    return {
        "python": _python_status(),
        "conda": _conda_status(),
        "env": _api_key_status(),
        "run_command": "conda run -n tianchi_agent python -m uvicorn agent:app --reload --host 0.0.0.0 --port 8000",
    }


@app.get("/dataset-info")
async def dataset_info(path: str | None = None) -> dict[str, Any]:
    dataset_path = Path(path or PROJECT_ROOT / "test" / "data_with_answer.jsonl")
    rows = _load_jsonl(dataset_path)
    return {
        "path": str(dataset_path),
        "count": len(rows),
        "sample": rows[:3],
    }


@app.post("/evaluate-stream")
async def evaluate_stream(req: EvaluationRequest) -> StreamingResponse:
    return StreamingResponse(_stream_evaluation(req), media_type="text/event-stream")


@app.post("/ag-ui")
async def ag_ui(payload: dict[str, Any]) -> StreamingResponse:
    """Compatibility endpoint that maps AG-UI-like payloads to a Deep Research run."""
    question = _extract_question_from_agui(payload)
    req = QueryRequest(question=question or "", deep_research=True)
    return StreamingResponse(_stream_result(req), media_type="text/event-stream")


async def _run_request(req: QueryRequest, event_sink=None):
    if not req.deep_research:
        # The old implicit coordinator is intentionally removed. Keep the API stable by
        # still running the deterministic Deep Research pipeline with a small budget.
        req = req.model_copy(update={"max_replans": 0, "max_adversarial_rounds": 0})
    runner = _build_runner(req, event_sink=event_sink)
    return await runner.run(req.question)


def _response_from_result(req: QueryRequest, result) -> QueryResponse:
    payload = result.to_dict()
    if req.return_markdown_only:
        return QueryResponse(
            answer=payload["answer"],
            report=None,
            metadata=payload.get("metadata"),
        )
    return QueryResponse(
        answer=payload["answer"],
        report=payload.get("report"),
        metadata=payload.get("metadata"),
    )


async def _stream_result(req: QueryRequest):
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def event_sink(event: dict[str, Any]) -> None:
        queue.put_nowait(event)

    async def run_and_finish() -> None:
        result = await _run_request(req, event_sink=event_sink)
        payload = result.to_dict()
        queue.put_nowait(
            {
                "type": "message",
                "answer": payload["answer"],
                "report": None if req.return_markdown_only else payload.get("report"),
                "metadata": payload.get("metadata"),
            }
        )
        queue.put_nowait(None)

    task = asyncio.create_task(run_and_finish())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield _sse(event)
    finally:
        await task


def _sse(event: dict[str, Any]) -> str:
    event_name = _event_name(str(event.get("type", "message")))
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event_name}\ndata: {data}\n\n"


def _event_name(event_type: str) -> str:
    return {
        "state": "State",
        "plan": "Plan",
        "dispatch": "Dispatch",
        "state_start": "TaskStart",
        "state_result": "TaskResult",
        "coverage": "Coverage",
        "replan_start": "Replan",
        "replan_result": "Replan",
        "final": "Final",
        "message": "Message",
        "eval_start": "EvalStart",
        "eval_checkpoint": "EvalCheckpoint",
        "eval_run_event": "EvalRunEvent",
        "eval_item": "EvalItem",
        "eval_summary": "EvalSummary",
        "eval_error": "EvalError",
    }.get(event_type, "Event")


async def _stream_evaluation(req: EvaluationRequest):
    dataset_path = Path(req.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    checkpoint_path = _eval_checkpoint_path(req, dataset_path)
    if req.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = _load_eval_checkpoint(checkpoint_path) if req.resume_from_checkpoint else {}
    rows = _load_jsonl(req.dataset_path)
    rows = rows[max(0, req.start) :]
    if req.limit is not None:
        rows = rows[: max(0, req.limit)]
    total = len(rows)
    correct = 0
    done_count = 0
    started = time.time()
    judge_llm = DashScopeLLM(model=req.judge_model) if req.judge_model else DashScopeLLM()
    yield _sse(
        {
            "type": "eval_start",
            "total": total,
            "start": req.start,
            "limit": req.limit,
            "checkpoint_path": str(checkpoint_path),
            "resume_from_checkpoint": req.resume_from_checkpoint,
            "rerun_incorrect_checkpoint_items": req.rerun_incorrect_checkpoint_items,
            "checkpoint_records": len(checkpoint),
        }
    )
    yield _sse(
        {
            "type": "eval_checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "loaded": len(checkpoint),
            "resume_from_checkpoint": req.resume_from_checkpoint,
            "reset_checkpoint": req.reset_checkpoint,
            "rerun_incorrect_checkpoint_items": req.rerun_incorrect_checkpoint_items,
        }
    )
    emit_runtime_log(
        "eval_checkpoint_load",
        checkpoint_path=str(checkpoint_path),
        loaded=len(checkpoint),
        resume_from_checkpoint=req.resume_from_checkpoint,
        reset_checkpoint=req.reset_checkpoint,
        rerun_incorrect_checkpoint_items=req.rerun_incorrect_checkpoint_items,
    )

    for offset, row in enumerate(rows, 1):
        row_id = row.get("id", req.start + offset - 1)
        item_index = req.start + offset - 1
        question = str(row.get("question", ""))
        gold = str(row.get("answer", ""))
        row_key = _eval_row_key(dataset_path, item_index, row)
        saved = checkpoint.get(row_key)
        rerun_saved = bool(saved and req.rerun_incorrect_checkpoint_items and not saved.get("item_correct"))
        if saved and not rerun_saved:
            done_count += 1
            saved_item = dict(saved.get("item", {}))
            if saved.get("item_correct"):
                correct += 1
            saved_item.update(
                {
                    "type": "eval_item",
                    "id": row_id,
                    "index": item_index,
                    "done": done_count,
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / done_count if done_count else 0.0,
                    "resumed": True,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            yield _sse(saved_item)
            emit_runtime_log("eval_checkpoint_skip", checkpoint_path=str(checkpoint_path), row_key=row_key, index=item_index)
            continue
        if rerun_saved:
            emit_runtime_log(
                "eval_checkpoint_rerun",
                checkpoint_path=str(checkpoint_path),
                row_key=row_key,
                index=item_index,
                previous_judgement=(saved.get("item") or {}).get("judgement"),
            )
        try:
            runner_req = QueryRequest(
                question=question,
                session_id=f"eval-{row_id}-{int(started)}",
                max_concurrent=req.max_concurrent,
                max_replans=req.max_replans,
                max_adversarial_rounds=req.max_adversarial_rounds,
                memory_db_path=str(_reset_eval_item_memory_db(checkpoint_path, row_key)),
            )
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            def event_sink(event: dict[str, Any]) -> None:
                queue.put_nowait(event)

            task = asyncio.create_task(_run_request(runner_req, event_sink=event_sink))
            last_keepalive = time.time()
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1)
                except asyncio.TimeoutError:
                    if time.time() - last_keepalive >= 15:
                        last_keepalive = time.time()
                        yield ": keepalive\n\n"
                    continue
                yield _sse(
                    {
                        "type": "eval_run_event",
                        "id": row_id,
                        "index": item_index,
                        "question": question,
                        "inner_type": event.get("type", "message"),
                        "inner_event": _event_name(str(event.get("type", "message"))),
                        "payload": event,
                    }
                )

            result = await task
            payload = result.to_dict()
            predicted = str(payload.get("answer", ""))
            judge_usage_before = _llm_usage_snapshot(judge_llm)
            verdict = await judge_one(
                llm=judge_llm,
                question=question,
                gold_answer=gold,
                predicted_answer=predicted,
            )
            judge_usage = _llm_usage_delta(judge_usage_before, _llm_usage_snapshot(judge_llm))
            metadata = dict(payload.get("metadata", {}))
            research_usage = dict(metadata.get("llm_usage", {})) if isinstance(metadata.get("llm_usage"), dict) else {}
            metadata["research_llm_usage"] = research_usage
            metadata["judge_llm_usage"] = judge_usage
            if research_usage or judge_usage:
                metadata["total_llm_usage"] = _merge_llm_usage(research_usage, judge_usage)
            is_correct = bool(verdict.get("is_same_object")) and verdict.get("judgement") == "correct"
            if is_correct:
                correct += 1
            done_count += 1
            eval_item = {
                "type": "eval_item",
                "id": row_id,
                "index": item_index,
                "done": done_count,
                "total": total,
                "question": question,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "judgement": verdict.get("judgement", "incorrect"),
                "is_same_object": bool(verdict.get("is_same_object")),
                "reason": verdict.get("reason", ""),
                "correct": correct,
                "accuracy": correct / done_count if done_count else 0.0,
                "metadata": metadata,
                "resumed": False,
                "checkpoint_rerun": rerun_saved,
                "previous_judgement": (saved.get("item") or {}).get("judgement") if rerun_saved else None,
                "checkpoint_path": str(checkpoint_path),
            }
            _append_eval_checkpoint(
                checkpoint_path,
                {
                    "row_key": row_key,
                    "dataset_path": str(dataset_path),
                    "index": item_index,
                    "id": row_id,
                    "item_correct": is_correct,
                    "item": eval_item,
                    "saved_at": round(time.time(), 3),
                },
            )
            yield _sse(eval_item)
        except Exception as exc:
            done_count += 1
            reason = f"{type(exc).__name__}: {exc}"
            yield _sse(
                {
                    "type": "eval_item",
                    "id": row_id,
                    "index": item_index,
                    "done": done_count,
                    "total": total,
                    "question": question,
                    "gold_answer": gold,
                    "predicted_answer": "",
                    "judgement": "error",
                    "is_same_object": False,
                    "reason": reason,
                    "correct": correct,
                    "accuracy": correct / done_count if done_count else 0.0,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_saved": False,
                }
            )
            if _should_stop_evaluation(exc):
                yield _sse(
                    {
                        "type": "eval_error",
                        "id": row_id,
                        "index": item_index,
                        "reason": reason,
                        "checkpoint_path": str(checkpoint_path),
                        "message": "Evaluation stopped before checkpointing this item. Fix the API key/quota and rerun with resume enabled.",
                    }
                )
                emit_runtime_log("eval_checkpoint_stop", checkpoint_path=str(checkpoint_path), index=item_index, error=reason)
                break

    elapsed = time.time() - started
    yield _sse(
        {
            "type": "eval_summary",
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "done": done_count,
            "elapsed_seconds": round(elapsed, 3),
            "checkpoint_path": str(checkpoint_path),
        }
    )


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    with dataset_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _eval_checkpoint_path(req: EvaluationRequest, dataset_path: Path) -> Path:
    if req.checkpoint_path:
        path = Path(req.checkpoint_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    signature = json.dumps(
        {
            "dataset_path": str(dataset_path.resolve()),
            "start": req.start,
            "limit": req.limit,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    stem = dataset_path.stem or "dataset"
    return PROJECT_ROOT / "data" / "eval_checkpoints" / f"{stem}-{digest}.jsonl"


def _eval_row_key(dataset_path: Path, index: int, row: dict[str, Any]) -> str:
    signature = json.dumps(
        {
            "dataset_path": str(dataset_path.resolve()),
            "index": index,
            "id": row.get("id", index),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()


def _load_eval_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_key = str(record.get("row_key", ""))
            item = record.get("item")
            if row_key and isinstance(item, dict) and item.get("judgement") != "error":
                records[row_key] = record
    return records


def _append_eval_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    emit_runtime_log(
        "eval_checkpoint_save",
        checkpoint_path=str(path),
        row_key=record.get("row_key"),
        index=record.get("index"),
        item_correct=record.get("item_correct"),
    )


def _reset_eval_item_memory_db(checkpoint_path: Path, row_key: str) -> Path:
    memory_dir = checkpoint_path.parent / "eval_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_path = memory_dir / f"{row_key}.db"
    for path in (memory_path, Path(str(memory_path) + "-wal"), Path(str(memory_path) + "-shm")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    emit_runtime_log("eval_memory_reset", memory_db_path=str(memory_path), row_key=row_key)
    return memory_path


def _should_stop_evaluation(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "api key",
        "apikey",
        "unauthorized",
        "forbidden",
        "invalid key",
        "invalid api",
        "quota",
        "insufficient",
        "balance",
        "billing",
        "payment",
        "expired",
        "401",
        "403",
        "429",
        "dashscope",
        "余额",
        "欠费",
        "续费",
        "额度",
        "限流",
        "认证",
        "鉴权",
    ]
    return any(marker in text for marker in markers)


def _llm_usage_snapshot(llm: Any) -> dict[str, Any]:
    getter = getattr(llm, "get_usage", None)
    if not callable(getter):
        return {}
    try:
        return dict(getter())
    except Exception:
        return {}


def _llm_usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not after:
        return {}
    keys = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
    delta = {key: max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)) for key in keys}
    delta["estimated"] = bool(after.get("estimated") or before.get("estimated"))
    return delta


def _merge_llm_usage(*items: dict[str, Any]) -> dict[str, Any]:
    merged = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": False}
    for item in items:
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
            merged[key] += int(item.get(key, 0) or 0)
        merged["estimated"] = bool(merged["estimated"] or item.get("estimated"))
    return merged


def _python_status() -> dict[str, Any]:
    modules = ["fastapi", "uvicorn", "pydantic", "numpy", "sklearn", "sentence_transformers", "openai", "dotenv"]
    missing = []
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:
            missing.append({"module": module, "error": type(exc).__name__})
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "missing_modules": missing,
        "runnable": not missing,
    }


def _conda_status() -> dict[str, Any]:
    conda = shutil.which("conda")
    if not conda:
        return {"available": False, "environments": [], "message": "conda not found on PATH"}
    try:
        completed = subprocess.run(
            [conda, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
        )
        data = json.loads(completed.stdout or "{}")
        environments = []
        active_prefix = Path(sys.prefix).resolve()
        for path in data.get("envs", []):
            env_path = Path(path).resolve()
            environments.append(
                {
                    "name": env_path.name,
                    "path": str(env_path),
                    "active": env_path == active_prefix,
                }
            )
        return {"available": True, "environments": environments}
    except Exception as exc:
        return {"available": True, "environments": [], "error": f"{type(exc).__name__}: {exc}"}


def _api_key_status() -> dict[str, Any]:
    groups = [
        {
            "name": "DashScope/Qwen LLM",
            "keys": ["DASHSCOPE_API_KEY"],
            "required": True,
            "purpose": "规划、worker 推理、答案合成和 judge",
        },
        {
            "name": "Search",
            "keys": ["SERPER_API_KEYS", "SERPER_API_KEY", "IQS_API_KEY"],
            "required": False,
            "purpose": "网页检索；至少配置其中一个可提升多跳 QA 召回",
        },
        {
            "name": "Jina",
            "keys": ["JINA_API_KEY"],
            "required": False,
            "purpose": "网页读取、页面分析和 Wikipedia fallback",
        },
        {
            "name": "Browser MCP",
            "keys": ["PLAYWRIGHT_MCP_URL", "PLAYWRIGHT_MCP_TOKEN"],
            "required": False,
            "purpose": "浏览器自动化；仅需要动态页面时配置",
        },
        {
            "name": "E2B",
            "keys": ["E2B_API_KEY"],
            "required": False,
            "purpose": "代码沙箱；多跳问答默认不强依赖",
        },
    ]
    statuses = []
    for group in groups:
        present_keys = [key for key in group["keys"] if _has_env_value(key)]
        statuses.append({**group, "configured": bool(present_keys), "configured_keys": present_keys})
    return {
        "env_file_exists": (PROJECT_ROOT / ".env").exists(),
        "template_file_exists": (PROJECT_ROOT / ".env.template").exists(),
        "groups": statuses,
        "missing_required": [item["name"] for item in statuses if item["required"] and not item["configured"]],
        "missing_recommended": [
            item["name"]
            for item in statuses
            if not item["required"] and not item["configured"] and item["name"] in {"Search", "Jina"}
        ],
    }


def _has_env_value(key: str) -> bool:
    value = os.getenv(key, "").strip().strip('"').strip("'")
    if not value:
        return False
    return value not in {"<KEY>", "<TOKEN>", "YOUR_API_KEY", "your_api_key", "None", "null"}


def _extract_question_from_agui(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
    return str(payload.get("question", ""))
