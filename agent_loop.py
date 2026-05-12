import asyncio
import datetime
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from inspect import iscoroutinefunction
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    List,
    Literal,
    Optional,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletionChunk
from tools_calling import (
    build_main_agent_system_prompt,
    build_sub_agent_system_prompt,
    generate_summarize_prompt,
)

try:
    from tools import SUB_AGENT_TOOLS
except ImportError:
    SUB_AGENT_TOOLS = []

logger = logging.getLogger(__name__)

MAX_MAIN_AGENT_TURNS = 30
MAX_SUB_AGENT_TURNS = 10
MAX_TOOL_TIMEOUT = 30
MAX_FACTS_PER_REPORT = 5
MAX_SOURCES_PER_REPORT = 4
MAX_CANONICAL_NAMES = 4
MAX_MAIN_TOOL_RESULT_CHARS = 16000
MAX_WORKER_PREVIEW_CHARS = 300
MEMORY_LEDGER_PATH = Path(__file__).resolve().parent / "MEMORY.md"

DEFAULT_SYSTEM_PROMPT = """
Accuracy is more important than speed.

When producing the final answer:
1. Return ONLY a JSON object with key "answer".
2. Match the question's requested identifier type exactly.
3. Prefer full official names unless the question asks for another form.
4. Keep the answer concise and exact-match friendly.
"""


@dataclass
class ToolCall:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None


@dataclass
class Chunk:
    step_index: int
    type: Literal["text", "tool_call", "tool_call_result"]
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[Any] = None


def python_type_to_json_type(t):
    """Map Python types to JSON types."""
    if t is str:
        return "string"
    if t is int:
        return "integer"
    if t is float:
        return "number"
    if t is bool:
        return "boolean"
    if t is list or get_origin(t) is list:
        return "array"
    if t is dict or get_origin(t) is dict:
        return "object"
    return "string"


def parse_docstring(docstring: str) -> dict:
    """Parse a docstring into a summary and param descriptions."""
    if not docstring:
        return {"description": "", "params": {}}

    lines = docstring.strip().split("\n")
    description_lines = []
    params = {}
    current_section = "description"
    current_param = None
    current_param_desc = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            current_section = "args"
            continue
        if stripped.lower() in (
            "returns:",
            "return:",
            "yields:",
            "raises:",
            "examples:",
            "example:",
            "note:",
            "notes:",
        ):
            if current_param and current_param_desc:
                params[current_param] = " ".join(current_param_desc).strip()
            current_section = "other"
            continue

        if current_section == "description":
            description_lines.append(stripped)
        elif current_section == "args":
            param_match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$", stripped)
            if param_match:
                if current_param and current_param_desc:
                    params[current_param] = " ".join(current_param_desc).strip()
                current_param = param_match.group(1)
                current_param_desc = [param_match.group(2)] if param_match.group(2) else []
            elif current_param and stripped:
                current_param_desc.append(stripped)

    if current_param and current_param_desc:
        params[current_param] = " ".join(current_param_desc).strip()

    while description_lines and not description_lines[-1]:
        description_lines.pop()

    return {
        "description": " ".join(description_lines).strip(),
        "params": params,
    }


def function_to_schema(func: Callable) -> dict:
    """Convert a Python function to an OpenAI tool schema."""
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    docstring_info = parse_docstring(func.__doc__ or "")

    parameters = {"type": "object", "properties": {}, "required": []}
    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = type_hints.get(name, str)
        param_type = python_type_to_json_type(annotation)
        param_info = {"type": param_type}

        if name in docstring_info["params"]:
            param_info["description"] = docstring_info["params"][name]

        if get_origin(annotation) == Literal:
            param_info["enum"] = list(get_args(annotation))
            param_info["type"] = python_type_to_json_type(type(get_args(annotation)[0]))

        parameters["properties"][name] = param_info
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": docstring_info["description"],
            "parameters": parameters,
        },
    }


def _contains_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    for char in text:
        if "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf":
            return True
    return False


def _compact_text(text: str, max_chars: int = 200) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_list(
    value: Any,
    *,
    limit: int,
    item_max_chars: int = 200,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = [str(value)]

    normalized = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            candidate = item.get("url") or item.get("source") or item.get("title") or json.dumps(item, ensure_ascii=False)
        else:
            candidate = str(item)
        candidate = _compact_text(candidate, item_max_chars)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
        if len(normalized) >= limit:
            break
    return normalized


def _strip_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _extract_json_object(text: str) -> Optional[dict]:
    stripped = _strip_code_fence(text)
    if not stripped:
        return None

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for index in range(start, len(stripped)):
        char = stripped[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end == -1:
        return None

    try:
        data = json.loads(stripped[start:end])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_urls(text: str, limit: int = MAX_SOURCES_PER_REPORT) -> list[str]:
    found = re.findall(r"https?://[^\s)>\]]+", text or "")
    unique = []
    seen = set()
    for item in found:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _markdown_blockquote(text: str) -> str:
    lines = str(text or "").strip().splitlines() or [""]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _append_memory_section(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(content)
        if not content.endswith("\n"):
            file.write("\n")


def _escape_markdown_table_cell(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _initialize_memory_ledger(path: Path, user_question: str) -> None:
    created_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    content = (
        "# MEMORY\n\n"
        "This ledger is regenerated for each top-level agent request. It records "
        "the main agent's delegated subtasks and the normalized JSON reports "
        "returned by sub-agents.\n\n"
        f"- Created at: {created_at}\n\n"
        "## Original Question\n\n"
        f"{_markdown_blockquote(user_question)}\n"
    )
    path.write_text(content, encoding="utf-8")


def _append_subtasks_to_memory_ledger(
    path: Path,
    round_index: int,
    subtasks: list[str],
) -> None:
    lines = [f"\n## Round {round_index} - Planned Subtasks\n"]
    if not subtasks:
        lines.append("- No subtasks provided.")
    else:
        for index, subtask in enumerate(subtasks, start=1):
            normalized = " ".join(str(subtask).strip().split())
            lines.append(f"{index}. {normalized}")
    _append_memory_section(path, "\n".join(lines) + "\n")


def _append_worker_reports_to_memory_ledger(
    path: Path,
    round_index: int,
    worker_reports: list[dict],
    ledger_update: dict,
) -> None:
    lines = [f"\n## Round {round_index} - Worker Reports\n"]
    if not worker_reports:
        lines.append("- No worker reports.")
    else:
        for index, report in enumerate(worker_reports, start=1):
            lines.extend(
                [
                    f"### Worker {index}",
                    "",
                    "```json",
                    json.dumps(report, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            f"## Round {round_index} - Subtask Status",
            "",
            "| # | Status | Confidence | Subtask | Answer |",
            "|---|---|---|---|---|",
        ]
    )
    if not worker_reports:
        lines.append("| - | unresolved | low | No worker reports. | |")
    else:
        for index, report in enumerate(worker_reports, start=1):
            status = _escape_markdown_table_cell(report.get("status", "unresolved"))
            confidence = _escape_markdown_table_cell(report.get("confidence", "low"))
            subtask = _escape_markdown_table_cell(report.get("subtask", ""))
            answer = _escape_markdown_table_cell(report.get("subtask_answer", ""))
            lines.append(f"| {index} | {status} | {confidence} | {subtask} | {answer} |")
    lines.append("")

    lines.extend(
        [
            f"## Round {round_index} - Ledger Update",
            "",
            "```json",
            json.dumps(ledger_update, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    _append_memory_section(path, "\n".join(lines))


def _normalize_worker_report(report_text: str, subtask: str, worker_index: int) -> dict:
    payload = _extract_json_object(report_text) or {}
    if not payload:
        payload = {
            "status": "unresolved",
            "subtask_answer": "",
            "confidence": "low",
            "facts": [
                _compact_text(
                    f"Worker {worker_index} returned unstructured output: {report_text}",
                    220,
                )
            ],
            "sources": _extract_urls(report_text),
            "canonical_names": [],
            "answer_form_hint": "",
            "unresolved": ["Worker output was not valid JSON."],
        }

    status = str(payload.get("status", "resolved" if payload.get("subtask_answer") else "unresolved")).lower()
    if status not in {"resolved", "unresolved"}:
        status = "unresolved"

    confidence = str(payload.get("confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    normalized = {
        "status": status,
        "subtask": _compact_text(subtask, 220),
        "subtask_answer": _compact_text(str(payload.get("subtask_answer", "")), 160),
        "confidence": confidence,
        "facts": _normalize_list(
            payload.get("facts"),
            limit=MAX_FACTS_PER_REPORT,
            item_max_chars=220,
        ),
        "sources": _normalize_list(
            payload.get("sources") or _extract_urls(report_text),
            limit=MAX_SOURCES_PER_REPORT,
            item_max_chars=160,
        ),
        "canonical_names": _normalize_list(
            payload.get("canonical_names"),
            limit=MAX_CANONICAL_NAMES,
            item_max_chars=120,
        ),
        "answer_form_hint": _compact_text(str(payload.get("answer_form_hint", "")), 160),
        "unresolved": _normalize_list(
            payload.get("unresolved"),
            limit=4,
            item_max_chars=180,
        ),
    }

    if not normalized["facts"] and normalized["subtask_answer"]:
        normalized["facts"] = [
            _compact_text(
                f"Worker resolved the node as: {normalized['subtask_answer']}",
                220,
            )
        ]

    return normalized


def _build_ledger_update(reports: list[dict]) -> dict:
    confirmed_facts = []
    candidate_answers = []
    canonical_names = []
    open_questions = []
    source_shortlist = []

    def _append_unique(bucket: list[str], values: list[str], limit: int, max_chars: int):
        for value in values:
            item = _compact_text(value, max_chars)
            if not item or item in bucket:
                continue
            bucket.append(item)
            if len(bucket) >= limit:
                break

    for report in reports:
        _append_unique(confirmed_facts, report["facts"], 10, 220)
        if report["subtask_answer"]:
            label = f"{report['subtask_answer']} [confidence={report['confidence']}]"
            _append_unique(candidate_answers, [label], 6, 180)
        _append_unique(canonical_names, report["canonical_names"], 8, 120)
        _append_unique(open_questions, report["unresolved"], 6, 180)
        _append_unique(source_shortlist, report["sources"], 8, 160)

    return {
        "confirmed_facts": confirmed_facts,
        "candidate_answers": candidate_answers,
        "canonical_names": canonical_names,
        "open_questions": open_questions,
        "source_shortlist": source_shortlist,
    }


def _build_subtask_result_payload(questions: list[str], results: list[Any]) -> dict:
    worker_reports = []
    for index, (subtask, result) in enumerate(zip(questions, results), start=1):
        if isinstance(result, Exception):
            report = {
                "status": "unresolved",
                "subtask": _compact_text(subtask, 220),
                "subtask_answer": "",
                "confidence": "low",
                "facts": [
                    _compact_text(
                        f"Worker {index} failed: {result}",
                        220,
                    )
                ],
                "sources": [],
                "canonical_names": [],
                "answer_form_hint": "",
                "unresolved": ["Worker execution failed."],
            }
        else:
            report = _normalize_worker_report(str(result), subtask, index)
        worker_reports.append(report)

    return {
        "worker_reports": worker_reports,
        "ledger_update": _build_ledger_update(worker_reports),
    }


def _serialize_subtask_payload(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized) <= MAX_MAIN_TOOL_RESULT_CHARS:
        return serialized
    worker_reports = payload.get("worker_reports", [])
    payload["worker_reports"] = worker_reports[: min(2, len(worker_reports))]
    payload["ledger_update"] = _build_ledger_update(worker_reports)
    return json.dumps(payload, ensure_ascii=False)


def _serialize_subtask_result(questions: list[str], results: list[Any]) -> str:
    return _serialize_subtask_payload(_build_subtask_result_payload(questions, results))


def _looks_like_answer_json(text: str) -> bool:
    data = _extract_json_object(text)
    return isinstance(data, dict) and "answer" in data


def _question_requires_research(question: str) -> bool:
    stripped = (question or "").strip()
    if len(stripped) > 35:
        return True

    lowered = stripped.lower()
    keywords = (
        "who",
        "which",
        "what",
        "when",
        "where",
        "whose",
        "英文",
        "全称",
        "名称",
        "名字",
        "哪一",
        "哪位",
        "哪座",
        "哪年",
        "哪个",
        "什么",
        "谁",
        "年份",
        "城市",
        "公司",
    )
    return any(keyword in lowered or keyword in stripped for keyword in keywords)


async def _finalize_sub_agent_report(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list,
    subtask: str,
    chinese_context: bool,
) -> str:
    final_messages = list(messages) + [
        {
            "role": "user",
            "content": generate_summarize_prompt(
                task_description=subtask,
                task_failed=False,
                is_main_agent=False,
                chinese_context=chinese_context,
            ),
        }
    ]
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=final_messages,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        return content
    except Exception as exc:
        logger.warning("Sub-agent finalize failed: %s", exc)
        return json.dumps(
            {
                "status": "unresolved",
                "subtask_answer": "",
                "confidence": "low",
                "facts": ["Failed to finalize worker report."],
                "sources": [],
                "canonical_names": [],
                "answer_form_hint": "",
                "unresolved": [str(exc)],
            },
            ensure_ascii=False,
        )


async def run_sub_agent(
    client: AsyncOpenAI,
    model: str,
    subtask: str,
    sub_agent_tool_functions: list,
    chinese_context: bool = False,
    progress_queue: Optional[asyncio.Queue] = None,
    worker_index: int = 0,
    user_question: str = "",
) -> str:
    """Run the sub-agent worker to complete a research subtask."""

    async def _emit_progress(text: str):
        if progress_queue is not None:
            await progress_queue.put(text)

    async def _llm_call_with_progress(coro, label: str):
        llm_task = asyncio.create_task(coro)

        async def _progress_ticker():
            waited = 0
            try:
                while True:
                    await asyncio.sleep(10)
                    waited += 10
                    await _emit_progress(
                        f"⏳ Worker {worker_index}: {label} ({waited}s)\n\n"
                    )
            except asyncio.CancelledError:
                pass

        ticker = asyncio.create_task(_progress_ticker())
        try:
            return await llm_task
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

    async def _tool_call_with_progress(coro, tool_name: str):
        tool_task = asyncio.create_task(coro)

        async def _progress_ticker():
            waited = 0
            try:
                while True:
                    await asyncio.sleep(15)
                    waited += 15
                    await _emit_progress(
                        f"⏳ Worker {worker_index}: `{tool_name}` running... ({waited}s)\n\n"
                    )
            except asyncio.CancelledError:
                pass

        ticker = asyncio.create_task(_progress_ticker())
        try:
            return await tool_task
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

    system_prompt = build_sub_agent_system_prompt(
        sub_agent_tool_functions,
        chinese_context,
    )
    if user_question:
        system_prompt += f"\n\n## User's Original Question\n{user_question}"

    tool_schema = [function_to_schema(func) for func in sub_agent_tool_functions]
    tool_functions_map = {func.__name__: func for func in sub_agent_tool_functions}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": subtask},
    ]

    logger.info("[Sub-Agent] Starting subtask: %s", subtask[:200])
    await _emit_progress(
        f"🔍 **Worker {worker_index}**: Starting research — {_compact_text(subtask, 100)}\n\n"
    )

    final_content = ""

    for turn in range(MAX_SUB_AGENT_TURNS):
        input_chars = sum(
            len(str(message.get("content", ""))) +
            sum(
                len(str(tc.get("function", {}).get("arguments", "")))
                for tc in message.get("tool_calls", [])
            )
            for message in messages
        )
        await _emit_progress(
            f"📊 Worker {worker_index}: Turn {turn + 1} — LLM input {input_chars:,} chars\n\n"
        )

        try:
            response = await _llm_call_with_progress(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_schema if tool_schema else None,
                ),
                label=f"Waiting for LLM response (turn {turn + 1})...",
            )
        except BadRequestError as exc:
            if exc.status_code == 400 and "data_inspection_failed" in (getattr(exc, "code", "") or ""):
                logger.warning("[Sub-Agent] Content filter triggered at turn %s", turn)
                await _emit_progress(
                    f"⚠️ Worker {worker_index}: Content filter triggered — sanitizing and retrying\n\n"
                )
                for idx in range(len(messages) - 1, -1, -1):
                    if messages[idx].get("role") == "tool":
                        messages[idx]["content"] = "[Tool result removed due to content filter.]"
                    elif messages[idx].get("role") == "assistant":
                        break
                continue
            logger.error("[Sub-Agent] LLM call failed at turn %s: %s", turn, exc)
            break
        except Exception as exc:
            logger.error("[Sub-Agent] LLM call failed at turn %s: %s", turn, exc)
            break

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            final_content = assistant_message.content or ""
            messages.append({"role": "assistant", "content": final_content})
            if final_content:
                preview = _compact_text(final_content, MAX_WORKER_PREVIEW_CHARS)
                await _emit_progress(
                    f"💬 Worker {worker_index}: Response ({len(final_content)} chars):\n{preview}\n\n"
                )
            break

        tool_calls_data = []
        for tool_call in assistant_message.tool_calls:
            arguments = tool_call.function.arguments
            try:
                json.loads(arguments)
            except Exception:
                arguments = json.dumps({})
            tool_calls_data.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": arguments,
                    },
                }
            )

        assistant_payload = {"role": "assistant", "tool_calls": tool_calls_data}
        if assistant_message.content:
            assistant_payload["content"] = assistant_message.content
        messages.append(assistant_payload)

        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            try:
                parsed_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                tool_result = f"Error: Failed to parse arguments: {exc}"
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result}
                )
                continue

            progress_detail = ""
            if parsed_args:
                first_value = next(iter(parsed_args.values()), "")
                if isinstance(first_value, str):
                    first_value = _compact_text(first_value, 80)
                progress_detail = f" | {first_value}"
            await _emit_progress(
                f"⚙️ Worker {worker_index}: `{func_name}`{progress_detail}\n\n"
            )

            try:
                if func_name not in tool_functions_map:
                    tool_result = f"Error: Tool '{func_name}' not found."
                else:
                    func = tool_functions_map[func_name]
                    coro = func(**parsed_args) if iscoroutinefunction(func) else asyncio.to_thread(func, **parsed_args)
                    try:
                        result = await asyncio.wait_for(
                            _tool_call_with_progress(coro, func_name),
                            timeout=MAX_TOOL_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        result = f"Error: Tool '{func_name}' timed out after {MAX_TOOL_TIMEOUT}s"
                        await _emit_progress(
                            f"⚠️ Worker {worker_index}: `{func_name}` timed out ({MAX_TOOL_TIMEOUT}s)\n\n"
                        )
                    tool_result = str(result)
            except Exception as exc:
                tool_result = f"Error: Execution failed - {exc}"

            logger.info(
                "[Sub-Agent] Turn %s: %s -> %s chars",
                turn,
                func_name,
                len(tool_result),
            )
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result}
            )

    if not final_content or not _extract_json_object(final_content):
        final_content = await _finalize_sub_agent_report(
            client=client,
            model=model,
            messages=messages,
            subtask=subtask,
            chinese_context=chinese_context,
        )

    normalized_preview = _normalize_worker_report(final_content, subtask, worker_index)
    final_content = json.dumps(normalized_preview, ensure_ascii=False)
    logger.info("[Sub-Agent] Result: %s chars", len(final_content))
    await _emit_progress(f"✅ **Worker {worker_index}**: Research complete.\n\n")
    return final_content


async def agent_loop(
    input_messages: list,
    tool_functions: List[Callable],
    skill_directories: Optional[List[str]] = None,
) -> AsyncIterator[Chunk]:
    """Main agent loop with multi-agent architecture."""
    del skill_directories

    assert os.getenv("DASHSCOPE_API_KEY"), "DASHSCOPE_API_KEY is not set"

    model = os.getenv("QWEN_MODEL") or "qwen-max"
    client = AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    user_question = ""
    for message in reversed(input_messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                user_question = content
            break

    _initialize_memory_ledger(MEMORY_LEDGER_PATH, user_question)

    chinese_context = _contains_cjk(user_question)
    yield Chunk(type="text", content="", step_index=0)

    sub_agent_tool_functions = list(SUB_AGENT_TOOLS)
    max_parallel = int(os.getenv("SUB_AGENT_NUM", "3"))
    progress_queue: asyncio.Queue = asyncio.Queue()
    delegation_round = 0

    async def execute_subtasks(subtasks_json: str) -> str:
        """Delegate one or more research subtasks to worker agents."""
        nonlocal delegation_round

        try:
            questions = json.loads(subtasks_json)
            if isinstance(questions, str):
                questions = [questions]
        except json.JSONDecodeError:
            questions = [subtasks_json]

        if not questions:
            return json.dumps(
                {
                    "worker_reports": [],
                    "ledger_update": {
                        "confirmed_facts": [],
                        "candidate_answers": [],
                        "canonical_names": [],
                        "open_questions": ["No subtasks provided."],
                        "source_shortlist": [],
                    },
                },
                ensure_ascii=False,
            )

        questions = [str(question) for question in questions[:max_parallel]]
        delegation_round += 1
        _append_subtasks_to_memory_ledger(
            MEMORY_LEDGER_PATH,
            delegation_round,
            questions,
        )
        logger.info("[Main Agent] Dispatching %s subtask(s) in parallel", len(questions))

        tasks = [
            run_sub_agent(
                client=client,
                model=model,
                subtask=question,
                sub_agent_tool_functions=sub_agent_tool_functions,
                chinese_context=chinese_context,
                progress_queue=progress_queue,
                worker_index=index + 1,
                user_question=user_question,
            )
            for index, question in enumerate(questions)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        payload = _build_subtask_result_payload(questions, results)
        _append_worker_reports_to_memory_ledger(
            MEMORY_LEDGER_PATH,
            delegation_round,
            payload["worker_reports"],
            payload["ledger_update"],
        )
        return _serialize_subtask_payload(payload)

    main_agent_tools = [execute_subtasks] + list(tool_functions or [])
    system_prompt = build_main_agent_system_prompt(
        main_agent_tools,
        chinese_context,
        max_parallel=max_parallel,
    )
    system_prompt = f"{system_prompt}\n\n{DEFAULT_SYSTEM_PROMPT}"

    prompt_messages = input_messages.copy()
    if prompt_messages and prompt_messages[0].get("role") == "system":
        original_content = prompt_messages[0].get("content", "")
        prompt_messages[0] = {
            "role": "system",
            "content": f"{original_content}\n\n{system_prompt}",
        }
    else:
        prompt_messages.insert(0, {"role": "system", "content": system_prompt})

    tool_schema = [function_to_schema(func) for func in main_agent_tools]
    tool_functions_map = {func.__name__: func for func in main_agent_tools}
    params = {
        "model": model,
        "stream": True,
        "tools": tool_schema,
    }

    step_index = 0

    for turn in range(MAX_MAIN_AGENT_TURNS):
        stream = await client.chat.completions.create(
            messages=prompt_messages,
            **params,
        )

        tool_calls_buffer = {}
        assistant_text_parts = []

        async for stream_chunk in stream:
            chunk = cast(ChatCompletionChunk, stream_chunk)
            delta = chunk.choices[0].delta

            if delta.content:
                assistant_text_parts.append(delta.content)
                yield Chunk(type="text", content=delta.content, step_index=step_index)

            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    idx = tool_call_chunk.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tool_call_chunk.id,
                            "function": {
                                "name": tool_call_chunk.function.name,
                                "arguments": "",
                            },
                        }
                    if tool_call_chunk.function.arguments:
                        tool_calls_buffer[idx]["function"]["arguments"] += (
                            tool_call_chunk.function.arguments
                        )

        assistant_text = "".join(assistant_text_parts).strip()

        if not tool_calls_buffer:
            if assistant_text:
                prompt_messages.append({"role": "assistant", "content": assistant_text})

            if turn == 0 and _question_requires_research(user_question):
                prompt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Research is required for this question. "
                            "Do not answer directly yet. First call execute_subtasks "
                            "to gather evidence, then answer in JSON."
                        ),
                    }
                )
                continue

            if assistant_text and not _looks_like_answer_json(assistant_text) and turn < MAX_MAIN_AGENT_TURNS - 1:
                prompt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your response must be ONLY a JSON object like "
                            "{\"answer\": \"...\"} with no extra text. "
                            "Restate the final answer in that exact format."
                        ),
                    }
                )
                continue
            break

        assistant_tool_calls_data = []
        for idx in sorted(tool_calls_buffer.keys()):
            raw_tool = tool_calls_buffer[idx]
            assistant_tool_calls_data.append(
                {
                    "id": raw_tool["id"],
                    "type": "function",
                    "function": {
                        "name": raw_tool["function"]["name"],
                        "arguments": raw_tool["function"]["arguments"],
                    },
                }
            )

        assistant_payload = {"role": "assistant", "tool_calls": assistant_tool_calls_data}
        if assistant_text:
            assistant_payload["content"] = assistant_text
        prompt_messages.append(assistant_payload)

        parsed_tool_calls = []
        for tool_data in assistant_tool_calls_data:
            call_id = tool_data["id"]
            func_name = tool_data["function"]["name"]
            func_args_str = tool_data["function"]["arguments"]
            tool_call = ToolCall(
                tool_call_id=call_id,
                tool_name=func_name,
                tool_arguments={},
            )
            try:
                parsed_args = json.loads(func_args_str)
                tool_call.tool_arguments = parsed_args
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )
                parsed_tool_calls.append((call_id, func_name, parsed_args, tool_call, None))
            except json.JSONDecodeError as exc:
                error_msg = f"Error: Failed to parse tool arguments JSON: {func_args_str}. Error: {exc}"
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )
                parsed_tool_calls.append((call_id, func_name, {}, tool_call, error_msg))

        async_tasks = {}
        sync_results = {}
        for call_id, func_name, parsed_args, _, error_msg in parsed_tool_calls:
            if error_msg:
                sync_results[call_id] = error_msg
                continue
            if func_name not in tool_functions_map:
                sync_results[call_id] = f"Error: Tool '{func_name}' not found."
                continue
            func = tool_functions_map[func_name]
            if iscoroutinefunction(func):
                async_tasks[call_id] = asyncio.create_task(func(**parsed_args))
            else:
                async_tasks[call_id] = asyncio.create_task(asyncio.to_thread(func, **parsed_args))

        if async_tasks:
            pending = set(async_tasks.values())
            while pending:
                while not progress_queue.empty():
                    try:
                        progress_text = progress_queue.get_nowait()
                        yield Chunk(type="text", content=progress_text, step_index=step_index)
                    except asyncio.QueueEmpty:
                        break

                _, pending = await asyncio.wait(pending, timeout=5)
                if pending:
                    drained = False
                    while not progress_queue.empty():
                        try:
                            progress_text = progress_queue.get_nowait()
                            yield Chunk(type="text", content=progress_text, step_index=step_index)
                            drained = True
                        except asyncio.QueueEmpty:
                            break
                    if not drained:
                        yield Chunk(type="text", content="", step_index=step_index)

            while not progress_queue.empty():
                try:
                    progress_text = progress_queue.get_nowait()
                    yield Chunk(type="text", content=progress_text, step_index=step_index)
                except asyncio.QueueEmpty:
                    break

            for call_id, task in async_tasks.items():
                try:
                    sync_results[call_id] = str(task.result())
                except Exception as exc:
                    sync_results[call_id] = f"Error: Execution failed - {exc}"

        for call_id, _, _, tool_call, _ in parsed_tool_calls:
            tool_result_content = sync_results[call_id]
            yield Chunk(
                type="tool_call_result",
                tool_result=tool_result_content,
                step_index=step_index,
                tool_call=tool_call,
            )
            prompt_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result_content,
                }
            )

        step_index += 1
    else:
        logger.warning(
            "Main agent reached max turns (%s), generating summary",
            MAX_MAIN_AGENT_TURNS,
        )
        summarize = generate_summarize_prompt(
            task_description=user_question,
            task_failed=True,
            is_main_agent=True,
            chinese_context=chinese_context,
        )
        prompt_messages.append({"role": "user", "content": summarize})
        stream = await client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            stream=True,
        )
        async for stream_chunk in stream:
            chunk = cast(ChatCompletionChunk, stream_chunk)
            delta = chunk.choices[0].delta
            if delta.content:
                yield Chunk(type="text", content=delta.content, step_index=step_index)
