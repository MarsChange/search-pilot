from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

from deep_research.runtime_logging import emit_runtime_log, summarize_result


def python_type_to_json_type(annotation: Any) -> str:
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is list or get_origin(annotation) is list:
        return "array"
    if annotation is dict or get_origin(annotation) is dict:
        return "object"
    return "string"


def function_to_schema(func: Callable[..., Any]) -> dict[str, Any]:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    parameters = {"type": "object", "properties": {}, "required": []}
    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        annotation = type_hints.get(name, str)
        param_schema = {"type": python_type_to_json_type(annotation)}
        if get_origin(annotation) is Literal:
            values = list(get_args(annotation))
            param_schema["enum"] = values
            if values:
                param_schema["type"] = python_type_to_json_type(type(values[0]))
        parameters["properties"][name] = param_schema
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or "",
            "parameters": parameters,
        },
    }


class FunctionToolExecutor:
    def __init__(self, functions: list[Callable[..., Any]] | None = None, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds
        if functions is None:
            functions = self._load_default_functions()
        self.functions = {func.__name__: func for func in functions}

    def schemas(self) -> list[dict[str, Any]]:
        return [function_to_schema(func) for func in self.functions.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.functions:
            emit_runtime_log("tool_call_error", tool=name, arguments=arguments, status="missing")
            return {"error": f"Tool '{name}' not found", "tool": name}
        func = self.functions[name]
        started = time.monotonic()
        emit_runtime_log("tool_call_start", tool=name, arguments=arguments)
        try:
            if inspect.iscoroutinefunction(func):
                task = func(**arguments)
            else:
                task = asyncio.to_thread(func, **arguments)
            result = await asyncio.wait_for(task, timeout=self.timeout_seconds)
            emit_runtime_log(
                "tool_call_end",
                tool=name,
                arguments=arguments,
                status="ok",
                elapsed_seconds=round(time.monotonic() - started, 3),
                result=summarize_result(result),
            )
            return {"tool": name, "arguments": arguments, "result": result}
        except asyncio.TimeoutError:
            emit_runtime_log(
                "tool_call_error",
                tool=name,
                arguments=arguments,
                status="timeout",
                elapsed_seconds=round(time.monotonic() - started, 3),
                error=f"timed out after {self.timeout_seconds}s",
            )
            return {"tool": name, "arguments": arguments, "error": f"timed out after {self.timeout_seconds}s"}
        except Exception as exc:
            emit_runtime_log(
                "tool_call_error",
                tool=name,
                arguments=arguments,
                status="error",
                elapsed_seconds=round(time.monotonic() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            return {"tool": name, "arguments": arguments, "error": f"{type(exc).__name__}: {exc}"}

    async def search(self, query: str, *, state_id: str = "", max_results: int = 5) -> dict[str, Any]:
        if "search_engine" in self.functions:
            raw = await self.execute(
                "search_engine",
                {"query": query, "num_results": max_results, "engine": "auto", "language": "zh-cn"},
            )
            raw["results"] = filter_search_results(extract_search_results(str(raw.get("result", ""))))
            raw["query"] = query
            raw["state_id"] = state_id
            emit_runtime_log(
                "search_results",
                tool=raw.get("tool", "search_engine"),
                state_id=state_id,
                query=query,
                results_count=len(raw.get("results") or []),
                status="error" if raw.get("error") else "ok",
            )
            return raw
        if "search_wikipedia" in self.functions:
            raw = await self.execute("search_wikipedia", {"entity": query, "first_sentences": 3})
            raw["results"] = filter_search_results(extract_search_results(str(raw.get("result", ""))))
            raw["query"] = query
            raw["state_id"] = state_id
            emit_runtime_log(
                "search_results",
                tool=raw.get("tool", "search_wikipedia"),
                state_id=state_id,
                query=query,
                results_count=len(raw.get("results") or []),
                status="error" if raw.get("error") else "ok",
            )
            return raw
        emit_runtime_log("search_results", tool="search", state_id=state_id, query=query, results_count=0, status="missing")
        return {"tool": "search", "query": query, "state_id": state_id, "results": [], "error": "No search tool configured."}

    async def enrich_url(self, url: str, *, question: str, state_id: str = "") -> dict[str, Any]:
        """Fetch or analyze one search result URL to get evidence beyond snippets."""
        url = (url or "").strip()
        if not url:
            return {"tool": "enrich_url", "state_id": state_id, "error": "empty URL", "results": []}
        if "analyze_webpage" in self.functions:
            raw = await self.execute("analyze_webpage", {"url": url, "question": question})
            raw["results"] = extract_content_evidence(str(raw.get("result", "")), url=url)
            raw["url"] = url
            raw["state_id"] = state_id
            return raw
        if "scrape_website" in self.functions:
            raw = await self.execute("scrape_website", {"url": url})
            raw["results"] = extract_content_evidence(str(raw.get("result", "")), url=url)
            raw["url"] = url
            raw["state_id"] = state_id
            return raw
        return {"tool": "enrich_url", "url": url, "state_id": state_id, "results": [], "error": "No webpage enrichment tool configured."}

    async def lookup_wikipedia(self, entity: str, *, state_id: str = "") -> dict[str, Any]:
        """Look up a likely entity in Wikipedia even when the general search tool is configured."""
        entity = (entity or "").strip()
        if not entity or "search_wikipedia" not in self.functions:
            return {"tool": "search_wikipedia", "query": entity, "state_id": state_id, "results": []}
        raw = await self.execute("search_wikipedia", {"entity": entity, "first_sentences": 5})
        raw["results"] = extract_content_evidence(str(raw.get("result", "")), url="")
        raw["query"] = entity
        raw["state_id"] = state_id
        return raw

    @staticmethod
    def _load_default_functions() -> list[Callable[..., Any]]:
        try:
            from tools import SUB_AGENT_TOOLS, SANDBOX_TOOLS

            return list(SUB_AGENT_TOOLS) + list(SANDBOX_TOOLS)
        except Exception:
            return []


def extract_search_results(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        title_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if title_match:
            if current:
                results.append(current)
            current = {"title": title_match.group(1).strip()}
            continue
        if stripped.startswith("URL:") or stripped.startswith("URL："):
            current["url"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Date:") or stripped.startswith("Date："):
            current["date"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Source:") or stripped.startswith("Source："):
            current["source"] = stripped.split(":", 1)[1].strip()
        elif current and stripped and "snippet" not in current and not stripped.startswith("-"):
            current["snippet"] = stripped
    if current:
        results.append(current)
    if not results:
        urls = re.findall(r"https?://[^\s)>\]]+", text or "")
        for url in urls[:5]:
            results.append({"title": url, "url": url, "snippet": ""})
    return results[:10]


def filter_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [item for item in results if not is_low_quality_search_result(item)]
    return filtered or results[:3]


def is_low_quality_search_result(item: dict[str, Any]) -> bool:
    text = " ".join(str(item.get(key, "")) for key in ("title", "snippet", "source", "url")).lower()
    if not text:
        return False
    low_quality_markers = (
        "deepresearch technical report",
        "yunque deepresearch",
        "test/data_with_answer",
        "crossword clue",
        "your search results for",
        "answers for ",
        "/popular/",
        "instagram.com/popular",
        "amphtml/",
    )
    return any(marker in text for marker in low_quality_markers)


def extract_content_evidence(text: str, *, url: str) -> list[dict[str, Any]]:
    payload = _try_json(text)
    if isinstance(payload, dict):
        evidence = []
        candidate = str(payload.get("candidate_entity") or "").strip()
        quote = str(payload.get("evidence_quote") or "").strip()
        for fact in payload.get("facts", []) if isinstance(payload.get("facts"), list) else []:
            claim = str(fact).strip()
            if claim:
                evidence.append(
                    {
                        "title": candidate or payload.get("source") or url,
                        "url": payload.get("url") or url,
                        "snippet": claim,
                        "source": payload.get("source") or url,
                    }
                )
        if quote:
            evidence.append(
                {
                    "title": candidate or payload.get("source") or url,
                    "url": payload.get("url") or url,
                    "snippet": quote,
                    "source": payload.get("source") or url,
                }
            )
        if evidence:
            return evidence[:6]

    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if len(line) < 30 or line.startswith(("http://", "https://", "Error:")):
            continue
        lines.append(line)
        if len(lines) >= 4:
            break
    return [{"title": url or "webpage", "url": url, "snippet": line, "source": url or "webpage"} for line in lines]


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def safe_json_dumps(value: Any, max_chars: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "\n[TRUNCATED]"
