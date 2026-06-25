from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol

from openai import AsyncOpenAI

from deep_research.runtime_logging import emit_runtime_log


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        ...


class DashScopeLLM:
    """OpenAI-compatible DashScope/Qwen client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model or os.getenv("QWEN_MODEL", "qwen-max")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        self.client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)
        self._usage_totals = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated": False,
        }

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        started = time.monotonic()
        model = kwargs.pop("model", self.model)
        temperature = kwargs.pop("temperature", 0.1)
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        usage = self._extract_usage(response, messages, content)
        self._record_usage(usage)
        emit_runtime_log(
            "llm_call_end",
            model=model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            estimated=usage["estimated"],
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        return content

    def get_usage(self) -> dict[str, Any]:
        return dict(self._usage_totals)

    def _record_usage(self, usage: dict[str, Any]) -> None:
        self._usage_totals["calls"] += 1
        self._usage_totals["prompt_tokens"] += int(usage["prompt_tokens"])
        self._usage_totals["completion_tokens"] += int(usage["completion_tokens"])
        self._usage_totals["total_tokens"] += int(usage["total_tokens"])
        self._usage_totals["estimated"] = bool(self._usage_totals["estimated"] or usage["estimated"])

    @staticmethod
    def _extract_usage(response: Any, messages: list[dict[str, Any]], content: str) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
        estimated = False
        if not total_tokens:
            prompt_tokens = estimate_message_tokens(messages)
            completion_tokens = estimate_text_tokens(content)
            total_tokens = prompt_tokens + completion_tokens
            estimated = True
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated": estimated,
        }


def estimate_text_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 3.5))


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("role", ""))) + len(str(message.get("content", "")))
    return max(1, int(total_chars / 3.5))


def strip_markdown_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = strip_markdown_fence(text)
    if not raw:
        return None
    for candidate in _json_candidates(raw):
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def extract_json_array(text: str) -> list[Any] | None:
    raw = strip_markdown_fence(text)
    if not raw:
        return None
    start = raw.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        char = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                candidate = re.sub(r",(\s*[}\]])", r"\1", raw[start:idx + 1])
                try:
                    data = json.loads(candidate)
                    return data if isinstance(data, list) else None
                except json.JSONDecodeError:
                    return None
    return None


def _json_candidates(raw: str) -> list[str]:
    candidates = [raw]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidates.extend(fenced)
    start = raw.find("{")
    if start == -1:
        return candidates
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        char = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(raw[start:idx + 1])
                break
    return candidates
