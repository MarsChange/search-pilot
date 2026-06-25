from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_SECRET_MARKERS = ("key", "token", "secret", "password", "authorization")
_SAFE_TOKEN_METRIC_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "original_tokens",
    "compressed_tokens",
}


def emit_runtime_log(event: str, **fields: Any) -> None:
    """Append one runtime diagnostic record without affecting the request path."""
    try:
        path = Path(os.getenv("TIANCHI_AGENT_LOG_FILE", "logs/agent_runtime.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": round(time.time(), 3),
            "event": event,
            **_sanitize(fields),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
    except Exception:
        pass


def summarize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, str):
        return {
            "result_type": "str",
            "chars": len(result),
            "looks_error": result.strip().lower().startswith("error:"),
        }
    if isinstance(result, dict):
        return {
            "result_type": "dict",
            "keys": sorted(str(key) for key in result.keys())[:20],
            "results_count": len(result.get("results") or []) if isinstance(result.get("results"), list) else None,
            "has_error": bool(result.get("error")),
        }
    if isinstance(result, list):
        return {"result_type": "list", "items": len(result)}
    return {"result_type": type(result).__name__}


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str.lower() in _SAFE_TOKEN_METRIC_KEYS:
                cleaned[key_str] = _sanitize(item)
            elif any(marker in key_str.lower() for marker in _SECRET_MARKERS):
                cleaned[key_str] = "[REDACTED]"
            else:
                cleaned[key_str] = _sanitize(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize(item) for item in value[:50]]
    if isinstance(value, str):
        return value if len(value) <= 1000 else value[:1000] + "\n[TRUNCATED]"
    return value
