"""
Dual-engine web search tool using Serper (Google) and Alibaba Cloud IQS.

Supports:
- explicit engine selection
- language-based auto routing
- cross-engine fallback on poor results
- query simplification retry
"""

import json
import logging
import os
import re
import urllib.parse
from typing import Literal

import requests

logger = logging.getLogger(__name__)

IQS_API_KEY = os.getenv("IQS_API_KEY", "").strip()
IQS_BASE = "https://cloud-iqs.aliyuncs.com"
IQS_TIMEOUT = 15

SERPER_BASE = "https://google.serper.dev/search"
SERPER_TIMEOUT = 30

_dead_keys: set[str] = set()


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    placeholder_markers = (
        "your",
        "replace_me",
        "example",
        "placeholder",
        "xxxx",
        "changeme",
    )
    return any(marker in lowered for marker in placeholder_markers)


def _is_valid_serper_key(value: str) -> bool:
    key = value.strip()
    if not key:
        return False
    if _looks_like_placeholder(key):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{20,}", key))


def _parse_serper_pool(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []

    parts = re.split(r"[\s,;]+", raw_value.strip())
    seen: set[str] = set()
    keys: list[str] = []
    for part in parts:
        key = part.strip()
        if not key or key in seen:
            continue
        if not _is_valid_serper_key(key):
            masked = (key[:6] + "..." + key[-4:]) if len(key) > 12 else key
            logger.warning(
                "[Serper] Ignoring invalid key in SERPER_API_KEYS: %s",
                masked,
            )
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _configured_key_pool() -> list[str]:
    return _parse_serper_pool(os.getenv("SERPER_API_KEYS"))


def _get_ordered_serper_keys() -> list[str]:
    keys = [key for key in _configured_key_pool() if key not in _dead_keys]
    env_key = os.getenv("SERPER_API_KEY", "").strip()
    if env_key:
        if not _is_valid_serper_key(env_key):
            masked = (env_key[:6] + "..." + env_key[-4:]) if len(env_key) > 12 else env_key
            logger.warning("[Serper] Ignoring invalid SERPER_API_KEY: %s", masked)
        elif env_key not in _dead_keys and env_key not in keys:
            keys.append(env_key)
    return keys


def _has_iqs_key() -> bool:
    return bool(IQS_API_KEY and not _looks_like_placeholder(IQS_API_KEY))


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text or "")


def _simplify_query(query: str) -> str:
    if not query or len(query) < 10:
        return query

    simplified = re.sub(
        r"\b(which|that|whose|who|when|where|what|how)\b|的|是|在|于|以及|并且",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if len(simplified) < len(query) * 0.35:
        return query
    return simplified or query


def _is_poor_result(result: str) -> bool:
    if not result:
        return True
    if len(result) < 120:
        return True
    poor_markers = (
        "No results found",
        "Search failed",
        "Error:",
        "Results Found: 0",
        "found 0 results",
    )
    return any(marker in result for marker in poor_markers)


def _format_result_header(query: str, results_count: int, engine_name: str) -> list[str]:
    return [
        f"Search Query: {query}",
        f"Engine: {engine_name}",
        f"Results Found: {results_count}",
        "-" * 50,
    ]


def _format_serper_results(query: str, data: dict, num_results: int) -> str:
    results = []
    for item in data.get("organic", [])[:num_results]:
        result = {
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        if "date" in item:
            result["date"] = item["date"]
        results.append(result)

    answer_box = data.get("answerBox")
    knowledge_graph = data.get("knowledgeGraph")
    lines = _format_result_header(query, len(results), "google/serper")

    if answer_box:
        lines.append("\n[Answer Box]")
        if answer_box.get("title"):
            lines.append(f"Title: {answer_box['title']}")
        answer = answer_box.get("answer", answer_box.get("snippet", ""))
        if answer:
            lines.append(f"Answer: {answer}")
        lines.append("")

    if knowledge_graph:
        lines.append("\n[Knowledge Graph]")
        if knowledge_graph.get("title"):
            lines.append(f"Title: {knowledge_graph['title']}")
        if knowledge_graph.get("type"):
            lines.append(f"Type: {knowledge_graph['type']}")
        if knowledge_graph.get("description"):
            lines.append(f"Description: {knowledge_graph['description']}")
        lines.append("")

    lines.append("\n[Search Results]")
    for index, result in enumerate(results, 1):
        lines.append(f"\n{index}. {result['title']}")
        lines.append(f"   URL: {result['link']}")
        if result.get("snippet"):
            lines.append(f"   {result['snippet']}")
        if result.get("date"):
            lines.append(f"   Date: {result['date']}")

    return "\n".join(lines)


def _format_iqs_results(query: str, page_items: list, num_results: int) -> str:
    trimmed_items = page_items[:num_results]
    if not trimmed_items:
        return f"No results found for '{query}'. Try with a more general query."

    lines = _format_result_header(query, len(trimmed_items), "alibaba-cloud-iqs")
    lines.append("\n[Search Results]")

    for index, item in enumerate(trimmed_items, 1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet") or item.get("htmlSnippet") or ""
        publish_time = item.get("publishTime", "")
        hostname = item.get("hostname", "")

        lines.append(f"\n{index}. {title}")
        lines.append(f"   URL: {link}")
        if hostname:
            lines.append(f"   Source: {hostname}")
        if publish_time:
            lines.append(f"   Date: {publish_time}")
        if snippet:
            lines.append(f"   {snippet}")

    return "\n".join(lines)


def _do_serper_request(api_key: str, payload: dict) -> requests.Response:
    return requests.post(
        SERPER_BASE,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=SERPER_TIMEOUT,
    )


def serper_search(query: str, num_results: int = 20, language: str = "en") -> str:
    keys = _get_ordered_serper_keys()
    if not keys:
        return "Error: No available Serper API keys."

    payload = {"q": query, "num": num_results, "hl": language}
    last_error = ""

    for key in keys:
        try:
            response = _do_serper_request(key, payload)

            if response.status_code in (400, 403):
                masked = key[:6] + "..." + key[-4:]
                content_type = response.headers.get("content-type", "")
                if response.status_code == 403 and "text/html" in content_type:
                    logger.warning("[Serper] Network-level HTML 403 encountered")
                logger.warning(
                    "[Serper] Key %s returned %s, marking as dead",
                    masked,
                    response.status_code,
                )
                _dead_keys.add(key)
                continue

            response.raise_for_status()
            return _format_serper_results(query, response.json(), num_results)
        except requests.exceptions.Timeout:
            last_error = "Request timed out"
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
        except json.JSONDecodeError:
            last_error = "Failed to parse API response"

    return f"Error: All Serper API keys failed. Last error: {last_error}"


def iqs_search(query: str, num_results: int = 20) -> str:
    if not _has_iqs_key():
        return "Error: IQS_API_KEY is not configured."

    headers = {"X-API-Key": IQS_API_KEY}
    params = {
        "query": query,
        "timeRange": "NoLimit",
    }

    last_error = ""
    for _ in range(3):
        try:
            response = requests.get(
                f"{IQS_BASE}/search/genericSearch",
                headers=headers,
                params=params,
                timeout=IQS_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return _format_iqs_results(query, data.get("pageItems", []), num_results)
        except requests.exceptions.Timeout:
            last_error = "Request timed out"
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
        except json.JSONDecodeError:
            last_error = "Failed to parse IQS response"

    return f"Error: IQS search failed. Last error: {last_error}"


def _run_engine(
    query: str,
    *,
    engine: Literal["google", "iqs"],
    num_results: int,
    language: str,
) -> str:
    if engine == "google":
        return serper_search(query, num_results=num_results, language=language)
    return iqs_search(query, num_results=num_results)


def _select_primary_engine(query: str, engine: Literal["auto", "google", "iqs"]) -> Literal["google", "iqs"]:
    if engine == "google":
        return "google"
    if engine == "iqs":
        return "iqs"
    return "iqs" if _contains_chinese(query) else "google"


def search_engine(
    query: str,
    num_results: int = 20,
    language: str = "en",
    engine: Literal["auto", "google", "iqs"] = "auto",
) -> str:
    """
    Search the web using Serper (Google) and Alibaba Cloud IQS with automatic routing and fallback.

    Args:
        query: The search query string.
        num_results: Number of results to return.
        language: Preferred language code for Google results, e.g. 'en' or 'zh-cn'.
        engine: Search engine selection. Use 'auto' to route by query language,
            'google' to force Serper, or 'iqs' to force Alibaba Cloud IQS.

    Returns:
        Formatted search results text.
    """
    query = (query or "").strip()
    if not query:
        return "Error: Query is empty."

    primary_engine = _select_primary_engine(query, engine)
    fallback_engine: Literal["google", "iqs"] = "iqs" if primary_engine == "google" else "google"

    logger.info(
        "[Search] Query=%r primary_engine=%s explicit_engine=%s",
        query[:80],
        primary_engine,
        engine,
    )

    primary_result = _run_engine(
        query,
        engine=primary_engine,
        num_results=num_results,
        language=language,
    )
    if not _is_poor_result(primary_result):
        return primary_result

    logger.info(
        "[Search] Poor %s result for query=%r, trying %s fallback",
        primary_engine,
        query[:80],
        fallback_engine,
    )
    fallback_result = _run_engine(
        query,
        engine=fallback_engine,
        num_results=num_results,
        language=language,
    )
    if not _is_poor_result(fallback_result):
        return fallback_result

    simplified_query = _simplify_query(query)
    if simplified_query != query:
        logger.info(
            "[Search] Both engines poor for query=%r, retrying simplified query=%r on %s",
            query[:80],
            simplified_query[:80],
            primary_engine,
        )
        simplified_result = _run_engine(
            simplified_query,
            engine=primary_engine,
            num_results=num_results,
            language=language,
        )
        if not _is_poor_result(simplified_result):
            return simplified_result

    return fallback_result if len(fallback_result) >= len(primary_result) else primary_result


SEARCH_ENGINE_TOOLS = []
if _configured_key_pool() or os.getenv("SERPER_API_KEY") or _has_iqs_key():
    SEARCH_ENGINE_TOOLS = [search_engine]
