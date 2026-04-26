"""
Webpage Analyzer sub-agent tool.

Reads webpage content and returns compact structured evidence for the worker.
"""

import asyncio
import json
import logging
import os
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 15000

ANALYSIS_SYSTEM_PROMPT = """You extract question-relevant evidence from a webpage.

Return ONLY one JSON object with this schema:
{
  "relevance": "high|medium|low|none",
  "candidate_entity": "best matching entity or empty string",
  "facts": ["short factual findings tied to the question"],
  "evidence_quote": "short direct quote or empty string",
  "evidence_source": "url or section label",
  "missing_info": ["what the page does not answer"]
}

Rules:
- No markdown fences.
- No prose outside JSON.
- Facts must be concise and useful for answering the question.
- If the page is not useful, say so with low/none relevance.
"""


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _fallback_payload(url: str, content: str, error: str = "") -> str:
    payload = {
        "relevance": "low",
        "candidate_entity": "",
        "facts": [f"Unable to produce structured page analysis for {url}."],
        "evidence_quote": "",
        "evidence_source": url,
        "missing_info": [error or "LLM analysis unavailable; inspect raw content manually."],
    }
    if content:
        payload["facts"].append(content[:300].replace("\n", " "))
    return json.dumps(payload, ensure_ascii=False)


async def analyze_webpage(url: str, question: str) -> str:
    """
    Read a webpage and analyze its content for information relevant to the research question.

    Args:
        url: The URL of the webpage to read and analyze.
        question: The research question for context.

    Returns:
        A compact JSON string containing structured evidence.
    """
    content, error = await _fetch_content(url)
    if error:
        return _fallback_payload(url, "", error)

    if not content or len(content.strip()) < 50:
        return _fallback_payload(url, "", "Page content is empty or too short.")

    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH] + "\n\n...[content truncated]"

    return await _analyze_with_llm(content, url, question)


async def _fetch_content(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fetch webpage content using scrape_website."""
    from tools.scrape_website import scrape_website

    try:
        result = await asyncio.to_thread(scrape_website, url)
        if result.startswith("Error:"):
            return None, result
        return result, None
    except Exception as exc:
        return None, str(exc)


async def _analyze_with_llm(content: str, url: str, question: str) -> str:
    """Use LLM to analyze webpage content against the research question."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return _fallback_payload(url, content[:800], "DASHSCOPE_API_KEY not set.")

    model = os.getenv("QWEN_MODEL") or "qwen-max"
    client = AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )

    user_message = f"""## Research Question
{question}

## Webpage URL
{url}

## Webpage Content
{content}
"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        cleaned = _strip_code_fence(raw)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Analysis result is not a JSON object")
        data.setdefault("evidence_source", url)
        return json.dumps(data, ensure_ascii=False)
    except Exception as exc:
        logger.warning("LLM analysis failed for %s: %s", url, exc)
        return _fallback_payload(url, content[:800], str(exc))


WEBPAGE_ANALYZER_TOOLS = [
    analyze_webpage,
]
