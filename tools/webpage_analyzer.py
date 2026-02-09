"""
Webpage Analyzer sub-agent tool.

A tool that fetches a webpage, reads its content, and uses an LLM to analyze
whether it contains information relevant to the research question.

This encapsulates the full read-analyze cycle into a single tool call,
so the main agent doesn't need separate navigate/snapshot/analyze steps per page.

Environment variables:
- DASHSCOPE_API_KEY: Required for LLM analysis
- QWEN_MODEL: Model to use (optional, defaults to qwen-max)
- JINA_API_KEY: Required for Jina Reader content extraction
- JINA_READER_URL: Jina Reader URL (optional, defaults to https://r.jina.ai)
"""

import logging
import os
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Max content length to send to LLM (characters)
MAX_CONTENT_LENGTH = 15000

# LLM analysis prompt
ANALYSIS_SYSTEM_PROMPT = """You are a research assistant specialized in extracting relevant information from web pages.

Your task:
1. Read the provided webpage content carefully
2. Identify ALL information that is relevant to the research question
3. Extract key facts, names, dates, numbers, and relationships
4. Clearly state what was found and what was NOT found on this page

Output format:
- **Relevance**: High/Medium/Low/None
- **Key Findings**: List specific facts found that relate to the question
- **Extracted Details**: Names, dates, numbers, quotes directly from the content
- **Missing Information**: What the question asks for that this page does NOT answer
- **Useful Leads**: Any links, references, or clues that could help find the answer elsewhere

Be thorough but concise. Focus only on information relevant to the question."""


async def analyze_webpage(url: str, question: str) -> str:
    """
    Read a webpage and analyze its content for information relevant to the research question.

    This sub-agent tool handles the full read-analyze cycle:
    1. Fetches the webpage content using Jina Reader (with fallback)
    2. Analyzes the content with AI to extract relevant information

    Use this tool after getting search results from search_engine to deeply
    analyze specific web pages. Call this tool for each relevant URL.

    Args:
        url: The URL of the webpage to read and analyze
        question: The original research question for context

    Returns:
        Analysis report with relevance assessment, key findings, and extracted details
    """
    # Step 1: Fetch webpage content
    content, error = _fetch_content(url)
    if error:
        return f"**Failed to fetch page**: {error}\n\nURL: {url}"

    if not content or len(content.strip()) < 50:
        return f"**Page content is empty or too short** for URL: {url}"

    # Step 2: Truncate if too long
    original_length = len(content)
    if original_length > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH] + "\n\n...[content truncated]"

    # Step 3: Analyze with LLM
    analysis = await _analyze_with_llm(content, url, question)
    return analysis


def _fetch_content(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fetch webpage content using scrape_website tool."""
    from tools.scrape_website import scrape_website

    try:
        result = scrape_website(url)
        if result.startswith("Error:"):
            return None, result
        return result, None
    except Exception as e:
        return None, str(e)


async def _analyze_with_llm(content: str, url: str, question: str) -> str:
    """Use LLM to analyze webpage content against the research question."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return f"**LLM analysis unavailable** (DASHSCOPE_API_KEY not set)\n\nRaw content preview:\n{content[:2000]}"

    model = os.getenv("QWEN_MODEL")

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

Please analyze this webpage content and extract all information relevant to the research question above."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or "No analysis generated."
    except Exception as e:
        logger.warning(f"LLM analysis failed for {url}: {e}")
        # Return raw content preview as fallback
        return f"**LLM analysis failed**: {e}\n\nRaw content preview:\n{content[:3000]}"


# Export tool functions
WEBPAGE_ANALYZER_TOOLS = [
    analyze_webpage,
]
