"""
Website scraping tool using Jina Reader API with fallback to requests + MarkItDown.

Requires JINA_API_KEY environment variable. Optionally uses JINA_READER_URL.
"""

import io
import os
from typing import Optional

import requests
from markitdown import MarkItDown


def _scrape_by_jina(url: str) -> tuple[Optional[str], Optional[str]]:
    """Scrape using Jina Reader API."""
    jina_api_key = os.getenv("JINA_API_KEY", "")
    if not jina_api_key:
        return None, "JINA_API_KEY is not set."

    jina_reader_url = os.getenv("JINA_READER_URL", "https://r.jina.ai")
    if not jina_reader_url:
        return None, "JINA_READER_URL is not set."

    jina_url = f"{jina_reader_url}/{url}"
    jina_headers = {
        "Authorization": f"Bearer {jina_api_key}",
        "X-Base": "final",
        "X-Engine": "browser",
        "X-With-Generated-Alt": "true",
        "X-With-Iframe": "true",
        "X-With-Shadow-Dom": "true",
    }

    try:
        response = requests.get(jina_url, headers=jina_headers, timeout=120)
        if response.status_code == 422:
            return (
                None,
                "Jina 422 error: URL may point to a file. This tool does not support files.",
            )
        response.raise_for_status()
        content = response.text

        if "Warning: This page maybe not yet fully loaded" in content:
            response = requests.get(jina_url, headers=jina_headers, timeout=300)
            if response.status_code == 422:
                return (
                    None,
                    "Jina 422 error: URL may point to a file.",
                )
            response.raise_for_status()
            content = response.text

        return content, None
    except Exception as e:
        return None, f"Jina scraping failed: {str(e)}"


def _scrape_request(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fallback scraping using requests + MarkItDown."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        try:
            stream = io.BytesIO(response.content)
            md = MarkItDown()
            content = md.convert_stream(stream).text_content
            return content, None
        except Exception:
            return response.text, None

    except Exception as e:
        return None, str(e)


def scrape_website(url: str) -> str:
    """
    Scrape and extract readable content from a webpage. Converts HTML to clean markdown text using Jina Reader API with fallback to direct requests. Use this to read the full content of a webpage.

    Args:
        url: The URL of the website to scrape (e.g. 'https://example.com/page').

    Returns:
        The webpage content converted to markdown text, or an error message.
    """
    if not url:
        return "Error: URL is empty. Please provide a valid URL."

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
        return "Error: Cannot scrape Hugging Face datasets/spaces. Use other tools for this."

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            content, error = _scrape_by_jina(url)
            if content is not None:
                return content

            # Fallback to requests
            content, error_requests = _scrape_request(url)
            if content is not None:
                return content

            return f"Error: Both scraping methods failed.\nJina: {error}\nRequests: {error_requests}"
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                return f"Error: Failed after {max_retries} retries: {str(e)}"

    return "Error: Exceeded maximum retries."


SCRAPE_WEBSITE_TOOLS = []
if os.getenv("JINA_API_KEY"):
    SCRAPE_WEBSITE_TOOLS = [scrape_website]
