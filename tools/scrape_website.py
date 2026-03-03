"""
Website scraping tool using Jina Reader API with fallback to requests + MarkItDown.

Requires JINA_API_KEY environment variable. Optionally uses JINA_READER_URL.
"""

import io
import logging
import os
import re
from typing import Optional

import requests
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

# Patterns that indicate Jina returned a blocked/invalid page
_JINA_BLOCKED_PATTERNS = [
    "Target URL returned error 403",
    "requiring CAPTCHA",
    "Just a moment...",
    "Verify you are human",
    "Access denied",
    "Enable JavaScript and cookies to continue",
]


def _is_blocked_content(content: str) -> bool:
    """Check if content looks like a CAPTCHA/anti-bot page rather than real content."""
    if len(content.strip()) < 200:
        return True
    return any(pattern in content for pattern in _JINA_BLOCKED_PATTERNS)


def _clean_scraped_markdown(text: str) -> str:
    """Remove noisy web elements from scraped markdown to reduce token count.

    Strips images, hyperlinks (keeping display text), navigation elements,
    Jina metadata headers, cookie/privacy banners, social media widgets,
    and other non-content boilerplate.
    """
    # --- Jina metadata header lines ---
    text = re.sub(r'^Title:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^URL Source:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Published Time:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Markdown Content:.*$', '', text, flags=re.MULTILINE)

    # --- Images ---
    # Linked images: [![alt](img_url)](page_url)
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    # Standalone images: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # --- Hyperlinks: keep display text, remove URL ---
    # [display text](url) → display text
    text = re.sub(r'\[([^\[\]]+)\]\(https?://[^\)]*\)', r'\1', text)
    # [display text](relative/path) → display text
    text = re.sub(r'\[([^\[\]]+)\]\(/[^\)]*\)', r'\1', text)
    # [display text](#anchor) → display text
    text = re.sub(r'\[([^\[\]]+)\]\(#[^\)]*\)', r'\1', text)

    # --- Bare URLs on their own line ---
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)

    # --- HTML remnants ---
    # Inline HTML tags
    text = re.sub(r'<(?:img|iframe|script|style|link|meta|noscript|svg|video|audio|source|picture|embed|object)[^>]*/?>', '', text, flags=re.IGNORECASE)
    # Block HTML tags with content
    text = re.sub(r'<(script|style|noscript|svg)\b[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Data attributes and aria labels (often left as noise)
    text = re.sub(r'\s+(?:data-\w+|aria-\w+)="[^"]*"', '', text)

    # --- Navigation / boilerplate lines ---
    # Lines that are just "Skip to content", "Menu", "Close", "Toggle navigation", etc.
    text = re.sub(
        r'^\s*(?:Skip to (?:main )?content|Toggle (?:navigation|menu)|'
        r'Close menu|Open menu|Back to top|'
        r'Cookie (?:Policy|Settings|Preferences)|Accept (?:all )?cookies|'
        r'Privacy Policy|Terms (?:of (?:Service|Use))|'
        r'Share (?:on|this)|Tweet|Pin it|'
        r'Follow us|Subscribe|Newsletter|'
        r'Previous|Next|Read more|Show more|Load more|'
        r'©\s*\d{4}.*$)\s*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )

    # --- Citation references like [1], [2], [edit], [citation needed] ---
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)

    # --- Empty markdown table separators ---
    text = re.sub(r'^\|[\s\-\|:]+\|$', '', text, flags=re.MULTILINE)
    # Empty table rows
    text = re.sub(r'^\|\s*\|\s*$', '', text, flags=re.MULTILINE)

    # --- Excessive whitespace ---
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Lines with only whitespace
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)

    return text.strip()


def _scrape_by_jina(url: str) -> tuple[Optional[str], Optional[str]]:
    """Scrape using Jina Reader API. Returns (content, None) or (None, error)."""
    jina_api_key = os.getenv("JINA_API_KEY", "")
    if not jina_api_key:
        return None, "JINA_API_KEY is not set"

    jina_reader_url = os.getenv("JINA_READER_URL", "https://r.jina.ai")
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
        logger.info(f"Jina scraping: {url}")
        response = requests.get(jina_url, headers=jina_headers, timeout=30)

        if response.status_code == 422:
            return None, "Jina 422: URL may point to a file, not supported"

        response.raise_for_status()
        content = response.text

        # Jina sometimes returns a partial page; retry once with longer timeout
        if "Warning: This page maybe not yet fully loaded" in content:
            logger.info(f"Jina: page not fully loaded, retrying: {url}")
            response = requests.get(jina_url, headers=jina_headers, timeout=60)
            response.raise_for_status()
            content = response.text

        # Check if we got a real page or a CAPTCHA/blocked page
        if _is_blocked_content(content):
            return None, f"Jina returned blocked/CAPTCHA page for {url}"

        content = _clean_scraped_markdown(content)
        logger.info(f"Jina success: {url} ({len(content)} chars after cleaning)")
        return content, None

    except requests.exceptions.ConnectionError as e:
        return None, f"Jina connection failed (network unreachable?): {e}"
    except requests.exceptions.HTTPError as e:
        return None, f"Jina HTTP error: {e}"
    except requests.exceptions.Timeout:
        return None, f"Jina request timed out (30s) for {url}"
    except Exception as e:
        return None, f"Jina scraping failed: {e}"


def _scrape_request(url: str) -> tuple[Optional[str], Optional[str]]:
    """Fallback scraping using requests + MarkItDown. Returns (content, None) or (None, error)."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        }
        logger.info(f"Requests fallback scraping: {url}")
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Try converting to markdown first
        try:
            stream = io.BytesIO(response.content)
            md = MarkItDown()
            content = md.convert_stream(stream).text_content
            if content and len(content.strip()) > 50:
                return _clean_scraped_markdown(content), None
        except Exception:
            pass

        # Fall back to raw HTML text
        if response.text and len(response.text.strip()) > 50:
            return _clean_scraped_markdown(response.text), None

        return None, "Page content is empty or too short"

    except requests.exceptions.ConnectionError as e:
        return None, f"Connection failed (network unreachable?): {e}"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e}"
    except requests.exceptions.Timeout:
        return None, f"Request timed out (60s) for {url}"
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

    # Try Jina Reader first (Jina servers can access foreign sites even from China)
    content, jina_error = _scrape_by_jina(url)
    if content is not None:
        return content

    logger.warning(f"Jina failed ({jina_error}), falling back to requests: {url}")

    # Fallback to requests + MarkItDown (needs direct network access or proxy)
    content, req_error = _scrape_request(url)
    if content is not None:
        return content

    return f"Error: Both scraping methods failed.\nJina: {jina_error}\nRequests: {req_error}"


SCRAPE_WEBSITE_TOOLS = []
if os.getenv("JINA_API_KEY"):
    SCRAPE_WEBSITE_TOOLS = [scrape_website]
