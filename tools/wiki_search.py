"""
Wikipedia search tools.

Provides functions to search current Wikipedia pages, retrieve historical
revisions, and list available revisions.

Falls back to Jina Reader when Wikipedia API is inaccessible (e.g. from China).
"""

import os
import re
import logging
from contextlib import contextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import requests
import wikipedia
import wikipedia.wikipedia as wiki_internal

logger = logging.getLogger(__name__)

# Set a proper User-Agent to avoid being blocked by Wikipedia API
wikipedia.set_user_agent("TianchiAgent/1.0 (Research Bot; Python/wikipedia)")

# Wikipedia configuration — always use English Wikipedia
_EN_API_URL = "https://en.wikipedia.org/w/api.php"
_EN_DOMAIN = "en.wikipedia.org"

# Set default API URL
wiki_internal.API_URL = _EN_API_URL


@contextmanager
def _wiki_lang(entity: str):
    """Context manager that yields the Wikipedia domain (always English)."""
    yield _EN_DOMAIN


# Jina fallback configuration
_JINA_API_KEY = os.getenv("JINA_API_KEY", "")
_JINA_READER_URL = os.getenv("JINA_READER_URL", "https://r.jina.ai")


def _clean_jina_wikipedia(text: str) -> str:
    """Remove Wikipedia-specific noise from Jina Reader markdown output."""
    # Remove Jina metadata header lines (Title:, URL Source:, Published Time:, Markdown Content:)
    text = re.sub(r'^Title:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^URL Source:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Published Time:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Markdown Content:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^From Wikipedia, the free encyclopedia\s*$', '', text, flags=re.MULTILINE)

    # Remove image links: [![...](image_url)](page_url) or ![...](url)
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Remove [edit] section links
    text = re.sub(r'\[edit\]\(https?://[^\)]*\)', '', text)
    text = re.sub(r'\[\[edit\]\(https?://[^\)]*\)\]', '', text)

    # Remove citation references like [[1]](url), [[2]](url)
    text = re.sub(r'\[\[\d+\]\]\(https?://[^\)]*\)', '', text)

    # Remove inline Wikipedia links but keep display text: [text](https://...wikipedia.org/...)
    text = re.sub(r'\[([^\[\]]+)\]\(https?://[a-z]{2,3}\.wikipedia\.org/[^\)]*\)', r'\1', text)

    # Remove any remaining Wikipedia links (other formats)
    text = re.sub(r'\[([^\[\]]+)\]\(https?://[^\)]*wikipedia[^\)]*\)', r'\1', text)

    # Remove markdown table separators: | --- | --- |
    text = re.sub(r'^\|[\s\-\|:]+\|$', '', text, flags=re.MULTILINE)

    # Clean up excessive pipe-delimited table rows that are just formatting noise
    # Keep rows that have actual content between pipes
    text = re.sub(r'^\|\s*\|\s*$', '', text, flags=re.MULTILINE)

    # Remove "See also", "References", "External links", "Further reading" sections and everything after
    text = re.sub(
        r'\n(?:See also|References|External links|Further reading|Notes|Citations|Bibliography)\s*\n[-=]*\n.*',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove multi-language Wikipedia link blocks at the end (lines with just language links)
    text = re.sub(r'\n\s*\*?\s*\[?https?://[a-z]{2,3}\.wikipedia\.org/[^\n]*', '', text)

    # Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _jina_fallback(entity: str, domain: str = "") -> str:
    """Fallback: fetch Wikipedia page content via Jina Reader when API is blocked."""
    if not _JINA_API_KEY:
        logger.warning("Jina fallback skipped: JINA_API_KEY not set")
        return ""
    if not domain:
        domain = _EN_DOMAIN
    wiki_url = f"https://{domain}/wiki/{entity.replace(' ', '_')}"
    jina_url = f"{_JINA_READER_URL}/{wiki_url}"
    headers = {
        "Authorization": f"Bearer {_JINA_API_KEY}",
        "Accept": "text/plain",
    }
    try:
        logger.info(f"Wikipedia API failed, trying Jina fallback for '{entity}'")
        resp = requests.get(jina_url, headers=headers, timeout=30)
        if resp.status_code == 200 and len(resp.text.strip()) > 50:
            content = _clean_jina_wikipedia(resp.text.strip())
            return (
                f"Page Title: {entity}\n\n"
                f"Content (via Jina fallback):\n{content[:50000]}\n\n"
                f"URL: {wiki_url}"
            )
        logger.warning(
            f"Jina fallback returned status {resp.status_code}, "
            f"body length {len(resp.text)}: {resp.text[:200]}"
        )
    except Exception as e:
        logger.warning(f"Jina fallback also failed for '{entity}': {e}")
    return ""

# Timeout (in seconds) for Wikipedia API calls before falling back to Jina
_WIKIPEDIA_TIMEOUT = 2

def search_wikipedia(entity: str, first_sentences: int = 0) -> str:
    """
    Get Wikipedia page content for a specific entity (people, places, concepts, events). Returns the page title, content (or first N sentences), and URL.

    Args:
        entity: The entity to search for in Wikipedia (e.g. a person's name, place, concept).
        first_sentences: Number of first sentences to return. Set to 0 for full content (default: 0).

    Returns:
        Formatted page content with title, text, and URL. Returns error message if not found.
    """

    def _core() -> str:
        """Core Wikipedia lookup logic (runs in a thread with timeout)."""
        with _wiki_lang(entity) as domain:
            return _core_inner(domain)

    def _core_inner(domain: str) -> str:
        try:
            page = wikipedia.page(title=entity, auto_suggest=False)

            result_parts = [f"Page Title: {page.title}"]

            if first_sentences > 0:
                try:
                    summary = wikipedia.summary(
                        entity, sentences=first_sentences, auto_suggest=False
                    )
                    result_parts.append(
                        f"First {first_sentences} sentences (introduction): {summary}"
                    )
                except Exception:
                    content_sentences = page.content.split(". ")[:first_sentences]
                    summary = (
                        ". ".join(content_sentences) + "."
                        if content_sentences
                        else page.content[:5000] + "..."
                    )
                    result_parts.append(
                        f"First {first_sentences} sentences (introduction): {summary}"
                    )
            else:
                result_parts.append(f"Content: {page.content}")

            result_parts.append(f"URL: {page.url}")

            return "\n\n".join(result_parts)

        except wikipedia.exceptions.DisambiguationError as e:
            options_list = "\n".join(
                [f"- {option}" for option in e.options[:10]]
            )
            output = (
                f"Disambiguation Error: Multiple pages found for '{entity}'.\n\n"
                f"Available options:\n{options_list}\n\n"
                f"Please be more specific in your search query."
            )
            try:
                search_results = wikipedia.search(entity, results=5)
                if search_results:
                    output += f"Try to search {entity} in Wikipedia: {search_results}"
                return output
            except Exception:
                pass
            return output

        except wikipedia.exceptions.PageError:
            try:
                search_results = wikipedia.search(entity, results=5)
                if search_results:
                    suggestion_list = "\n".join(
                        [f"- {result}" for result in search_results[:5]]
                    )
                    return (
                        f"Page Not Found: No Wikipedia page found for '{entity}'.\n\n"
                        f"Similar pages found:\n{suggestion_list}\n\n"
                        f"Try searching for one of these suggestions instead."
                    )
                else:
                    return (
                        f"Page Not Found: No Wikipedia page found for '{entity}' "
                        f"and no similar pages were found."
                    )
            except Exception as search_error:
                return (
                    f"Page Not Found: No Wikipedia page found for '{entity}'. "
                    f"Search for alternatives also failed: {str(search_error)}"
                )

        except wikipedia.exceptions.RedirectError:
            return f"Redirect Error: Failed to follow redirect for '{entity}'"

        except wikipedia.exceptions.WikipediaException as e:
            return f"Wikipedia Error: {str(e)}"

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_core)
        return future.result(timeout=_WIKIPEDIA_TIMEOUT)
    except FuturesTimeoutError:
        logger.warning(
            f"Wikipedia API timed out after {_WIKIPEDIA_TIMEOUT}s for '{entity}', "
            f"falling back to Jina"
        )
        fallback = _jina_fallback(entity)
        if fallback:
            return fallback
        return (
            f"Timeout Error: Wikipedia API did not respond within {_WIKIPEDIA_TIMEOUT}s "
            f"for '{entity}' and Jina fallback is unavailable."
        )
    except (requests.exceptions.RequestException, ConnectionError) as e:
        # Network error — try Jina fallback
        fallback = _jina_fallback(entity)
        if fallback:
            return fallback
        return f"Network Error: Failed to connect to Wikipedia: {str(e)}"
    except Exception as e:
        # Catch-all — also try Jina fallback for unexpected connection issues
        fallback = _jina_fallback(entity)
        if fallback:
            return fallback
        return f"Unexpected Error: {str(e)}"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def search_wikipedia_revision(
    entity: str,
    date: str = "",
    revision_id: int = 0,
) -> str:
    """
    Get historical Wikipedia page content as it appeared at a specific date or revision. Use this when you need past information that may have changed over time.
    Use 'list_wikipedia_revisions' function FIRST when you need historical information but don't know the exact date or revision ID.
    Args:
        entity: The entity/page title to search for in Wikipedia.
        date: Target date in 'YYYY-MM-DD' format. Returns the revision closest to (but not after) this date. Leave empty if using revision_id.
        revision_id: Specific revision ID to retrieve. If provided, date is ignored. Set to 0 to use date instead.

    Returns:
        Historical page content with revision metadata (ID, date, editor, URL).
    """
    if not date and not revision_id:
        return "Error: Either 'date' (YYYY-MM-DD) or 'revision_id' must be provided."

    def _core() -> str:
        return _search_wikipedia_revision_inner(entity, date, revision_id)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_core)
        return future.result(timeout=_WIKIPEDIA_TIMEOUT)
    except FuturesTimeoutError:
        logger.warning(
            f"Wikipedia revision API timed out after {_WIKIPEDIA_TIMEOUT}s for '{entity}', "
            f"falling back to Jina"
        )
        fallback = _jina_fallback(entity)
        if fallback:
            return fallback
        return (
            f"Timeout Error: Wikipedia API did not respond within {_WIKIPEDIA_TIMEOUT}s "
            f"for '{entity}'."
        )
    except Exception as e:
        fallback = _jina_fallback(entity)
        if fallback:
            return fallback
        return f"Unexpected Error: {str(e)}"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _search_wikipedia_revision_inner(entity: str, date: str, revision_id: int) -> str:
    """Inner implementation of search_wikipedia_revision (runs in thread with timeout)."""
    try:
        with _wiki_lang(entity) as domain:
            # Step 1: Resolve page title
            try:
                page = wikipedia.page(title=entity, auto_suggest=False)
                page_title = page.title
            except wikipedia.exceptions.DisambiguationError as e:
                options_list = "\n".join([f"- {option}" for option in e.options[:10]])
                return (
                    f"Disambiguation Error: Multiple pages found for '{entity}'.\n\n"
                    f"Available options:\n{options_list}\n\n"
                    f"Please be more specific."
                )
            except wikipedia.exceptions.PageError:
                return f"Page Not Found: No Wikipedia page found for '{entity}'."

            # Step 2: Get revision ID
            target_revid = revision_id if revision_id else None
            rev_timestamp = None
            rev_comment = None

            if not target_revid and date:
                try:
                    target_date = datetime.strptime(date, "%Y-%m-%d")
                    rvstart = target_date.strftime("%Y-%m-%dT23:59:59Z")
                except ValueError:
                    return f"Error: Invalid date format '{date}'. Use 'YYYY-MM-DD'."

                rev_params = {
                    "action": "query",
                    "prop": "revisions",
                    "titles": page_title,
                    "rvlimit": 1,
                    "rvprop": "ids|timestamp|comment",
                    "rvstart": rvstart,
                    "rvdir": "older",
                }
                rev_data = wiki_internal._wiki_request(rev_params)

                rev_pages = rev_data.get("query", {}).get("pages", {})
                rev_page = list(rev_pages.values())[0]
                revisions = rev_page.get("revisions", [])

                if not revisions:
                    return (
                        f"No revision found for '{entity}' on or before {date}. "
                        f"The page may not have existed at that time."
                    )

                target_revid = revisions[0]["revid"]
                rev_timestamp = revisions[0]["timestamp"]
                rev_comment = revisions[0].get("comment", "No comment")

            # Step 3: Get page content at specific revision
            content_params = {
                "action": "query",
                "prop": "revisions",
                "revids": target_revid,
                "rvprop": "content|timestamp|comment|user",
                "rvslots": "main",
            }
            content_data = wiki_internal._wiki_request(content_params)

            content_pages = content_data.get("query", {}).get("pages", {})
            content_page = list(content_pages.values())[0]

            if "revisions" not in content_page:
                return f"Error: Could not retrieve revision {target_revid} for '{entity}'."

            revision = content_page["revisions"][0]
            raw_content = revision.get("slots", {}).get("main", {}).get("*", "")
            timestamp = revision.get("timestamp", rev_timestamp or "Unknown")
            user = revision.get("user", "Unknown")
            comment = revision.get("comment", rev_comment or "No comment")

            # Step 4: Clean wikitext
            cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", raw_content, flags=re.DOTALL)
            cleaned = re.sub(r"<ref[^/]*/>", "", cleaned)
            cleaned = re.sub(r"\{\{[^}]*\}\}", "", cleaned)
            cleaned = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", cleaned)
            cleaned = re.sub(r"'{2,}", "", cleaned)
            cleaned = re.sub(r"<[^>]+>", "", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = cleaned.strip()

            result_parts = [
                f"Page Title: {page_title}",
                f"Revision ID: {target_revid}",
                f"Revision Date: {timestamp}",
                f"Editor: {user}",
                f"Edit Comment: {comment}",
                f"URL: https://{domain}/w/index.php?title={page_title.replace(' ', '_')}&oldid={target_revid}",
                "",
                f"Content:\n{cleaned[:50000]}"
            ]

            if len(raw_content) > 50000:
                result_parts.append("\n... (content truncated)")

            return "\n\n".join(result_parts)

    except requests.exceptions.RequestException as e:
        return f"Network Error: Failed to connect to Wikipedia API: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


def list_wikipedia_revisions(
    entity: str,
    start_date: str = "",
    end_date: str = "",
    limit: int = 20,
) -> str:
    """
    List available historical revisions for a Wikipedia page. Use this FIRST when you need historical information but don't know the exact date or revision ID.

    Args:
        entity: The entity/page title to search for in Wikipedia.
        start_date: Start of date range in 'YYYY-MM-DD' format (optional, leave empty for no filter).
        end_date: End of date range in 'YYYY-MM-DD' format (optional, leave empty for no filter).
        limit: Maximum number of revisions to return (default: 20, max: 50).

    Returns:
        List of revisions with revision ID, date, editor, and edit comment. Use the revision_id with search_wikipedia_revision() to get full content.
    """
    limit = min(limit, 50)

    def _core() -> str:
        return _list_wikipedia_revisions_inner(entity, start_date, end_date, limit)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_core)
        return future.result(timeout=_WIKIPEDIA_TIMEOUT)
    except FuturesTimeoutError:
        logger.warning(
            f"Wikipedia revisions list API timed out after {_WIKIPEDIA_TIMEOUT}s for '{entity}'"
        )
        return (
            f"Timeout Error: Wikipedia API did not respond within {_WIKIPEDIA_TIMEOUT}s "
            f"for '{entity}'."
        )
    except Exception as e:
        return f"Unexpected Error: {str(e)}"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _list_wikipedia_revisions_inner(
    entity: str, start_date: str, end_date: str, limit: int
) -> str:
    """Inner implementation of list_wikipedia_revisions (runs in thread with timeout)."""
    try:
        with _wiki_lang(entity) as domain:
            try:
                page = wikipedia.page(title=entity, auto_suggest=False)
                page_title = page.title
            except wikipedia.exceptions.DisambiguationError as e:
                options_list = "\n".join([f"- {option}" for option in e.options[:10]])
                return (
                    f"Disambiguation Error: Multiple pages found for '{entity}'.\n\n"
                    f"Available options:\n{options_list}\n\n"
                    f"Please be more specific."
                )
            except wikipedia.exceptions.PageError:
                return f"Page Not Found: No Wikipedia page found for '{entity}'."

            rev_params = {
                "action": "query",
                "prop": "revisions",
                "titles": page_title,
                "rvlimit": limit,
                "rvprop": "ids|timestamp|user|comment|size",
            }

            if start_date:
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    rev_params["rvend"] = start_dt.strftime("%Y-%m-%dT00:00:00Z")
                except ValueError:
                    return f"Error: Invalid start_date format '{start_date}'. Use 'YYYY-MM-DD'."

            if end_date:
                try:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    rev_params["rvstart"] = end_dt.strftime("%Y-%m-%dT23:59:59Z")
                except ValueError:
                    return f"Error: Invalid end_date format '{end_date}'. Use 'YYYY-MM-DD'."

            rev_params["rvdir"] = "older"

            rev_data = wiki_internal._wiki_request(rev_params)

            rev_pages = rev_data.get("query", {}).get("pages", {})
            rev_page = list(rev_pages.values())[0]
            revisions = rev_page.get("revisions", [])

            if not revisions:
                date_range_msg = ""
                if start_date or end_date:
                    date_range_msg = f" in the specified date range ({start_date or 'beginning'} to {end_date or 'now'})"
                return f"No revisions found for '{entity}'{date_range_msg}."

            result_parts = [
                f"Page Title: {page_title}",
                f"Revisions Found: {len(revisions)}",
                f"History URL: https://{domain}/w/index.php?title={page_title.replace(' ', '_')}&action=history",
                "",
                "Available Revisions:",
                "-" * 80,
            ]

            for rev in revisions:
                rev_id = rev.get("revid", "Unknown")
                timestamp = rev.get("timestamp", "Unknown")
                user = rev.get("user", "Unknown")
                comment = rev.get("comment", "")[:100]
                size = rev.get("size", 0)

                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = timestamp

                result_parts.append(
                    f"  ID: {rev_id} | Date: {date_str} | Editor: {user} | Size: {size} bytes"
                )
                if comment:
                    result_parts.append(f"    Comment: {comment}")
                result_parts.append("")

            result_parts.append("-" * 80)
            result_parts.append(
                "To get content from a specific revision, use search_wikipedia_revision "
                f"with entity=\"{page_title}\" and revision_id=<ID>"
            )

            return "\n".join(result_parts)

    except requests.exceptions.RequestException as e:
        return f"Network Error: Failed to connect to Wikipedia API: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


WIKI_SEARCH_TOOLS = [search_wikipedia, search_wikipedia_revision, list_wikipedia_revisions]
