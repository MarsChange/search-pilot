"""
Wikipedia search tools.

Provides functions to search current Wikipedia pages, retrieve historical
revisions, and list available revisions.
"""

import re
import time
import logging
from datetime import datetime
import requests
import wikipedia
import wikipedia.wikipedia as wiki_internal

logger = logging.getLogger(__name__)

# Set a proper User-Agent to avoid being blocked by Wikipedia API
wikipedia.set_user_agent("TianchiAgent/1.0 (Research Bot; Python/wikipedia)")

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # seconds, doubles each retry


def _retry_on_network_error(func, *args, **kwargs):
    """Retry a function call on network errors with exponential backoff."""
    last_error: Exception = RuntimeError("No retries attempted")
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ConnectionResetError,
                ConnectionAbortedError) as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                wait_time = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"Wikipedia network error (attempt {attempt + 1}/{_MAX_RETRIES}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Wikipedia network error after {_MAX_RETRIES} attempts: {e}")
    raise last_error


def search_wikipedia(entity: str, first_sentences: int = 0) -> str:
    """
    Get Wikipedia page content for a specific entity (people, places, concepts, events). Returns the page title, content (or first N sentences), and URL.

    Args:
        entity: The entity to search for in Wikipedia (e.g. a person's name, place, concept).
        first_sentences: Number of first sentences to return. Set to 0 for full content (default: 0).

    Returns:
        Formatted page content with title, text, and URL. Returns error message if not found.
    """
    try:
        page = _retry_on_network_error(wikipedia.page, title=entity, auto_suggest=False)

        result_parts = [f"Page Title: {page.title}"]

        if first_sentences > 0:
            try:
                summary = _retry_on_network_error(
                    wikipedia.summary,
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
            search_results = _retry_on_network_error(wikipedia.search, entity, results=5)
            if search_results:
                output += f"Try to search {entity} in Wikipedia: {search_results}"
            return output
        except Exception:
            pass
        return output

    except wikipedia.exceptions.PageError:
        try:
            search_results = _retry_on_network_error(wikipedia.search, entity, results=5)
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

    except requests.exceptions.RequestException as e:
        return f"Network Error: Failed to connect to Wikipedia: {str(e)}"

    except wikipedia.exceptions.WikipediaException as e:
        return f"Wikipedia Error: {str(e)}"

    except Exception as e:
        return f"Unexpected Error: {str(e)}"


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

    try:
        # Step 1: Resolve page title
        try:
            page = _retry_on_network_error(wikipedia.page, title=entity, auto_suggest=False)
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
            rev_data = _retry_on_network_error(wiki_internal._wiki_request, rev_params)

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
        content_data = _retry_on_network_error(wiki_internal._wiki_request, content_params)

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
            f"URL: https://en.wikipedia.org/w/index.php?title={page_title.replace(' ', '_')}&oldid={target_revid}",
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

    try:
        try:
            page = _retry_on_network_error(wikipedia.page, title=entity, auto_suggest=False)
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

        rev_data = _retry_on_network_error(wiki_internal._wiki_request, rev_params)

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
            f"History URL: https://en.wikipedia.org/w/index.php?title={page_title.replace(' ', '_')}&action=history",
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
