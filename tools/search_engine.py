"""
Google Search tool using Serper API.

Requires SERPER_API_KEY environment variable.
"""

import json
import os

import requests


def search_engine(
    query: str,
    num_results: int = 20,
    language: str = "en",
) -> str:
    """
    Search the web using Google via Serper API. Returns formatted search results including titles, URLs, snippets, answer boxes, and knowledge graph data.

    Args:
        query: The search query string (use Chinese keywords for Chinese questions, English for English questions).
        num_results: Number of results to return (default: 20).
        language: Language code for results, e.g. 'en' for English, 'zh-cn' for Chinese (default: 'en').

    Returns:
        Formatted search results text with titles, URLs, and snippets.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"

    payload = {
        "q": query,
        "num": num_results,
        "hl": language,
    }

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Extract organic search results
        results = []
        organic_results = data.get("organic", [])

        for item in organic_results[:num_results]:
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            if "date" in item:
                result["date"] = item["date"]
            results.append(result)

        # Include answer box if available
        answer_box = data.get("answerBox")
        knowledge_graph = data.get("knowledgeGraph")

        # Format results for display
        lines = []
        lines.append(f"Search Query: {query}")
        lines.append(f"Results Found: {len(results)}")
        lines.append("-" * 50)

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
        for i, result in enumerate(results, 1):
            lines.append(f"\n{i}. {result['title']}")
            lines.append(f"   URL: {result['link']}")
            if result.get("snippet"):
                lines.append(f"   {result['snippet']}")
            if result.get("date"):
                lines.append(f"   Date: {result['date']}")

        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Failed to parse API response"


SEARCH_ENGINE_TOOLS = []
if os.getenv("SERPER_API_KEY"):
    SEARCH_ENGINE_TOOLS = [search_engine]
