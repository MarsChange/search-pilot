"""
Tool functions for the agent.

Import all tools from submodules and expose them as a unified list.
"""

import logging
import os

BROWSER_TOOLS = []
SANDBOX_TOOLS = []
WEBPAGE_ANALYZER_TOOLS = []
SEARCH_ENGINE_TOOLS = []
WIKI_SEARCH_TOOLS = []
SCRAPE_WEBSITE_TOOLS = []

# Only load browser tools if PLAYWRIGHT_MCP_URL is configured
if os.getenv("PLAYWRIGHT_MCP_URL"):
    try:
        from tools.browser_session import BROWSER_TOOLS
    except ImportError as e:
        logging.warning(f"Browser tools unavailable (missing dependency): {e}")
        BROWSER_TOOLS = []

# Load webpage analyzer tools (requires DASHSCOPE_API_KEY + JINA_API_KEY)
if os.getenv("DASHSCOPE_API_KEY") and os.getenv("JINA_API_KEY"):
    try:
        from tools.webpage_analyzer import WEBPAGE_ANALYZER_TOOLS
    except ImportError as e:
        logging.warning(f"Webpage analyzer tools unavailable (missing dependency): {e}")
        WEBPAGE_ANALYZER_TOOLS = []

# Load sandbox tools if E2B_API_KEY is configured
if os.getenv("E2B_API_KEY"):
    try:
        from tools.code_sandbox import SANDBOX_TOOLS
    except ImportError as e:
        logging.warning(f"Sandbox tools unavailable (missing dependency): {e}")
        SANDBOX_TOOLS = []

# Load search tools if SERPER_API_KEY is configured
if os.getenv("SERPER_API_KEY"):
    try:
        from tools.search_engine import SEARCH_ENGINE_TOOLS
    except ImportError as e:
        logging.warning(f"Search tools unavailable (missing dependency): {e}")
        SEARCH_ENGINE_TOOLS = []

# Wiki search tools (no API key required, uses public Wikipedia API)
try:
    from tools.wiki_search import WIKI_SEARCH_TOOLS
except ImportError as e:
    logging.warning(f"Wiki search tools unavailable (missing dependency): {e}")
    WIKI_SEARCH_TOOLS = []

# Load scrape website tools if JINA_API_KEY is configured
if os.getenv("JINA_API_KEY"):
    try:
        from tools.scrape_website import SCRAPE_WEBSITE_TOOLS
    except ImportError as e:
        logging.warning(f"Scrape website tools unavailable (missing dependency): {e}")
        SCRAPE_WEBSITE_TOOLS = []

# Aggregate all tools from different modules
TOOLS = [
    *BROWSER_TOOLS,
    *WEBPAGE_ANALYZER_TOOLS,
    *SANDBOX_TOOLS,
    *SEARCH_ENGINE_TOOLS,
    *WIKI_SEARCH_TOOLS,
    *SCRAPE_WEBSITE_TOOLS,
]

# Sub-agent gets search/scrape/analyze/browser tools
SUB_AGENT_TOOLS = [
    *SEARCH_ENGINE_TOOLS,
    *WIKI_SEARCH_TOOLS,
    *SCRAPE_WEBSITE_TOOLS,
    *WEBPAGE_ANALYZER_TOOLS,
    *BROWSER_TOOLS,
]

# Main agent gets sandbox only; execute_subtask is injected in agent_loop
MAIN_AGENT_TOOLS = [*SANDBOX_TOOLS]

__all__ = [
    "TOOLS",
    "BROWSER_TOOLS",
    "WEBPAGE_ANALYZER_TOOLS",
    "SANDBOX_TOOLS",
    "SEARCH_ENGINE_TOOLS",
    "WIKI_SEARCH_TOOLS",
    "SCRAPE_WEBSITE_TOOLS",
    "SUB_AGENT_TOOLS",
    "MAIN_AGENT_TOOLS",
]
