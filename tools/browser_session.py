"""
Playwright browser tools via MCP (Model Context Protocol).

Connects to MCP server via SSE transport for cloud platforms like LangStudio.

Environment variables:
- PLAYWRIGHT_MCP_URL: SSE endpoint URL for MCP server (required)
- PLAYWRIGHT_MCP_TOKEN: Optional Bearer token for authentication
"""

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class PlaywrightMCPSession:
    """
    Persistent MCP client session connected to Playwright MCP server via SSE.
    """

    _instance: Optional["PlaywrightMCPSession"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    @classmethod
    async def get_instance(cls) -> "PlaywrightMCPSession":
        """Get or create the singleton instance, initializing MCP connection if needed."""
        async with cls._lock:
            if cls._instance is None or cls._instance._session is None:
                instance = cls()
                await instance._connect()
                cls._instance = instance
            return cls._instance

    async def _connect(self):
        """Connect to MCP server via SSE transport."""
        mcp_url = os.getenv("PLAYWRIGHT_MCP_URL")
        if not mcp_url:
            raise RuntimeError("PLAYWRIGHT_MCP_URL environment variable is required")

        self._exit_stack = AsyncExitStack()

        # Prepare headers for authentication if token is provided
        headers = {}
        token = os.getenv("PLAYWRIGHT_MCP_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        logger.info(f"Connecting to MCP server via SSE: {mcp_url}")

        transport = await self._exit_stack.enter_async_context(
            sse_client(
                mcp_url,
                headers=headers,
                timeout=30,  # Connection timeout (default 5s is too short)
                sse_read_timeout=600,  # SSE read timeout for long operations
            )
        )
        read_stream, write_stream = transport

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        # Log available tools
        response = await self._session.list_tools()
        tool_names = [t.name for t in response.tools]
        logger.info(f"Playwright MCP connected. Available tools: {tool_names}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server and return the result as string."""
        if self._session is None:
            raise RuntimeError("MCP session not initialized")

        result = await self._session.call_tool(tool_name, arguments=arguments)

        # Extract text content from result
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            elif hasattr(content, "data"):
                parts.append(f"[Binary data: {content.mimeType}]")
        return "\n".join(parts) if parts else "OK"

    async def close(self):
        """Close the MCP session."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        PlaywrightMCPSession._instance = None


# ============== Tool Functions ==============
async def browser_navigate(url: str) -> str:
    """
    Navigate to a URL in the browser.

    Args:
        url: The URL to navigate to (must include http:// or https://)

    Returns:
        Page accessibility snapshot after navigation
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_navigate", {"url": url})


async def browser_navigate_back() -> str:
    """
    Navigate back to the previous page in browser history.

    Returns:
        Page accessibility snapshot after navigation
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_navigate_back", {})


async def browser_navigate_forward() -> str:
    """
    Navigate forward to the next page in browser history.

    Returns:
        Page accessibility snapshot after navigation
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_navigate_forward", {})


async def browser_click(element: str, ref: str) -> str:
    """
    Click an element on the page using its accessibility reference.

    Args:
        element: Human-readable element description
        ref: Exact accessibility reference (e.g. "link[123]", "button[456]") from the page snapshot

    Returns:
        Page accessibility snapshot after clicking
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_click", {"element": element, "ref": ref})


async def browser_type(element: str, ref: str, text: str) -> str:
    """
    Type text into an editable element on the page.

    Args:
        element: Human-readable element description
        ref: Exact accessibility reference from the page snapshot
        text: Text to type into the element

    Returns:
        Page accessibility snapshot after typing
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_type", {
        "element": element, "ref": ref, "text": text
    })


async def browser_select_option(element: str, ref: str, values: str) -> str:
    """
    Select an option from a dropdown element.

    Args:
        element: Human-readable element description
        ref: Exact accessibility reference from the page snapshot
        values: Comma-separated list of option values to select

    Returns:
        Page accessibility snapshot after selection
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_select_option", {
        "element": element, "ref": ref, "values": values
    })


async def browser_snapshot() -> str:
    """
    Capture the current page's accessibility snapshot.

    Returns:
        Accessibility snapshot with element references
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_snapshot", {})


async def browser_screenshot() -> str:
    """
    Take a screenshot of the current page.

    Returns:
        Screenshot data or confirmation message
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_screenshot", {})


async def browser_press_key(key: str) -> str:
    """
    Press a keyboard key in the browser.

    Args:
        key: Key to press (e.g. "Enter", "Escape", "ArrowDown", "Tab")

    Returns:
        Page accessibility snapshot after key press
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_press_key", {"key": key})


async def browser_hover(element: str, ref: str) -> str:
    """
    Hover over an element on the page.

    Args:
        element: Human-readable element description
        ref: Exact accessibility reference from the page snapshot

    Returns:
        Page accessibility snapshot after hovering
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_hover", {"element": element, "ref": ref})


async def browser_evaluate(expression: str) -> str:
    """
    Execute JavaScript in the browser console.

    Args:
        expression: JavaScript expression to evaluate

    Returns:
        Result of the JavaScript evaluation
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_evaluate", {"expression": expression})


async def browser_tab_list() -> str:
    """
    List all open browser tabs.

    Returns:
        List of open tabs with their titles and URLs
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_tab_list", {})


async def browser_tab_new(url: str = "") -> str:
    """
    Open a new browser tab, optionally navigating to a URL.

    Args:
        url: URL to navigate to in the new tab (optional)

    Returns:
        Page accessibility snapshot of the new tab
    """
    args = {}
    if url:
        args["url"] = url
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_tab_new", args)


async def browser_tab_close(index: int = -1) -> str:
    """
    Close a browser tab by index.

    Args:
        index: Tab index to close (default: current tab)

    Returns:
        Confirmation message
    """
    session = await PlaywrightMCPSession.get_instance()
    return await session.call_tool("browser_tab_close", {"index": index})


async def browser_close() -> str:
    """
    Close the browser and MCP session.

    Returns:
        Confirmation message
    """
    session = await PlaywrightMCPSession.get_instance()
    await session.close()
    return "Browser session closed."


# List of all browser tool functions
BROWSER_TOOLS = [
    browser_navigate,
    browser_navigate_back,
    browser_navigate_forward,
    browser_click,
    browser_type,
    browser_select_option,
    browser_snapshot,
    browser_screenshot,
    browser_press_key,
    browser_hover,
    browser_evaluate,
    browser_tab_list,
    browser_tab_new,
    browser_tab_close,
    browser_close,
]
