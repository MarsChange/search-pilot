import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Union

from ag_ui.core import (
    Event,
    Message,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from agent_loop import Chunk

logger = logging.getLogger(__name__)


def to_openai_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert AG-UI protocol messages to OpenAI-compatible message format."""
    openai_messages = []

    for msg in messages:
        openai_msg: Dict[str, Any] = {
            "role": msg.role,
            "content": msg.content or "",
        }

        # Add optional name field if present
        if hasattr(msg, "name") and msg.name:
            openai_msg["name"] = msg.name

        # Handle tool_calls for assistant messages
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            openai_msg["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in msg.tool_calls
            ]

        # Handle tool_call_id for tool messages
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            openai_msg["tool_call_id"] = msg.tool_call_id

        openai_messages.append(openai_msg)

    return openai_messages


# Keepalive as actual SSE data event (not comment).
# Alibaba Cloud proxy only counts "data:" events as activity.
KEEPALIVE_SSE = "data: \n\n"


async def stream_agui_events(
    chunks: AsyncIterator[Chunk], run_agent_input: RunAgentInput
) -> AsyncIterator[Union[Event, str]]:
    """
    Stream AG-UI protocol events from a stream of Chunk objects.
    """
    yield RunStartedEvent(
        thread_id=run_agent_input.thread_id,
        run_id=run_agent_input.run_id,
        parent_run_id=run_agent_input.parent_run_id,
        input=run_agent_input,
    )

    msg_id = str(uuid.uuid4())
    text_message_started = False
    try:
        async for chunk in chunks:
            if chunk.type == "text" and chunk.content:
                # Start text message if not already started
                if not text_message_started:
                    yield TextMessageStartEvent(message_id=msg_id)
                    text_message_started = True

                yield TextMessageContentEvent(
                    message_id=msg_id,
                    delta=chunk.content,
                )

            elif chunk.type == "text" and not chunk.content:
                # Empty text chunk = keepalive signal from agent_loop
                yield KEEPALIVE_SSE

            elif chunk.type == "tool_call" and chunk.tool_call:
                # End any pending text message before tool call
                if text_message_started:
                    yield TextMessageEndEvent(message_id=msg_id)
                    text_message_started = False
                    msg_id = str(uuid.uuid4())

                tc = chunk.tool_call
                yield ToolCallStartEvent(
                    tool_call_id=tc.tool_call_id,
                    tool_call_name=tc.tool_name,
                )
                yield ToolCallArgsEvent(
                    tool_call_id=tc.tool_call_id,
                    delta=json.dumps(tc.tool_arguments, ensure_ascii=False),
                )

                yield ToolCallEndEvent(
                    tool_call_id=tc.tool_call_id,
                )

            elif chunk.type == "tool_call_result":
                yield ToolCallResultEvent(
                    message_id=str(uuid.uuid4()),
                    tool_call_id=chunk.tool_call.tool_call_id,
                    content=str(chunk.tool_result),
                    role="tool",
                )
    except Exception as e:
        logger.warning(f"Error streaming AG-UI events: {e}", exc_info=True)
        # End any pending text message
        if text_message_started:
            yield TextMessageEndEvent(message_id=msg_id)
        yield RunErrorEvent(
            message=f"Failed to stream events due to an unexpected error: {str(e)}",
            code="error_in_agent_loop",
        )
        return

    # End any pending text message
    if text_message_started:
        yield TextMessageEndEvent(message_id=msg_id)

    yield RunFinishedEvent(
        thread_id=run_agent_input.thread_id,
        run_id=run_agent_input.run_id,
    )


def to_sse_data(event: Event) -> str:
    """
    Convert an AG-UI Event to Server-Sent Events (SSE) data format.

    """
    data = event.model_dump(
        mode="json",
        exclude_none=True,
        by_alias=True,
    )
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
