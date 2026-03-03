import asyncio
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from inspect import iscoroutinefunction
from typing import (
    Any,
    AsyncIterator,
    Callable,
    List,
    Literal,
    Optional,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from tools_calling import (
    build_main_agent_system_prompt,
    build_sub_agent_system_prompt,
    generate_summarize_prompt,
)

# Import sub-agent tools; fallback to empty if unavailable
try:
    from tools import SUB_AGENT_TOOLS
except ImportError:
    SUB_AGENT_TOOLS = []

logger = logging.getLogger(__name__)

# --- Constants ---

MAX_MAIN_AGENT_TURNS = 30
MAX_SUB_AGENT_TURNS = 10


DEFAULT_SYSTEM_PROMPT = """You are an AI assistant designed to help with a variety of tasks. You have access to several tools that can assist you in providing accurate and relevant information.

Your task is to actively gather detailed information from the internet and generate answers to users' questions. Your goal is not to rush to a definitive answer or conclusion, but to collect complete information and present all reasonable candidate answers you find, accompanied by clearly documented supporting evidence, reasoning steps, uncertainty factors, and explicit intermediate findings.

The user has no intention of deliberately setting traps or creating confusion. Please use the most common, reasonable, and direct explanation to handle the task, and do not overthink or focus on rare or far-fetched explanations.

Important Note: - Gather comprehensive information from reliable sources to fully understand all aspects of the issue.
- Present all possible candidate answers you identified during the information gathering process, regardless of uncertainty, ambiguity, or incomplete verification. Avoid jumping to conclusions or omitting any discovered possibilities.
- Clearly record the detailed facts, evidence, and reasoning steps supporting each candidate answer, and carefully preserve the intermediate analysis results.
- During the information collection process, clearly mark and retain all uncertainties, conflicting interpretations, or different understandings that are discovered. Do not arbitrarily discard or resolve these issues on your own.
- In cases where there is inconsistency, ambiguity, errors, or potential mismatches with general guidelines or provided examples in the explicit instructions of a problem (such as numerical accuracy, formatting, specific requirements), all reasonable explanations and corresponding candidate answers should be clearly documented and presented.

Recognize that the original task description itself may inadvertently contain errors, imprecision, inaccuracy, or conflicts due to user carelessness, misunderstanding, or limited professional knowledge. Do not attempt to internally question or "correct" these instructions; instead, present the survey results transparently based on every reasonable interpretation.

Your goal is to achieve the highest degree of completeness, transparency, and detailed documentation, enabling users to make independent judgments and choose their preferred answers. Even in the presence of uncertainty, explicitly recording the existence of possible answers can significantly enhance the user experience, ensuring that no reasonable solutions are irreversibly omitted due to early misunderstandings or premature filtering.

When generating responses, it is crucial to pay attention to the following points:
1. Keep your response as concise as possible. The response content should be returned as a JSON dictionary, with the answer to the question corresponding to the key "answer". For example, if the user inputs {"question": "Where is the capital of France?"}, you only need to respond with {"answer": "Paris"}
2. To minimize misjudgments caused by format differences, the following preprocessing is applied to the output responses:
- Convert English letters to lowercase;
- Remove leading and trailing spaces;
- All numerical questions involve integers;
- If the answer contains multiple entities, please follow English grammar, with a comma or semicolon followed by a space. The specific symbol should be based on the user's question.
"""


# --- Data Classes ---


@dataclass
class ToolCall:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None


@dataclass
class Chunk:
    step_index: int
    type: Literal["text", "tool_call", "tool_call_result"]
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[Any] = None


# --- Utility Functions ---


def python_type_to_json_type(t):
    """Map Python types to JSON types."""
    if t is str:
        return "string"
    elif t is int:
        return "integer"
    elif t is float:
        return "number"
    elif t is bool:
        return "boolean"
    elif t is list or get_origin(t) is list:
        return "array"
    elif t is dict or get_origin(t) is dict:
        return "object"
    return "string"


def parse_docstring(docstring: str) -> dict:
    """
    Parse a docstring to extract description and parameter descriptions.

    Supports Google-style docstrings:
        Args:
            param_name: Description of the parameter
            param_name (type): Description of the parameter

    Returns:
        dict with 'description' and 'params' keys
    """
    if not docstring:
        return {"description": "", "params": {}}

    lines = docstring.strip().split("\n")
    description_lines = []
    params = {}
    current_section = "description"
    current_param = None
    current_param_desc = []

    for line in lines:
        stripped = line.strip()

        # Check for section headers
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            current_section = "args"
            continue
        elif stripped.lower() in ("returns:", "return:", "yields:", "raises:", "examples:", "example:", "note:", "notes:"):
            # Save current param if any
            if current_param and current_param_desc:
                params[current_param] = " ".join(current_param_desc).strip()
            current_section = "other"
            continue

        if current_section == "description":
            description_lines.append(stripped)
        elif current_section == "args":
            # Check if this is a new parameter definition
            # Patterns: "param_name: description" or "param_name (type): description"
            param_match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$", stripped)
            if param_match:
                # Save previous param
                if current_param and current_param_desc:
                    params[current_param] = " ".join(current_param_desc).strip()
                current_param = param_match.group(1)
                current_param_desc = [param_match.group(2)] if param_match.group(2) else []
            elif current_param and stripped:
                # Continuation of current param description
                current_param_desc.append(stripped)

    # Save last param
    if current_param and current_param_desc:
        params[current_param] = " ".join(current_param_desc).strip()

    # Clean up description - remove empty lines at end
    while description_lines and not description_lines[-1]:
        description_lines.pop()

    return {
        "description": " ".join(description_lines).strip(),
        "params": params,
    }


def function_to_schema(func: Callable) -> dict:
    """
    Convert a Python function to an OpenAI API Tool Schema.

    Extracts function description and parameter descriptions from docstring.
    """
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)

    # Parse docstring for descriptions
    docstring_info = parse_docstring(func.__doc__ or "")

    parameters = {"type": "object", "properties": {}, "required": []}

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = type_hints.get(name, str)
        param_type = python_type_to_json_type(annotation)

        param_info = {"type": param_type}

        # Add parameter description from docstring if available
        if name in docstring_info["params"]:
            param_info["description"] = docstring_info["params"][name]

        if get_origin(annotation) == Literal:
            param_info["enum"] = list(get_args(annotation))
            param_info["type"] = python_type_to_json_type(type(get_args(annotation)[0]))

        parameters["properties"][name] = param_info
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": docstring_info["description"],
            "parameters": parameters,
        },
    }


# --- CJK Detection ---
def _contains_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
            return True
    return False

# --- Sub-Agent Runner ---
async def run_sub_agent(
    client: AsyncOpenAI,
    model: str,
    subtask: str,
    sub_agent_tool_functions: list,
    chinese_context: bool = False,
    progress_queue: Optional[asyncio.Queue] = None,
    worker_index: int = 0,
) -> str:
    """Run the sub-agent worker to complete a research subtask.

    Non-streaming, bounded turns.

    Args:
        client: OpenAI-compatible async client
        model: Model name
        subtask: The research subtask description
        sub_agent_tool_functions: Tool functions available to the sub-agent
        chinese_context: Whether CJK context is detected
        progress_queue: Optional queue to push progress chunks for streaming
        worker_index: Worker number for progress display

    Returns:
        Summary report string from the sub-agent
    """
    async def _emit_progress(text: str):
        """Push a progress message to the queue for streaming."""
        if progress_queue is not None:
            await progress_queue.put(text)

    async def _llm_call_with_progress(coro, label: str):
        """Run an LLM coroutine with a background progress emitter.

        Spawns a separate task that emits ⏳ every 10s while the LLM call
        is in progress. This avoids nested asyncio.wait issues.
        """
        llm_task = asyncio.create_task(coro)

        async def _progress_ticker():
            wait_seconds = 0
            try:
                while True:
                    await asyncio.sleep(10)
                    wait_seconds += 10
                    await _emit_progress(
                        f"⏳ Worker {worker_index}: {label} ({wait_seconds}s)\n\n"
                    )
            except asyncio.CancelledError:
                pass

        ticker = asyncio.create_task(_progress_ticker())
        try:
            return await llm_task
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

    async def _tool_call_with_progress(coro, tool_name: str):
        """Run a tool coroutine with a background progress emitter.

        Emits ⏳ every 15s so the UI doesn't appear stuck during long tool calls.
        """
        tool_task = asyncio.create_task(coro)

        async def _progress_ticker():
            wait_seconds = 0
            try:
                while True:
                    await asyncio.sleep(15)
                    wait_seconds += 15
                    await _emit_progress(
                        f"⏳ Worker {worker_index}: `{tool_name}` running... ({wait_seconds}s)\n\n"
                    )
            except asyncio.CancelledError:
                pass

        ticker = asyncio.create_task(_progress_ticker())
        try:
            return await tool_task
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass
    # Build sub-agent system prompt
    system_prompt = build_sub_agent_system_prompt(
        sub_agent_tool_functions, chinese_context
    )

    # Build tool schema and map
    tool_schema = [function_to_schema(f) for f in sub_agent_tool_functions]
    tool_functions_map = {f.__name__: f for f in sub_agent_tool_functions}

    # Initialize messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": subtask},
    ]

    logger.info(f"[Sub-Agent] Starting subtask: {subtask[:200]}...")
    short_task = subtask[:100] + ("..." if len(subtask) > 100 else "")
    await _emit_progress(f"🔍 **Worker {worker_index}**: Starting research — {short_task}\n\n")

    turn = 0

    for turn in range(MAX_SUB_AGENT_TURNS):
        # Calculate total input chars for this LLM call
        input_chars = sum(
            len(str(m.get("content", "")))
            + sum(
                len(str(tc.get("function", {}).get("arguments", "")))
                for tc in m.get("tool_calls", [])
            )
            for m in messages
        )
        await _emit_progress(
            f"📊 Worker {worker_index}: Turn {turn + 1} — LLM input {input_chars:,} chars\n\n"
        )

        try:
            response = await _llm_call_with_progress(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_schema if tool_schema else None,
                ),
                label=f"Waiting for LLM response (turn {turn + 1})...",
            )
        except Exception as e:
            logger.error(f"[Sub-Agent] LLM call failed at turn {turn}: {e}")
            break

        choice = response.choices[0]
        assistant_message = choice.message

        # If no tool calls, this is the final response
        if not assistant_message.tool_calls:
            content = assistant_message.content or ""
            logger.info(f"[Sub-Agent] Completed at turn {turn} with {len(content)} chars")
            messages.append({"role": "assistant", "content": content})
            if content:
                # Show a truncated preview of the final response
                preview = content[:300] + ("..." if len(content) > 300 else "")
                await _emit_progress(
                    f"💬 Worker {worker_index}: Response ({len(content)} chars):\n{preview}\n\n"
                )
            break

        # Build assistant message for history
        tool_calls_data = []
        for tc in assistant_message.tool_calls:
            args_str = tc.function.arguments
            # Ensure arguments is valid JSON for DashScope API compatibility
            try:
                json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_str = json.dumps({})
            tool_calls_data.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": args_str,
                },
            })
        assistant_msg = {"role": "assistant", "tool_calls": tool_calls_data}
        if assistant_message.content:
            assistant_msg["content"] = assistant_message.content
        messages.append(assistant_msg)

        # Execute tool calls
        for tc in assistant_message.tool_calls:
            func_name = tc.function.name
            try:
                parsed_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                tool_result = f"Error: Failed to parse arguments: {e}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
                )
                continue

            # Emit progress BEFORE tool execution so UI shows what's happening
            progress_detail = ""
            if parsed_args:
                # Show the first meaningful argument value as context
                first_val = next(iter(parsed_args.values()), "")
                if isinstance(first_val, str) and len(first_val) > 80:
                    first_val = first_val[:80] + "..."
                progress_detail = f" | {first_val}"
            await _emit_progress(
                f"⚙️ Worker {worker_index}: `{func_name}`{progress_detail}\n\n"
            )

            # Execute the tool with timeout and progress ticker
            TOOL_TIMEOUT = 30  # seconds
            try:
                if func_name in tool_functions_map:
                    func = tool_functions_map[func_name]
                    if iscoroutinefunction(func):
                        coro = func(**parsed_args)
                    else:
                        coro = asyncio.to_thread(func, **parsed_args)
                    try:
                        result = await asyncio.wait_for(
                            _tool_call_with_progress(
                                coro, func_name
                            ),
                            timeout=TOOL_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        result = f"Error: Tool '{func_name}' timed out after {TOOL_TIMEOUT}s"
                        await _emit_progress(
                            f"⚠️ Worker {worker_index}: `{func_name}` timed out ({TOOL_TIMEOUT}s)\n\n"
                        )
                    tool_result = str(result)
                else:
                    tool_result = f"Error: Tool '{func_name}' not found."
            except Exception as e:
                tool_result = f"Error: Execution failed - {str(e)}"

            logger.info(
                f"[Sub-Agent] Turn {turn}: {func_name} -> {len(tool_result)} chars"
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
            )
    else:
        # Loop completed without break = max turns exhausted
        logger.warning(
            f"[Sub-Agent] Reached max turns ({MAX_SUB_AGENT_TURNS})"
        )

    # Extract last assistant content directly (no extra summary LLM call)
    result = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            result = msg["content"]
            break

    if not result:
        result = "No findings were produced for this subtask."

    logger.info(f"[Sub-Agent] Result: {len(result)} chars")
    await _emit_progress(f"✅ **Worker {worker_index}**: Research complete.\n\n")
    return result


# --- Main Agent Loop ---


async def agent_loop(
    input_messages: list,
    tool_functions: List[Callable],
    skill_directories: Optional[List[str]] = None,
) -> AsyncIterator[Chunk]:
    """
    Main agent loop with multi-agent architecture.

    The main agent decomposes tasks and delegates research to sub-agent workers
    via execute_subtask. The sub-agent has direct access to search, scrape, and
    analyze tools.

    Args:
        input_messages: List of chat messages
        tool_functions: List of tool functions for the MAIN agent (e.g., sandbox tools)
        skill_directories: Deprecated, ignored. Kept for API compatibility.
    """

    assert os.getenv("DASHSCOPE_API_KEY"), "DASHSCOPE_API_KEY is not set"

    model = os.getenv("QWEN_MODEL") or "qwen-max"

    client = AsyncOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # --- Extract user question ---
    user_question = ""
    for msg in reversed(input_messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_question = content
            break

    # --- Detect Chinese context ---
    chinese_context = _contains_cjk(user_question)

    # Yield an initial chunk immediately so the SSE connection has data
    yield Chunk(type="text", content="", step_index=0)

    # --- Sub-agent tool functions come from SUB_AGENT_TOOLS (imported from tools/) ---
    sub_agent_tool_functions = list(SUB_AGENT_TOOLS)
    max_parallel = int(os.getenv("SUB_AGENT_NUM", "3"))

    # --- Progress queue for sub-agent streaming ---
    progress_queue: asyncio.Queue = asyncio.Queue()

    # --- Create execute_subtasks closure (parallel sub-agent dispatch) ---
    async def execute_subtasks(subtasks_json: str) -> str:
        """
        Delegate one or more research subtasks to worker agents, executed in parallel. Each worker has independent access to web search, webpage analysis, Wikipedia, website scraping, and browser tools. Workers run concurrently and return structured research reports.

        Args:
            subtasks_json: A JSON array of subtask description strings. Each element is a self-contained research question that includes ALL relevant context (workers have no shared memory). For a single subtask, use a one-element array.
        """
        try:
            questions = json.loads(subtasks_json)
            if isinstance(questions, str):
                questions = [questions]
        except json.JSONDecodeError:
            # Fallback: treat as a single question
            questions = [subtasks_json]

        if not questions:
            return "Error: No subtasks provided."

        # Cap parallelism
        questions = questions[:max_parallel]

        logger.info(
            f"[Main Agent] Dispatching {len(questions)} subtask(s) in parallel"
        )

        # Run all sub-agents concurrently
        tasks = [
            run_sub_agent(
                client=client,
                model=model,
                subtask=q,
                sub_agent_tool_functions=sub_agent_tool_functions,
                chinese_context=chinese_context,
                progress_queue=progress_queue,
                worker_index=i + 1,
            )
            for i, q in enumerate(questions)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format combined results
        output_parts = []
        for i, (q, r) in enumerate(zip(questions, results)):
            if isinstance(r, Exception):
                output_parts.append(
                    f"## Subtask {i + 1}\n**Question**: {q}\n**Result**: Error - {str(r)}"
                )
            else:
                output_parts.append(
                    f"## Subtask {i + 1}\n**Question**: {q}\n**Result**:\n{r}"
                )

        return "\n\n---\n\n".join(output_parts)

    # --- Main agent tools: [execute_subtasks] + passed-in tool_functions ---
    main_agent_tools = [execute_subtasks] + list(tool_functions or [])

    # --- Build main agent system prompt ---
    system_prompt = build_main_agent_system_prompt(
        main_agent_tools, chinese_context, max_parallel=max_parallel
    )
    # Append the default system prompt (JSON answer format requirement)
    system_prompt = f"{system_prompt}\n\n{DEFAULT_SYSTEM_PROMPT}"

    # --- Prepare messages ---
    prompt_messages = input_messages.copy()
    if prompt_messages and prompt_messages[0].get("role") == "system":
        original_content = prompt_messages[0].get("content", "")
        prompt_messages[0] = {
            "role": "system",
            "content": f"{original_content}\n\n{system_prompt}",
        }
    else:
        prompt_messages.insert(
            0,
            {"role": "system", "content": system_prompt},
        )

    # --- Build tool schema and function map ---
    tool_schema = [function_to_schema(f) for f in main_agent_tools]
    tool_functions_map = {f.__name__: f for f in main_agent_tools}

    params = {
        "model": model,
        "stream": True,
        "tools": tool_schema,
    }

    step_index = 0

    # --- Main Agent Loop (bounded) ---
    for turn in range(MAX_MAIN_AGENT_TURNS):
        # Make the streaming request
        stream = await client.chat.completions.create(
            messages=prompt_messages,
            **params,
            # extra_body={"enable_thinking": True},
        )

        tool_calls_buffer = {}

        # Process the stream
        async for chunk in stream:  # type: ChatCompletionChunk
            chunk = cast(ChatCompletionChunk, chunk)

            delta = chunk.choices[0].delta

            # Case A: Standard text content
            if delta.content:
                yield Chunk(
                    type="text", content=delta.content, step_index=step_index
                )

            # Case B: Tool call fragments (accumulate them)
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc_chunk.id,
                            "function": {
                                "name": tc_chunk.function.name,
                                "arguments": "",
                            },
                        }
                    # Append tool arguments fragment
                    if tc_chunk.function.arguments:
                        tool_calls_buffer[idx]["function"]["arguments"] += (
                            tc_chunk.function.arguments
                        )

        # If no tool calls, the model returned a final text response
        if not tool_calls_buffer:
            break

        assistant_tool_calls_data = []
        sorted_indices = sorted(tool_calls_buffer.keys())

        for idx in sorted_indices:
            raw_tool = tool_calls_buffer[idx]
            assistant_tool_calls_data.append(
                {
                    "id": raw_tool["id"],
                    "type": "function",
                    "function": {
                        "name": raw_tool["function"]["name"],
                        "arguments": raw_tool["function"]["arguments"],
                    },
                }
            )

        # Append the assistant's tool call request to history
        prompt_messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls_data,
            }
        )

        # Execute tools and yield results — parallel for async tools
        # Phase 1: Parse all tool calls and yield tool_call notifications
        parsed_tool_calls = []  # (call_id, func_name, parsed_args, tool_call, error_msg)
        for tool_data in assistant_tool_calls_data:
            call_id = tool_data["id"]
            func_name = tool_data["function"]["name"]
            func_args_str = tool_data["function"]["arguments"]

            tool_call = ToolCall(
                tool_call_id=call_id,
                tool_name=func_name,
                tool_arguments={},
            )

            try:
                parsed_args = json.loads(func_args_str)
                tool_call.tool_arguments = parsed_args
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )
                parsed_tool_calls.append(
                    (call_id, func_name, parsed_args, tool_call, None)
                )
            except json.JSONDecodeError as e:
                error_msg = f"Error: Failed to parse tool arguments JSON: {func_args_str}. Error: {e}"
                yield Chunk(
                    step_index=step_index,
                    type="tool_call",
                    tool_call=tool_call,
                )
                parsed_tool_calls.append(
                    (call_id, func_name, {}, tool_call, error_msg)
                )

        # Phase 2: Launch all tools — async ones run concurrently
        async_tasks = {}  # call_id -> asyncio.Task
        sync_results = {}  # call_id -> result string

        for call_id, func_name, parsed_args, tool_call, error_msg in parsed_tool_calls:
            if error_msg:
                sync_results[call_id] = error_msg
                continue

            if func_name not in tool_functions_map:
                sync_results[call_id] = f"Error: Tool '{func_name}' not found."
                continue

            func = tool_functions_map[func_name]
            if iscoroutinefunction(func):
                async_tasks[call_id] = asyncio.create_task(func(**parsed_args))
            else:
                # Run sync tools in thread to avoid blocking event loop
                async_tasks[call_id] = asyncio.create_task(
                    asyncio.to_thread(func, **parsed_args)
                )

        # Wait for all async tasks, draining sub-agent progress queue
        if async_tasks:
            pending = set(async_tasks.values())
            while pending:
                # Drain any progress messages from sub-agents
                while not progress_queue.empty():
                    try:
                        progress_text = progress_queue.get_nowait()
                        yield Chunk(
                            type="text",
                            content=progress_text,
                            step_index=step_index,
                        )
                    except asyncio.QueueEmpty:
                        break

                _, pending = await asyncio.wait(pending, timeout=5)
                if pending:
                    # Drain again after wait
                    drained = False
                    while not progress_queue.empty():
                        try:
                            progress_text = progress_queue.get_nowait()
                            yield Chunk(
                                type="text",
                                content=progress_text,
                                step_index=step_index,
                            )
                            drained = True
                        except asyncio.QueueEmpty:
                            break
                    # If no progress was emitted, send keepalive
                    if not drained:
                        yield Chunk(
                            type="text", content="", step_index=step_index
                        )

            # Final drain after all tasks complete
            while not progress_queue.empty():
                try:
                    progress_text = progress_queue.get_nowait()
                    yield Chunk(
                        type="text",
                        content=progress_text,
                        step_index=step_index,
                    )
                except asyncio.QueueEmpty:
                    break

            # Collect async results
            for call_id, task in async_tasks.items():
                try:
                    sync_results[call_id] = str(task.result())
                except Exception as e:
                    sync_results[call_id] = f"Error: Execution failed - {str(e)}"

        # Phase 3: Yield all results and update message history
        for call_id, func_name, parsed_args, tool_call, error_msg in parsed_tool_calls:
            tool_result_content = sync_results[call_id]

            yield Chunk(
                type="tool_call_result",
                tool_result=tool_result_content,
                step_index=step_index,
                tool_call=tool_call,
            )

            prompt_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result_content,
                }
            )
        step_index += 1
    else:
        # Reached max turns — inject summarize prompt for final answer
        logger.warning(
            f"Main agent reached max turns ({MAX_MAIN_AGENT_TURNS}), generating summary"
        )
        summarize = generate_summarize_prompt(
            task_description=user_question,
            task_failed=True,
            is_main_agent=True,
            chinese_context=chinese_context,
        )
        prompt_messages.append({"role": "user", "content": summarize})

        # Final LLM call without toodi
        stream = await client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            stream=True,
            # extra_body={"enable_thinking": True},
        )
        async for chunk in stream:
            chunk = cast(ChatCompletionChunk, chunk)
            delta = chunk.choices[0].delta
            if delta.content:
                yield Chunk(
                    type="text", content=delta.content, step_index=step_index
                )
