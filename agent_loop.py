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
MAX_SUB_AGENT_TURNS = 25

QUESTION_ANALYSIS_PROMPT = """Carefully analyze the given task description (question) without attempting to solve it directly. Your role is to identify potential challenges and areas that require special attention during the solving process, and provide practical guidance for someone who will solve this task by actively gathering and analyzing information from the web.

Identify and concisely list key points in the question that could potentially impact subsequent information collection or the accuracy and completeness of the problem solution, especially those likely to cause mistakes, carelessness, or confusion during problem-solving.

The question author does not intend to set traps or intentionally create confusion. Interpret the question in the most common, reasonable, and straightforward manner, without speculating about hidden meanings or unlikely scenarios. However, be aware that mistakes, imprecise wording, or inconsistencies may exist due to carelessness or limited subject expertise, rather than intentional ambiguity.

Additionally, when considering potential answers or interpretations, note that question authors typically favor more common and familiar expressions over overly technical, formal, or obscure terminology. They generally prefer straightforward and common-sense interpretations rather than being excessively cautious or academically rigorous in their wording choices.

Also, consider additional flagging issues such as:
- Potential mistakes or oversights introduced unintentionally by the question author due to his misunderstanding, carelessness, or lack of attention to detail.
- Terms or instructions that might have multiple valid interpretations due to ambiguity, imprecision, outdated terminology, or subtle wording nuances.
- Numeric precision, rounding requirements, formatting, or units that might be unclear, erroneous, or inconsistent with standard practices or provided examples.
- Contradictions or inconsistencies between explicit textual instructions and examples or contextual clues provided within the question itself.

Avoid overanalyzing or listing trivial details that would not materially affect the task outcome.

## Knowledge-Based Entity Hypothesis Generation (CRITICAL)

For each descriptive element, indirect reference, or oblique characterization in the question, you MUST leverage your internal knowledge to generate the most likely candidate entities. This step transforms vague multi-hop questions into concrete, targeted research tasks.

For each descriptive element in the question:
1. **Extract the descriptive clue** from the question text
2. **Propose the most likely candidate entity** based on your broad knowledge
3. **Explain the match** — briefly state why this candidate fits the description
4. **Assign confidence** — high / medium / low
5. **List alternatives** — if multiple candidates are plausible, list the top 2-3 with reasoning for each

**Chain derivation**: Once you identify a high-confidence candidate for one element, immediately use it as a known condition to derive candidates for subsequent elements. Build a complete candidate reasoning chain from the question's starting point to its final target.

After generating all hypotheses, clearly categorize:
- **High-confidence hypotheses** — can be treated as near-facts, only need quick verification
- **Medium/low-confidence hypotheses** — require dedicated search effort to confirm or eliminate
- **Unknown elements** — no strong candidate available, require open-ended search from scratch

## Reasoning Chain Decomposition (CRITICAL)

If the question involves multi-hop reasoning (requiring multiple information retrieval steps to reach an answer):

1. **Identify the reasoning chain** — Decompose the question into independent sub-questions, each involving exactly ONE specific information retrieval task
2. **Pre-judge each node using internal knowledge** — For each sub-question, provide the most likely candidate answer based on your existing knowledge, with confidence level
3. **Provide a search plan for each sub-question**:
   - Recommended search keywords (3-7 keywords, short and precise)
   - Alternative keywords (to use if the first search fails)
4. **Mark dependencies between sub-questions** — Which sub-questions can be searched independently (parallelizable), and which require prior sub-questions to be resolved first (sequential)
5. **Specify the key fact each sub-question must confirm** — What information must be established at each step before proceeding to the next
6. **Identify skippable nodes** — If a node's hypothesis has high confidence, recommend the main agent adopt it directly rather than spending time on verification

Note: Search keywords for each sub-question must be independent — NEVER mix keywords from multiple sub-questions into a single query.

## 中文分析指导

如果问题涉及中文语境，请特别注意：

- **语言理解**：识别可能存在的中文表达歧义、方言差异或特定语境下的含义
- **文化背景**：考虑可能需要中文文化背景知识才能正确理解的术语或概念
- **信息获取**：标注需要使用中文搜索关键词才能获得准确信息的方面
- **格式要求**：识别中文特有的格式要求、表达习惯或答案形式
- **翻译风险**：标记直接翻译可能导致误解或信息丢失的关键术语
- **时效性**：注意中文信息源的时效性和地域性特征
- **分析输出**：使用中文进行分析和提示，确保语言一致性

Here is the question:

"""


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
) -> str:
    """Run the sub-agent worker to complete a research subtask.

    Non-streaming, bounded turns.

    Args:
        client: OpenAI-compatible async client
        model: Model name
        subtask: The research subtask description
        sub_agent_tool_functions: Tool functions available to the sub-agent
        chinese_context: Whether CJK context is detected

    Returns:
        Summary report string from the sub-agent
    """
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

    task_failed = False
    turn = 0

    for turn in range(MAX_SUB_AGENT_TURNS):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schema if tool_schema else None,
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

            # Execute the tool (sync tools run in thread to avoid blocking event loop)
            try:
                if func_name in tool_functions_map:
                    func = tool_functions_map[func_name]
                    if iscoroutinefunction(func):
                        result = await func(**parsed_args)
                    else:
                        result = await asyncio.to_thread(func, **parsed_args)
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
        task_failed = True
        logger.warning(
            f"[Sub-Agent] Reached max turns ({MAX_SUB_AGENT_TURNS})"
        )

    # Generate summary via summarize prompt
    summarize = generate_summarize_prompt(
        task_description=subtask,
        task_failed=task_failed,
        is_main_agent=False,
        chinese_context=chinese_context,
    )
    messages.append({"role": "user", "content": summarize})

    try:
        summary_response = await client.chat.completions.create(
            model=model,
            messages=messages,
            # No tools parameter — force text-only response
        )
        summary = summary_response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[Sub-Agent] Summary generation failed: {e}")
        # Fall back to last assistant message
        summary = "Error generating summary. "
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                summary += msg["content"]
                break

    logger.info(f"[Sub-Agent] Summary: {len(summary)} chars")
    return summary


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
    # and the client won't idle-timeout during Phase 0 pre-analysis
    yield Chunk(type="text", content="", step_index=0)

    # --- Phase 0: Question Pre-Analysis (optional, controlled by ENABLE_QUESTION_ANALYSIS) ---
    question_analysis = ""
    enable_analysis = os.getenv("ENABLE_QUESTION_ANALYSIS", "true").lower() in ("true", "1", "yes")
    if enable_analysis and user_question:
        try:
            analysis_task = asyncio.create_task(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{QUESTION_ANALYSIS_PROMPT}{user_question}",
                        }
                    ],
                )
            )
            # Wait with periodic keepalive to prevent SSE timeout
            pending = {analysis_task}
            while pending:
                _, pending = await asyncio.wait(pending, timeout=15)
                if pending:
                    yield Chunk(type="text", content="", step_index=0)

            analysis_response = analysis_task.result()
            question_analysis = analysis_response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Phase 0 question pre-analysis failed: {e}, skipping")

    # --- Sub-agent tool functions come from SUB_AGENT_TOOLS (imported from tools/) ---
    sub_agent_tool_functions = list(SUB_AGENT_TOOLS)
    max_parallel = int(os.getenv("SUB_AGENT_NUM", "3"))

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
            )
            for q in questions
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

    # Inject question pre-analysis as context
    if question_analysis:
        prompt_messages.append(
            {
                "role": "user",
                "content": (
                    f"<question_analysis>\n{question_analysis}\n</question_analysis>\n\n"
                    "Based on the above analysis, now solve the original question. "
                    "Start by outlining your decomposition plan, then delegate subtasks step by step."
                ),
            }
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

        # Wait for all async tasks with periodic keepalive
        if async_tasks:
            pending = set(async_tasks.values())
            while pending:
                _, pending = await asyncio.wait(pending, timeout=15)
                if pending:
                    yield Chunk(
                        type="text", content="", step_index=step_index
                    )

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
