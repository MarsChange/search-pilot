"""
Prompt builders and tool calling utilities for the agent.

Provides system prompt generation for main agent and sub-agent, tool function
descriptions, and summarize prompt generation.
"""

import datetime
import logging
from typing import List

logger = logging.getLogger(__name__)


def build_tool_functions_prompt(tool_functions: list) -> str:
    """
    Build system prompt section describing tool functions.

    Args:
        tool_functions: List of tool function objects

    Returns:
        System prompt text describing tool functions by category
    """
    if not tool_functions:
        return ""

    # Group tools by category based on function name
    categories = {}
    for func in tool_functions:
        name = func.__name__
        if name.startswith("browser_"):
            category = "browser"
        elif name == "analyze_webpage":
            category = "webpage_analyzer"
        elif name == "google_search":
            category = "search"
        elif name.startswith("search_wikipedia") or name == "list_wikipedia_revisions":
            category = "wiki"
        elif name == "scrape_website":
            category = "scrape"
        else:
            category = "general"

        if category not in categories:
            categories[category] = []
        categories[category].append(name)

    lines = ["# Available Tools"]
    lines.append("")

    if "search" in categories:
        lines.append("**Google Search** (`google_search`):")
        lines.append("- Search the web using Google. Returns titles, URLs, snippets, answer boxes, and knowledge graphs.")
        lines.append("- Use Chinese keywords for Chinese questions, English for English questions.")
        lines.append("- Craft SHORT queries: 3-7 discriminative keywords targeting ONE specific aspect.")
        lines.append("")

    if "wiki" in categories:
        lines.append("**Wikipedia Tools** (`search_wikipedia`, `search_wikipedia_revision`, `list_wikipedia_revisions`):")
        lines.append("- `search_wikipedia(entity)` — Get current Wikipedia page content")
        lines.append("- `search_wikipedia_revision(entity, date)` — Get historical page content at a specific date")
        lines.append("- `list_wikipedia_revisions(entity)` — List available revisions (use FIRST when exploring history)")
        lines.append("")

    if "scrape" in categories:
        lines.append("**Website Scraper** (`scrape_website`):")
        lines.append("- Extract readable content from any webpage, converts HTML to markdown")
        lines.append("- Use as alternative to analyze_webpage for full page content")
        lines.append("")

    if "webpage_analyzer" in categories:
        lines.append("**Webpage Analyzer** (`analyze_webpage`):")
        lines.append("- `analyze_webpage(url, question)` — Fetches a webpage and uses AI to extract information relevant to your question")
        lines.append("- Use AFTER search to deeply analyze URLs from search results")
        lines.append("- Returns: relevance assessment, key findings, extracted details")
        lines.append("")

    if "browser" in categories:
        lines.append("**Browser Tools** (persistent browser session):")
        lines.append("- For web page interaction: navigation, clicking, typing, screenshots")
        lines.append("- Browser state persists across calls (cookies, sessions maintained)")
        lines.append("- Use `browser_snapshot` to get page structure and element references")
        lines.append(f"- Available: {', '.join(categories['browser'])}")
        lines.append("")

    if "general" in categories:
        lines.append("**Other Tools:**")
        lines.append(f"- {', '.join(categories['general'])}")
        lines.append("")

    return "\n".join(lines)


def build_main_agent_system_prompt(
    tool_functions: list,
    chinese_context: bool = False,
) -> str:
    """
    Build the system prompt for the main agent (task decomposition + delegation).

    The main agent does NOT directly search/scrape — it delegates to the sub-agent
    worker via execute_subtask.

    Args:
        tool_functions: List of tool function objects available to the main agent
        chinese_context: Whether the question involves CJK content

    Returns:
        System prompt text for the main agent
    """
    formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")

    prompt = f"""You are a research coordinator agent. Today is: {formatted_date}

# General Objective

You accomplish a given task by decomposing it into research subtasks and delegating them to a worker agent via the `execute_subtask` tool. The worker agent has access to search engines, webpage analysis, and browser tools. It will execute each subtask and return a structured research report.

You do NOT have direct access to search or web browsing tools. You MUST use `execute_subtask` for all information gathering.

## Task Strategy

1. Before taking any action, carefully analyze the question and the pre-analysis hints provided.
2. **Multi-hop decomposition**: For complex questions involving multiple reasoning steps, decompose the question into independent sub-questions. Each sub-question should target ONE specific piece of information.
3. Delegate each sub-question to the worker agent via `execute_subtask`. Each subtask description must be:
   - **Self-contained**: Include ALL relevant context from previous subtask results (the worker has no memory of previous subtasks)
   - **Specific**: Target ONE well-defined information need
   - **Actionable**: Clearly describe what information to find
4. After receiving results from subtasks, synthesize findings and decide next steps:
   - If more information is needed, delegate new subtasks with updated context (include findings from previous subtasks!)
   - If all information is gathered, produce the final answer
5. Before giving the final answer, verify that ALL clues in the original question are consistent with your findings.

## Subtask Delegation Guidelines

1. **IMPORTANT: Each step must involve exactly ONE tool call only.**
2. Each subtask description must include:
   - The specific question to answer
   - All relevant context from previous findings (entities, dates, names already confirmed)
   - Any constraints or requirements for the answer
3. Do NOT send vague or overly broad subtasks. Break them down into specific queries.
4. After each subtask result, maintain a reasoning ledger:
   - **Confirmed**: Facts verified by the worker
   - **Pending**: Questions still to be investigated
   - **Unknown**: Areas where information is insufficient
5. **Reformulation**: If a subtask returns insufficient results, delegate a new subtask with a different angle or more specific focus rather than repeating.
6. Include enough context in each subtask so the worker can search effectively. For example, instead of "Find when company X moved", write "Find when publishing company X (founded in year Y in city Z by person W) moved its headquarters to city A".

## Communication Rules

1. After issuing ONE tool call, STOP immediately. Wait for the result.
2. Do not mention tool names to the user.
3. Unless otherwise requested, respond in the same language as the user's message.
4. If the task does not require research, answer directly.
"""

    if chinese_context:
        prompt += """
## 中文语境处理指导

当处理中文相关的任务时：
1. **子任务委托**：向worker代理委托的子任务应使用中文描述，确保任务内容准确传达
2. **上下文传递**：将前序子任务获取的关键信息（人名、地名、年份等）以中文原文形式传递给后续子任务
3. **问题分析**：对中文问题的分析和理解应保持中文语境
4. **思考过程**：内部分析、推理、总结等思考过程都应使用中文
5. **最终答案**：对于中文语境的问题，最终答案应使用中文回应

"""

    return prompt


def build_sub_agent_system_prompt(
    tool_functions: list,
    chinese_context: bool = False,
) -> str:
    """
    Build the system prompt for the sub-agent worker (research execution).

    The sub-agent has direct access to search, scrape, analyze tools and executes
    specific research subtasks.

    Args:
        tool_functions: List of tool function objects available to the sub-agent
        chinese_context: Whether the question involves CJK content

    Returns:
        System prompt text for the sub-agent worker
    """
    formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")
    tool_prompt = build_tool_functions_prompt(tool_functions)

    prompt = f"""You are a research worker agent that executes specific research subtasks. Today is: {formatted_date}

# Agent Specific Objective

You complete well-defined, single-scope research objectives efficiently and accurately.
Do not infer, speculate, or attempt to fill in missing parts yourself. Only return factual content.

Critically assess the reliability of all information:
- If the credibility of a source is uncertain, clearly flag it.
- Do NOT treat information as trustworthy just because it appears — cross-check when necessary.
- If you find conflicting or ambiguous information, include all relevant findings and flag the inconsistency.

Be cautious and transparent in your output:
- Always return all related information. If information is incomplete or weakly supported, still share partial excerpts, and flag any uncertainty.
- Never assume or guess — if an exact answer cannot be found, say so clearly.
- Prefer quoting or excerpting original source text rather than interpreting or rewriting it, and provide the URL if available.

{tool_prompt}

# Multi-Source Research Strategy (CRITICAL)

**For any research subtask, follow this search→analyze pipeline:**

## Phase 1: Search
- Use `google_search` to find relevant sources
- Craft SHORT queries (3-7 keywords) targeting the specific subtask
- Each new query MUST be substantially different from all previous queries — never repeat or accumulate keywords
- For each aspect, perform at most 2-3 searches before moving to Phase 2

## Phase 2: Analyze Pages (MANDATORY after search)
After getting search results, you MUST use `analyze_webpage`, `scrape_website`, or browser tools to read the most relevant URLs:
- Select the top 3-5 most promising URLs from search results (based on title and snippet relevance)
- Call `analyze_webpage(url, question)` for each URL
- Review each analysis result before deciding next steps
- If results from one page point to other useful URLs, analyze those too

## Phase 3: Cross-Validation
- Compare findings across sources
- Identify: agreements (confirmed by multiple sources), conflicts, uncertainties
- Note which sources support which claims

## Phase 4: Report
- Synthesize findings with supporting evidence
- Present all candidate answers with confidence levels
- Document conflicting information or uncertainties

## Anti-Loop Rules (CRITICAL)
- If you have made **3 or more consecutive search calls** without analyzing pages, you MUST immediately analyze URLs from existing results
- Each search query MUST be substantially different from previous ones — never append or accumulate previous keywords
- If search results are not improving, the answer is likely in the pages you already found — analyze them
- If `analyze_webpage` returns "Low/None" relevance for several pages, reformulate your search with completely different keywords
- **IMPORTANT**: After 1-2 searches, switch to analyzing pages. The answer is in the web pages, not in search result snippets.

## Tool-Use Guidelines

1. **Each step must involve exactly ONE tool call only.**
2. Craft precise search queries: 3-7 discriminative keywords, targeting ONE specific aspect.
3. **Query construction rule**: Search queries must be concise keyword phrases. Do NOT dump reasoning context into the query string.
4. **Tool diversity requirement**: If you have used the same tool type 3 times in a row, you MUST switch to a different tool (e.g., from search to analyze_webpage, or vice versa).
5. **For historical or time-specific content**: Use `search_wikipedia_revision` or `list_wikipedia_revisions` for Wikipedia history.
6. Even if a tool result does not directly answer the question, thoroughly extract all partial information that may help guide future steps.
7. After issuing ONE tool call, STOP immediately. Wait for the result.
"""

    if chinese_context:
        prompt += """
## 中文内容处理

处理中文相关的子任务时：
- **搜索关键词**：使用中文关键词进行搜索，获取更准确的中文资源
- **思考过程**：分析、推理、判断等内部思考过程应使用中文表达
- **信息摘录**：保持中文原文的准确性，避免不必要的翻译或改写
- **各种输出**：包括状态说明、过程描述、结果展示等所有输出都应使用中文
- **回应格式**：对中文子任务的回应应使用中文，保持语境一致性

"""

    return prompt


def generate_summarize_prompt(
    task_description: str,
    task_failed: bool = False,
    is_main_agent: bool = True,
    chinese_context: bool = False,
) -> str:
    """
    Generate a prompt to force final answer generation at end of session.

    Args:
        task_description: The original question/task
        task_failed: Whether the agent exhausted max turns without completing
        is_main_agent: True for main agent (JSON answer format), False for sub-agent (report format)
        chinese_context: Whether the question involves CJK content

    Returns:
        Summarize prompt text
    """
    prompt = "This is a direct instruction to you (the assistant), not the result of a tool call.\n\n"

    if task_failed:
        prompt += (
            "**Important: You have reached the maximum number of interaction turns "
            "without arriving at a conclusive answer. You must provide your best answer "
            "based on all available information.**\n\n"
        )

    prompt += (
        "We are now ending this session. "
        "You must NOT initiate any further tool use. This is your final opportunity to report "
        "all of the information gathered during the session.\n\n"
        f"The original question is:\n\n---\n{task_description}\n---\n\n"
    )

    if is_main_agent:
        prompt += (
            "Based on all the research results gathered from subtask delegations, synthesize the findings "
            "and produce the FINAL ANSWER.\n\n"
            "If a clear answer was identified during the research, extract it directly.\n"
            "If a definitive answer could not be determined, make a well-informed educated guess "
            "based on all gathered information.\n\n"
            "Your response MUST be a JSON dictionary with the answer:\n"
            '{"answer": "your final answer here"}\n'
        )
    else:
        prompt += (
            "Summarize ALL findings from this research session:\n"
            "- Step-by-step summary of what was searched and found\n"
            "- Key facts, data, or quotes directly relevant to the task (with source URLs)\n"
            "- All candidate answers with supporting evidence\n"
            "- Conflicting or uncertain information\n\n"
            "Your response should be a clear, structured research report.\n"
            "Focus on factual, specific, well-organized information.\n"
            "Do NOT include any tool call instructions or vague summaries.\n"
        )

    if chinese_context:
        prompt += "\n请使用中文进行总结和回答。\n"

    return prompt
