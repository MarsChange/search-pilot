hao"""
Prompt builders and tool calling utilities for the agent.

Provides system prompt generation for main agent and sub-agent, tool function
descriptions, and summarize prompt generation.
"""

import datetime
import logging

logger = logging.getLogger(__name__)


def build_tool_functions_prompt(tool_functions: list) -> str:
    """Build a compact tool description block."""
    if not tool_functions:
        return ""

    categories = {}
    for func in tool_functions:
        name = func.__name__
        if name.startswith("browser_"):
            category = "browser"
        elif name == "analyze_webpage":
            category = "webpage_analyzer"
        elif name == "search_engine":
            category = "search"
        elif name.startswith("search_wikipedia") or name == "list_wikipedia_revisions":
            category = "wiki"
        elif name == "scrape_website":
            category = "scrape"
        else:
            category = "general"
        categories.setdefault(category, []).append(name)

    lines = ["# Available Tools", ""]

    if "search" in categories:
        lines.append("**Search** (`search_engine`)")
        lines.append("- Use short, discriminative queries.")
        lines.append("- Supports `engine=auto|google|iqs`.")
        lines.append("- Use `google` for international / English entities, `iqs` for Chinese entities and Chinese websites, and `auto` when routing can follow query language.")
        lines.append("")

    if "wiki" in categories:
        lines.append("**Wikipedia** (`search_wikipedia`, `search_wikipedia_revision`, `list_wikipedia_revisions`)")
        lines.append("- Use for entity grounding, dates, and historical snapshots.")
        lines.append("")

    if "scrape" in categories:
        lines.append("**Website Scraper** (`scrape_website`)")
        lines.append("- Use when snippets are insufficient and you need raw page text.")
        lines.append("")

    if "webpage_analyzer" in categories:
        lines.append("**Webpage Analyzer** (`analyze_webpage`)")
        lines.append("- Use for compact, question-focused evidence extraction from a page.")
        lines.append("")

    if "browser" in categories:
        lines.append("**Browser Tools**")
        lines.append("- Use only for pages that require interaction or dynamic rendering.")
        lines.append(f"- Available: {', '.join(categories['browser'])}")
        lines.append("")

    if "general" in categories:
        lines.append("**Other Tools**")
        lines.append(f"- {', '.join(categories['general'])}")
        lines.append("")

    return "\n".join(lines).strip()


def build_main_agent_system_prompt(
    tool_functions: list,
    chinese_context: bool = False,
    max_parallel: int = 3,
) -> str:
    """Build the system prompt for the main coordinator agent."""
    formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")

    prompt = f"""You are a research coordinator agent. Today is: {formatted_date}

Your job is to solve research questions by delegating focused subtasks through `execute_subtasks`, then deciding the final answer from structured evidence.

You do NOT have direct web search access. For research questions, do not answer on the first turn from prior knowledge alone. Use `execute_subtasks` first unless the task is obviously non-research.

## Core Policy

1. Decompose the question into nodes before the first tool call.
2. Track six items internally at all times:
   - `constraints`
   - `confirmed_facts`
   - `open_nodes`
   - `candidate_answer`
   - `answer_form`
   - `verification_status`
3. Only mark a node as confirmed when the worker result clearly resolves it and there is no meaningful contradiction.
4. Before the final answer, re-read the original question and verify every constraint one by one.
5. Exact-match accuracy matters more than speed.

## How to Use `execute_subtasks`

Call `execute_subtasks` with a JSON array of self-contained subtask strings.
- Maximum {max_parallel} subtasks in one call.
- Each subtask must include the local goal, relevant confirmed context, and answer-format requirement.
- Only parallelize independent nodes.

The tool returns compact JSON with this shape:
{{
  "worker_reports": [
    {{
      "status": "resolved|unresolved",
      "subtask": "...",
      "subtask_answer": "...",
      "confidence": "high|medium|low",
      "facts": ["..."],
      "sources": ["..."],
      "canonical_names": ["..."],
      "answer_form_hint": "...",
      "unresolved": ["..."]
    }}
  ],
  "ledger_update": {{
    "confirmed_facts": ["..."],
    "candidate_answers": ["..."],
    "canonical_names": ["..."],
    "open_questions": ["..."],
    "source_shortlist": ["..."]
  }}
}}

Read the tool result as a ledger update, not as free-form prose.

## Final Answer Guardrails

- Never output the final answer before `answer_form` is clear.
- Never output a name, title, era name, nickname, abbreviation, or translation variant unless it exactly matches the requested identifier type.
- Use full official names by default unless the question asks for another form.
- Chinese question -> Chinese answer. English question -> English answer. If the question explicitly asks for an English official name or another format, follow that request exactly.
- If the answer is numeric, return digits only unless the question says otherwise.
- Output ONLY JSON in the form {{"answer": "..."}}.
"""

    if chinese_context:
        prompt += """

## 中文语境附加要求

- 向 worker 派发子任务时优先使用中文描述。
- 对中文实体，最终答案优先使用权威中文标准译名或官方中文全称。
- 若题目明确要求英文全称、官方英文名或特定格式，以题目要求为准。
"""

    return prompt


def build_sub_agent_system_prompt(
    tool_functions: list,
    chinese_context: bool = False,
) -> str:
    """Build the system prompt for the sub-agent worker."""
    formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")
    tool_prompt = build_tool_functions_prompt(tool_functions)

    prompt = f"""You are a research worker agent. Today is: {formatted_date}

You execute one focused research subtask and return a compact structured report for the coordinator.

## Mission

- Resolve exactly one node.
- Gather evidence, not broad discussion.
- Prefer reliable sources.
- Keep output compact and structured.

{tool_prompt}

## Search Strategy

1. Start with targeted search.
2. Pick the search engine deliberately:
   - Prefer `engine="google"` for international entities, English names, foreign institutions, science, and cross-border topics.
   - Prefer `engine="iqs"` for Chinese entities, Chinese organizations, Chinese place names, and Chinese-language sites.
   - Use `engine="auto"` when the query language already strongly indicates the right source pool.
3. If snippets already answer the subtask clearly, report immediately.
4. If the current engine returns weak, off-topic, or empty results, switch to the other engine and retry with a tighter query.
5. If snippets are insufficient, inspect only the most promising 1-2 pages.
6. For historical/time-sensitive Wikipedia questions, use revision tools.
7. Stop once you can support the subtask answer with strong evidence.

## Output Contract

Return ONLY one JSON object with this schema:
{{
  "status": "resolved|unresolved",
  "subtask_answer": "best answer for this node or empty string",
  "confidence": "high|medium|low",
  "facts": ["short evidence-backed facts"],
  "sources": ["url or source label"],
  "canonical_names": ["official or canonical entity names relevant to the answer"],
  "answer_form_hint": "what identifier or format the coordinator should prefer",
  "unresolved": ["what is still ambiguous or missing"]
}}

Rules:
- No Markdown fences.
- No prose outside the JSON object.
- `facts` must be concise and evidence-like, not chain-of-thought.
- `sources` should be short and useful.
- If you have only one authoritative source, say so in `facts` or `unresolved`.
- If the node is not resolved, set `status` to `unresolved` and leave `subtask_answer` empty or tentative.
- Do not emit many alternative candidates unless the ambiguity is real and important.
"""

    if chinese_context:
        prompt += """

## 中文处理

- 中文子任务优先使用中文检索和中文输出。
- 中文实体优先保留标准中文名称，同时在 `canonical_names` 中补充必要的官方全称。
"""

    return prompt


def generate_summarize_prompt(
    task_description: str,
    task_failed: bool = False,
    is_main_agent: bool = True,
    chinese_context: bool = False,
) -> str:
    """Generate the final no-more-tools prompt."""
    prompt = "This is a direct instruction to you, not a tool result.\n\n"

    if task_failed:
        prompt += (
            "You have reached the maximum number of turns. "
            "Use the evidence already collected and finish carefully.\n\n"
        )

    prompt += (
        "No more tool use is allowed.\n\n"
        f"Original task:\n---\n{task_description}\n---\n\n"
    )

    if is_main_agent:
        prompt += (
            "Re-read the original question.\n"
            "Check every constraint explicitly.\n"
            "Choose the answer form that exactly matches the question.\n"
            "Return ONLY JSON: {\"answer\": \"...\"}\n"
        )
    else:
        prompt += (
            "Return ONLY one JSON object using this schema:\n"
            "{"
            "\"status\": \"resolved|unresolved\", "
            "\"subtask_answer\": \"\", "
            "\"confidence\": \"high|medium|low\", "
            "\"facts\": [], "
            "\"sources\": [], "
            "\"canonical_names\": [], "
            "\"answer_form_hint\": \"\", "
            "\"unresolved\": []"
            "}\n"
        )

    if chinese_context:
        prompt += "\n请使用与原问题一致的语言完成输出。\n"

    return prompt
