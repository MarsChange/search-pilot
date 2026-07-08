<p align="center">
  <h1 align="center">Search Pilot - A Deep Research Agent</h1>
  <p align="center"><b>面向复杂多跳问答的状态图驱动深度研究 Agent</b></p>
  <em><p align="center">阿里云 Data+AI 工程师全球大奖赛：高校赛道 Research Agent 参赛项目</p></em>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#agent-workflow">Agent Workflow</a> •
  <a href="#quickstart">Quick Start</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#evaluation">Evaluation</a>
</p>

---

## Update on 2026.6.25

在原来比赛的基础之上进行重构，不依赖 LangChain/LangGraph 而是重构了一套基于 FSM 驱动的 dynamic state graph 调度并发范式，由原来的全局记忆账本改为 SQLite + numpy 向量索引的共享记忆存储，并添加上下文压缩机制，缓解原先框架上下文长度爆炸的问题。同时引入 LLM-as-judge的评测脚本，使用天池大赛复赛 100 道赛题，实现了64%的 Deep Research正确率。

## Overview

Search Pilot 是一个基于 LLM 的 Deep Research 系统，用于回答需要检索、验证、跨来源推理和证据综合的复杂问题。当前版本采用 `DeepResearchRunner` 作为统一执行引擎：先把问题规划为可调度的 research states，再用状态图管理依赖、并发执行、覆盖检查、必要时重规划，最后综合证据并输出答案。

系统提供三类入口：

- **问答 API**：`POST /` 和 `POST /stream`，返回最终答案、研究报告和元数据。
- **可视化界面**：`GET /visualize`，实时查看计划、状态路径、工具调用、网页解析和最终报告。
- **评测入口**：`POST /evaluate-stream`，对 JSONL 数据集逐题评测，支持断点续跑和每题独立记忆库。

核心目标不是一次性生成答案，而是把一次研究过程拆成可观察、可恢复、可调参的执行链路。

## Features

- **状态图研究流程**：将问题拆解为 `search`、`analyze`、`verify`、`backtrack`、`synthesize` 等节点，并按依赖关系调度。
- **并发检索与网页分析**：每轮最多并发执行 `max_concurrent` 个 ready states，每个 state 自动搜索、打开高价值 URL、补充 Wikipedia 查询。
- **覆盖检查与重规划**：根据 coverage checklist、失败节点、冲突、开放问题和来源质量判断是否需要补充验证。
- **证据记忆库**：把每个 state 的 evidence 写入 SQLite，支持后续 state 检索上下文；评测模式下按题目/session 隔离并在每题开始前重置。
- **上下文压缩**：工具结果、历史结果和记忆上下文进入 worker 前会经过压缩，降低长链路 token 压力。
- **对抗审阅**：可选 Red/Blue 审阅轮次，用于在最终报告前检查薄弱点。
- **断点续跑评测**：评测结果写入 JSONL checkpoint，API key 失效、欠费或外部工具异常后可继续跳过已完成题目。
- **本地运行日志**：工具调用、搜索结果、网页解析、LLM 调用、评测 checkpoint 等事件写入 `logs/agent_runtime.jsonl`。

<h2 id="agent-workflow">Agent Workflow</h2>

当前 Agent 工作流由 `agent.py` 接收请求，并交给 `deep_research.runner.DeepResearchRunner` 执行。

```text
HTTP Request
    |
    v
FastAPI agent.py
    |
    v
DeepResearchRunner
    |
    +--> PLANNING
    |       使用 DeepResearchPlanner 生成 ResearchPlan：
    |       - research_intent
    |       - success_criteria
    |       - coverage_checklist
    |       - risk_flags
    |       - ResearchState 列表
    |
    +--> DISPATCHING
    |       ResearchStateGraph 找出依赖已满足的 pending states，
    |       按 priority 和 max_concurrent 并发调度。
    |
    +--> WORKER EXECUTION
    |       每个 DeepResearchWorker 针对一个 state：
    |       - 准备 search_queries
    |       - 调用 search_engine 或 Wikipedia
    |       - 选择搜索结果 URL 做 scrape/analyze
    |       - 压缩工具结果、记忆和 prior_results
    |       - 让 LLM 输出结构化 WorkerResult
    |
    +--> COLLECTING
    |       把 WorkerResult 中的 evidence 写入 SharedMemoryStore。
    |
    +--> COVERAGE_CHECK
    |       检查 coverage、failed states、open questions、
    |       conflicts、一手来源和时效信息。
    |
    +--> REPLANNING
    |       若覆盖不完整或存在失败/冲突，且未超过 max_replans，
    |       生成新的 verify/backtrack states 并加入状态图。
    |
    +--> SYNTHESIZING
    |       DeepResearchSynthesizer 基于 plan、results、state_history、
    |       coverage 和 sources 生成最终研究报告。
    |
    +--> ADVERSARIAL_REVIEW
    |       可选 Red/Blue 审阅，检查最终报告是否有漏洞。
    |
    +--> FINALIZING
            提取最终 answer，返回 report 和 metadata。
```

### State Graph

`ResearchStateGraph` 是当前执行流的核心调度结构：

- `pending`：等待依赖完成。
- `running`：正在执行。
- `success`：节点已完成并产生可用结果。
- `failed`：节点超时、工具失败或 worker 结果失败。
- `skipped`：依赖不可用，或最终答案已足够时跳过。

调度时只执行依赖全部 `success` 的节点；失败率过高或 coverage 不完整时进入重规划。这样可以避免无依赖的节点串行等待，也可以把失败补救集中到新的验证节点中。

### Worker Tool Flow

每个 worker 不再让 LLM 任意循环调用工具，而是采用受控工具流程：

1. 从 state 的 `search_queries` 或 `description` 生成最多 4 个候选查询。
2. 对前 3 个查询调用搜索工具。
3. 从搜索结果里选择最多 2 个 URL 调用 `analyze_webpage` 或 `scrape_website`。
4. 对需要实体核验的节点补充 Wikipedia 查询。
5. 压缩工具结果后交给 LLM，要求输出严格 JSON 格式的 `WorkerResult`。

`WorkerResult` 包含：

- `status`: `resolved`、`partial` 或 `failed`
- `candidate_answer`
- `key_findings`
- `evidence`
- `conflicts`
- `open_questions`
- `canonical_names`
- `answer_form_hint`
- `tool_calls`

### Memory Design

记忆库由 `SharedMemoryStore` 管理，默认路径为：

```text
data/deep_research_memory.db
```

每条 evidence 会被转换为 `MemoryEntry`，记录 claim、source、url、confidence、agent/state id、topic 和 metadata。普通问答会复用同一个 memory db；评测模式会为每个样本使用独立 db：

```text
data/eval_checkpoints/eval_memory/<row_key>.db
```

在每道评测题开始前，系统会删除同名 `.db`、`.db-wal`、`.db-shm`，保证题目之间不共享历史 evidence 和 conflicts。

### Runtime Events

流式接口会通过 SSE 输出研究过程事件：

| Event            | 含义                            |
| ---------------- | ------------------------------- |
| `State`        | FSM 状态切换                    |
| `Plan`         | 初始研究计划和状态列表          |
| `Dispatch`     | 本轮可并发执行的 ready states   |
| `TaskStart`    | 单个 state 开始                 |
| `TaskResult`   | 单个 state 完成、失败或部分完成 |
| `Coverage`     | 覆盖检查结果                    |
| `Replan`       | 重规划开始和结果                |
| `Final`        | 最终答案和元数据                |
| `EvalRunEvent` | 评测时包装的单题内部研究事件    |
| `EvalItem`     | 单题评测结果                    |
| `EvalSummary`  | 整轮评测汇总                    |
| `EvalError`    | 评测中断错误                    |

<h2 id="quickstart">Quick Start</h2>

### 环境要求

- Python 3.12 推荐；Python 3.10+ 通常也可运行。
- DashScope API Key，用于调用 Qwen 兼容接口。
- 可选：Serper、Jina、E2B、Playwright MCP 等工具 Key。

### Conda 环境

```bash
conda create -n tianchi_agent python=3.12 -y
conda activate tianchi_agent
pip install -r requirements.txt
```

### 环境变量

```bash
cp .env.template .env
```

至少需要在 `.env` 中配置：

```env
DASHSCOPE_API_KEY=sk-xxxxx
```

### 启动服务

```bash
conda run -n tianchi_agent python -m uvicorn agent:app --reload --host 0.0.0.0 --port 8000
```

启动后可访问：

- API: `http://127.0.0.1:8000/`
- 可视化界面: `http://127.0.0.1:8000/visualize`
- 环境状态: `http://127.0.0.1:8000/env-status`

如果 `Ctrl-C` 无法停止 `--reload` 进程，可以先查 PID：

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <PID>
```

必要时再使用：

```bash
kill -9 <PID>
```

<h2 id="configuration">Configuration</h2>

| 环境变量                    | 必需 | 说明                                                     |
| --------------------------- | ---- | -------------------------------------------------------- |
| `DASHSCOPE_API_KEY`       | 是   | LLM 调用 Key                                             |
| `QWEN_MODEL`              | 否   | 默认`qwen-max`                                         |
| `DEEP_RESEARCH_MEMORY_DB` | 否   | 普通问答的 SQLite 记忆库路径                             |
| `SERPER_API_KEYS`         | 否   | Serper Key 池，支持逗号或换行分隔                        |
| `SERPER_API_KEY`          | 否   | 单个 Serper Key                                          |
| `IQS_API_KEY`             | 否   | 可选搜索服务 Key                                         |
| `JINA_API_KEY`            | 否   | Jina Reader，用于网页解析和降级                          |
| `JINA_READER_URL`         | 否   | 默认`https://r.jina.ai`                                |
| `E2B_API_KEY`             | 否   | E2B Python 沙箱                                          |
| `PLAYWRIGHT_MCP_URL`      | 否   | Playwright MCP 服务地址                                  |
| `PLAYWRIGHT_MCP_TOKEN`    | 否   | Playwright MCP Token                                     |
| `SUB_AGENT_NUM`           | 否   | 兼容旧配置，当前主要使用请求里的`max_concurrent`       |
| `TIANCHI_AGENT_LOG_FILE`  | 否   | runtime JSONL 日志路径，默认`logs/agent_runtime.jsonl` |

工具按环境变量条件加载。没有配置对应 Key 时，相关工具不会注册，worker 会根据可用工具降级执行。

## API

### 普通问答

```bash
curl -X POST http://127.0.0.1:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请研究某个复杂问题",
    "max_concurrent": 3,
    "max_replans": 2,
    "max_adversarial_rounds": 0
  }'
```

返回：

```json
{
  "answer": "...",
  "report": {
    "query": "...",
    "content": "...",
    "sources": [],
    "confidence": 0.0,
    "state_history": [],
    "states": [],
    "coverage": {},
    "critique_history": []
  },
  "metadata": {
    "mode": "deep_research",
    "session_id": "...",
    "num_states": 0,
    "num_searches": 0,
    "num_replans": 0,
    "elapsed_seconds": 0.0
  }
}
```

### 流式问答

```bash
curl -N -X POST http://127.0.0.1:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"..."}'
```

### AG-UI 兼容入口

```text
POST /ag-ui
```

该入口会从 AG-UI 风格 payload 中提取用户问题，并映射到 Deep Research 流式执行。

<h2 id="evaluation">Evaluation</h2>

评测接口读取 JSONL 数据集，每行需要包含 `question` 和 `answer` 字段。接口会逐题运行 Deep Research，再调用 LLM judge 判断预测答案是否等价。

```bash
curl -N -X POST http://127.0.0.1:8000/evaluate-stream \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "test/data_with_answer.jsonl",
    "start": 0,
    "limit": 200,
    "max_concurrent": 3,
    "max_replans": 2,
    "max_adversarial_rounds": 0,
    "resume_from_checkpoint": true,
    "rerun_incorrect_checkpoint_items": false
  }'
```

### Checkpoint

默认 checkpoint 路径：

```text
data/eval_checkpoints/<dataset-stem>-<dataset-hash>.jsonl
```

每条完成记录包含：

- `row_key`
- `dataset_path`
- `index`
- `id`
- `item_correct`
- 原始题目、标准答案、预测答案和 judge 结果
- Deep Research metadata
- `checkpoint_path`

重新运行时，`resume_from_checkpoint=true` 会跳过已完成题目。`rerun_incorrect_checkpoint_items=true` 会只重新测试 checkpoint 中 `item_correct=false` 的错误样本，已正确样本仍跳过；新结果会追加写入 checkpoint，并在后续 resume 时覆盖旧记录。`reset_checkpoint=true` 会删除 checkpoint 后从头开始。

如果遇到 API key 失效、欠费或鉴权类错误，评测流会发出 `EvalError` 并停止，避免把错误题继续写成完成结果。

### 可视化评测

打开：

```text
http://127.0.0.1:8000/visualize
```

可在页面里启动评测，并查看每道题的：

- plan 和 coverage checklist
- state 路径和状态变化
- 搜索、网页解析、工具调用结果
- gold answer、predicted answer 和 judge reason
- checkpoint/resume 状态

## Tool System

默认工具由 `tools/__init__.py` 根据环境变量导入，并通过 `FunctionToolExecutor` 统一执行。

| 工具                          | 用途                                                     |
| ----------------------------- | -------------------------------------------------------- |
| `search_engine`             | Google/Serper 搜索                                       |
| `search_wikipedia`          | Wikipedia 当前页面检索                                   |
| `search_wikipedia_revision` | Wikipedia 历史版本检索                                   |
| `list_wikipedia_revisions`  | Wikipedia 修订历史                                       |
| `scrape_website`            | 网页解析，Jina Reader 优先，失败后 requests + MarkItDown |
| `analyze_webpage`           | 解析网页后用 LLM 针对问题抽取证据                        |
| `code_sandbox`              | E2B Python 沙箱                                          |
| `browser_session`           | Playwright MCP 浏览器自动化                              |

工具调用会被包裹为 30 秒超时，并写入 runtime log。网页解析中的 Jina 失败会自动尝试 requests 降级；如果两条链路都失败，会记录 `webpage_parse_error`。

## Module Structure

```text
.
├── agent.py                         # FastAPI 入口、SSE、评测、可视化页面挂载
├── deep_research/
│   ├── runner.py                    # DeepResearchRunner 主流程
│   ├── planner.py                   # 初始计划与重规划
│   ├── state_graph.py               # ResearchStateGraph 调度
│   ├── worker.py                    # 单 state 检索、解析和结构化输出
│   ├── synthesizer.py               # 最终报告综合与答案抽取
│   ├── tool_adapter.py              # 工具注册、执行、超时和日志
│   ├── runtime_logging.py           # JSONL 本地日志
│   ├── schemas.py                   # ResearchPlan/State/Result 数据结构
│   ├── memory/                      # SQLite 记忆库和 embedding
│   ├── compressor/                  # 上下文压缩
│   └── adversarial/                 # Red/Blue 对抗审阅
├── evaluation/
│   └── llm_judge.py                 # 评测答案等价判断
├── tools/                           # 搜索、Wikipedia、网页解析、浏览器和沙箱工具
├── web/
│   └── research_visualizer.html     # 研究和评测可视化界面
├── requirements.txt
└── .env.template
```

## Logs

默认日志文件：

```text
logs/agent_runtime.jsonl
```

常见事件：

- `tool_call_start`
- `tool_call_end`
- `tool_call_error`
- `search_results`
- `webpage_parse_start`
- `webpage_parse_fallback`
- `webpage_parse_end`
- `webpage_parse_error`
- `webpage_analysis_start`
- `webpage_analysis_end`
- `llm_call_end`
- `eval_checkpoint_load`
- `eval_checkpoint_save`
- `eval_memory_reset`

这些日志用于定位工具调用、网页解析、Jina 额度、搜索质量、超时和评测断点问题。

## 比赛成绩

### 初赛

![初赛成绩截图](docs/preliminary-result.png)

### 复赛

![复赛成绩截图](docs/final-result.png)

## License

[MIT](LICENSE)
