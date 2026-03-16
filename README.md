<p align="center">
  <h1 align="center">🔍 Search Pilot - A Deep Research Agent</h1>
  <p align="center"><b>基于多智能体协作的深度研究系统，专攻复杂多跳推理问答</b></p>
  <em><p align="center">本项目为<a href=https://tianchi.aliyun.com/competition/entrance/532448/customize823>阿里云 Data+AI 工程师全球大奖赛：高校赛道</a>的 <b>Research Agent</b> 参赛项目 (Rank 15)</p></em>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#features">Features</a> •
  <a href="#quickstart">Quick Start</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#tools">Tool System</a>
</p>

---

## Overview

一个基于 LLM 的多智能体深度研究系统，任务场景为解答复杂的多跳推理问题。它将复杂问题自动分解为多个子任务，分派给并行工作的研究型子智能体，最终汇聚各方结论生成高质量答案。

本项目构建了 **双层 Agent 架构**，实现多跳推理链式解题——主智能体负责任务规划与推理链推进，子智能体负责并行执行搜索、爬取、分析等具体研究操作，多轮协作直至问题完全解决。

### 核心理念

```
复杂问题 → 推理链分解 → 并行子任务研究 → 证据汇聚 → 链式推进 → 最终答案
```

## Features

- **🧠 多跳推理 (Multi-Hop Reasoning)** — 将复杂问题分解为推理链，逐层解决依赖关系，支持高达 30 轮主智能体交互
- **👥 多智能体并行研究 (Multi-Agent Parallelism)** — 主智能体协调任务分解，最多同时派出 N 个子智能体并行执行研究任务（N 可配置，默认 3）
- **🔧 丰富的工具生态 (Rich Tool Ecosystem)** — 搜索引擎、Wikipedia（含历史版本回溯）、网页爬取、网页智能分析、沙箱代码执行、浏览器自动化
- **📡 实时进度流式推送** — 子智能体研究进度通过 `asyncio.Queue` 实时推送至主循环，前端可追踪每个 Worker 的搜索、分析、思考过程
- **🇨🇳 中文语境优化** — 自动检测 CJK 内容，切换中文搜索策略和中文提示词，确保中文问题的准确理解与回答
- **🛡️ 多层容错与鲁棒性** — Wikipedia API → Jina Reader 降级链、Jina Reader → requests+MarkItDown 爬取降级、DashScope 内容过滤自动清洗重试、工具调用超时保护

<h2 id="architecture">Architecture</h2>

### 两层智能体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户请求 (User Request)                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (agent.py)                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 🧠 Main Agent (主智能体)                          │
│                                                                  │
│  · 接收用户问题，构建完整推理链                                       │
│  · 识别可并行的独立推理节点                                           │
│  · 调用 execute_subtasks 分派子任务（JSON 数组）                     │
│  · 根据子任务结果推进推理链，必要时发起新一轮子任务                       │
│  · 汇总所有研究结果，生成 {"answer": "..."} 最终答案                   │
│                                                                  │
│  可用工具: execute_subtasks, code_sandbox                         │
│  最大交互轮次: 30                                                  │
└──────────┬──────────┬──────────┬────────────────────────────────┘
           │          │          │    asyncio.gather 并发启动
           ▼          ▼          ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ 🔎 Sub-    │ │ 🔎 Sub-    │ │ 🔎 Sub-    │    ← 最多 N 个并行
│ Agent #1   │ │ Agent #2   │ │ Agent #3   │      (SUB_AGENT_NUM)
│            │ │            │ │            │
│ · 搜索引擎  │ │ · 搜索引擎  │ │ · 搜索引擎  │
│ · Wikipedia│ │ · Wikipedia│ │ · Wikipedia│
│ · 网页爬取  │ │ · 网页爬取  │ │ · 网页爬取  │
│ · 网页分析  │ │ · 网页分析  │ │ · 网页分析  │
│ · 浏览器   │ │ · 浏览器    │ │ · 浏览器   │
│            │ │            │ │            │
│ 最大轮次:10 │ │ 最大轮次:10 │ │ 最大轮次:10 │
└────────────┘ └────────────┘ └────────────┘
       │              │              │
       └──────────────┼──────────────┘
                      │ 研究报告汇总
                      ▼
              主智能体继续推理
```

### 请求处理流程

```
用户问题
    → agent_loop() 异步生成器
        → LLM 流式调用 (Qwen via DashScope)
            → 检测到 tool_call: execute_subtasks
                → 解析子任务 JSON 数组
                → asyncio.gather 并发启动 N 个 run_sub_agent()
                    → 各子智能体独立执行: LLM ↔ Tool 循环
                    → 实时进度通过 asyncio.Queue 推送至主循环
                    → 返回结构化研究报告
                → 汇总所有子智能体报告
            → 主智能体继续推理链推进
            → ... (可多次调用 execute_subtasks)
        → 最终 LLM 合成答案
    → 流式响应返回
```

### 模块结构

```
deep-research-agent/
├── agent.py              # FastAPI 服务入口
├── agent_loop.py          # 核心编排引擎：流式 LLM 调用、工具执行、子智能体管理
├── tools_calling.py       # 提示词构建器：主/子智能体系统提示、工具描述格式化
├── tools/
│   ├── __init__.py        # 工具注册中心，按环境变量条件加载，分配主/子智能体工具集
│   ├── search_engine.py   # Google 搜索 (Serper API)
│   ├── wiki_search.py     # Wikipedia 搜索（含历史版本查询 + Jina 降级）
│   ├── scrape_website.py  # 网页爬取与内容提取（双重降级策略）
│   ├── webpage_analyzer.py# LLM 驱动的网页智能分析
│   ├── code_sandbox.py    # E2B 沙箱代码执行环境
│   └── browser_session.py # Playwright MCP 浏览器自动化
├── requirements.txt       # Python 依赖
└── .env.template          # 环境变量模板
```

<h2 id="quickstart">Quick Start</h2>

### 环境要求

- Python 3.10+
- [DashScope API Key](https://dashscope.aliyun.com/)（必需，用于调用 Qwen 大模型）

### 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.template .env
# 编辑 .env 文件，填入你的 API Keys
```

### 启动服务

```bash
python -m uvicorn agent:app --reload --host 0.0.0.0 --port 8000
```

<h2 id="configuration">Configuration</h2>

将 `.env.template` 复制为 `.env` 并填入相应的 API Keys。工具模块按需加载——只有配置了对应 Key 的工具才会被激活。

| 环境变量                 | 必需 | 说明                                                      |
| ------------------------ | ---- | --------------------------------------------------------- |
| `DASHSCOPE_API_KEY`    | ✅   | 阿里云 DashScope API Key，用于调用 Qwen 大模型            |
| `QWEN_MODEL`           |      | 模型名称，默认 `qwen-max`                               |
| `SERPER_API_KEYS`      |      | Google 搜索服务 Key 池（可选，支持逗号/换行分隔多个 key） |
| `SERPER_API_KEY`       |      | Google 搜索服务 (Serper)，启用搜索引擎工具                |
| `JINA_API_KEY`         |      | Jina AI，启用网页爬取、网页分析、Wikipedia 降级           |
| `E2B_API_KEY`          |      | E2B 沙箱，启用安全的 Python 代码执行                      |
| `PLAYWRIGHT_MCP_URL`   |      | Playwright MCP 服务 URL，启用浏览器自动化                 |
| `PLAYWRIGHT_MCP_TOKEN` |      | Playwright MCP 认证令牌                                   |
| `SUB_AGENT_NUM`        |      | 并行子智能体数量，默认 `3`                              |

### 最小配置

只需 `DASHSCOPE_API_KEY` 即可启动服务。搜索、爬取等工具将根据配置的 Key 自动启用：

```env
DASHSCOPE_API_KEY=sk-xxxxx
SERPER_API_KEYS=key_a,key_b,key_c
SERPER_API_KEY=xxxxx
JINA_API_KEY=jina_xxxxx
```

<h2 id="tools">Tool System</h2>

工具系统采用 **条件加载** 机制——每个工具模块仅在对应的环境变量配置后才会被导入和注册。工具函数通过 Python 类型注解和 docstring 自动生成 JSON Schema，无需手动维护工具定义。

### 工具分配

| 工具                          | 主智能体 | 子智能体 | 说明                                   |
| ----------------------------- | -------- | -------- | -------------------------------------- |
| `execute_subtasks`          | ✅       | —       | 向子智能体派发研究子任务（运行时注入） |
| `code_sandbox`              | ✅       | —       | E2B Python 沙箱执行                    |
| `search_engine`             | —       | ✅       | Google 搜索 (Serper API)               |
| `search_wikipedia`          | —       | ✅       | Wikipedia 当前内容查询                 |
| `search_wikipedia_revision` | —       | ✅       | Wikipedia 历史版本内容                 |
| `list_wikipedia_revisions`  | —       | ✅       | Wikipedia 修订历史列表                 |
| `scrape_website`            | —       | ✅       | 网页爬取 (Jina Reader)                 |
| `analyze_webpage`           | —       | ✅       | LLM 驱动的网页智能分析                 |
| `browser_*`                 | —       | ✅       | 浏览器自动化 (Playwright MCP)          |

### 工具详情

#### 🔍 搜索引擎 (`search_engine`)

通过 Serper API 进行 Google 搜索，返回标题、URL、摘要、答案框和知识图谱等结构化结果。

- **依赖**: `SERPER_API_KEYS`（推荐，防止单个KEY使用额度不满足问答需求）或 `SERPER_API_KEY`
- **参数**: `query` (查询词), `num_results` (结果数, 默认 20), `language` (语言, 默认 "en")

#### 📚 Wikipedia (`wiki_search`)

完整的 Wikipedia 访问能力，支持当前内容查询和历史版本回溯。

- **依赖**: 无（公开 API），`JINA_API_KEY` 可启用降级通道
- **容错**: Wikipedia API 超时(2s) → 自动降级至 Jina Reader 抓取对应页面
- **函数**:
  - `search_wikipedia(entity)` — 获取当前页面内容
  - `search_wikipedia_revision(entity, date)` — 获取指定日期的历史内容
  - `list_wikipedia_revisions(entity)` — 列出修订历史

#### 🌐 网页爬取 (`scrape_website`)

将任意网页内容转换为干净的 Markdown 格式文本。

- **依赖**: `JINA_API_KEY`
- **容错**: Jina Reader → requests + MarkItDown
- **特性**: 自动检测 CAPTCHA/反爬页面并触发降级

#### 🧪 网页分析 (`analyze_webpage`)

先爬取网页内容，再利用 LLM 根据研究问题进行针对性分析，生成结构化研究报告。

- **依赖**: `DASHSCOPE_API_KEY` + `JINA_API_KEY`
- **输出**: 相关性评级、关键发现、提取详情、缺失信息、有用线索

#### 💻 代码沙箱 (`code_sandbox`)

基于 E2B 的安全 Python 执行环境，预装 pandas、numpy 等数据分析库。

- **依赖**: `E2B_API_KEY`
- **函数**: `create_sandbox`, `run_python_code`, `download_file_to_sandbox`, `run_shell_command`, `close_sandbox`

#### 🖥️ 浏览器自动化 (`browser_session`)

通过 Playwright MCP 协议远程控制浏览器，支持导航、点击、输入、截图等完整交互。

- **依赖**: `PLAYWRIGHT_MCP_URL`
- **特性**: 单例会话, SSE 传输, 线程安全

## 多跳推理策略

Agent 的核心推理策略遵循 **链式解题 (Chain Resolution)** 模式：

1. **推理链分解** — 将复杂问题分解为完整的推理节点链
2. **并行加速** — 识别无依赖关系的独立节点，同时派发至多个子智能体并行研究
3. **前向推进** — 已确认的事实不再重复验证，只向前推进未解决的节点
4. **矛盾验证** — 仅在搜索结果出现矛盾或多个候选答案时才进行交叉验证
5. **即时作答** — 推理链的最后一个节点解决后立即生成答案

这种策略有效避免了传统多跳问答中的冗余查询和循环验证问题，显著提升了复杂问题的解答效率和准确率。

## 容错与鲁棒性

系统在多个层面实现了容错机制，确保在网络受限或内容审核场景下仍能稳定运行：

| 层面           | 机制                                       | 说明                                                                      |
| -------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| Wikipedia 访问 | API 超时(2s) → Jina Reader 降级           | 应对国内网络无法直连 Wikipedia 的场景                                     |
| 网页爬取       | Jina Reader → requests + MarkItDown       | CAPTCHA/反爬自动检测并切换策略                                            |
| 内容过滤       | `data_inspection_failed` 自动清洗重试    | Wikipedia 等外部内容触发 DashScope 内容审核时，自动替换敏感内容并继续推理 |
| 工具执行       | 30s 超时保护                               | 单个工具调用超时后返回错误信息，不阻塞整体流程                            |
| 子智能体异常   | `asyncio.gather` + `return_exceptions` | 单个子智能体失败不影响其他并行子智能体的正常返回                          |
| 主智能体兜底   | 最大轮次耗尽 → 强制总结生成答案           | 即使未完全解答也会基于已有信息给出最佳猜测                                |

## 比赛成绩（初赛27/188，复赛15/49）
### 初赛

![初赛成绩截图](docs/preliminary-result.png)
### 复赛

![复赛成绩截图](docs/final-result.png)

**初次参赛，尚有许多不足，欢迎大家提 Issue 一起交流、讨论方案和改进思路。**


## License

[MIT](LICENSE)
