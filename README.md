<p align="center">
  <h1 align="center">🔍 SearchPilot</h1>
  <p align="center"><b>Multi-Hop Reasoning Deep Research Agent for Complex Question Answering</b></p>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#features">Features</a> •
  <a href="#quickstart">Quick Start</a> •
  <a href="#api">API Reference</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#tools">Tool System</a>
</p>

---

## Overview

SearchPilot 是一个基于 LLM 的多智能体深度研究系统，专为解答复杂的多跳推理问题而设计。它将复杂问题自动分解为多个子任务，分派给并行工作的研究型子智能体，最终汇聚各方结论生成高质量答案。

与传统的单轮检索-生成（RAG）方案不同，SearchPilot 采用 **两层 Agent 架构**，实现了真正意义上的 **多跳推理链式解题**——主智能体负责任务规划与推理链推进，子智能体负责并行执行搜索、爬取、分析等具体研究操作，多轮协作直至问题完全解决。

### 核心理念

```
复杂问题 → 推理链分解 → 并行子任务研究 → 证据汇聚 → 链式推进 → 最终答案
```

## Features

- **🧠 多跳推理 (Multi-Hop Reasoning)** — 将复杂问题分解为推理链，逐层解决依赖关系，支持高达 30 轮主智能体交互
- **👥 多智能体并行研究 (Multi-Agent Parallelism)** — 主智能体协调任务分解，最多同时派出多个子智能体并行执行研究任务
- **🔧 丰富的工具生态 (Rich Tool Ecosystem)** — 搜索引擎、Wikipedia（含历史版本）、网页爬取、网页智能分析、沙箱代码执行、浏览器自动化
- **🌊 全链路流式输出 (End-to-End Streaming)** — 支持 SSE 流式响应，工具调用过程可实时追踪
- **🔌 AG-UI 协议支持** — 兼容 [AG-UI Protocol](https://docs.ag-ui.com) 标准，可无缝对接 UI 前端
- **🇨🇳 中文语境优化** — 自动检测 CJK 内容，切换中文搜索策略和中文提示词
- **🔄 多层容错降级 (Fallback Chains)** — Wikipedia API → Jina Reader、Jina Reader → requests+MarkItDown，确保信息获取的鲁棒性

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
│           POST /  ·  POST /stream  ·  POST /ag-ui               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 🧠 Main Agent (主智能体)                          │
│                                                                  │
│  · 接收用户问题，构建完整推理链                                       │
│  · 识别可并行的独立节点                                              │
│  · 调用 execute_subtasks 分派子任务                                 │
│  · 根据子任务结果推进推理链                                           │
│  · 汇总所有研究结果生成最终答案                                       │
│                                                                  │
│  可用工具: execute_subtasks, code_sandbox                         │
│  最大交互轮次: 30                                                  │
└──────────┬──────────┬──────────┬────────────────────────────────┘
           │          │          │
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
│ 最大轮次:25 │ │ 最大轮次:25 │ │ 最大轮次:25 │
└────────────┘ └────────────┘ └────────────┘
```

### 请求处理流程

```
HTTP Request
    → FastAPI Endpoint
        → agent_loop() 异步生成器
            → LLM 流式调用 (Qwen via DashScope)
                → 检测到 tool_call: execute_subtasks
                    → 解析子任务 JSON 数组
                    → asyncio.gather 并发启动 N 个 run_sub_agent()
                        → 各子智能体独立执行: LLM ↔ Tool 循环
                        → 生成结构化研究报告
                    → 汇总所有子智能体报告
                → 主智能体继续推理链推进
                → ... (可多次调用 execute_subtasks)
            → 最终 LLM 合成答案
        → SSE 流式响应返回
```

### 模块结构

```
SearchPilot/
├── agent.py              # FastAPI 服务入口，定义三个 API 端点
├── agent_loop.py          # 核心编排引擎：流式 LLM 调用、工具执行、子智能体管理
├── tools_calling.py       # 提示词构建器：主/子智能体系统提示、工具描述格式化
├── agui.py                # AG-UI Protocol 适配层，Chunk 流 → AG-UI SSE 事件
├── tools/
│   ├── __init__.py        # 工具注册中心，按环境变量条件加载
│   ├── search_engine.py   # Google/Bing 搜索 (Serper API)
│   ├── wiki_search.py     # Wikipedia 搜索（含历史版本查询）
│   ├── scrape_website.py  # 网页爬取与内容提取
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
# 克隆项目
git clone https://github.com/your-repo/SearchPilot.git
cd SearchPilot

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

### 快速测试

```bash
# 基础问答（完整 JSON 响应）
curl -X POST "http://localhost:8000/" \
  -H "Content-Type: application/json" \
  -d '{"question": "谁是第一个登上月球的人？"}'

# 流式响应
curl -N -X POST "http://localhost:8000/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "量子计算的基本原理是什么？"}'
```

<h2 id="api">API Reference</h2>

### `POST /` — 完整响应

接收问题，返回完整的 JSON 答案。

**请求体：**
```json
{
  "question": "你的问题",
  "chat_history": []   // 可选，历史对话消息
}
```

**响应：**
```json
{
  "answer": "最终答案"
}
```

### `POST /stream` — SSE 流式响应

以 Server-Sent Events 格式逐块返回答案文本。

**响应示例：**
```text
data: {"answer": "第一个登上月球的人是"}
data: {"answer": "尼尔·阿姆斯特朗"}
data: {"answer": "，1969年7月20日。"}
```

### `POST /ag-ui` — AG-UI Protocol

兼容 [AG-UI Protocol](https://docs.ag-ui.com) 标准的 SSE 事件流。支持文本消息、工具调用、工具结果等完整事件类型，可直接对接支持 AG-UI 的前端 UI。

**事件类型：**

| 事件 | 说明 |
|------|------|
| `RunStartedEvent` | 运行开始 |
| `TextMessageStartEvent` / `ContentEvent` / `EndEvent` | 文本消息流 |
| `ToolCallStartEvent` / `ArgsEvent` / `EndEvent` | 工具调用事件 |
| `ToolCallResultEvent` | 工具执行结果 |
| `RunFinishedEvent` | 运行结束 |

<h2 id="configuration">Configuration</h2>

将 `.env.template` 复制为 `.env` 并填入相应的 API Keys。工具模块按需加载——只有配置了对应 Key 的工具才会被激活。

| 环境变量 | 必需 | 说明 |
|---------|------|------|
| `DASHSCOPE_API_KEY` | ✅ | 阿里云 DashScope API Key，用于调用 Qwen 大模型 |
| `QWEN_MODEL` | | 模型名称，默认 `qwen-max` |
| `SERPER_API_KEY` | | Google 搜索服务 (Serper)，启用搜索引擎工具 |
| `JINA_API_KEY` | | Jina AI，启用网页爬取、网页分析、Wikipedia 降级 |
| `E2B_API_KEY` | | E2B 沙箱，启用安全的 Python 代码执行 |
| `PLAYWRIGHT_MCP_URL` | | Playwright MCP 服务 URL，启用浏览器自动化 |
| `PLAYWRIGHT_MCP_TOKEN` | | Playwright MCP 认证令牌 |
| `SUB_AGENT_NUM` | | 并行子智能体数量，默认 `3` |

### 最小配置

只需 `DASHSCOPE_API_KEY` 即可启动服务。搜索、爬取等工具将根据配置的 Key 自动启用：

```env
DASHSCOPE_API_KEY=sk-xxxxx
SERPER_API_KEY=xxxxx
JINA_API_KEY=jina_xxxxx
```

<h2 id="tools">Tool System</h2>

SearchPilot 的工具系统采用 **条件加载** 机制——每个工具模块仅在对应的环境变量配置后才会被导入和注册。工具函数通过 Python 类型注解和 docstring 自动生成 JSON Schema，无需手动维护工具定义。

### 工具分配

| 工具 | 主智能体 | 子智能体 | 说明 |
|------|---------|---------|------|
| `execute_subtasks` | ✅ | — | 派发子任务（运行时注入） |
| `code_sandbox` | ✅ | — | Python 沙箱执行 |
| `search_engine` | — | ✅ | Google/Bing 搜索 |
| `search_wikipedia` | — | ✅ | Wikipedia 当前内容 |
| `search_wikipedia_revision` | — | ✅ | Wikipedia 历史版本 |
| `list_wikipedia_revisions` | — | ✅ | Wikipedia 修订历史 |
| `scrape_website` | — | ✅ | 网页爬取 |
| `analyze_webpage` | — | ✅ | LLM 网页分析 |
| `browser_*` | — | ✅ | 浏览器自动化 (15 个函数) |

### 工具详情

#### 🔍 搜索引擎 (`search_engine`)

通过 Serper API 进行 Google 搜索，返回标题、URL、摘要、答案框和知识图谱等结构化结果。

- **依赖**: `SERPER_API_KEY`
- **参数**: `query` (查询词), `num_results` (结果数, 默认 20), `language` (语言, 默认 "en")

#### 📚 Wikipedia (`wiki_search`)

完整的 Wikipedia 访问能力，支持当前内容查询和历史版本回溯。

- **依赖**: 无（公开 API），`JINA_API_KEY` 可启用降级通道
- **容错**: Wikipedia API 超时(5s) → 自动降级至 Jina Reader
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
- **特性**: 单例会话, SSE 传输, 线程安全, 15 个操作函数

## Streaming & Keepalive

SearchPilot 的流式响应基于 `Chunk` 数据类实现，支持三种类型：

| 类型 | 说明 |
|------|------|
| `text` | 文本内容块 |
| `tool_call` | 工具调用通知（含工具名和参数） |
| `tool_call_result` | 工具执行结果 |

在子智能体执行耗时操作期间（如复杂搜索和网页分析），系统每 **15 秒** 发送空 SSE 事件作为心跳保活，防止连接超时断开。

## 多跳推理策略

SearchPilot 的核心推理策略遵循 **链式解题 (Chain Resolution)** 模式：

1. **推理链分解** — 将复杂问题分解为完整的推理节点链
2. **并行加速** — 识别无依赖关系的独立节点，同时派发研究
3. **前向推进** — 已确认的事实不再重复验证，只向前推进
4. **矛盾验证** — 仅在搜索结果出现矛盾或多个候选答案时才进行验证
5. **即时作答** — 推理链的最后一个节点解决后立即生成答案

这种策略有效避免了传统多跳问答中的冗余查询和循环验证问题，显著提升了复杂问题的解答效率和准确率。

## License

[MIT](LICENSE)
