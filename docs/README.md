# Deep Research 框架细节文档

本目录维护当前 Deep Research 框架的模块级设计说明。文档按功能拆分，便于后续单独更新。

## 文档索引

| 功能 | 文档 | 主要实现 |
| --- | --- | --- |
| FSM 状态编排机制 | [fsm-state-orchestration.md](fsm-state-orchestration.md) | `deep_research/runner.py`, `deep_research/state_graph.py`, `deep_research/schemas.py` |
| 三级语义压缩 | [semantic-compression.md](semantic-compression.md) | `deep_research/compressor/` |
| 共享记忆数据库 SQLite + numpy | [shared-memory-store.md](shared-memory-store.md) | `deep_research/memory/` |
| Red-Blue 双 Agent 攻防评审 | [red-blue-adversarial-review.md](red-blue-adversarial-review.md) | `deep_research/adversarial/` |

## 总体执行链路

当前框架以 `DeepResearchRunner` 为统一入口。HTTP 请求由 `agent.py` 转换为 `QueryRequest` 后构造 runner，并通过 `runner.run(question)` 启动研究流程。

```text
agent.py
  -> DeepResearchRunner.run()
      -> DeepResearchPlanner.create_plan()
      -> ResearchStateGraph.add_state()
      -> DeepResearchWorker.run() 并发执行 ready states
      -> SharedMemoryStore.put() 收集 evidence
      -> coverage check / replan
      -> DeepResearchSynthesizer.synthesize()
      -> AdversarialLoop.run()
      -> DeepResearchResult
```

框架里有两层状态概念：

- 外层 `FSMState` 描述 runner 当前处于规划、调度、收集、检查、合成等流程阶段。
- 内层 `ResearchStateGraph` 描述被规划出的研究节点、依赖关系和每个节点的执行状态。

这两层组合后，系统既能对外输出可视化事件，也能在内部按依赖并发执行、失败补救和覆盖检查。

## 入口与关键参数

`agent.py` 中 `QueryRequest` 暴露以下与框架能力直接相关的参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `deep_research` | `true` | 保持 API 兼容；即使为 `false` 也会走 Deep Research，只是关闭重规划和攻防评审。 |
| `session_id` | 自动 UUID | 共享记忆的会话隔离键。 |
| `max_concurrent` | `3` | 每轮最多并发执行的 ready states 数量。 |
| `max_replans` | `2` | 覆盖不足、失败或冲突时最多补规划轮数。 |
| `max_adversarial_rounds` | `2` | 最终报告 Red-Blue 攻防评审最大轮数；设为 `0` 可关闭。 |
| `memory_db_path` | `DEEP_RESEARCH_MEMORY_DB` 或 `data/deep_research_memory.db` | SQLite 记忆库路径。 |

## 流式事件

runner 通过 `event_sink` 输出内部事件，`agent.py` 将其映射为 SSE event name。常用事件包括：

| 内部 type | SSE event | 含义 |
| --- | --- | --- |
| `state` | `State` | FSM 阶段切换。 |
| `plan` | `Plan` | 初始计划、状态列表和覆盖清单。 |
| `dispatch` | `Dispatch` | 本轮可执行的 ready states。 |
| `state_start` | `TaskStart` | 单个 research state 开始执行。 |
| `state_result` | `TaskResult` | 单个 research state 完成、失败或部分完成。 |
| `coverage` | `Coverage` | 覆盖检查结果。 |
| `replan_start` / `replan_result` | `Replan` | 补规划开始和结果。 |
| `final` | `Final` | 最终答案、置信度和 metadata。 |

这些事件同时服务 `/stream`、`/ag-ui` 和可视化页面。
