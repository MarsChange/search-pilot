# FSM 状态编排机制

当前框架的编排由 `DeepResearchRunner` 驱动，核心文件是 `deep_research/runner.py`、`deep_research/state_graph.py` 和 `deep_research/schemas.py`。

## 两层状态模型

### 外层 FSMState

`FSMState` 是 runner 级别的流程阶段，定义在 `schemas.py`：

| 状态 | 触发位置 | 说明 |
| --- | --- | --- |
| `PLANNING` | `run()` 开始后 | 读取长期记忆，调用 planner 生成 `ResearchPlan`。 |
| `DISPATCHING` | 每轮循环开始 | 从状态图中取出依赖已满足的 pending states。 |
| `COLLECTING` | worker batch 完成后 | 将 worker evidence 写入共享记忆库。 |
| `COVERAGE_CHECK` | collecting 后 | 检查覆盖清单、失败节点、开放问题、冲突和来源质量。 |
| `REPLANNING` | 覆盖不足或失败率过高时 | 调用 replanner 追加补救 states。 |
| `SYNTHESIZING` | 调度循环结束后 | 基于 evidence、coverage、state history 合成最终报告。 |
| `ADVERSARIAL_REVIEW` | synthesizing 后 | 可选 Red-Blue 攻防评审。 |
| `FINALIZING` | 评审后 | 提取短答案，组装 metadata。 |
| `DONE` | 正常完成 | 返回 `DeepResearchResult`。 |
| `FAILED` | 捕获异常 | 返回失败报告和已收集 evidence。 |

每次 `_transition()` 都会追加 `StateHistoryEvent`，并通过 `event_sink` 发出 `{"type": "state", ...}`。

### 内层 ResearchStateGraph

`ResearchStateGraph` 管理 planner 生成的研究节点。每个节点是 `ResearchState`：

| 字段 | 作用 |
| --- | --- |
| `state_id` | 节点唯一 ID。为空时由 graph 自动生成。 |
| `state_type` | `search`、`analyze`、`verify`、`backtrack`、`synthesize`。 |
| `description` | 当前节点要解决的单一研究目标。 |
| `dependencies` | 依赖的前置 state IDs。 |
| `search_queries` | worker 优先使用的检索词。 |
| `expected_output` | 期望产物，如 facts、candidate answer。 |
| `coverage_tags` | 节点成功后可覆盖的检查项。 |
| `priority` | ready states 排序依据，数字越小越优先。 |
| `timeout_seconds` | 单节点超时时间。 |
| `parent_state_id` / `backtrack_reason` | 补规划、验证、回溯时记录来源。 |

节点执行状态为 `pending`、`running`、`success`、`failed`、`skipped`。

## 调度规则

`ResearchStateGraph.ready_states(limit)` 只返回满足以下条件的节点：

1. 当前状态是 `pending`。
2. `dependencies` 中的每个前置节点都已 `success`。
3. 按 `(priority, state_id)` 排序。
4. 最多返回 `max_concurrent` 个。

runner 将 ready states 标记为 `running` 后，用 `asyncio.gather()` 并发执行。单个节点通过 `asyncio.wait_for()` 施加超时，默认使用 runner 的 `state_timeout_seconds`。

```text
pending + dependencies success
  -> ready_states()
  -> mark_running()
  -> DeepResearchWorker.run()
      -> success / failed
```

如果没有 ready states 但仍有 pending states，说明依赖链已经断裂，runner 会将剩余 pending 节点标记为 `skipped`，原因是 `dependencies unavailable`。

## Coverage 检查

`_check_coverage()` 综合以下信号生成 `CoverageReport`：

- `coverage_checklist` 是否被成功节点的 `coverage_tags` 覆盖。
- 是否存在 failed states。
- worker 是否留下 `open_questions`。
- worker 或记忆库是否发现与答案相关的 conflicts。
- 是否至少有一手来源；没有时标记 `missing_authoritative_sources`。
- 如果计划要求时效性，evidence 是否包含 date；没有时标记 `missing_recency`。

只有覆盖完整、无失败、无冲突、无开放问题，并满足来源和时效约束时，`complete` 才为 `true`。

## 重规划与回溯

runner 在以下条件下进入 `REPLANNING`：

- 状态图失败率超过阈值，默认 `failure_rate > 0.5`。
- 或 coverage 未完成。
- 且 `num_replans < max_replans`。

`DeepResearchPlanner.replan()` 会接收成功结果、失败节点、覆盖缺口、冲突和记忆上下文，输出新的 states。LLM 重规划失败时使用 fallback：

- 对 coverage gaps 创建补充 `search` state。
- 对与高置信候选答案相关的 conflicts 创建 `verify` state。
- 对 failed states 创建重试 `search` state。

新增 state 会加入同一个 `ResearchStateGraph`，因此可以继续复用原有状态历史和共享记忆。

## 提前停止

`_has_sufficient_final_answer()` 用于多跳短答案场景的提前停止。满足以下条件时，runner 会跳过剩余 pending states：

- missing coverage 只剩 `verification`、`official`、`recency`、`source`、`sources` 这类可容忍项。
- 已有 worker 返回 `resolved`。
- 存在非空且有效的 `candidate_answer`。
- 候选答案与 state 的 `final_answer`、`answer_format`、`candidate_answer` 或 `answer_form_hint` 匹配。
- 最高 evidence confidence 不低于 `0.75`。
- 没有与候选答案相关的冲突。

触发后会发出 `early_stop` 事件，并将剩余 pending states 标记为 `skipped`，原因是 `final answer already resolved`。

## Metadata

最终返回的 metadata 来自 `_metadata()`：

| 字段 | 含义 |
| --- | --- |
| `mode` | 固定为 `deep_research`。 |
| `session_id` | 本次运行的会话 ID。 |
| `num_states` | 状态图中的节点总数，包括补规划节点。 |
| `num_searches` | worker 工具调用数量。 |
| `num_replans` | 实际重规划次数。 |
| `adversarial_rounds` | Red-Blue 评审记录数。 |
| `elapsed_seconds` | runner 总耗时。 |
| `failed` | 是否进入失败报告路径。 |
| `llm_usage` | LLM 客户端可返回的用量统计。 |
| `compression` | `ContextCompressor.get_stats()` 的压缩统计。 |

## 维护建议

- 新增 FSM 阶段时，需要同时更新 `FSMState`、runner 中的 `_transition()` 调用点、可视化事件消费逻辑和本文档。
- 新增 `ResearchStateType` 时，需要同步更新 planner prompt、worker 工具策略和 coverage 语义。
- 调整提前停止策略时，应优先补充单元测试或评测样例，因为该逻辑会直接影响答案准确率与工具成本。
