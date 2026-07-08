# Red-Blue 双 Agent 攻防评审

Red-Blue 攻防评审由 `deep_research/adversarial/` 维护，在 `DeepResearchRunner` 的 `SYNTHESIZING` 之后、`FINALIZING` 之前执行。

## 目标

该机制用于在最终答案返回前，让一个 Red Agent 对报告进行严格审查，再让 Blue Agent 基于已有 evidence 修订报告。它不重新检索外部资料，主要降低以下风险：

- unsupported claim。
- 引用质量不足。
- 逻辑跳跃。
- 覆盖缺口。
- 时效性遗漏。
- 候选答案与题目要求格式不匹配。

## 调用位置

`DeepResearchRunner.run()` 中的顺序：

```text
SYNTHESIZING
  -> DeepResearchSynthesizer.synthesize()
ADVERSARIAL_REVIEW
  -> AdversarialLoop.run(final_report)  # max_adversarial_rounds > 0
FINALIZING
  -> extract_final_answer(final_report.content)
```

如果请求参数 `max_adversarial_rounds` 为 `0`，runner 会跳过 `AdversarialLoop.run()`。

## Red Agent

实现位置：`red_agent.py`。

Red Agent 输入：

- 原始问题 `report.query`。
- 报告内容前 8000 字符。
- 最多 30 条 sources。
- coverage 状态。

Red Agent 输出 `RedVerdict`：

| 字段 | 说明 |
| --- | --- |
| `overall_score` | 0 到 10 的总分。 |
| `dimension_scores` | 各评审维度分数。 |
| `issues` | 结构化问题列表。 |
| `passed` | 是否直接通过。 |
| `raw_feedback` | LLM 原始反馈，便于日志和调试。 |

评审维度定义在 `verdict.py`：

- `factual_accuracy`
- `hallucination_risk`
- `citation_quality`
- `logical_consistency`
- `coverage`
- `recency`
- `business_usefulness`

每个 `RedIssue` 包含：

| 字段 | 说明 |
| --- | --- |
| `severity` | `critical`、`major` 或 `minor`。 |
| `dimension` | 命中的评审维度。 |
| `location` | 问题所在位置。 |
| `problem` | 问题描述。 |
| `required_fix` | 建议修复动作，如 `add_evidence`、`remove_claim`、`clarify_uncertainty`。 |

## Blue Agent

实现位置：`blue_agent.py`。

Blue Agent 输入：

- 当前报告内容前 8000 字符。
- Red Agent 的 `RedVerdict`。
- 最多 30 条现有 sources。

Blue Agent 只能基于已有 evidence 修复：

- 不能新增无来源事实。
- 无法证实的内容必须删除或标注不确定。
- 修复后返回 `fixed_report`、`changes` 和 `remaining_risks`。

如果 Blue Agent LLM 调用失败，系统保留原报告，并把异常写入 `remaining_risks`。

## 循环终止条件

`AdversarialLoop` 默认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_rounds` | runner 传入，默认请求为 `2` | 最大攻防轮次。 |
| `score_threshold` | `8.0` | Red 总分达到阈值即停止。 |
| `delta_threshold` | `0.3` | 与上一轮分数差小于阈值时视为收敛。 |

每轮流程：

```text
RedAgent.attack(current_report)
  -> RedVerdict
  -> 判断是否停止
  -> BlueAgent.defend(current_report, verdict)
  -> 更新 current_report
```

停止原因写入 history：

| stop_reason | 含义 |
| --- | --- |
| `oscillation_detected` | 新一轮问题与历史问题重复，避免来回震荡。 |
| `score_threshold_met` | Red 总分达标或 verdict 标记 passed。 |
| `delta_converged` | 分数改善小于 `delta_threshold`。 |
| `max_rounds_reached` | 达到最大轮次。 |

## critique_history

每轮都会写入一条记录：

| 字段 | 说明 |
| --- | --- |
| `round` | 第几轮。 |
| `overall_score` | Red 总分。 |
| `dimension_scores` | 维度分。 |
| `issues` | Red issues。 |
| `oscillation_detected` | 是否检测到重复问题。 |
| `stop_reason` | 本轮停止原因。 |
| `changes` | Blue 修订动作。 |
| `remaining_risks` | Blue 无法完全修复的风险。 |
| `raw_feedback` | Red 原始反馈。 |

`AdversarialLoop.run()` 会将 history 写回 `ResearchReport.critique_history`。如果 history 非空，报告置信度会更新为：

```text
max(current_confidence, min(1.0, last_overall_score / 10.0))
```

## 当前边界

- 攻防评审不触发新的搜索、网页解析或 Wikipedia 查询。
- Blue Agent 只修改报告内容，不直接修改 `sources`、`coverage` 或状态图。
- Red Agent 评审的报告内容截断到前 8000 字符，sources 截断到前 30 条。
- `AdversarialLoop.__init__()` 会将 `max_rounds` 至少设为 1；关闭评审应在 runner 层通过 `max_adversarial_rounds = 0` 实现。

## 维护建议

- 如果新增评审维度，需要同步更新 `DIMENSION_KEYS`、Red prompt、可视化展示和本文档。
- 如果希望 Red 发现问题后触发真实补检索，应把输出接入 runner 的 replan 状态图，而不是让 Blue 直接编写新事实。
- 如果发现短答案任务被 Blue 改成 Markdown 长报告，应检查 `BLUE_AGENT_SYSTEM_PROMPT` 与 synthesizer 输出格式是否一致。
