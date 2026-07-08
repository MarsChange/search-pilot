# 三级语义压缩

上下文压缩由 `deep_research/compressor/` 维护，入口类是 `ContextCompressor`。它主要服务两个场景：

- worker 将工具结果、长期记忆和 prior results 放入 LLM 前。
- synthesizer 将多节点 evidence 放入最终合成 prompt 前。

## 预算模型

`ContextCompressor` 默认配置：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `budget` | `16000` | 总 token 预算估算值。 |
| `output_reserve` | `2048` | 预留给模型输出的 token。 |
| `available_budget` | `budget - output_reserve`，最低 `1000` | 压缩后输入可用预算。 |
| `system_prompt_tokens` | 调用时传入，默认 `0` | 额外扣除系统提示词预算。 |

token 估算使用 `len(text) / 3.5`，是轻量估算，不依赖 tokenizer。

## 自动分级

当调用 `compress(texts, query)` 且未显式传入 `level` 时，系统按原始上下文占用比例自动选择压缩层级：

| usage | 行为 |
| --- | --- |
| `<= 0.60` | 不压缩，只记录 level 0。 |
| `> 0.60` | 启用 L1 相关性过滤。 |
| `> 0.80` | 启用 L1 + L2 抽取式压缩。 |
| `> 0.95` | 启用 L1 + L2 + L3 LLM 摘要。 |

usage 的计算方式是：

```text
original_tokens / available
```

## L1：相关性过滤

实现位置：`ContextCompressor._l1_filter()`。

流程：

1. 用 `Embedder.encode(query)` 得到 query 向量。
2. 对每段文本的前 1200 字符编码。
3. 使用 numpy 计算 cosine similarity。
4. 保留相似度 `>= 0.15` 的文本。
5. 如果全部低于阈值，至少保留得分最高的一段。
6. 如果仍超过 `available * 0.85`，按得分从低到高逐步丢弃。

L1 适合大量网页片段、搜索结果、历史 evidence 混在一起的场景，目标是先剔除明显不相关材料。

## L2：抽取式压缩

实现位置：`ExtractiveCompressor`。

L2 对每段文本做句子级抽取：

1. 使用中英文标点切分句子。
2. 通过 embedder 对句子批量编码。
3. 构造句子相似度矩阵，使用类似 TextRank 的中心性分数。
4. 如果有 query，则用 query 相似度重新加权。
5. 对高价值句子加权，包括数字、百分比、年份、URL、官方、财报、公告、reported、according 等模式。
6. 按 `target_ratio` 选择分数最高的句子，并按原文顺序输出。

`target_ratio` 在 `ContextCompressor.compress()` 中动态计算：

```text
max(0.15, min(0.4, available / compressed_tokens * 0.35))
```

因此上下文越紧，抽取比例越低，但不会低于 15%。

## L3：LLM 语义摘要

实现位置：`LLMSummarizer`。

L3 会把 L2 后的多段文本合并，再要求 LLM 在不引入新事实的前提下压缩，并明确保留：

- 与查询相关的关键事实。
- 数字。
- 日期。
- 来源。

如果没有注入 LLM，或者 LLM 调用失败，L3 会退化为 `text[:max_chars]` 截断，保证压缩链路不会因为摘要失败阻塞主流程。

## 最终滑窗兜底

三层压缩后，如果 `calculate_tokens(compressed) > available`，系统会走 `SlidingWindowCompressor`：

- system message 永远保留。
- 非 system message 优先保留最近内容。
- 至少保留 `min_recent_turns`，默认 `3`。
- 如果最后一条仍超长，则截断最后一条并追加 `[CONTENT_TRUNCATED]`。

这一层不是语义压缩，而是硬预算保护，避免 prompt 超长。

## 统计与日志

每次压缩都会记录：

| 字段 | 含义 |
| --- | --- |
| `level` | 本次压缩层级。 |
| `original_tokens` | 压缩前估算 token。 |
| `compressed_tokens` | 压缩后估算 token。 |
| `compression_ratio` | `compressed_tokens / original_tokens`。 |
| `information_retention` | 基于数字、年份、百分比和英文实体的轻量保留率。 |

记录会进入 `_history`，并通过 `emit_runtime_log("context_compression", ...)` 写入运行日志。最终 metadata 中的 `compression` 来自 `get_stats()`。

## Embedder 降级策略

压缩链路使用 `deep_research/memory/embedder.py`：

- 优先加载 `sentence-transformers/all-MiniLM-L6-v2`。
- 模型加载或编码失败时，使用基于 md5 seed 的确定性随机向量。
- 向量维度固定为 `384`。

这种设计保证本地没有 embedding 模型或网络不可用时，测试仍可稳定运行；但语义质量会低于真实 embedding。

## 维护建议

- 调整 L1 相似度阈值 `0.15` 时，应同时观察召回率和上下文长度。
- 调整自动分级阈值时，应关注 worker 输出 JSON 的稳定性，因为压缩过重会丢失格式提示和候选实体。
- 新增高价值句子模式时，优先修改 `HIGH_VALUE_RE`，避免在 worker prompt 中重复写规则。
- 如果后续引入真实 tokenizer，应保持 `calculate_tokens()` 的接口不变，减少调用点改动。
