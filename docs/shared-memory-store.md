# 共享记忆数据库 SQLite + numpy

共享记忆由 `deep_research/memory/` 维护，入口类是 `SharedMemoryStore`。它把 worker 产生的 evidence 持久化到 SQLite，并在内存中维护 numpy 向量索引，用于后续规划、执行和补规划检索相关上下文。

## 数据流

```text
WorkerResult.evidence
  -> DeepResearchRunner._collect_to_memory()
  -> MemoryEntry
  -> SharedMemoryStore.put()
      -> SQLite entries
      -> numpy normalized embedding matrix
      -> conflict detection
```

读取路径：

```text
query / state.description
  -> SharedMemoryStore.get_context_for_query()
  -> query_by_similarity()
  -> Markdown 片段
  -> planner / replanner / worker prompt
```

## SQLite 表结构

`LongTermMemory._ensure_tables()` 会自动创建两个表。

### entries

| 字段 | 说明 |
| --- | --- |
| `entry_id` | 主键。runner 默认使用 `state_id:index:hash(claim)`。 |
| `session_id` | 会话隔离键。 |
| `claim` | 证据断言文本。 |
| `source` | 来源名称。 |
| `url` | 来源 URL。 |
| `confidence` | 0 到 1 的置信度。 |
| `agent_id` | 产生该 evidence 的 state id。 |
| `timestamp` | 写入时间。 |
| `evidence_type` | `primary`、`secondary` 或 `inference`。 |
| `topic` | 当前 query 的前 80 个字符。 |
| `embedding_json` | embedding 向量 JSON。 |
| `metadata_json` | 额外信息，如 date、state_id。 |

索引：

- `idx_entries_session`
- `idx_entries_topic`

### conflicts

| 字段 | 说明 |
| --- | --- |
| `conflict_id` | 主键 UUID。 |
| `entry_id_1` / `entry_id_2` | 冲突的两条记忆。 |
| `claim_1` / `claim_2` | 冲突断言文本快照。 |
| `similarity` | 两条 claim 的向量相似度。 |
| `status` | 默认 `open`，可更新为 `resolved` 或 `dismissed`。 |
| `resolution` | 解析出的胜出 entry id。 |
| `created_at` | 创建时间。 |

索引：

- `idx_conflicts_status`

## numpy 向量索引

`SharedMemoryStore._rebuild_index()` 会把当前 session 的 entries 加载到内存：

- `_entry_ids`: entry id 顺序表。
- `_entries`: entry id 到 `MemoryEntry` 的字典。
- `_embeddings`: shape 为 `(N, dim)` 的 float32 numpy 矩阵。

写入新 entry 后，`_add_to_index()` 会把 normalized vector 追加到 `_embeddings`。查询时：

1. 对 query 编码并归一化。
2. 使用矩阵乘法 `self._embeddings.dot(query_vec)` 计算相似度。
3. 按相似度倒序取 `top_k`。
4. 过滤低于 `min_sim` 的结果。

默认上下文检索参数：

```text
top_k = 10
min_sim = 0.25
max_tokens = 4000
```

返回上下文格式是 Markdown 列表，包含 claim、来源、URL、置信度、证据类型、相关度和距今天数。

## 写入过滤与去重

`SharedMemoryStore.put()` 写入前会先过滤低价值条目：

- claim 长度小于 24。
- confidence 小于 0.3。
- 命中错误、API key、闲聊、中文追问等 junk patterns。

去重阈值：

```text
DEDUP_THRESHOLD = 0.92
```

如果新 entry 与已有 entry 相似度超过阈值：

- 若新 entry 置信度更高，则用相同 `entry_id` 替换旧记录。
- 否则直接返回已有 duplicate id。

这样可以减少重复 evidence 对后续上下文的污染。

## 冲突检测

冲突检测发生在新 entry 追加索引后。触发条件：

1. 新旧 claim 的相似度在 `CONFLICT_LOW` 和 `CONFLICT_HIGH` 之间。
2. 当前阈值为：

```text
CONFLICT_LOW = 0.65
CONFLICT_HIGH = 0.92
```

3. `semantically_opposite()` 判断两条 claim 存在语义相反关系。

相反关系判断目前是轻量启发式：

- 中英文否定词，如 不、没、无、未、非、not、no、never、without。
- 若干反义词组，如 increase/decrease、success/failure、high/low 及对应中文词。

发现冲突后写入 `conflicts` 表，状态为 `open`。coverage 检查会读取 open conflicts，并只保留与高置信候选答案相关的冲突。

## 冲突解析

`resolve_conflict()` 支持两种策略：

| strategy | 行为 |
| --- | --- |
| `source_weight` | 默认策略。按 `EVIDENCE_TYPE_WEIGHT * confidence` 选择胜出 entry。 |
| `llm_judge` | 调用外部 judge 函数判断；异常或无效时回退到 `source_weight`。 |

证据类型权重定义在 `schemas.py`：

| evidence_type | weight |
| --- | --- |
| `primary` | `1.0` |
| `secondary` | `0.8` |
| `inference` | `0.55` |

## session 隔离

`SharedMemoryStore` 初始化时接收 `session_id`。`_rebuild_index()` 默认只加载当前 session 的 entries，因此普通查询不会把其他 session 的记忆加载到向量索引里。

默认 DB 路径：

```text
data/deep_research_memory.db
```

可通过以下方式覆盖：

- 请求参数 `memory_db_path`。
- 环境变量 `DEEP_RESEARCH_MEMORY_DB`。

## 并发与线程安全

SQLite 连接使用 `check_same_thread=False`，`LongTermMemory` 和 `SharedMemoryStore` 都使用 `threading.RLock()` 保护关键区。

runner 当前在同一事件循环中并发执行 worker，collecting 阶段集中写入 memory，因此常规路径下写竞争较少；RLock 主要用于保护后续多线程或外部访问扩展。

## 维护建议

- 如果新增 evidence 字段，应优先放入 `metadata_json`，避免频繁迁移主表 schema。
- 如果改动去重或冲突阈值，应同时观察 coverage 中 conflicts 数量，避免过多无关冲突触发重规划。
- 如果后续记忆量明显增大，可考虑把 numpy 全量矩阵替换为 Faiss、SQLite 向量扩展或按 topic 分片，但保持 `query_by_similarity()` 接口稳定。
