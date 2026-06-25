from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Any

import numpy as np

from deep_research.memory.embedder import Embedder
from deep_research.memory.long_term import ConflictRecord, LongTermMemory, MemoryEntry
from deep_research.schemas import EVIDENCE_TYPE_WEIGHT

DEDUP_THRESHOLD = 0.92
CONFLICT_LOW = 0.65
CONFLICT_HIGH = 0.92

NEGATION_WORDS = {"不", "没", "无", "未", "非", "否", "not", "no", "never", "without"}
ANTONYM_PAIRS = [
    ({"increase", "increased", "增长", "增加", "上升", "提高"}, {"decrease", "decreased", "下降", "减少", "降低"}),
    ({"success", "成功", "支持", "positive", "增长"}, {"failure", "失败", "反对", "negative", "下滑"}),
    ({"high", "高"}, {"low", "低"}),
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def semantically_opposite(claim_a: str, claim_b: str) -> bool:
    a = claim_a.lower()
    b = claim_b.lower()
    a_neg = any(word in a for word in NEGATION_WORDS)
    b_neg = any(word in b for word in NEGATION_WORDS)
    if a_neg != b_neg:
        return True
    for left, right in ANTONYM_PAIRS:
        a_left = any(word in a for word in left)
        a_right = any(word in a for word in right)
        b_left = any(word in b for word in left)
        b_right = any(word in b for word in right)
        if (a_left and b_right) or (a_right and b_left):
            return True
    return False


class SharedMemoryStore:
    def __init__(
        self,
        db_path: str = "data/deep_research_memory.db",
        *,
        session_id: str = "",
        embedder: Any | None = None,
    ) -> None:
        self.session_id = session_id
        self.lt = LongTermMemory(db_path)
        self.embedder = embedder or Embedder()
        self._lock = threading.RLock()
        self._entry_ids: list[str] = []
        self._entries: dict[str, MemoryEntry] = {}
        self._embeddings = np.zeros((0, int(getattr(self.embedder, "dim", 384))), dtype=np.float32)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        entries = self.lt.get_all_entries(session_id=self.session_id or None)
        with self._lock:
            self._entry_ids = [entry.entry_id for entry in entries]
            self._entries = {entry.entry_id: entry for entry in entries}
            if entries:
                mat = np.array([entry.embedding for entry in entries], dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self._embeddings = mat / norms
            else:
                self._embeddings = np.zeros((0, int(getattr(self.embedder, "dim", 384))), dtype=np.float32)

    def put(self, entry: MemoryEntry) -> str:
        entry.session_id = self.session_id or entry.session_id
        if self._is_junk(entry):
            return entry.entry_id
        if not entry.embedding:
            entry.embedding = self.embedder.encode(entry.claim)

        duplicate_id = self._find_duplicate(entry)
        if duplicate_id:
            existing = self.lt.get_entry(duplicate_id)
            if existing and entry.confidence > existing.confidence:
                entry.entry_id = duplicate_id
                entry.timestamp = max(entry.timestamp, existing.timestamp)
                self.lt.insert_entry(entry)
                self._rebuild_index()
            return duplicate_id

        self.lt.insert_entry(entry)
        self._add_to_index(entry)
        self._detect_conflicts(entry)
        return entry.entry_id

    def query_by_similarity(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_sim: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        if self._embeddings.shape[0] == 0:
            return []
        query_vec = self._normalized_vec(self.embedder.encode(query))
        if query_vec is None:
            return []
        with self._lock:
            sims = self._embeddings.dot(query_vec)
            ranked = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in ranked:
                sim = float(sims[int(idx)])
                if sim < min_sim:
                    continue
                entry = self._entries.get(self._entry_ids[int(idx)])
                if entry:
                    results.append((entry, sim))
            return results

    def get_context_for_query(self, query: str, max_tokens: int = 4000) -> str:
        matches = self.query_by_similarity(query, top_k=10, min_sim=0.25)
        if not matches:
            return ""
        max_chars = int(max_tokens * 3.5)
        parts = ["## 相关长期记忆\n"]
        total = len(parts[0])
        now = time.time()
        for entry, sim in matches:
            age_days = max((now - entry.timestamp) / 86400.0, 0.0)
            block = (
                f"- {entry.claim}\n"
                f"  来源: {entry.source} {entry.url} | 置信度: {entry.confidence:.2f} | "
                f"证据类型: {entry.evidence_type} | 相关度: {sim:.2f} | 距今: {age_days:.1f}天\n"
            )
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "".join(parts)

    def get_conflicts(self, status: str | None = None) -> list[ConflictRecord]:
        return self.lt.get_conflicts(status)

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        strategy: str = "source_weight",
        llm_judge: Any | None = None,
    ) -> MemoryEntry | None:
        target = next((conflict for conflict in self.lt.get_conflicts() if conflict.conflict_id == conflict_id), None)
        if target is None:
            return None
        entry_1 = self.lt.get_entry(target.entry_id_1)
        entry_2 = self.lt.get_entry(target.entry_id_2)
        if entry_1 is None or entry_2 is None:
            self.lt.update_conflict_resolution(conflict_id, "dismissed")
            return None
        if strategy == "llm_judge" and llm_judge is not None:
            winner = self._resolve_by_llm(entry_1, entry_2, llm_judge)
        elif strategy == "source_weight":
            winner = self._resolve_by_source_weight(entry_1, entry_2)
        else:
            winner = self._resolve_by_source_weight(entry_1, entry_2)
        self.lt.update_conflict_resolution(conflict_id, "resolved", winner.entry_id)
        return winner

    def __len__(self) -> int:
        return self.lt.count_entries(self.session_id or None)

    def _is_junk(self, entry: MemoryEntry) -> bool:
        claim = (entry.claim or "").strip()
        if len(claim) < 24:
            return True
        if entry.confidence < 0.3:
            return True
        junk_patterns = [
            r"^error\s*:",
            r"api key",
            r"i'?m ready to help",
            r"请问您想",
        ]
        return any(re.search(pattern, claim, re.I) for pattern in junk_patterns)

    def _add_to_index(self, entry: MemoryEntry) -> None:
        vec = self._normalized_vec(entry.embedding)
        if vec is None:
            return
        with self._lock:
            self._entry_ids.append(entry.entry_id)
            self._entries[entry.entry_id] = entry
            if self._embeddings.shape[0] == 0:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])

    def _find_duplicate(self, entry: MemoryEntry) -> str | None:
        if self._embeddings.shape[0] == 0:
            return None
        vec = self._normalized_vec(entry.embedding)
        if vec is None:
            return None
        with self._lock:
            sims = self._embeddings.dot(vec)
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) > DEDUP_THRESHOLD:
            return self._entry_ids[best_idx]
        return None

    def _detect_conflicts(self, new_entry: MemoryEntry) -> None:
        vec = self._normalized_vec(new_entry.embedding)
        if vec is None or self._embeddings.shape[0] <= 1:
            return
        with self._lock:
            sims = self._embeddings.dot(vec)
            existing_ids = list(self._entry_ids[:-1])
        for idx, existing_id in enumerate(existing_ids):
            sim = float(sims[idx])
            if not (CONFLICT_LOW < sim < CONFLICT_HIGH):
                continue
            existing = self._entries.get(existing_id)
            if existing and semantically_opposite(existing.claim, new_entry.claim):
                self.lt.insert_conflict(
                    ConflictRecord(
                        conflict_id=str(uuid.uuid4()),
                        entry_id_1=existing.entry_id,
                        entry_id_2=new_entry.entry_id,
                        claim_1=existing.claim,
                        claim_2=new_entry.claim,
                        similarity=sim,
                    )
                )

    def _resolve_by_source_weight(self, entry_1: MemoryEntry, entry_2: MemoryEntry) -> MemoryEntry:
        score_1 = EVIDENCE_TYPE_WEIGHT.get(entry_1.evidence_type, 0.5) * entry_1.confidence
        score_2 = EVIDENCE_TYPE_WEIGHT.get(entry_2.evidence_type, 0.5) * entry_2.confidence
        return entry_1 if score_1 >= score_2 else entry_2

    def _resolve_by_llm(self, entry_1: MemoryEntry, entry_2: MemoryEntry, llm_judge: Any) -> MemoryEntry:
        try:
            decision = llm_judge(entry_1, entry_2)
            if str(decision).strip().upper().startswith("B"):
                return entry_2
        except Exception:
            pass
        return self._resolve_by_source_weight(entry_1, entry_2)

    @staticmethod
    def _normalized_vec(values: list[float]) -> np.ndarray | None:
        vec = np.array(values, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return vec / norm
