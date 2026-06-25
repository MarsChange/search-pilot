from __future__ import annotations

import re
from typing import Any

import numpy as np

from deep_research.compressor.extractive import ExtractiveCompressor
from deep_research.compressor.sliding_window import SlidingWindowCompressor
from deep_research.compressor.summarizer import LLMSummarizer
from deep_research.memory.embedder import Embedder
from deep_research.runtime_logging import emit_runtime_log


class ContextCompressor:
    """L1 relevance filter, L2 extractive compression, L3 optional LLM summary."""

    def __init__(
        self,
        llm: Any | None = None,
        embedder: Any | None = None,
        budget: int = 16000,
        output_reserve: int = 2048,
    ) -> None:
        self.llm = llm
        self.embedder = embedder or Embedder()
        self.budget = budget
        self.output_reserve = output_reserve
        self.available_budget = max(1000, budget - output_reserve)
        self.extractive = ExtractiveCompressor(self.embedder)
        self.sliding = SlidingWindowCompressor(max_tokens=self.available_budget)
        self.summarizer = LLMSummarizer(llm)
        self._history: list[dict[str, Any]] = []

    def calculate_tokens(self, texts: list[str]) -> int:
        return int(sum(len(text) for text in texts) / 3.5)

    async def compress(
        self,
        texts: list[str],
        query: str = "",
        *,
        level: int | None = None,
        system_prompt_tokens: int = 0,
    ) -> list[str]:
        if not texts:
            return []
        available = max(500, self.available_budget - system_prompt_tokens)
        original_tokens = self.calculate_tokens(texts)
        usage = original_tokens / max(available, 1)
        if level is None:
            if usage > 0.95:
                level = 3
            elif usage > 0.80:
                level = 2
            elif usage > 0.60:
                level = 1
            else:
                self._record(texts, texts, 0)
                return texts
        compressed = list(texts)
        if level >= 1:
            compressed = self._l1_filter(compressed, query, available)
        if level >= 2:
            ratio = max(0.15, min(0.4, available / max(self.calculate_tokens(compressed), 1) * 0.35))
            compressed = [self.extractive.compress(text, query, target_ratio=ratio) for text in compressed]
        if level >= 3:
            compressed = [await self.summarizer.summarize_documents(compressed, query, max_chars=available * 3)]
        if self.calculate_tokens(compressed) > available:
            messages = [{"role": "user", "content": text} for text in compressed]
            compressed = [message["content"] for message in self.sliding.compress(messages)]
        self._record(texts, compressed, level)
        return compressed

    def _l1_filter(self, texts: list[str], query: str, available: int) -> list[str]:
        if not query:
            return texts
        q_vec = self._norm(self.embedder.encode(query))
        if q_vec is None:
            return texts
        scored = []
        for text in texts:
            vec = self._norm(self.embedder.encode(text[:1200]))
            sim = float(vec.dot(q_vec)) if vec is not None else 0.0
            scored.append((text, sim))
        scored.sort(key=lambda item: item[1], reverse=True)
        kept = [text for text, sim in scored if sim >= 0.15]
        if not kept and scored:
            kept = [scored[0][0]]
        while self.calculate_tokens(kept) > available * 0.85 and len(kept) > 1:
            kept.pop()
        return kept

    def get_stats(self) -> dict[str, Any]:
        if not self._history:
            return {
                "total_compresses": 0,
                "avg_compression_ratio": 1.0,
                "avg_retention": 1.0,
                "history": [],
            }
        return {
            "total_compresses": len(self._history),
            "avg_compression_ratio": round(sum(item["compression_ratio"] for item in self._history) / len(self._history), 3),
            "avg_retention": round(sum(item["information_retention"] for item in self._history) / len(self._history), 3),
            "history": list(self._history),
        }

    def _record(self, original: list[str], compressed: list[str], level: int) -> None:
        before = self.calculate_tokens(original)
        after = self.calculate_tokens(compressed)
        self._history.append(
            record := {
                "level": level,
                "original_tokens": before,
                "compressed_tokens": after,
                "compression_ratio": round(after / max(before, 1), 3),
                "information_retention": round(self._retention(original, compressed), 3),
            }
        )
        emit_runtime_log("context_compression", **record)

    @staticmethod
    def _retention(original: list[str], compressed: list[str]) -> float:
        source = "\n".join(original)
        target = "\n".join(compressed)
        entities = set(re.findall(r"\d+[\d,]*\.?\d*%?|20\d{2}|[A-Z][A-Za-z0-9-]{2,}", source))
        if not entities:
            return 1.0
        return sum(1 for entity in entities if entity in target) / len(entities)

    @staticmethod
    def _norm(values: list[float]) -> np.ndarray | None:
        vec = np.array(values, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return vec / norm
