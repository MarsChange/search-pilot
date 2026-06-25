from __future__ import annotations

import re
from typing import Any

import numpy as np

from deep_research.memory.embedder import Embedder

SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？\n]*")
HIGH_VALUE_RE = re.compile(r"(\d+[\d,]*\.?\d*%?|20\d{2}|https?://|官方|财报|公告|数据显示|reported|according)", re.I)


def tokenize_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.findall(text or "") if len(part.strip()) > 8]


class ExtractiveCompressor:
    def __init__(self, embedder: Any | None = None) -> None:
        self.embedder = embedder or Embedder()

    def compress(self, text: str, query: str = "", target_ratio: float = 0.3) -> str:
        sentences = tokenize_sentences(text)
        if len(sentences) <= 3:
            return text
        chosen = self.textrank_sentences(sentences, query, top_ratio=target_ratio)
        return " ".join(chosen)

    def textrank_sentences(
        self,
        sentences: list[str],
        query: str,
        top_ratio: float = 0.3,
    ) -> list[str]:
        embeddings = np.array(self.embedder.encode_batch(sentences), dtype=np.float32)
        if embeddings.size == 0:
            return []
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        normalized = embeddings / norms
        sim_matrix = normalized.dot(normalized.T)
        scores = sim_matrix.sum(axis=1)
        if query:
            q = np.array(self.embedder.encode(query), dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm > 1e-9:
                scores = scores * normalized.dot(q / q_norm)
        bonuses = np.array([1.25 if HIGH_VALUE_RE.search(sentence) else 1.0 for sentence in sentences])
        scores = scores * bonuses
        k = max(1, int(len(sentences) * top_ratio))
        selected = set(np.argsort(scores)[::-1][:k].tolist())
        return [sentence for idx, sentence in enumerate(sentences) if idx in selected]

    def get_stats(self, original: str, compressed: str) -> dict[str, Any]:
        return {
            "compression_ratio": round(len(compressed) / max(len(original), 1), 3),
            "original_chars": len(original),
            "compressed_chars": len(compressed),
        }
