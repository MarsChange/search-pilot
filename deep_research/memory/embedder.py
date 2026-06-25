from __future__ import annotations

import hashlib
import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class Embedder:
    """Sentence-transformers embedder with deterministic fallback for tests."""

    _model_cache: Any = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.dim = 384
        self._available = True

    def _model(self) -> Any:
        if not self._available:
            return None
        if Embedder._model_cache is not None:
            return Embedder._model_cache
        try:
            from sentence_transformers import SentenceTransformer

            Embedder._model_cache = SentenceTransformer(self.model_name)
            return Embedder._model_cache
        except Exception as exc:  # pragma: no cover - depends on local model/network
            logger.warning("Embedding model unavailable, using deterministic fallback: %s", exc)
            self._available = False
            return None

    def encode(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * self.dim
        model = self._model()
        if model is not None:
            try:
                return model.encode(text, normalize_embeddings=True).tolist()
            except Exception as exc:  # pragma: no cover
                logger.warning("Embedding encode failed, using fallback: %s", exc)
        return self._fallback(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._model()
        if model is not None:
            try:
                return [vec.tolist() for vec in model.encode(texts, normalize_embeddings=True)]
            except Exception as exc:  # pragma: no cover
                logger.warning("Embedding batch encode failed, using fallback: %s", exc)
        return [self.encode(text) for text in texts]

    def _fallback(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        vec = np.array([rng.gauss(0.0, 1.0) for _ in range(self.dim)], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        return vec.tolist()
