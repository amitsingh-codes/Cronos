"""Embedding model wrapper. The model is loaded once, lazily, and reused for every request."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

from app.config import Settings
from app.errors import AppError

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(AppError):
    status_code, code = 503, "embedding_unavailable"
    message = "The embedding model could not be loaded on this server."


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised embeddings."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self._model_name)
                    self._model = SentenceTransformer(self._model_name)
                except Exception as exc:
                    logger.error("Embedding model load failed: %s", type(exc).__name__)
                    raise EmbeddingUnavailableError() from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        self.load()
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)
