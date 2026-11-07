"""
Cross-encoder reranker (Phase 3).

The embedder scores the question and each chunk separately; a cross-encoder reads
them together, which is slower but more accurate. It only scores the ~30 fused
candidates, never the whole index.

If the model cannot be loaded (e.g. offline before the first download) the
retriever simply skips reranking instead of failing.
"""

import logging
from typing import Optional

import numpy as np

from config.settings import RERANKER_MODEL
from src.embeddings.embedder import _pick_device

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    _instance = None
    _model = None
    _load_failed = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Reranker._model is not None or Reranker._load_failed:
            return
        try:
            from sentence_transformers import CrossEncoder

            device = _pick_device()
            logger.info(f"Loading reranker on {device}: {RERANKER_MODEL}")
            Reranker._model = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
        except Exception as exc:
            Reranker._load_failed = True
            logger.warning(f"Reranker unavailable, continuing without it: {exc}")

    @property
    def available(self) -> bool:
        return Reranker._model is not None

    def _outputs_probabilities(self) -> bool:
        # Depending on the sentence-transformers version and model config,
        # predict() returns raw logits or sigmoid probabilities.
        model = Reranker._model
        activation = getattr(model, "activation_fn", None) or getattr(model, "default_activation_function", None)
        return activation is not None and type(activation).__name__ == "Sigmoid"

    def score(self, query: str, texts: list[str]) -> Optional[list[float]]:
        """Relevance probabilities in 0..1 (higher = more relevant), or None if unavailable."""
        if not self.available or not texts:
            return None
        try:
            raw = Reranker._model.predict(
                [(query, text) for text in texts],
                batch_size=32,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning(f"Reranking failed, using fused order: {exc}")
            return None

        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        probabilities = raw if self._outputs_probabilities() else _sigmoid(raw)
        return probabilities.tolist()
