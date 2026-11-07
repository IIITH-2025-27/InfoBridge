"""
Embedding Generation Module (FINAL)
- Device-aware (CUDA / Apple MPS / CPU)
- Model-aware query/passage prefixes (e5 models need them)
- Filters bad chunks
- Returns embeddings + valid_chunks (critical fix)
"""

import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    if EMBEDDING_DEVICE != "auto":
        return EMBEDDING_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU: noticeably faster index builds on an M1
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _prefixes(model_name: str) -> tuple[str, str]:
    """(query_prefix, passage_prefix) the model was trained with."""
    if "e5" in model_name.lower():
        return "query: ", "passage: "
    return "", ""


class Embedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Embedder._model is None:
            device = _pick_device()
            logger.info(f"Loading embedding model on {device}: {EMBEDDING_MODEL}")

            Embedder._model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device
            )

            actual_dim = Embedder._model.get_sentence_embedding_dimension()
            if actual_dim != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"{EMBEDDING_MODEL} produces {actual_dim}-d vectors but "
                    f"EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}. Set EMBEDDING_DIMENSION in .env."
                )

            logger.info(f"Embedding dimension: {EMBEDDING_DIMENSION}")

        self.query_prefix, self.passage_prefix = _prefixes(EMBEDDING_MODEL)

    @property
    def model(self):
        return Embedder._model

    @property
    def model_name(self) -> str:
        return EMBEDDING_MODEL

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            self.query_prefix + query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(emb, dtype=np.float32)

    def encode_documents(self, texts, batch_size=64, show_progress=True):
        if not texts:
            return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)

        embeddings = self.model.encode(
            [self.passage_prefix + text for text in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )

        embeddings = np.array(embeddings, dtype=np.float32)

        assert embeddings.shape[1] == EMBEDDING_DIMENSION

        logger.info(f"Encoded {len(texts)} texts → {embeddings.shape}")
        return embeddings

    def encode_chunks(self, chunks, batch_size=64):
        # Filter bad chunks
        valid_chunks = [c for c in chunks if c["text"].strip()]
        # embed_text = title + chunk (see chunker); fall back to the raw text
        texts = [c.get("embed_text", c["text"]) for c in valid_chunks]

        embeddings = self.encode_documents(texts, batch_size=batch_size)

        assert len(embeddings) == len(valid_chunks)

        return embeddings, valid_chunks
