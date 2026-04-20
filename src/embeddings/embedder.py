"""
Embedding Generation Module (FINAL)
- Device-aware (CPU/GPU)
- Filters bad chunks
- Returns embeddings + valid_chunks (critical fix)
"""

import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class Embedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Embedder._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading embedding model on {device}: {EMBEDDING_MODEL}")

            Embedder._model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device
            )

            logger.info(f"Embedding dimension: {EMBEDDING_DIMENSION}")

    @property
    def model(self):
        return Embedder._model

    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(emb, dtype=np.float32)

    def encode_documents(self, texts, batch_size=64, show_progress=True):
        if not texts:
            return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
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
        texts = [c["text"] for c in valid_chunks]

        embeddings = self.encode_documents(texts, batch_size=batch_size)

        assert len(embeddings) == len(valid_chunks)

        return embeddings, valid_chunks