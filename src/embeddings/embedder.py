"""
Embedding Generation Module
Generates vector embeddings using Sentence Transformers (MiniLM).
"""

import logging
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class Embedder:
    """Generates text embeddings using a pre-trained sentence transformer model."""

    _instance = None
    _model = None

    def __new__(cls):
        """Singleton pattern to avoid reloading the model."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Load the embedding model (only once due to singleton)."""
        if Embedder._model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            Embedder._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(
                f"Model loaded. Embedding dimension: {EMBEDDING_DIMENSION}"
            )

    @property
    def model(self) -> SentenceTransformer:
        return Embedder._model

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a vector.

        Args:
            query: User query text.

        Returns:
            Normalized embedding vector of shape (EMBEDDING_DIMENSION,).
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of document texts into vectors.

        Args:
            texts: List of text chunks to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            Normalized embedding matrix of shape (n_texts, EMBEDDING_DIMENSION).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIMENSION)

        logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Encoded embeddings shape: {embeddings.shape}")
        return embeddings

    def encode_chunks(self, chunks: list[dict], batch_size: int = 64) -> np.ndarray:
        """
        Encode chunks from the chunking pipeline.

        Args:
            chunks: List of chunk dicts (must have 'text' key).
            batch_size: Batch size for encoding.

        Returns:
            Embedding matrix of shape (n_chunks, EMBEDDING_DIMENSION).
        """
        texts = [chunk["text"] for chunk in chunks]
        return self.encode_documents(texts, batch_size=batch_size)
