"""
FAISS Vector Store Module
Manages vector storage, persistence, and similarity search.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.settings import (
    EMBEDDING_DIMENSION,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    VECTORSTORE_DIR,
    TOP_K,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store with metadata management."""

    def __init__(self):
        """Initialize an empty FAISS index and metadata store."""
        # Use IndexFlatIP for inner product (cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata: list[dict] = []
        self.texts: list[str] = []

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self.index.ntotal

    def add_documents(
        self,
        embeddings: np.ndarray,
        chunks: list[dict],
    ) -> None:
        """
        Add document embeddings and their metadata to the store.

        Args:
            embeddings: Embedding matrix of shape (n, EMBEDDING_DIMENSION).
            chunks: List of chunk dicts with 'text' and 'metadata' keys.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
            )

        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected shape (n, {EMBEDDING_DIMENSION}), got {embeddings.shape}"
            )

        # Ensure float32
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(embeddings)

        for chunk in chunks:
            self.texts.append(chunk["text"])
            self.metadata.append(chunk["metadata"])

        logger.info(
            f"Added {len(chunks)} vectors. Total vectors: {self.size}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
        service_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for the most similar documents to a query.

        Args:
            query_embedding: Query vector of shape (EMBEDDING_DIMENSION,).
            top_k: Number of results to return.
            service_filter: Optional service type to filter results.

        Returns:
            List of result dicts with keys: text, metadata, score.
        """
        if self.size == 0:
            logger.warning("Vector store is empty. No results.")
            return []

        # Reshape for FAISS
        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )

        # If filtering by service, we need to search more and then filter
        search_k = top_k * 3 if service_filter else top_k

        scores, indices = self.index.search(query, min(search_k, self.size))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            metadata = self.metadata[idx]

            # Apply service filter if specified
            if service_filter and metadata.get("service_type") != service_filter:
                continue

            results.append({
                "text": self.texts[idx],
                "metadata": metadata,
                "score": float(score),
            })

            if len(results) >= top_k:
                break

        logger.info(
            f"Search returned {len(results)} results "
            f"(filter: {service_filter or 'none'})"
        )
        return results

    def save(
        self,
        index_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
    ) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Args:
            index_path: Path to save the FAISS index.
            metadata_path: Path to save the metadata pickle.
        """
        index_path = Path(index_path or FAISS_INDEX_FILE)
        metadata_path = Path(metadata_path or METADATA_FILE)

        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save metadata and texts
        with open(metadata_path, "wb") as f:
            pickle.dump({"metadata": self.metadata, "texts": self.texts}, f)

        logger.info(
            f"Saved vector store: {self.size} vectors → {index_path}"
        )

    def load(
        self,
        index_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
    ) -> bool:
        """
        Load a persisted FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the metadata pickle file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        index_path = Path(index_path or FAISS_INDEX_FILE)
        metadata_path = Path(metadata_path or METADATA_FILE)

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(
                f"Vector store files not found at {index_path.parent}. "
                "Run 'python scripts/build_index.py' first."
            )
            return False

        try:
            self.index = faiss.read_index(str(index_path))

            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.texts = data["texts"]

            logger.info(f"Loaded vector store: {self.size} vectors from {index_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    def get_service_stats(self) -> dict[str, int]:
        """Get count of chunks per service type."""
        stats: dict[str, int] = {}
        for meta in self.metadata:
            svc = meta.get("service_type", "unknown")
            stats[svc] = stats.get(svc, 0) + 1
        return stats
