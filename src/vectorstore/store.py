"""
FAISS Vector Store (FINAL - FIXED)
"""

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.settings import (
    SIMILARITY_THRESHOLD,
    EMBEDDING_DIMENSION,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    TOP_K,
)

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata = []
        self.texts = []

    @property
    def size(self):
        return self.index.ntotal

    def get_service_stats(self) -> dict[str, int]:
        """Return chunk counts grouped by service_type for sidebar display."""
        if not self.metadata:
            return {}

        counts = Counter(
            item.get("service_type", "unknown")
            for item in self.metadata
        )
        return dict(counts)

    
    
    
    def add_documents(self, embeddings, chunks):
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding-chunk mismatch")

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError("Wrong embedding dimension")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(embeddings)

        for c in chunks:
            self.texts.append(c["text"])
            self.metadata.append(c["metadata"])

        logger.info(f"Added {len(chunks)} vectors → total {self.size}")

    def search(self, query_embedding, top_k=TOP_K, service_filter=None):

        if self.size == 0:
            return []

        if query_embedding is None or len(query_embedding) == 0:
            return []

        # Normalize query (safety)
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            return []
        query_embedding = query_embedding / norm

        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1),
            dtype=np.float32
        )

        search_k = top_k * 3 if service_filter else top_k

        scores, indices = self.index.search(query, min(search_k, self.size))

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            meta = self.metadata[idx]

            if service_filter and meta.get("service_type") != service_filter:
                continue

            # 🔥 Proper threshold filtering
            if score < SIMILARITY_THRESHOLD:
                continue

            results.append({
                "text": self.texts[idx],
                "metadata": meta,
                "score": float(score)
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, index_path=None, metadata_path=None):
        index_path = Path(index_path or FAISS_INDEX_FILE)
        metadata_path = Path(metadata_path or METADATA_FILE)

        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "texts": self.texts
            }, f)

        logger.info(f"Saved {self.size} vectors")

    def load(self, index_path=None, metadata_path=None):
        index_path = Path(index_path or FAISS_INDEX_FILE)
        metadata_path = Path(metadata_path or METADATA_FILE)

        if not index_path.exists() or not metadata_path.exists():
            logger.warning("Vector store not found")
            return False

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.texts = data["texts"]

        logger.info(f"Loaded {self.size} vectors")
        return True