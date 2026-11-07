"""
FAISS Vector Store (FINAL - FIXED)
"""

import logging
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.settings import (
    SIMILARITY_THRESHOLD,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    TOP_K,
)
from src.data_processing.doc_metadata import infer_doc_dates, title_from_filename

logger = logging.getLogger(__name__)

# Indexes built before the manifest existed were all made with this model.
_LEGACY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class IndexMismatchError(RuntimeError):
    """The saved index was built with a different embedding model."""


class VectorStore:

    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata = []
        self.texts = []
        self.manifest = {}
        self._position_lookup: dict[tuple, int] = {}

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

        self._build_position_lookup()
        logger.info(f"Added {len(chunks)} vectors → total {self.size}")

    # ─── Lookups ─────────────────────────────────────────────────────────────

    def _build_position_lookup(self) -> None:
        self._position_lookup = {
            (m.get("source_file"), m.get("page"), m.get("chunk_index")): i
            for i, m in enumerate(self.metadata)
        }

    def neighbor_positions(self, position: int, window: int = 1) -> list[int]:
        """Positions of the chunks just before/after `position` on the same page."""
        meta = self.metadata[position]
        neighbors = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            key = (meta.get("source_file"), meta.get("page"), meta.get("chunk_index", 0) + offset)
            if key in self._position_lookup:
                neighbors.append(self._position_lookup[key])
        return neighbors

    def dense_scores(self, query_embedding: np.ndarray, positions: list[int]) -> dict[int, float]:
        """Cosine similarity of the query with specific stored vectors."""
        if not positions:
            return {}
        query = self._normalize(query_embedding)
        if query is None:
            return {}
        vectors = np.vstack([self.index.reconstruct(int(p)) for p in positions])
        scores = vectors @ query.reshape(-1)
        return {p: float(s) for p, s in zip(positions, scores)}

    def make_result(self, position: int, score: float) -> dict:
        return {
            "text": self.texts[position],
            "metadata": self.metadata[position],
            "score": float(score),
            "position": position,
        }

    # ─── Search ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(query_embedding) -> Optional[np.ndarray]:
        if query_embedding is None or len(query_embedding) == 0:
            return None
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            return None
        return np.ascontiguousarray((query_embedding / norm).reshape(1, -1), dtype=np.float32)

    def search(self, query_embedding, top_k=TOP_K, service_filter=None, apply_threshold=True):

        if self.size == 0:
            return []

        query = self._normalize(query_embedding)
        if query is None:
            return []

        # With a service filter, scan the whole index: a flat index of ~20k
        # vectors is cheap, and top_k * 3 often missed small services entirely.
        search_k = self.size if service_filter else top_k

        scores, indices = self.index.search(query, min(search_k, self.size))

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            meta = self.metadata[idx]

            if service_filter and meta.get("service_type") != service_filter:
                continue

            # 🔥 Proper threshold filtering
            if apply_threshold and score < SIMILARITY_THRESHOLD:
                continue

            results.append(self.make_result(int(idx), score))

            if len(results) >= top_k:
                break

        return results

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self, index_path=None, metadata_path=None):
        index_path = Path(index_path or FAISS_INDEX_FILE)
        metadata_path = Path(metadata_path or METADATA_FILE)

        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        self.manifest = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "num_chunks": self.size,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "texts": self.texts,
                "manifest": self.manifest,
            }, f)

        logger.info(f"Saved {self.size} vectors")

    def load(self, index_path=None, metadata_path=None):
        index_path = Path(index_path or FAISS_INDEX_FILE)
        metadata_path = Path(metadata_path or METADATA_FILE)

        if not index_path.exists() or not metadata_path.exists():
            logger.warning("Vector store not found")
            return False

        with open(metadata_path, "rb") as f:
            data = pickle.load(f)

        manifest = data.get("manifest") or {"embedding_model": _LEGACY_EMBEDDING_MODEL}
        if manifest.get("embedding_model") != EMBEDDING_MODEL:
            # Same dimension (384) would load silently and return nonsense results.
            raise IndexMismatchError(
                f"The index was built with '{manifest.get('embedding_model')}' but "
                f"EMBEDDING_MODEL is '{EMBEDDING_MODEL}'. Rebuild it with "
                f"`python scripts/build_index.py`, or set EMBEDDING_MODEL in .env to match."
            )

        self.index = faiss.read_index(str(index_path))
        self.metadata = data["metadata"]
        self.texts = data["texts"]
        self.manifest = manifest

        self._backfill_metadata()
        self._build_position_lookup()

        logger.info(f"Loaded {self.size} vectors")
        return True

    def _backfill_metadata(self) -> None:
        """Add title/year fields to indexes built before they existed."""
        cache: dict[str, dict] = {}
        for meta in self.metadata:
            if "doc_year" in meta:
                continue
            source_file = meta.get("source_file", "")
            if source_file not in cache:
                cache[source_file] = infer_doc_dates(source_file)
            meta.setdefault("title", title_from_filename(source_file))
            meta.update(cache[source_file])
