"""
Retrieval Module
Orchestrates query embedding and search to retrieve relevant document chunks.

Pipeline (each step can be switched off in config/settings.py for ablations):
    dense (FAISS) + BM25 candidates  →  Reciprocal Rank Fusion
    →  cross-encoder rerank  →  recency preference  →  max N chunks per PDF
    →  optional neighbour expansion
"""

import logging
from typing import Optional

import numpy as np

from src.data_processing.doc_metadata import document_family, extract_query_year
from src.embeddings.embedder import Embedder
from src.retrieval.bm25 import BM25Index
from src.retrieval.reranker import Reranker
from src.vectorstore.store import VectorStore
from config.settings import (
    TOP_K,
    CANDIDATE_POOL,
    RRF_K,
    RETRIEVAL_MODE,
    USE_RERANKER,
    EXPAND_NEIGHBORS,
    MAX_CHUNKS_PER_SOURCE,
    MAX_CONTEXT_CHARS,
    MIN_ANSWER_SCORE,
    MIN_RERANK_SCORE,
    PREFER_RECENT_DOCS,
    RECENCY_WEIGHT,
    CHUNK_OVERLAP,
)


logger = logging.getLogger(__name__)


def _merge_overlapping(first: str, second: str) -> str:
    """Join two consecutive chunks, removing the overlap the splitter repeated."""
    max_overlap = min(len(first), len(second), CHUNK_OVERLAP + 50)
    for size in range(max_overlap, 10, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return first + "\n" + second


class Retriever:
    """Retrieves relevant document chunks for a given query."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Initialized and loaded VectorStore instance.
            embedder: Embedder instance (creates a new one if not provided).
            reranker: Reranker instance (created lazily when USE_RERANKER is on).
        """
        self.vector_store = vector_store
        self.embedder = embedder or Embedder()
        self._reranker = reranker
        self._bm25: Optional[BM25Index] = None
        self._service_masks: dict[str, np.ndarray] = {}

    # ─── Lazy components ─────────────────────────────────────────────────────

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            texts = [
                f"{meta.get('title', '')}\n{text}"
                for meta, text in zip(self.vector_store.metadata, self.vector_store.texts)
            ]
            self._bm25 = BM25Index(texts)
        return self._bm25

    @property
    def reranker(self) -> Optional[Reranker]:
        if not USE_RERANKER:
            return None
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker if self._reranker.available else None

    def _service_mask(self, service: str) -> np.ndarray:
        if service not in self._service_masks:
            self._service_masks[service] = np.array(
                [m.get("service_type") == service for m in self.vector_store.metadata]
            )
        return self._service_masks[service]

    # ─── Main entry point ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        service_filter: Optional[str] = None,
        unique_sources: bool = False,
        preferred_service: Optional[str] = None,
        max_per_source: int = MAX_CHUNKS_PER_SOURCE,
    ) -> list[dict]:
        """
        Args:
            query: Retrieval query (English, standalone).
            top_k: Number of chunks to return.
            service_filter: Hard filter chosen by the user in the sidebar.
            unique_sources: Old behaviour (one chunk per PDF); same as max_per_source=1.
            preferred_service: Soft preference from the service router: that
                service's results are counted twice in the fusion, but other
                services can still win.
            max_per_source: Maximum chunks returned from the same PDF.

        Returns:
            Result dicts with text, metadata, score (0..1 confidence), position and
            the individual dense / bm25 / rrf / rerank scores.
        """
        if self.vector_store.size == 0:
            return []

        if unique_sources:
            max_per_source = 1

        pool = max(CANDIDATE_POOL, top_k)
        query_embedding = self.embedder.encode_query(query)

        # STEP 1: candidate lists from each retriever
        ranked_lists: list[list[int]] = []
        bm25_scores: dict[int, float] = {}

        services_to_search = [service_filter]
        if preferred_service and not service_filter:
            services_to_search.append(preferred_service)

        for service in services_to_search:
            if RETRIEVAL_MODE in ("dense", "hybrid"):
                dense_hits = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=pool,
                    service_filter=service,
                    apply_threshold=RETRIEVAL_MODE == "dense",
                )
                ranked_lists.append([hit["position"] for hit in dense_hits])

            if RETRIEVAL_MODE in ("bm25", "hybrid"):
                mask = self._service_mask(service) if service else None
                bm25_hits = self.bm25.search(query, top_k=pool, mask=mask)
                ranked_lists.append([pos for pos, _ in bm25_hits])
                bm25_scores.update(dict(bm25_hits))

        # STEP 2: Reciprocal Rank Fusion — combines rankings without having to
        # calibrate cosine scores against BM25 scores.
        rrf_scores: dict[int, float] = {}
        for ranking in ranked_lists:
            for rank, position in enumerate(ranking):
                rrf_scores[position] = rrf_scores.get(position, 0.0) + 1.0 / (RRF_K + rank + 1)

        if not rrf_scores:
            logger.warning(f"No results found for query: '{query}'")
            return []

        positions = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:pool]
        dense_scores = self.vector_store.dense_scores(query_embedding, positions)

        results = []
        for position in positions:
            result = self.vector_store.make_result(position, dense_scores.get(position, 0.0))
            result["dense_score"] = dense_scores.get(position, 0.0)
            result["bm25_score"] = bm25_scores.get(position, 0.0)
            result["rrf_score"] = rrf_scores[position]
            results.append(result)

        # STEP 3: rerank
        rerank_scores = None
        reranker = self.reranker
        if reranker is not None:
            rerank_scores = reranker.score(query, [r["text"] for r in results])

        if rerank_scores is not None:
            for result, score in zip(results, rerank_scores):
                result["rerank_score"] = float(score)
                result["confidence"] = float(score)
                result["confidence_type"] = "rerank"
            base_key = "rerank_score"
        else:
            for result in results:
                result["confidence"] = result["dense_score"]
                result["confidence_type"] = "dense"
            base_key = "dense_score" if RETRIEVAL_MODE == "dense" else "rrf_score"

        # STEP 4: ordering (+ prefer the newest documents)
        self._rank(results, base_key, query)

        # STEP 5: at most N chunks from the same PDF
        results = self._limit_per_source(results, top_k, max_per_source)

        # STEP 6: optionally attach neighbouring chunks
        if EXPAND_NEIGHBORS:
            self._expand_neighbors(results)

        for result in results:
            # "score" is what the UI shows as relevance (0..1)
            result["score"] = result["confidence"]

        if not results:
            logger.warning(f"No results found for query: '{query}'")

        logger.info(
            f"Query: '{query[:60]}...' → {len(results)} results "
            f"(mode: {RETRIEVAL_MODE}, reranked: {rerank_scores is not None}, "
            f"filter: {service_filter or 'all'}, preferred: {preferred_service or '-'})"
        )

        return results

    # ─── Ranking helpers ─────────────────────────────────────────────────────

    def _rank(self, results: list[dict], base_key: str, query: str) -> None:
        base = np.array([r[base_key] for r in results], dtype=np.float32)
        span = float(base.max() - base.min()) if len(base) else 0.0
        normalized = (base - base.min()) / span if span > 0 else np.ones_like(base)

        bonuses = self._recency_bonus(results, query) if PREFER_RECENT_DOCS else [0.0] * len(results)

        for result, norm, bonus in zip(results, normalized, bonuses):
            result["rank_score"] = float(norm) + bonus

        results.sort(key=lambda r: r["rank_score"], reverse=True)

    @staticmethod
    def _recency_bonus(results: list[dict], query: str) -> list[float]:
        """
        Small bonus for newer editions of the same document (Phase 5).

        Only documents that are versions of each other are compared: same service
        and same document family (title without years/versions, e.g. the AY 2024-25
        and AY 2025-26 ITR FAQs). Comparing years across unrelated documents
        (a 2024 electoral manual vs a 2016 passport manual) pushed correct pages
        down in the eval (Hit@1 0.52 → 0.49).

        If the user names a year, editions for that year get the bonus instead.
        """
        bonuses = [0.0] * len(results)
        asked_year = extract_query_year(query)

        families: dict[tuple[str, str], list[int]] = {}
        for i, result in enumerate(results):
            meta = result["metadata"]
            if not meta.get("doc_year"):
                continue
            key = (str(meta.get("service_type")), document_family(str(meta.get("source_file", ""))))
            families.setdefault(key, []).append(i)

        for indices in families.values():
            years = {results[i]["metadata"]["doc_year"] for i in indices}
            if len(years) < 2:
                continue  # only one edition among the candidates: nothing to prefer

            oldest, newest = min(years), max(years)
            for i in indices:
                year = results[i]["metadata"]["doc_year"]
                if asked_year:
                    bonuses[i] = RECENCY_WEIGHT if year == asked_year else 0.0
                else:
                    bonuses[i] = RECENCY_WEIGHT * (year - oldest) / (newest - oldest)

        return bonuses

    @staticmethod
    def _limit_per_source(results: list[dict], top_k: int, max_per_source: int) -> list[dict]:
        kept = []
        per_source: dict[tuple[str, str], int] = {}

        for result in results:
            meta = result.get("metadata", {})
            key = (str(meta.get("source_file", "")), str(meta.get("service_type", "")))
            if per_source.get(key, 0) >= max_per_source:
                continue
            per_source[key] = per_source.get(key, 0) + 1
            kept.append(result)
            if len(kept) >= top_k:
                break

        return kept

    def _expand_neighbors(self, results: list[dict]) -> None:
        used = {r["position"] for r in results}
        for result in results:
            position = result["position"]
            neighbors = [p for p in self.vector_store.neighbor_positions(position) if p not in used]
            if not neighbors:
                continue
            used.update(neighbors)

            ordered = sorted([position, *neighbors], key=lambda p: self.vector_store.metadata[p].get("chunk_index", 0))
            text = self.vector_store.texts[ordered[0]]
            for p in ordered[1:]:
                text = _merge_overlapping(text, self.vector_store.texts[p])

            result["core_text"] = result["text"]
            result["text"] = text
            result["expanded"] = True

    # ─── Answerability ───────────────────────────────────────────────────────

    @staticmethod
    def answer_threshold(result: dict) -> float:
        """Minimum confidence for answering, depending on how it was computed."""
        if result.get("confidence_type") == "rerank":
            return MIN_RERANK_SCORE
        return MIN_ANSWER_SCORE

    @classmethod
    def is_answerable(cls, results: list[dict]) -> bool:
        if not results:
            return False
        best = max(results, key=lambda r: r.get("confidence", 0.0))
        return best.get("confidence", 0.0) >= cls.answer_threshold(best)

    # ─── Context building ────────────────────────────────────────────────────

    @staticmethod
    def _source_header(number: int, result: dict) -> str:
        meta = result["metadata"]
        parts = [
            f"Source {number}",
            str(meta.get("source_file", "Unknown")),
            f"page {meta.get('page', '?')}",
            f"service: {meta.get('service_type', 'Unknown')}",
        ]
        if meta.get("assessment_year"):
            parts.append(f"AY {meta['assessment_year']}")
        elif meta.get("doc_year"):
            parts.append(f"year {meta['doc_year']}")
        return "[" + " | ".join(parts) + "]"

    def select_for_context(self, results: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
        """
        Keep whole chunks, best first, until the character budget is used.
        Cutting at chunk boundaries (instead of mid-text) keeps citation numbers
        meaningful: every numbered source the LLM sees is complete.
        """
        selected = []
        used = 0
        for number, result in enumerate(results, start=1):
            block = len(self._source_header(number, result)) + len(result["text"]) + 7
            if selected and used + block > max_chars:
                break
            selected.append(result)
            used += block
        return selected

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved results into a context string for the LLM.

        Args:
            results: List of retrieved result dicts (already passed through select_for_context).

        Returns:
            Formatted context string; each block starts with a numbered source header.
        """
        if not results:
            return "No relevant information found in the documents."

        context_parts = [
            f"{self._source_header(i, result)}\n{result['text']}"
            for i, result in enumerate(results, start=1)
        ]
        context = "\n\n---\n\n".join(context_parts)

        # Safety net if a single chunk is longer than the whole budget
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

        return context

    def get_source_citations(self, results: list[dict]) -> list[dict]:
        """
        Extract source citation information from results (one entry per PDF).

        Args:
            results: List of retrieved result dicts.

        Returns:
            List of citation dicts with source details.
        """
        citation_map: dict[tuple[str, str], dict] = {}

        for result in results:
            metadata = result.get("metadata", {})
            source_file = str(metadata.get("source_file", "Unknown"))
            source_path = str(metadata.get("source_path", ""))
            page = str(metadata.get("page", "?"))
            service_type = str(metadata.get("service_type", "unknown"))
            relevance_score = float(result.get("score", 0.0))

            key = (source_file, service_type)
            existing = citation_map.get(key)

            if existing is None or relevance_score > float(existing.get("relevance_score", 0.0)):
                citation_map[key] = {
                    "source_file": source_file,
                    "source_path": source_path,
                    "page": page,
                    "service_type": service_type,
                    "relevance_score": relevance_score,
                }

        citations = sorted(
            citation_map.values(),
            key=lambda item: float(item.get("relevance_score", 0.0)),
            reverse=True,
        )

        return citations
