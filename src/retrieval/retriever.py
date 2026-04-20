"""
Retrieval Module
Orchestrates query embedding and vector search to retrieve relevant document chunks.
"""

import logging
from typing import Optional

from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from config.settings import TOP_K


logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks for a given query."""

    def __init__(self, vector_store: VectorStore, embedder: Optional[Embedder] = None):
        """
        Initialize the retriever.

        Args:
            vector_store: Initialized and loaded VectorStore instance.
            embedder: Embedder instance (creates a new one if not provided).
        """
        self.vector_store = vector_store
        self.embedder = embedder or Embedder()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        service_filter: Optional[str] = None,
    ) -> list[dict]:

        # STEP 1: Normalize query using LLM
        

        # DEBUG (very useful)
        # logger.info(f"Query: {query}")

        # STEP 2: Embedding
        query_embedding = self.embedder.encode_query(query)

        #  STEP 3: Retrieval
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            service_filter=service_filter,
        )

        if not results:
            logger.warning(f"No results found for query: '{query}'")

        logger.info(
            f"Query: '{query[:60]}...' → {len(results)} results "
            f"(filter: {service_filter or 'all'})"
        )

        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved results into a context string for the LLM.

        Args:
            results: List of retrieved result dicts.

        Returns:
            Formatted context string with source citations.
        """
        if not results:
            return "No relevant information found in the documents."

        context_parts = []

        for i, result in enumerate(results, start=1):
            source = result["metadata"].get("source_file", "Unknown")
            service = result["metadata"].get("service_type", "Unknown")
            score = result["score"]

            context_parts.append(
                f"[Source {i}: {source} | Service: {service} | Relevance: {score:.2f}]\n"
                f"{result['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # 🔥 Optional safety: limit context size (prevents LLM overflow)
        MAX_CONTEXT_CHARS = 2000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

        return context

    def get_source_citations(self, results: list[dict]) -> list[dict]:
        """
        Extract source citation information from results.

        Args:
            results: List of retrieved result dicts.

        Returns:
            List of citation dicts with source details.
        """
        citation_map: dict[tuple[str, str, str], dict] = {}

        for result in results:
            metadata = result.get("metadata", {})
            source_file = str(metadata.get("source_file", "Unknown"))
            page = str(metadata.get("page", "?"))
            service_type = str(metadata.get("service_type", "unknown"))
            relevance_score = float(result.get("score", 0.0))

            key = (source_file, page, service_type)
            existing = citation_map.get(key)

            if existing is None or relevance_score > float(existing.get("relevance_score", 0.0)):
                citation_map[key] = {
                    "source_file": source_file,
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