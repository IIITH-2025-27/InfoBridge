"""
Retrieval Module
Orchestrates query embedding and vector search to retrieve relevant document chunks.
"""

import logging
from typing import Optional

from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from config.settings import TOP_K, SIMILARITY_THRESHOLD

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
        min_score: float = SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User query string.
            top_k: Number of results to return.
            service_filter: Optional service category to filter by.
            min_score: Minimum similarity score threshold.

        Returns:
            List of result dicts: {text, metadata, score}.
        """
        # Encode the query
        query_embedding = self.embedder.encode_query(query)

        # Search the vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            service_filter=service_filter,
        )

        # Filter by minimum score
        filtered_results = [r for r in results if r["score"] >= min_score]

        if not filtered_results and results:
            # If all results are below threshold, return the best one anyway
            filtered_results = [results[0]]
            logger.warning(
                f"All results below threshold ({min_score}). "
                f"Returning best match with score {results[0]['score']:.4f}"
            )

        logger.info(
            f"Query: '{query[:60]}...' → {len(filtered_results)} results "
            f"(filter: {service_filter or 'all'})"
        )
        return filtered_results

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

        return "\n\n---\n\n".join(context_parts)

    def get_source_citations(self, results: list[dict]) -> list[dict]:
        """
        Extract source citation information from results.

        Args:
            results: List of retrieved result dicts.

        Returns:
            List of citation dicts with source details.
        """
        seen_sources = set()
        citations = []

        for result in results:
            source_file = result["metadata"].get("source_file", "Unknown")
            if source_file not in seen_sources:
                seen_sources.add(source_file)
                citations.append({
                    "source_file": source_file,
                    "service_type": result["metadata"].get("service_type", "Unknown"),
                    "relevance_score": result["score"],
                })

        return citations
