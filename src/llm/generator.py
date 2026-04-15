"""
Response Generation Module
Orchestrates the full RAG pipeline: retrieve → format → generate → respond.
"""

import logging
from typing import Optional

from src.retrieval.retriever import Retriever
from src.llm.gemini_client import GeminiClient
from src.llm.prompts import build_prompt, get_system_instruction
from src.memory.chat_memory import ChatMemory
from config.settings import TOP_K

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Orchestrates retrieval and LLM response generation."""

    def __init__(
        self,
        retriever: Retriever,
        gemini_client: Optional[GeminiClient] = None,
        memory: Optional[ChatMemory] = None,
    ):
        """
        Initialize the response generator.

        Args:
            retriever: Initialized Retriever instance.
            gemini_client: Gemini client (creates new if not provided).
            memory: ChatMemory instance (creates new if not provided).
        """
        self.retriever = retriever
        self.gemini = gemini_client or GeminiClient()
        self.memory = memory or ChatMemory()

    def generate(
        self,
        query: str,
        language: str = "en",
        service_filter: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> dict:
        """
        Generate a complete response for a user query.

        Pipeline:
        1. Retrieve relevant chunks
        2. Format context with citations
        3. Build prompt with conversation history
        4. Call Gemini API
        5. Return structured response

        Args:
            query: User's question.
            language: Language code ('en' or 'hi').
            service_filter: Optional service category filter.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with keys: answer, sources, language, query.
        """
        # Step 1: Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            service_filter=service_filter,
        )

        # Step 2: Format context
        context = self.retriever.format_context(results)
        citations = self.retriever.get_source_citations(results)

        # Step 3: Get conversation history
        chat_history = self.memory.format_history()

        # Step 4: Build prompt
        prompt = build_prompt(
            query=query,
            context=context,
            chat_history=chat_history,
            language=language,
        )

        system_instruction = get_system_instruction(language)

        # Step 5: Generate response
        answer = self.gemini.generate(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        # If Hindi is requested but the answer is not, retry with a stronger hint
        if language == "hi":
            devanagari_chars = sum(1 for ch in answer if "\u0900" <= ch <= "\u097F")
            total_letters = sum(1 for ch in answer if ch.isalpha())
            devanagari_ratio = devanagari_chars / max(total_letters, 1)
            if devanagari_ratio < 0.25:
                retry_instruction = system_instruction + "\n\nIMPORTANT: जवाब केवल हिंदी में दें।"
                answer = self.gemini.generate(
                    prompt=prompt,
                    system_instruction=retry_instruction,
                )

        # Ensure answers always include deterministic citations
        if citations:
            answer_lower = answer.lower()
            has_sources = "sources:" in answer_lower or "स्रोत" in answer_lower
            sources_label = "स्रोत:" if language == "hi" else "Sources:"
            if not has_sources:
                sources_lines = [sources_label]
                for citation in citations:
                    sources_lines.append(
                        f"- {citation['source_file']} ({citation['service_type']})"
                    )
                answer = f"{answer}\n\n" + "\n".join(sources_lines)

        # Step 6: Update conversation memory
        self.memory.add_turn(role="user", content=query)
        self.memory.add_turn(role="assistant", content=answer)

        response = {
            "answer": answer,
            "sources": citations,
            "language": language,
            "query": query,
            "num_chunks_retrieved": len(results),
        }

        logger.info(
            f"Generated response for query: '{query[:50]}...' "
            f"({len(citations)} sources, lang={language})"
        )

        return response

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
