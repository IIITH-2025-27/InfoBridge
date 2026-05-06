"""
Response Generation Module
Orchestrates the full RAG pipeline: retrieve → format → generate → respond.
"""

import logging
import re
from typing import Optional

from src.retrieval.retriever import Retriever
from src.llm.client import llmClient
from src.llm.general_responses import get_general_query_response
from src.llm.prompts import build_prompt, get_output_language_guard, get_system_instruction
from src.memory.chat_memory import ChatMemory
from config.settings import TOP_K, MIN_ANSWER_SCORE
from utility.language import normalize_query_llm_cached

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Orchestrates retrieval and LLM response generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: Optional[llmClient] = None,
        memory: Optional[ChatMemory] = None,
    ):
        """
        Initialize the response generator.

        Args:
            retriever: Initialized Retriever instance.
            llm_client: LLM client (creates new if not provided).
            memory: ChatMemory instance (creates new if not provided).
        """
        self.retriever = retriever
        self.llm = llm_client or llmClient()
        self.memory = memory or ChatMemory()

    @staticmethod
    def _llm_unavailable(answer: str) -> bool:
        lower = answer.lower()
        return (
            "quota is exhausted" in lower
            or "model configuration issue" in lower
            or "encountered an error while generating" in lower
        )

    @staticmethod
    def _build_retrieval_fallback(results: list[dict], language: str) -> str:
        if not results:
            if language == "hi":
                return (
                    "अभी AI उत्तर उपलब्ध नहीं है और प्रलेखों में पर्याप्त मिलान नहीं मिला। "
                    "कृपया प्रश्न को स्पष्ट करके फिर पूछें।"
                )
            return (
                "AI response generation is temporarily unavailable, and no strong document match was found. "
                "Please rephrase your question and try again."
            )

        max_snippets = min(3, len(results))

        if language == "hi":
            lines = [
                "AI उत्तर फिलहाल उपलब्ध नहीं है, इसलिए नीचे प्रलेखों से सीधे सबसे प्रासंगिक जानकारी दी जा रही है:",
            ]
            snippet_label = "अंश"
        else:
            lines = [
                "AI response generation is temporarily unavailable, so here is the most relevant information directly from the documents:",
            ]
            snippet_label = "Snippet"

        for i in range(max_snippets):
            result = results[i]
            source = result["metadata"].get("source_file", "Unknown")
            service = result["metadata"].get("service_type", "Unknown")
            excerpt = " ".join(result["text"].split())
            excerpt = excerpt[:420] + ("..." if len(excerpt) > 420 else "")
            lines.append(f"\n{snippet_label} {i + 1} ({source} | {service}):\n{excerpt}")

        return "\n".join(lines)

    @staticmethod
    def _strip_inline_sources(answer: str) -> str:
        """Remove inline source/citation lines from model output so sources stay in the UI dropdown."""
        if not answer:
            return answer

        cleaned_lines = []
        for line in answer.splitlines():
            normalized = line.strip().lower()
            if not normalized:
                cleaned_lines.append(line)
                continue

            if (
                normalized.startswith("source:")
                or normalized.startswith("sources:")
                or normalized.startswith("[source:")
                or normalized.startswith("[sources:")
                or normalized.startswith("स्रोत:")
                or normalized.startswith("[स्रोत:")
                or "document reference" in normalized
                or "document references" in normalized
            ):
                continue

            cleaned_lines.append(line)

        cleaned_answer = "\n".join(cleaned_lines).strip()
        return cleaned_answer or answer

    @staticmethod
    def _looks_hinglish_romanized(text: str) -> bool:
        tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
        if not tokens:
            return False

        markers = {
            "main",
            "mujhe",
            "mera",
            "meri",
            "aap",
            "kaise",
            "kya",
            "kyun",
            "kyu",
            "hai",
            "hain",
            "nahi",
            "nahin",
            "chahiye",
            "karna",
            "banwana",
            "banaye",
            "banvao",
            "banwao",
        }
        marker_count = sum(1 for token in tokens if token in markers)
        return marker_count >= 2

    @staticmethod
    def _contains_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text or ""))

    @staticmethod
    def _is_hindi_output(text: str) -> bool:
        devanagari_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
        total_letters = sum(1 for ch in text if ch.isalpha())
        devanagari_ratio = devanagari_chars / max(total_letters, 1)
        return devanagari_ratio >= 0.20

    def _enforce_answer_language(self, answer: str, language: str) -> str:
        """Force final answer language to match selected app language."""
        if self._llm_unavailable(answer):
            return answer

        if language == "en":
            if not self._contains_devanagari(answer) and not self._looks_hinglish_romanized(answer):
                return answer

            prompt = (
                "Rewrite the following answer in clear, natural English only. "
                "Keep meaning and structure. Keep source labels/sources unchanged if present.\n\n"
                f"{answer}"
            )
            instruction = (
                "Return only the rewritten English answer. "
                "Do not include Hindi words in Devanagari or Romanized Hindi."
            )
            rewritten = self.llm.generate(prompt=prompt, system_instruction=instruction)
            return rewritten if rewritten and not self._llm_unavailable(rewritten) else answer

        if language == "hi":
            if self._is_hindi_output(answer):
                return answer

            prompt = (
                "Rewrite the following answer in Hindi (Devanagari script) only. "
                "Keep meaning and structure. Keep source labels/sources unchanged if present.\n\n"
                f"{answer}"
            )
            instruction = (
                "केवल देवनागरी हिंदी में उत्तर लौटाएं। "
                "स्रोत सूची/लेबल को यथावत रखें।"
            )
            rewritten = self.llm.generate(prompt=prompt, system_instruction=instruction)
            return rewritten if rewritten and not self._llm_unavailable(rewritten) else answer

        return answer

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
        4. Call llm API
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
        original_query = query

        selected_language = (language or "en").strip().lower()
        response_language = selected_language if selected_language in {"en", "hi"} else "en"

        general_response = get_general_query_response(original_query, response_language)
        if general_response:
            self.memory.add_turn(role="user", content=original_query)
            self.memory.add_turn(role="assistant", content=general_response)
            return {
                "answer": general_response,
                "sources": [],
                "show_sources": False,
                "language": response_language,
                "query": original_query.strip().lower(),
                "num_chunks_retrieved": 0,
            }

        query = query.strip().lower()
        query = normalize_query_llm_cached(query)
        logger.debug(f"Normalized Query: {query}")
        
        logger.info(f"Original Query: {original_query}")
        logger.info(f"Processed Query: {query}")
        
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            service_filter=service_filter,
        )

        # If retrieval returns only weak matches (top score below MIN_ANSWER_SCORE),
        # treat as no results so the model will explicitly say information is unavailable.
        if results:
            top_score = max(r.get("score", 0.0) for r in results)
            if top_score < MIN_ANSWER_SCORE:
                results = []

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
            language=response_language,
        )
        output_language_guard = get_output_language_guard(response_language)
        prompt = f"{prompt}\n\n--- OUTPUT LANGUAGE CONSTRAINT ---\n{output_language_guard}"

        system_instruction = get_system_instruction(response_language)

        # Step 5: Generate response
        answer = self.llm.generate(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        # If Hindi is requested but answer is not in Devanagari, retry with a stronger hint.
        if response_language == "hi" and not self._llm_unavailable(answer):
            devanagari_chars = sum(1 for ch in answer if "\u0900" <= ch <= "\u097F")
            total_letters = sum(1 for ch in answer if ch.isalpha())
            devanagari_ratio = devanagari_chars / max(total_letters, 1)
            if devanagari_ratio < 0.25:
                retry_instruction = system_instruction + "\n\nIMPORTANT: जवाब केवल हिंदी में दें।"
                answer = self.llm.generate(
                    prompt=prompt,
                    system_instruction=retry_instruction,
                )

        if (
            response_language == "en"
            and not self._llm_unavailable(answer)
            and self._looks_hinglish_romanized(answer)
        ):
            retry_instruction = (
                system_instruction
                + "\n\nIMPORTANT: Respond only in English. "
                + "Do not use Hindi words written in Latin script."
            )
            answer = self.llm.generate(
                prompt=prompt,
                system_instruction=retry_instruction,
            )

        answer = self._enforce_answer_language(answer=answer, language=response_language)
        answer = self._strip_inline_sources(answer)

        used_retrieval_fallback = False
        if self._llm_unavailable(answer):
            used_retrieval_fallback = True
            answer = self._build_retrieval_fallback(results=results, language=response_language)

        # Step 6: Update conversation memory
        self.memory.add_turn(role="user", content=original_query)
        self.memory.add_turn(role="assistant", content=answer)

        response = {
            "answer": answer,
            "sources": citations,
            "show_sources": bool(citations) and not used_retrieval_fallback,
            "language": response_language,
            "query": query,
            "num_chunks_retrieved": len(results),
        }

        logger.info(
            f"Generated response for query: '{query[:50]}...' "
            f"({len(citations)} sources, lang={response_language})"
        )

        return response

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
