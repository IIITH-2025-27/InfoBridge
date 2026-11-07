"""
Response Generation Module
Orchestrates the full RAG pipeline: retrieve → format → generate → respond.
"""

import logging
import re
import time
from typing import Optional

from src.retrieval.retriever import Retriever
from src.retrieval.query_processor import QueryProcessor
from src.llm.citations import apply_citations, strip_markers
from src.llm.client import is_llm_error, llmClient
from src.llm.general_responses import get_general_query_response
from src.llm.prompts import build_prompt, get_output_language_guard, get_system_instruction
from src.memory.chat_memory import ChatMemory
from src.guardrails import detect_prompt_injection, injection_refusal, pii_notice, redact_pii
from config.settings import (
    TOP_K,
    ENABLE_INLINE_CITATIONS,
    ENABLE_PII_REDACTION,
    ENABLE_INJECTION_GUARD,
)

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
        self.query_processor = QueryProcessor(llm=self.llm)

    @staticmethod
    def _llm_unavailable(answer: str) -> bool:
        # Was a check for "quota is exhausted", which the client never returned,
        # so the document-snippet fallback below never ran.
        return is_llm_error(answer)

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
                "Keep meaning and structure. Keep citation markers like [1] or [2][3] exactly where they are.\n\n"
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
                "Keep meaning and structure. Keep citation markers like [1] or [2][3] exactly where they are.\n\n"
                f"{answer}"
            )
            instruction = (
                "केवल देवनागरी हिंदी में उत्तर लौटाएं। "
                "स्रोत सूची/लेबल को यथावत रखें।"
            )
            rewritten = self.llm.generate(prompt=prompt, system_instruction=instruction)
            return rewritten if rewritten and not self._llm_unavailable(rewritten) else answer

        return answer

    def _simple_response(self, user_text: str, answer: str, language: str, query: str) -> dict:
        """Response that skipped retrieval (greeting, refusal, ...)."""
        self.memory.add_turn(role="user", content=user_text)
        self.memory.add_turn(role="assistant", content=answer)
        return {
            "answer": answer,
            "sources": [],
            "show_sources": False,
            "language": language,
            "query": query,
            "retrieval_query": query,
            "routed_service": None,
            "num_chunks_retrieved": 0,
            "contexts": [],
            "answerable": False,
            "unsupported_sentences": [],
            "pii_redacted": [],
            "timings": {},
        }

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
        1. Guardrails (prompt-injection refusal, PII redaction)
        2. Query understanding (follow-up rewrite, translation, service routing)
        3. Retrieve relevant chunks (hybrid + rerank) and decide if answerable
        4. Build prompt with numbered sources and conversation history
        5. Call LLM, enforce output language
        6. Turn [n] markers into page-level citations
        7. Return structured response

        Args:
            query: User's question.
            language: Language code ('en' or 'hi').
            service_filter: Optional service category filter (from the sidebar).
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with keys: answer, sources, language, query, retrieval_query,
            contexts (texts shown to the LLM), answerable, timings, ...
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()

        original_query = query

        selected_language = (language or "en").strip().lower()
        response_language = selected_language if selected_language in {"en", "hi"} else "en"

        general_response = get_general_query_response(original_query, response_language)
        if general_response:
            return self._simple_response(
                original_query, general_response, response_language, original_query.strip().lower()
            )

        # Step 1: Guardrails
        if ENABLE_INJECTION_GUARD and detect_prompt_injection(original_query):
            logger.warning("Prompt-injection attempt refused")
            return self._simple_response(
                original_query, injection_refusal(response_language), response_language, original_query
            )

        pii_found: list[str] = []
        if ENABLE_PII_REDACTION:
            redaction = redact_pii(original_query)
            if redaction.redacted:
                pii_found = redaction.found
                original_query = redaction.text
                logger.info(f"Redacted PII from query: {pii_found}")

        # Step 2: Query understanding
        t0 = time.perf_counter()
        plan = self.query_processor.process(original_query, self.memory.get_history())
        query = plan.retrieval_query
        timings["query_processing_ms"] = (time.perf_counter() - t0) * 1000

        logger.info(f"Original Query: {original_query}")
        logger.info(f"Processed Query: {query}")

        # Step 3: Retrieve
        t0 = time.perf_counter()
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            service_filter=service_filter,
            preferred_service=None if service_filter else plan.service,
        )
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # If retrieval returns only weak matches, treat as no results so the
        # model will explicitly say information is unavailable.
        answerable = self.retriever.is_answerable(results)
        if not answerable:
            results = []

        # Step 4: Format context (whole chunks only, within the size budget)
        context_results = self.retriever.select_for_context(results)
        context = self.retriever.format_context(context_results)

        chat_history = self.memory.format_history()

        question_for_llm = original_query
        if plan.rewritten:
            question_for_llm = f"{original_query}\n(Interpreted as: {query})"

        prompt = build_prompt(
            query=question_for_llm,
            context=context,
            chat_history=chat_history,
            language=response_language,
        )
        output_language_guard = get_output_language_guard(response_language)
        prompt = f"{prompt}\n\n--- OUTPUT LANGUAGE CONSTRAINT ---\n{output_language_guard}"

        system_instruction = get_system_instruction(response_language)

        # Step 5: Generate response
        t0 = time.perf_counter()
        answer = self.llm.generate(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        # If Hindi is requested but answer is not in Devanagari, retry with a stronger hint.
        if response_language == "hi" and not self._llm_unavailable(answer):
            devanagari_chars = sum(1 for ch in answer if "ऀ" <= ch <= "ॿ")
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
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000

        # Step 6: Citations
        used_retrieval_fallback = False
        unsupported_sentences: list[str] = []
        if self._llm_unavailable(answer):
            used_retrieval_fallback = True
            answer = self._build_retrieval_fallback(results=context_results, language=response_language)
            citations = []
        elif ENABLE_INLINE_CITATIONS:
            cited = apply_citations(answer, context_results, response_language)
            answer = cited.answer
            citations = cited.sources
            unsupported_sentences = cited.unsupported_sentences
        else:
            answer = strip_markers(answer)
            citations = self.retriever.get_source_citations(context_results)

        if pii_found:
            answer = f"{pii_notice(pii_found, response_language)}\n\n{answer}"

        # Step 7: Update conversation memory (markers removed: numbering is per answer)
        self.memory.add_turn(role="user", content=original_query)
        self.memory.add_turn(role="assistant", content=strip_markers(answer))

        timings["total_ms"] = (time.perf_counter() - started) * 1000

        response = {
            "answer": answer,
            "sources": citations,
            "show_sources": bool(citations) and not used_retrieval_fallback,
            "language": response_language,
            "query": original_query,
            "retrieval_query": query,
            "routed_service": plan.service,
            "num_chunks_retrieved": len(context_results),
            "contexts": [r["text"] for r in context_results],
            "retrieved": context_results,
            "answerable": answerable,
            "used_retrieval_fallback": used_retrieval_fallback,
            "unsupported_sentences": unsupported_sentences,
            "pii_redacted": pii_found,
            "timings": timings,
        }

        logger.info(
            f"Generated response for query: '{query[:50]}...' "
            f"({len(citations)} sources, lang={response_language}, "
            f"{timings['total_ms']:.0f} ms)"
        )

        return response

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
