"""
Response Generation Module
Orchestrates the full RAG pipeline: retrieve → format → generate → respond.
"""

import logging
import re
from typing import Optional

from src.retrieval.retriever import Retriever
from src.llm.client import llmClient
from src.llm.prompts import build_prompt, get_output_language_guard, get_system_instruction, get_general_system_instruction
from src.memory.chat_memory import ChatMemory
from config.settings import TOP_K
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

    # Patterns that indicate a conversational / general-purpose query that
    # does NOT need document retrieval.
    _CONVERSATIONAL_PATTERNS = [
        r"^(hi|hello|hey|namaste|namaskar|hola|howdy|good\s+(morning|afternoon|evening|day))[\s!.,?]*$",
        r"^(how are you|how r u|how do you do|what's up|wassup|sup)[\s!.,?]*$",
        r"^(thanks|thank you|thank you so much|thx|ty|dhanyavaad)[\s!.,?]*$",
        r"^(bye|goodbye|good bye|see you|see ya|alvida)[\s!.,?]*$",
        r"(what (can|do) you (do|help|assist|know)|what are your capabilities|what services do you (cover|offer|support|provide))",
        r"(who are you|what are you|tell me about yourself|introduce yourself)",
        r"^(ok|okay|got it|i see|sure|alright|great|nice|cool|perfect)[\s!.,?]*$",
        r"^(yes|no|yep|nope|yeah|nah)[\s!.,?]*$",
        # Website / portal / contact queries — not answerable from PDF docs
        r"(website|web site|official\s+site|portal|link|url|web\s+address)",
        r"(helpline|toll.?free|contact\s+number|phone\s+number|call\s+centre|customer\s+care)",
        r"(give me the|share the|what is the|provide the).{0,30}(link|url|website|portal|number)",
    ]

    @classmethod
    def _is_conversational_query(cls, query: str) -> bool:
        """Return True if the query is a simple greeting or general question."""
        import re as _re
        q = query.strip().lower()
        # Very short inputs with no government keywords are likely conversational
        gov_keywords = {
            "passport", "voter", "driving", "licence", "license", "tax", "itr",
            "ayushman", "pmjay", "epic", "aadhaar", "pan", "document", "apply",
            "registration", "certificate", "income", "form", "fee", "eligib",
            "पासपोर्ट", "वोटर", "लाइसेंस", "आयकर", "आयुष्मान",
        }
        has_gov_keyword = any(kw in q for kw in gov_keywords)
        if has_gov_keyword:
            return False
        for pattern in cls._CONVERSATIONAL_PATTERNS:
            if _re.search(pattern, q, _re.IGNORECASE):
                return True
        # Treat very short inputs (≤4 words, no gov keyword) as conversational
        if len(q.split()) <= 4 and not has_gov_keyword:
            return True
        return False

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
        # Step 0: Language normalisation
        original_query = query

        selected_language = (language or "en").strip().lower()
        response_language = selected_language if selected_language in {"en", "hi"} else "en"

        query_stripped = query.strip()
        query = query_stripped.lower()
        query = normalize_query_llm_cached(query)
        logger.debug(f"Normalized Query: {query}")
        logger.info(f"Original Query: {original_query}")
        logger.info(f"Processed Query: {query}")

        # Step 1a: Fast path — conversational / general queries bypass RAG
        if self._is_conversational_query(query_stripped):
            logger.info("Conversational query detected — skipping retrieval")
            general_instruction = get_general_system_instruction(response_language)
            answer = self.llm.generate(
                prompt=query_stripped,
                system_instruction=general_instruction,
            )
            answer = self._enforce_answer_language(answer=answer, language=response_language)
            self.memory.add_turn(role="user", content=original_query)
            self.memory.add_turn(role="assistant", content=answer)
            return {
                "answer": answer,
                "sources": [],
                "language": response_language,
                "query": query,
                "num_chunks_retrieved": 0,
            }

        # Step 1b: Retrieve relevant chunks for document-based questions
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

        if self._llm_unavailable(answer):
            answer = self._build_retrieval_fallback(results=results, language=response_language)

        # Step 6: Update conversation memory
        self.memory.add_turn(role="user", content=original_query)
        self.memory.add_turn(role="assistant", content=answer)

        response = {
            "answer": answer,
            "sources": citations,
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
