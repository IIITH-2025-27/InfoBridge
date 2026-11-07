import logging
import re
from functools import lru_cache
from src.llm.client import is_llm_error, llmClient

logger = logging.getLogger(__name__)

HINGLISH_MARKERS = {
    "main",
    "ka",
    "ki",
    "ke",
    "kya",
    "kaise",
    "kr",
    "kar",
    "karo",
    "krna",
    "karna",
    "mujhe",
    "mera",
    "meri",
    "mere",
    "aap",
    "apna",
    "apni",
    "apne",
    "hai",
    "hain",
    "nahi",
    "nahin",
    "kyu",
    "kyun",
    "kaun",
    "kab",
    "kahan",
    "banana",
    "banwana",
    "banwaye",
    "banvao",
    "banwao",
    "banwau",
    "banau",
    "banao",
    "banbau",
    "banbau",
    "chahiye",
    "btao",
    "batao",
    "mujhko",
    "tum",
    "tumhara",
    "ham",
    "hume",
    "karke",
    "karne",
    "karunga",
    "karungi",
    "karu",
    "krunga",
    "krungi",
    "se",
    "ko",
    "mein",
    "mai",
}

STRONG_HINGLISH_MARKERS = {
    "main",
    "mujhe",
    "mujhko",
    "mera",
    "meri",
    "mere",
    "aap",
    "kaise",
    "kya",
    "kyu",
    "kyun",
    "kaun",
    "kab",
    "kahan",
    "hai",
    "hain",
    "nahi",
    "nahin",
    "chahiye",
}

ENGLISH_HINT_WORDS = {
    "how",
    "what",
    "when",
    "where",
    "why",
    "who",
    "can",
    "could",
    "should",
    "would",
    "please",
    "tell",
    "guide",
    "steps",
    "process",
    "apply",
    "application",
    "document",
    "documents",
    "required",
    "need",
    "passport",
    "voter",
    "license",
    "licence",
}


@lru_cache(maxsize=1)
def _get_llm() -> llmClient:
    return llmClient()


def _script_counts(text: str) -> tuple[int, int]:
    """Return (latin_count, devanagari_count) for a text snippet."""
    latin_chars = len(re.findall(r"[A-Za-z]", text or ""))
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text or ""))
    return latin_chars, devanagari_chars


def has_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def looks_hinglish(query: str) -> bool:
    tokens = re.findall(r"[a-zA-Z']+", query.lower())
    if not tokens:
        return False

    marker_count = sum(1 for t in tokens if t in HINGLISH_MARKERS)
    strong_marker_count = sum(1 for t in tokens if t in STRONG_HINGLISH_MARKERS)
    english_hint_count = sum(1 for t in tokens if t in ENGLISH_HINT_WORDS)

    # Any strong marker is usually enough to classify as Hinglish.
    if strong_marker_count >= 1:
        return True

    # Strong Hinglish signal from common romanized Hindi particles/verbs.
    if marker_count >= 2:
        return True

    # Single marker with no strong English structure is likely Hinglish.
    if marker_count == 1 and english_hint_count == 0:
        return True

    return False


def normalize_query_llm(query: str) -> str:
    # Always normalize to English for retrieval regardless of selected output language.
    # The documents are English, and BM25 + the English reranker need English words
    # (the multilingual embedder alone would cope, but the other two would not).
    # Output language is enforced separately by the LLM system instruction.
    # Used directly when ENABLE_QUERY_REWRITE is off; otherwise QueryProcessor
    # does translation + rewriting in a single call.
    query = (query or "").strip()

    if not query:
        return query

    if not has_devanagari(query) and not looks_hinglish(query):
        return query  # Already looks like English — skip normalization.

    prompt = f"""Translate the following query into English.

Query:
{query}

Output rules:
- Return only the English translation of the query.
- No explanation, no quotes, no extra text.
"""
    normalizer_instruction = (
        "You are a strict text translator. Return exactly one short translated query line only. "
        "Do not answer the user question. Do not add bullets, labels, or explanation."
    )
    try:
        normalized = _get_llm().generate(
            prompt=prompt,
            system_instruction=normalizer_instruction,
        ).strip().strip('"').strip("'")

        if is_llm_error(normalized):
            return query

        latin_count, devanagari_count = _script_counts(normalized)
        if devanagari_count > 0 and devanagari_count >= latin_count:
            logger.warning("Discarded normalization: Devanagari still present in output.")
            return query

        if len(normalized) > max(len(query) * 3, len(query) + 120):
            logger.warning("Discarded normalization output due to excessive length.")
            return query

        return normalized
    except Exception:
        return query
    


@lru_cache(maxsize=1000)
def normalize_query_llm_cached(query: str) -> str:
    return normalize_query_llm(query)