import logging
import re
from functools import lru_cache
from src.llm.client import llmClient

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


def _has_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def _is_llm_error_text(text: str) -> bool:
    lower = (text or "").lower()
    return (
        lower.startswith("error:")
        or "quota exceeded" in lower
        or "model not found" in lower
        or "failed to generate response" in lower
    )


def _looks_hinglish_heuristic(query: str) -> bool:
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
    # all-MiniLM-L6-v2 is an English embedding model; Hindi/Hinglish queries embed
    # poorly and produce low similarity scores against the indexed documents.
    # Output language is enforced separately by the LLM system instruction.
    query = (query or "").strip()

    if not query:
        return query

    if not _has_devanagari(query) and not _looks_hinglish_heuristic(query):
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

        if not normalized or _is_llm_error_text(normalized):
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