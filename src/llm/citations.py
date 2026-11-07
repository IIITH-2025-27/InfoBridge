"""
Sentence-level citations (Phase 5)

The LLM is asked to end each sentence with markers such as [1] or [2][3] that
refer to the numbered sources in its context. This module:

1. keeps only valid markers and renumbers them 1..n in order of first use,
2. builds the matching source list (file + page) for the UI,
3. checks that each cited English sentence is supported by its cited chunk(s)
   (content-word overlap), so unsupported claims can be flagged.
"""

import re
from dataclasses import dataclass, field

from config.settings import CITATION_SUPPORT_THRESHOLD

_MARKER_GROUP = re.compile(r"(?:\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])+")
_NUMBERS = re.compile(r"\d+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "been", "by", "with", "as", "at", "from", "it", "this", "that",
    "these", "those", "which", "you", "your", "can", "will", "shall", "should",
    "must", "may", "has", "have", "had", "if", "any", "all", "not", "also", "such",
    "their", "they", "its", "into", "about", "under", "per", "then", "than", "there",
    "do", "does", "need", "needs", "required", "following", "following:", "etc",
    "please", "visit", "official", "portal", "i", "we", "our",
}


@dataclass
class CitationResult:
    answer: str
    sources: list[dict]
    unsupported_sentences: list[str] = field(default_factory=list)
    checked_sentences: int = 0
    used_inline_markers: bool = False


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _source_entry(number: int, result: dict) -> dict:
    meta = result.get("metadata", {})
    return {
        "number": number,
        "source_file": str(meta.get("source_file", "Unknown")),
        "source_path": str(meta.get("source_path", "")),
        "page": str(meta.get("page", "?")),
        "service_type": str(meta.get("service_type", "unknown")),
        "assessment_year": meta.get("assessment_year"),
        "relevance_score": float(result.get("score", 0.0)),
    }


def strip_markers(text: str) -> str:
    """Remove [n] markers (used before storing an answer in chat memory)."""
    return _MARKER_GROUP.sub("", text or "").strip()


def apply_citations(answer: str, context_results: list[dict], language: str = "en") -> CitationResult:
    """
    Args:
        answer: Raw LLM answer, possibly containing [n] markers.
        context_results: The results shown to the LLM, in source-number order.
        language: Answer language; the support check only runs for English.
    """
    valid_range = range(1, len(context_results) + 1)
    renumber: dict[int, int] = {}

    def _rewrite(match: re.Match) -> str:
        numbers = []
        for raw in _NUMBERS.findall(match.group(0)):
            original = int(raw)
            if original not in valid_range:
                continue
            if original not in renumber:
                renumber[original] = len(renumber) + 1
            if renumber[original] not in numbers:
                numbers.append(renumber[original])
        if not numbers:
            return ""
        return " " + "".join(f"[{n}]" for n in numbers)

    rewritten = _MARKER_GROUP.sub(_rewrite, answer or "")
    # "...fee is ₹1500. [2]" → "...fee is ₹1500 [2]." so the marker stays with its sentence
    rewritten = re.sub(r"([.!?।])((?:\s*\[\d+\])+)", r"\2\1", rewritten)
    rewritten = re.sub(r"[ \t]+([.,;:!?।])", r"\1", rewritten)

    if not renumber:
        # Model ignored the citation instruction: list every source it was given,
        # one entry per (file, page), so the user still sees where it came from.
        sources, seen = [], set()
        for result in context_results:
            key = (result["metadata"].get("source_file"), result["metadata"].get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(_source_entry(len(sources) + 1, result))
        return CitationResult(answer=rewritten.strip(), sources=sources)

    by_new_number = {new: context_results[old - 1] for old, new in renumber.items()}
    sources = [_source_entry(n, by_new_number[n]) for n in sorted(by_new_number)]

    unsupported, checked = [], 0
    if language == "en":
        unsupported, checked = _check_support(rewritten, by_new_number)

    return CitationResult(
        answer=rewritten.strip(),
        sources=sources,
        unsupported_sentences=unsupported,
        checked_sentences=checked,
        used_inline_markers=True,
    )


def _check_support(answer: str, by_number: dict[int, dict]) -> tuple[list[str], int]:
    """
    A cited sentence counts as supported when at least CITATION_SUPPORT_THRESHOLD
    of its content words appear in the chunk(s) it cites. Cheap, local and
    deterministic; the LLM-judge faithfulness metric in eval/ is the thorough check.
    """
    unsupported, checked = [], 0

    for sentence in _SENTENCE_SPLIT.split(answer):
        cited = [int(n) for group in _MARKER_GROUP.findall(sentence) for n in _NUMBERS.findall(group)]
        cited = [n for n in cited if n in by_number]
        if not cited:
            continue

        words = _content_words(strip_markers(sentence))
        if len(words) < 3:
            continue

        source_words: set[str] = set()
        for n in cited:
            source_words |= _content_words(by_number[n]["text"])

        checked += 1
        coverage = len(words & source_words) / len(words)
        if coverage < CITATION_SUPPORT_THRESHOLD:
            unsupported.append(strip_markers(sentence).strip())

    return unsupported, checked
