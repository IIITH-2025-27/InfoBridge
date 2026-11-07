"""
Document metadata helpers (Phase 5: version / date awareness).

Tax rules change every year, so each document gets a `doc_year` (and an
`assessment_year` like "2024-25" when it mentions one). The retriever uses these to
prefer the newest document, and the context shown to the LLM includes them.
"""

import re
from pathlib import Path
from typing import Optional

# "AY 2024-25", "A.Y. 2024-2025", "Assessment Year 2024-25", "AY2024-25", "AY 25-26"
_AY_PATTERN = re.compile(
    r"(?:\bA\.?\s?Y\.?|assessment\s+year)\s*[:\-]?\s*((?:20)?\d{2})\s*[-–/]\s*(\d{2,4})",
    re.IGNORECASE,
)
# "FY 2023-24", "Financial Year 2023-24"
_FY_PATTERN = re.compile(
    r"(?:\bF\.?\s?Y\.?|financial\s+year|previous\s+year)\s*[:\-]?\s*(20\d{2})\s*[-–/]\s*(\d{2,4})",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"(?<!\d)(20[0-3]\d)(?!\d)")


def _format_range(start: str, end: str) -> str:
    return f"{start}-{end[-2:]}"


def title_from_filename(source_file: str) -> str:
    """'Common ITR Filing FAQs AY 2024-25.pdf' → 'Common ITR Filing FAQs AY 2024-25'."""
    stem = Path(source_file).stem
    return re.sub(r"[_]+", " ", stem).strip()


_VERSION_TOKENS = re.compile(
    r"\b(?:a\.?\s?y\.?|f\.?\s?y\.?|assessment\s+year|financial\s+year)\s*(?:(?:20)?\d{2}\s*[-–/]\s*\d{2,4})?"
    r"|\(\d+\)"  # "(1)" added to downloaded copies
    r"|\b20\d{2}(?:\s*[-–/]\s*\d{2,4})?\b"
    r"|\bv(?:er(?:sion)?)?\s*\d+(?:\.\d+)*\b"
    r"|\b(?:latest|updated|revised|new|old)\b",
    re.IGNORECASE,
)


def document_family(source_file: str) -> str:
    """
    Title with years and version markers removed, so yearly editions of the same
    document share a key: "Common ITR Filing FAQs AY 2024-25" and "... AY 2025-26"
    → "common itr filing faqs". Different forms stay different ("itr form 2" vs
    "itr form 3"), because only years / versions are removed, not other numbers.
    """
    title = _VERSION_TOKENS.sub(" ", title_from_filename(source_file).lower())
    return re.sub(r"[^a-z0-9ऀ-ॿ]+", " ", title).strip()


def infer_doc_dates(source_file: str, text: str = "") -> dict:
    """
    Infer year information from a file name and (optionally) its first pages.

    Returns:
        {"doc_year": int | None, "assessment_year": str | None}
        doc_year is the start year of the assessment year when one is found
        (AY 2025-26 → 2025), otherwise the latest plausible year mentioned.
    """
    sources = [title_from_filename(source_file), (text or "")[:3000]]

    for source in sources:
        match = _AY_PATTERN.search(source)
        if match:
            start = match.group(1)
            start = start if len(start) == 4 else f"20{start}"
            return {
                "doc_year": int(start),
                "assessment_year": _format_range(start, match.group(2)),
            }

    for source in sources:
        match = _FY_PATTERN.search(source)
        if match:
            # FY 2023-24 is assessed in AY 2024-25
            start = int(match.group(1)) + 1
            return {
                "doc_year": start,
                "assessment_year": f"{start}-{str(start + 1)[-2:]}",
            }

    # Fall back to plain years, file name first (more reliable than body text).
    for source in sources:
        years = [int(y) for y in _YEAR_PATTERN.findall(source)]
        if years:
            return {"doc_year": max(years), "assessment_year": None}

    return {"doc_year": None, "assessment_year": None}


def extract_query_year(query: str) -> Optional[int]:
    """Year the user explicitly asks about (AY/FY or a bare year), if any."""
    dates = infer_doc_dates("", query)
    return dates["doc_year"]
