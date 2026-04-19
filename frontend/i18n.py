"""Simple CSV-based UI translation helpers."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

_TRANSLATION_FILE = Path(__file__).resolve().parent / "translations_en_hi.csv"


@lru_cache(maxsize=1)
def _load_en_hi_map() -> dict[str, str]:
    """Load English to Hindi mappings from CSV once per process."""
    if not _TRANSLATION_FILE.exists():
        return {}

    mapping: dict[str, str] = {}
    with _TRANSLATION_FILE.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            english = (row.get("english") or "").strip()
            hindi = (row.get("hindi") or "").strip()
            if english and hindi:
                mapping[english] = hindi
    return mapping


def t(text: str, language: str) -> str:
    """Translate English UI text to Hindi if language is hi, else return as-is."""
    if language != "hi":
        return text
    return _load_en_hi_map().get(text, text)
