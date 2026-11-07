#!/usr/bin/env python3
"""
Find PDFs in data/ that are missing from the index, and say why (Phase 2).

Usage:
    python scripts/check_unindexed.py
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF

from config.settings import DATA_DIR, METADATA_FILE, SERVICE_CATEGORIES
from src.data_processing.pdf_extractor import PDFExtractor, TESSERACT_AVAILABLE


def diagnose(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        return f"cannot open ({exc})"

    if doc.needs_pass:
        return "password-protected"

    pages = len(doc)
    text_pages = 0
    image_only_pages = 0
    total_chars = 0

    for page in doc:
        text = page.get_text()
        total_chars += len(text.strip())
        if not PDFExtractor._is_bad_text(text):
            text_pages += 1
        elif page.get_images():
            image_only_pages += 1

    if text_pages == 0 and image_only_pages > 0:
        hint = "will be OCR'd on next build" if TESSERACT_AVAILABLE else (
            "install Tesseract (brew install tesseract tesseract-lang) and pytesseract, then rebuild"
        )
        return f"scanned / image-only ({image_only_pages}/{pages} pages) → {hint}"

    if text_pages == 0:
        return f"no extractable text ({total_chars} chars over {pages} pages)"

    return (
        f"has text on {text_pages}/{pages} pages → probably dropped by the old chunk filter; "
        "will be indexed after rebuilding"
    )


def main() -> None:
    indexed: set[str] = set()
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "rb") as f:
            data = pickle.load(f)
        indexed = {m.get("source_file") for m in data.get("metadata", [])}
    else:
        print(f"No index found at {METADATA_FILE}; reporting on every PDF.\n")

    print(f"Tesseract OCR available: {TESSERACT_AVAILABLE}\n")

    missing_total = 0
    all_total = 0
    for service_key in SERVICE_CATEGORIES:
        service_dir = DATA_DIR / service_key
        pdfs = sorted(service_dir.glob("*.pdf")) if service_dir.exists() else []
        all_total += len(pdfs)
        missing = [p for p in pdfs if p.name not in indexed]
        missing_total += len(missing)

        print(f"{service_key}: {len(pdfs)} PDFs, {len(missing)} not indexed")
        for pdf in missing:
            print(f"   - {pdf.name}: {diagnose(pdf)}")

    print(f"\nTotal: {all_total} PDFs, {all_total - missing_total} indexed, {missing_total} missing")


if __name__ == "__main__":
    main()
