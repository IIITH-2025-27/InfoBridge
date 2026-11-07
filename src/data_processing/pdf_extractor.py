"""
PDF Text Extraction Module (PRODUCTION SAFE)
- PyMuPDF for primary extraction
- OPTIONAL OCR fallback (auto-detected)
- Graceful degradation if OCR not installed
"""

import re
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# =========================
# OPTIONAL OCR SETUP
# =========================

try:
    import pytesseract
    from PIL import Image

    TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

    if TESSERACT_AVAILABLE:
        logger.info("✓ Tesseract OCR available")
    else:
        logger.warning("⚠️ Tesseract not found → OCR disabled")

except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("⚠️ pytesseract/PIL not installed → OCR disabled")


OCR_DPI = 200


def _ocr_languages() -> str:
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    return "eng+hin" if "hin" in installed else "eng"


class PDFExtractor:
    """Extracts structured text from PDFs with optional OCR fallback."""

    # =========================
    # PUBLIC METHODS
    # =========================

    @staticmethod
    def extract_from_pdf(pdf_path: str | Path) -> List[Dict]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        documents = []

        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc, start=1):

                text = page.get_text()
                used_ocr = False

                # Step 1: Check if text is weak
                if PDFExtractor._is_bad_text(text):

                    # Step 2: OCR fallback (only if available). Scanned pages are
                    # images; a page with no text and no images is truly blank.
                    if TESSERACT_AVAILABLE and len(page.get_images()) > 0:
                        logger.warning(f"⚠️ OCR fallback on page {page_num}")
                        text = PDFExtractor._run_ocr(page)
                        used_ocr = True
                    else:
                        logger.warning(
                            f"⚠️ Skipping OCR on page {page_num} (not available or no images)"
                        )

                # Step 3: Final check
                if PDFExtractor._is_bad_text(text):
                    logger.warning(f"Skipping page {page_num} (no useful text)")
                    continue

                cleaned = PDFExtractor._clean_text(text)

                documents.append({
                    "text": cleaned,
                    "page": page_num,
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "service_type": pdf_path.parent.name,
                    "ocr": used_ocr,
                })

            logger.info(
                f"✓ Extracted {len(documents)} usable pages from {pdf_path.name}"
            )

        except Exception as e:
            logger.error(f"✗ Error processing {pdf_path}: {e}")
            raise

        return documents

    @staticmethod
    def extract_from_directory(
        dir_path: str | Path,
        service_type: Optional[str] = None
    ) -> List[Dict]:

        dir_path = Path(dir_path)

        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        all_docs = []
        pdf_files = sorted(dir_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDFs found in {dir_path}")
            return []

        for pdf_file in pdf_files:
            try:
                docs = PDFExtractor.extract_from_pdf(pdf_file)

                for d in docs:
                    if service_type:
                        d["service_type"] = service_type

                all_docs.extend(docs)
                logger.info(f"✓ Processed: {pdf_file.name}")

            except Exception as e:
                logger.error(f"✗ Failed: {pdf_file.name} | {e}")

        logger.info(f"Total extracted pages: {len(all_docs)}")
        return all_docs

    # =========================
    # INTERNAL HELPERS
    # =========================

    @staticmethod
    def _is_bad_text(text: str) -> bool:
        """Check if extracted text is too weak."""
        if not text:
            return True

        text = text.strip()

        # Phase 2: was < 50 chars / < 10 words, which skipped short form pages.
        if len(text) < 20:
            return True

        if len(text.split()) < 4:
            return True

        return False

    @staticmethod
    def _run_ocr(page) -> str:
        """Run OCR safely."""
        try:
            # Default pixmap is 72 DPI, too low for Tesseract; 200 DPI reads well.
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # eng+hin when the Hindi language pack is installed, else English only
            return pytesseract.image_to_string(img, lang=_ocr_languages())
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean noisy government PDF text."""

        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove page numbers
        text = re.sub(r"\n\s*Page\s*\d+.*?\n", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # Remove decorative symbols
        text = re.sub(r"\*{2,}", "", text)
        text = re.sub(r"[⮚•►]", "", text)

        # Remove bracket headers
        text = re.sub(r"\[.*?\]", "", text)

        # Remove only noisy parentheses (safe)
        text = re.sub(r"\((CPV.*?|Division.*?)\)", "", text)

        # Fix excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()