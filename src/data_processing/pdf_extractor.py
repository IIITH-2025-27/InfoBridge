"""
PDF Text Extraction Module
Extracts and cleans text from government PDF documents.
"""

import re
import logging
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Handles extraction of text content from PDF files."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str | Path) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text as a single string.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            reader = PdfReader(str(pdf_path))
            pages_text = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    cleaned = PDFExtractor._clean_page_text(text, page_num)
                    if cleaned.strip():
                        pages_text.append(cleaned)

            full_text = "\n\n".join(pages_text)
            logger.info(
                f"Extracted {len(pages_text)} pages from {pdf_path.name} "
                f"({len(full_text)} characters)"
            )
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

    @staticmethod
    def _clean_page_text(text: str, page_num: int) -> str:
        """
        Clean extracted page text by removing noise.

        Args:
            text: Raw extracted text from a page.
            page_num: Page number (for reference).

        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)

        # Remove page numbers (common patterns)
        text = re.sub(r"\n\s*Page\s*\d+\s*(?:of\s*\d+)?\s*\n", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # Remove common header/footer patterns
        text = re.sub(r"(?i)^\s*(confidential|draft|internal)\s*$", "", text, flags=re.MULTILINE)

        # Remove URLs that are just noise (but keep informative ones)
        # text = re.sub(r"https?://\S+", "", text)  # Keep URLs as they may be useful

        # Fix multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix spacing after periods
        text = re.sub(r"\.([A-Z])", r". \1", text)

        return text.strip()

    @staticmethod
    def extract_from_directory(
        dir_path: str | Path,
        service_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Extract text from all PDFs in a directory.

        Args:
            dir_path: Path to directory containing PDFs.
            service_type: Optional service category label.

        Returns:
            List of dicts with keys: text, source_file, service_type.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        documents = []
        pdf_files = sorted(dir_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {dir_path}")
            return []

        for pdf_file in pdf_files:
            try:
                text = PDFExtractor.extract_text_from_pdf(pdf_file)
                if text.strip():
                    documents.append({
                        "text": text,
                        "source_file": pdf_file.name,
                        "source_path": str(pdf_file),
                        "service_type": service_type or dir_path.name,
                    })
                    logger.info(f"✓ Processed: {pdf_file.name}")
                else:
                    logger.warning(f"✗ Empty text from: {pdf_file.name}")
            except Exception as e:
                logger.error(f"✗ Failed to process {pdf_file.name}: {e}")

        logger.info(
            f"Extracted {len(documents)} documents from {dir_path} "
            f"(service: {service_type or dir_path.name})"
        )
        return documents
