"""
Text Chunking Module (RAG-Ready)
Splits PAGE-WISE text into high-quality chunks.
"""

import logging
from collections import defaultdict
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS
from src.data_processing.doc_metadata import infer_doc_dates, title_from_filename

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into overlapping chunks with metadata."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_words: int = MIN_CHUNK_WORDS,
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.min_chunk_words = min_chunk_words

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        # Year / AY metadata is per file, inferred from its name and first pages.
        first_pages: dict[str, list[str]] = defaultdict(list)
        for doc in documents:
            if len(first_pages[doc["source_file"]]) < 2:
                first_pages[doc["source_file"]].append(doc["text"])

        doc_dates = {
            source_file: infer_doc_dates(source_file, "\n".join(pages))
            for source_file, pages in first_pages.items()
        }

        all_chunks = []
        dropped = 0

        for doc in documents:
            chunks, doc_dropped = self._chunk_single(doc, doc_dates[doc["source_file"]])
            all_chunks.extend(chunks)
            dropped += doc_dropped

        logger.info(f"✓ Total chunks created: {len(all_chunks)} (dropped {dropped} tiny chunks)")
        return all_chunks

    def _chunk_single(self, doc: Dict, dates: Dict) -> tuple[List[Dict], int]:
        text = doc["text"]
        title = title_from_filename(doc["source_file"])

        raw_chunks = self.splitter.split_text(text)
        chunks = []
        dropped = 0

        for idx, chunk in enumerate(raw_chunks):
            chunk = chunk.strip()

            # Only drop near-empty fragments. The old 30-word rule removed most of
            # the form PDFs, whose useful content is short field lines.
            if len(chunk.split()) < self.min_chunk_words:
                dropped += 1
                continue

            chunks.append({
                "text": chunk,
                # Embedded text gets the document title so short chunks such as
                # form fields still match queries like "form 6" or "ITR-4".
                "embed_text": f"{title}\n{chunk}",
                "metadata": {
                    "chunk_id": f"{doc['source_file']}_p{doc['page']}_c{idx}",
                    "source_file": doc["source_file"],
                    "source_path": doc["source_path"],
                    "service_type": doc["service_type"],
                    "page": doc["page"],
                    "chunk_index": idx,
                    "char_count": len(chunk),
                    "title": title,
                    "doc_year": dates.get("doc_year"),
                    "assessment_year": dates.get("assessment_year"),
                    "ocr": doc.get("ocr", False),
                }
            })

        return chunks, dropped
