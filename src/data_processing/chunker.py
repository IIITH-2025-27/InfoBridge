"""
Text Chunking Module (RAG-Ready)
Splits PAGE-WISE text into high-quality chunks.
"""

import logging
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into overlapping chunks with metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []

        for doc in documents:
            chunks = self._chunk_single(doc)
            all_chunks.extend(chunks)

        logger.info(f"✓ Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _chunk_single(self, doc: Dict) -> List[Dict]:
        text = doc["text"]

        raw_chunks = self.splitter.split_text(text)
        chunks = []

        for idx, chunk in enumerate(raw_chunks):

            # 🔥 Filter weak chunks
            if len(chunk.split()) < 30:
                continue

            chunks.append({
                "text": chunk.strip(),
                "metadata": {
                    "chunk_id": f"{doc['source_file']}_p{doc['page']}_c{idx}",
                    "source_file": doc["source_file"],
                    "source_path": doc["source_path"],
                    "service_type": doc["service_type"],
                    "page": doc["page"],
                    "chunk_index": idx,
                    "char_count": len(chunk)
                }
            })

        return chunks