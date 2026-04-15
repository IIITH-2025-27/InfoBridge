"""
Text Chunking Module
Splits extracted text into overlapping chunks with metadata for embedding.
"""

import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into overlapping chunks with metadata."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            is_separator_regex=False,
        )

    def chunk_text(
        self,
        text: str,
        source_file: str,
        service_type: str,
        source_path: Optional[str] = None,
    ) -> list[dict]:
        """
        Split text into chunks and attach metadata.

        Args:
            text: Full text content to chunk.
            source_file: Name of the source PDF file.
            service_type: Category of government service.
            source_path: Optional full path to source file.

        Returns:
            List of chunk dicts with keys: text, metadata.
        """
        if not text.strip():
            return []

        chunks_text = self.splitter.split_text(text)

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            if not chunk_text.strip():
                continue

            chunk = {
                "text": chunk_text.strip(),
                "metadata": {
                    "chunk_id": f"{source_file}_chunk_{idx}",
                    "source_file": source_file,
                    "source_path": source_path or "",
                    "service_type": service_type,
                    "chunk_index": idx,
                    "total_chunks": len(chunks_text),
                    "char_count": len(chunk_text),
                },
            }
            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from {source_file} "
            f"(avg {sum(c['metadata']['char_count'] for c in chunks) // max(len(chunks), 1)} chars/chunk)"
        )
        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Chunk all documents from the extraction pipeline.

        Args:
            documents: List of document dicts from PDFExtractor.

        Returns:
            List of all chunks with metadata.
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(
                text=doc["text"],
                source_file=doc["source_file"],
                service_type=doc["service_type"],
                source_path=doc.get("source_path"),
            )
            all_chunks.extend(chunks)

        logger.info(
            f"Total chunks created: {len(all_chunks)} from {len(documents)} documents"
        )
        return all_chunks
