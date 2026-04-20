#!/usr/bin/env python3
"""
Build Index Script
One-time script to process PDFs, generate embeddings, and build the FAISS index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --data-dir /path/to/pdfs
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, SERVICE_CATEGORIES, VECTORSTORE_DIR
from src.data_processing.pdf_extractor import PDFExtractor
from src.data_processing.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_index(data_dir: Path = DATA_DIR) -> None:
    """
    Build the FAISS index from PDFs in the data directory.

    Steps:
    1. Extract text from all PDFs organized by service
    2. Chunk text into overlapping segments
    3. Generate embeddings
    4. Build and persist FAISS index
    """
    start_time = time.time()

    print("\n" + "=" * 60)
    print("🏛️  InfoBridge — Building Vector Index")
    print("=" * 60)

    # ── Step 1: Extract text from PDFs ────────────────────────────────────
    print(f"\n Data directory: {data_dir}")
    print("─" * 40)

    all_documents = []
    for service_key, service_info in SERVICE_CATEGORIES.items():
        service_dir = data_dir / service_key
        if not service_dir.exists():
            print(f"  {service_info['icon']} {service_info['name']}: "
                  f"directory not found ({service_dir})")
            continue

        print(f"\n  {service_info['icon']} Processing: {service_info['name']}")
        docs = PDFExtractor.extract_from_directory(
            dir_path=service_dir,
            service_type=service_key,
        )
        all_documents.extend(docs)
        print(f"     → {len(docs)} documents extracted")

    if not all_documents:
        print("\n No documents found! Please add PDFs to the data/ directory.")
        print("   Expected structure:")
        for key in SERVICE_CATEGORIES:
            print(f"     data/{key}/*.pdf")
        sys.exit(1)

    print(f"\n📄 Total documents: {len(all_documents)}")

    # ── Step 2: Chunk documents ───────────────────────────────────────────
    print("\n Chunking documents...")
    chunker = TextChunker()
    all_chunks = chunker.chunk_documents(all_documents)
    print(f"   → {len(all_chunks)} chunks created")

    if not all_chunks:
        print(" No chunks created. Check if PDF text extraction is working.")
        sys.exit(1)

    # ── Step 3: Generate embeddings ───────────────────────────────────────
    print("\nGenerating embeddings (this may take a moment)...")
    embedder = Embedder()
    embeddings, valid_chunks = embedder.encode_chunks(all_chunks)
    print(f"   → Embeddings shape: {embeddings.shape}")
    if len(valid_chunks) != len(all_chunks):
        print(f"   → Filtered weak/empty chunks: {len(all_chunks) - len(valid_chunks)}")

    # ── Step 4: Build and save FAISS index ────────────────────────────────
    print("\n📦 Building FAISS index...")
    vector_store = VectorStore()
    vector_store.add_documents(embeddings, valid_chunks)

    # Save to disk
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    stats = vector_store.get_service_stats()

    print("\n" + "=" * 60)
    print("✅ Index built successfully!")
    print("=" * 60)
    print(f"\n  📊 Summary:")
    print(f"     Total documents: {len(all_documents)}")
    print(f"     Total chunks:    {len(valid_chunks)}")
    print(f"     Index size:      {vector_store.size} vectors")
    print(f"     Time elapsed:    {elapsed:.1f}s")
    print(f"\n  📁 Chunks per service:")
    for svc, count in sorted(stats.items()):
        icon = SERVICE_CATEGORIES.get(svc, {}).get("icon", "📄")
        name = SERVICE_CATEGORIES.get(svc, {}).get("name", svc.title())
        print(f"     {icon} {name}: {count}")
    print(f"\n  💾 Saved to: {VECTORSTORE_DIR}")
    print(f"\n  🚀 Run the app: streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build InfoBridge FAISS index")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to data directory containing service subfolders",
    )
    args = parser.parse_args()

    build_index(data_dir=args.data_dir)
