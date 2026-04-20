import sys
from pathlib import Path

# 🔥 Fix import path FIRST (very important)
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_processing.pdf_extractor import PDFExtractor
from src.data_processing.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore

from src.llm.generator import ResponseGenerator
from src.llm.client import llmClient
from src.retrieval.retriever import Retriever



# =========================
# RETRIEVER (bridge class)
# =========================
# class Retriever:
#     def __init__(self, store, embedder):
#         self.store = store
#         self.embedder = embedder

#     def retrieve(self, query, top_k=5, service_filter=None):
#         q_emb = self.embedder.encode_query(query)
#         return self.store.search(q_emb, top_k, service_filter)

#     def format_context(self, results):
#         return "\n\n".join([r["text"] for r in results])

#     def get_source_citations(self, results):
#         return [r["metadata"] for r in results]


# =========================
# INITIALIZATION
# =========================
embedder = Embedder()
store = VectorStore()

# =========================
# LOAD OR BUILD VECTOR DB
# =========================
loaded = store.load()

if not loaded:
    print("\n🔨 Building vector store (first run only)...\n")

    docs = PDFExtractor.extract_from_directory("data/passport")
    print(f"Extracted pages: {len(docs)}")

    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings, valid_chunks = embedder.encode_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    store.add_documents(embeddings, valid_chunks)
    store.save()

    print("✅ Vector store built and saved!\n")

else:
    print("⚡ Loaded existing vector store!\n")


# =========================
# SETUP RAG PIPELINE
# =========================
retriever = Retriever(store, embedder)
generator = ResponseGenerator(retriever, llmClient())


# =========================
# QUERY LOOP
# =========================
print("💬 Ask your question (type 'exit' to quit)\n")

while True:
    query = input("👉 You: ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break

    response = generator.generate(query)
    temp = retriever.retrieve(query)

    print("\n🤖 Answer:\n")
    print(response["answer"])
    print("\n" + "=" * 60 + "\n")