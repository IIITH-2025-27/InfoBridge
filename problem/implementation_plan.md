# InfoBridge: T10.7 Multi-Service Government RAG Chatbot

Building a Retrieval-Augmented Generation (RAG) chatbot that answers queries about Indian government services by retrieving information from official PDFs. This is the T10.7 variant — a Tier 2 multi-service unified chatbot with Hindi support, conversation memory, and source citations.

**Key requirements from assignment:**
- 5+ government service sources combined (Passport, Voter ID, Driving Licence, Income Tax, Ayushman Bharat)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (free, CPU)
- Vector store: `faiss-cpu` or `chromadb`
- LLM: Gemini API (free Google AI Studio)
- Language toggle (English/Hindi)
- Conversation memory
- Source PDF citations in every answer
- Streamlit chat UI

> [!IMPORTANT]
> The **Data Team** part (collecting/downloading actual government PDFs) is excluded from this implementation — the user will handle that separately. The code will be structured to work with any PDFs placed in the `data/` directory.

## Proposed Changes

### Project Structure

```
InfoBridge/
├── app.py                          # Main Streamlit entry point
├── config/
│   └── settings.py                 # All configuration constants
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py        # Extract text from PDFs
│   │   └── chunker.py              # Split text into overlapping chunks with metadata
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py             # Generate embeddings using MiniLM
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── store.py                # FAISS vector store operations
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py            # Query processing & top-k retrieval
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── gemini_client.py        # Gemini API wrapper
│   │   ├── prompts.py              # Prompt templates for RAG + Hindi
│   │   └── generator.py            # Response generation orchestrator
│   └── memory/
│       ├── __init__.py
│       └── chat_memory.py          # Conversation memory management
├── scripts/
│   └── build_index.py              # One-time script to index all PDFs
├── data/                           # PDFs go here (organized by service)
│   ├── passport/
│   ├── voter/
│   ├── driving/
│   ├── tax/
│   └── health/
├── vectorstore_data/               # Persisted FAISS index (auto-generated)
├── requirements.txt
├── .env.example                    # API key template
└── README.md
```

---

### Configuration

#### [NEW] [settings.py](file:///home/vikash/Desktop/git_folders/InfoBridge/config/settings.py)

Central configuration for all components:
- `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2`
- `CHUNK_SIZE`: 500 words, `CHUNK_OVERLAP`: 50 words
- `TOP_K`: 5 (number of retrieved chunks)
- `FAISS_INDEX_PATH`: path to persisted index
- `GEMINI_MODEL`: `gemini-1.5-flash`
- `SERVICE_CATEGORIES`: mapping of service names to data directories
- `MAX_MEMORY_TURNS`: 10 (conversation memory window)

---

### Data Processing Module

#### [NEW] [pdf_extractor.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/data_processing/pdf_extractor.py)

- Use `PyPDF2` to extract text from PDFs
- Handle encoding issues, multi-page documents
- Clean extracted text: remove headers/footers/page numbers, fix whitespace
- Return structured output: `{text, source_file, service_type}` per PDF

#### [NEW] [chunker.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/data_processing/chunker.py)

- Split cleaned text into overlapping chunks (500 words, 50-word overlap)
- Attach metadata to each chunk: `{chunk_id, source_pdf, service_type, page_numbers}`
- Use `langchain.text_splitter.RecursiveCharacterTextSplitter` for robust splitting

---

### Embedding Module

#### [NEW] [embedder.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/embeddings/embedder.py)

- Load `sentence-transformers/all-MiniLM-L6-v2` model
- Batch-encode text chunks into 384-dimensional vectors
- Provide `encode_query()` and `encode_documents()` methods
- Cache model to avoid reloading

---

### Vector Store Module

#### [NEW] [store.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/vectorstore/store.py)

- Build FAISS `IndexFlatIP` (inner product for cosine similarity on normalized vectors)
- Store metadata alongside vectors using a parallel pickle file
- Provide `add_documents()`, `search()`, `save()`, `load()` methods
- Persist index to disk at `vectorstore_data/`

---

### Retrieval Module

#### [NEW] [retriever.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/retrieval/retriever.py)

- Accept user query → embed with MiniLM → search FAISS → return top-k chunks with metadata
- Optional: filter by service category before retrieval
- Re-rank results by relevance score
- Return formatted context + source citations

---

### LLM Integration Module

#### [NEW] [gemini_client.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/llm/gemini_client.py)

- Wrapper around `google-generativeai` SDK
- Configure `gemini-1.5-flash` model
- Handle API key from environment variable
- Implement `generate()` with temperature/safety settings

#### [NEW] [prompts.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/llm/prompts.py)

- **English RAG prompt**: System instruction to answer ONLY from provided context, cite sources, say "I don't have information about this" if not found
- **Hindi RAG prompt**: Same logic but instruct to respond in Hindi
- **Conversation-aware prompt**: Include recent chat history for context continuity
- **Service-specific prompt prefix**: Adapt tone based on service category

#### [NEW] [generator.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/llm/generator.py)

- Orchestrate the full generation pipeline:
  1. Retrieve relevant chunks via retriever
  2. Format context with source citations
  3. Build prompt with conversation history + language preference
  4. Call Gemini API
  5. Format response with source references
- Return structured response: `{answer, sources[], language}`

---

### Conversation Memory

#### [NEW] [chat_memory.py](file:///home/vikash/Desktop/git_folders/InfoBridge/src/memory/chat_memory.py)

- Maintain sliding window of last N conversation turns
- Store as list of `{role, content}` dicts
- Integrate with Streamlit `session_state`
- Provide `add_turn()`, `get_history()`, `clear()` methods

---

### Frontend - Streamlit App

#### [NEW] [app.py](file:///home/vikash/Desktop/git_folders/InfoBridge/app.py)

Full Streamlit chat application:
- **Header**: App title, description, branding
- **Sidebar**:
  - Service category filter (All / Passport / Voter ID / Driving Licence / Income Tax / Ayushman Bharat)
  - Language toggle (🇬🇧 English / 🇮🇳 Hindi)
  - "Clear Chat" button
  - Information about available services
- **Main area**:
  - Chat interface using `st.chat_input` and `st.chat_message`
  - Each bot response shows expandable source citations
  - Conversation history maintained across messages
- **Styling**: Custom CSS for a polished, government-portal look

---

### Scripts & Configuration Files

#### [NEW] [build_index.py](file:///home/vikash/Desktop/git_folders/InfoBridge/scripts/build_index.py)

One-time script to:
1. Scan `data/` directory for PDFs
2. Extract text from each PDF
3. Chunk text with metadata
4. Generate embeddings
5. Build and save FAISS index

#### [NEW] [requirements.txt](file:///home/vikash/Desktop/git_folders/InfoBridge/requirements.txt)

```
streamlit>=1.30.0
PyPDF2>=3.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
langchain>=0.1.0
langchain-text-splitters>=0.0.1
numpy>=1.24.0
```

#### [NEW] [.env.example](file:///home/vikash/Desktop/git_folders/InfoBridge/.env.example)

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

#### [MODIFY] [README.md](file:///home/vikash/Desktop/git_folders/InfoBridge/README.md)

Complete project README with:
- Project description & T10.7 context
- Features list
- Installation steps
- How to add PDFs and build the index
- How to run the app
- Architecture diagram
- Tech stack

---

## Verification Plan

### Automated Tests

1. **Test PDF Processing Pipeline**
   ```bash
   # Create a sample test PDF, then run:
   python -c "from src.data_processing.pdf_extractor import PDFExtractor; print(PDFExtractor.extract('data/test_sample.pdf'))"
   ```

2. **Test Chunking**
   ```bash
   python -c "from src.data_processing.chunker import TextChunker; chunks = TextChunker.chunk('long test text...', 'test.pdf', 'passport'); print(f'{len(chunks)} chunks created')"
   ```

3. **Test Embedding Generation**
   ```bash
   python -c "from src.embeddings.embedder import Embedder; e = Embedder(); v = e.encode_query('How to apply for passport?'); print(f'Vector shape: {v.shape}')"
   ```

4. **Test FAISS Index Build & Search**
   ```bash
   python -c "from src.vectorstore.store import VectorStore; vs = VectorStore(); vs.add_documents(['test chunk'], [{'source': 'test.pdf'}]); results = vs.search('test query'); print(results)"
   ```

5. **Test Full End-to-End Pipeline** (requires Gemini API key)
   ```bash
   python scripts/build_index.py
   streamlit run app.py
   ```

### Manual Verification

1. **Run the Streamlit app** (`streamlit run app.py`) and test:
   - Ask "How do I apply for a passport?" → should get relevant answer with source citations
   - Toggle language to Hindi → response should be in Hindi
   - Ask a follow-up question → should use conversation memory for context
   - Switch service filter → retrieval should prioritize that service
   - Ask an out-of-scope question → should respond with "I don't have information about this"
   - Verify chat history displays correctly
   - Click "Clear Chat" → should reset conversation

2. **User to provide sample PDFs**: Place at least 1-2 PDFs per service in `data/` subfolders, then run `python scripts/build_index.py` to build the index.
