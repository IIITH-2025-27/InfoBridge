# 🏛️ InfoBridge: Smart Access to Public Services

**SMAI Assignment 3 — T10.7: Multi-Service Government RAG Chatbot**

InfoBridge is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indian government services by retrieving information from official PDF documents and generating accurate, source-backed responses.

## ✨ Features

- 🔍 **RAG Pipeline** — Retrieves relevant information from official government PDFs
- 🤖 **Gemini AI** — Powered by Google Gemini 2.0 Flash for accurate responses
- 🌐 **Hindi Support** — Language toggle for English and Hindi responses
- 💬 **Conversation Memory** — Context-aware follow-up questions
- 📎 **Source Citations** — Every answer comes with document references
- 🛟 **Graceful Fallback** — Shows retrieval-based, cited snippets when LLM quota/model issues occur
- 🗂️ **Multi-Service** — Covers 5 government services:
  - 🛂 Passport Services
  - 🗳️ Voter ID / EPIC
  - 🚗 Driving Licence & RC
  - 💰 Income Tax
  - 🏥 Ayushman Bharat (PM-JAY)

## 🏗️ Architecture

```
PDFs → Text Extraction → Chunking → Embeddings (MiniLM) → FAISS Index
                                                              ↓
User Query → Embedding → Similarity Search → Context → Gemini LLM → Response + Sources
```

## 📁 Project Structure

```
InfoBridge/
├── app.py                      # Streamlit chat interface
├── config/settings.py          # Configuration constants
├── src/
│   ├── data_processing/
│   │   ├── pdf_extractor.py    # PDF text extraction
│   │   └── chunker.py          # Text chunking with metadata
│   ├── embeddings/
│   │   └── embedder.py         # MiniLM embedding generation
│   ├── vectorstore/
│   │   └── store.py            # FAISS vector store
│   ├── retrieval/
│   │   └── retriever.py        # Query & retrieval pipeline
│   ├── llm/
│   │   ├── gemini_client.py    # Gemini API wrapper
│   │   ├── prompts.py          # EN/HI prompt templates
│   │   └── generator.py        # Response generation orchestrator
│   └── memory/
│       └── chat_memory.py      # Conversation memory
├── scripts/
│   └── build_index.py          # Index building script
├── data/                       # PDF documents (organized by service)
├── vectorstore_data/           # Persisted FAISS index
├── requirements.txt
└── .env.example
```

##  Quick Start

> **⚠️ All commands should be run inside a virtual environment.**

### 1. Create & Activate Virtual Environment

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it (Linux / macOS)
source .venv/bin/activate

# Activate it (Windows)
# .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Key

```bash
cp .env.example .env
# Edit .env and add your Google AI Studio API key
# Optional: set GEMINI_MODEL (default is gemini-2.0-flash)
# Optional: set GEMINI_FALLBACK_MODELS for automatic model retry
# Get one free at: https://aistudio.google.com/app/apikey
```

### 4. Add PDF Documents

Place government PDFs in the appropriate service folders:

```
data/
├── passport/    # Passport instruction booklets, FAQs
├── voter/       # Voter registration forms, guides
├── driving/     # DL application guides, Parivahan docs
├── tax/         # ITR form instructions, tax guides
└── health/      # Ayushman Bharat guidelines, PM-JAY docs
```

### 5. Build the Index

```bash
python scripts/build_index.py
```

### 6. Run the App

```bash
streamlit run app.py
```

## 🛠️ Technology Stack

| Component       | Technology                           |
|-----------------|---------------------------------------|
| Frontend        | Streamlit                             |
| Backend         | Python                                |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2|
| Vector Store    | FAISS (faiss-cpu)                     |
| LLM             | Google Gemini 2.0 Flash               |
| PDF Parsing     | PyPDF2                                |
| Text Splitting  | LangChain Text Splitters              |

## 📊 Evaluation

- **Accuracy**: Responses are grounded in retrieved context only
- **Relevance**: FAISS cosine similarity for chunk retrieval
- **Latency**: Optimized for < 3 second response time
- **Source Transparency**: Every response includes document citations


## 🙏 Acknowledgements

- Google Gemini API for response generation
- Sentence Transformers for embedding models
- FAISS by Meta AI for vector similarity search
- Streamlit for the web interface
