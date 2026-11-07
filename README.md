# 🏛️ InfoBridge: Smart Access to Public Services

**SMAI Assignment 3 — T10.7: Multi-Service Government RAG Chatbot**

InfoBridge is a Retrieval-Augmented Generation (RAG) chatbot for Indian government services. It retrieves information from official PDFs, formats citations, and generates grounded answers in a Streamlit interface.

## Features

- RAG pipeline over five service areas: Passport, Voter ID / EPIC, Driving Licence & RC, Income Tax, and Ayushman Bharat.
- Gemini-powered LLM responses through `src/llm/client.py` (REST API, no extra SDK), with `MODEL` and `FALLBACK_MODELS` support. Groq still works with `LLM_PROVIDER=groq`.
- Hybrid retrieval: BM25 keywords and FAISS vectors, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder.
- Multilingual embeddings (`multilingual-e5-small`) for Hindi.
- Sentence-level citations `[1]` that link to the exact PDF page, plus a check that each cited sentence is supported by its source.
- Follow-up questions are rewritten into standalone questions before search, and each question is routed to the most likely service.
- Newer documents are preferred (assessment-year metadata for tax documents).
- Guardrails: Aadhaar, PAN, mobile and e-mail redaction, and prompt-injection refusal.
- An automatic evaluation harness in `eval/` that needs no hand-labelled data (see `eval/README.md`).
- Conversation memory for follow-up questions.
- English/Hindi UI support through the CSV translation layer in `frontend/translations_en_hi.csv`.
- Sidebar with a collapsible left strip, theme persistence, service filtering, and a localized service selector.
- Streamlit chat interface with reusable styling in `frontend/styles.py`.

## Architecture

```
PDFs -> PDF extraction -> chunking -> embeddings -> FAISS index
                                         |
                                         v
User query -> guardrails -> query rewrite/translate/route
           -> BM25 + FAISS -> RRF -> cross-encoder rerank -> max 3 chunks/PDF
           -> numbered context -> Gemini LLM -> [n] citations + support check -> answer + sources
```

## Project Structure

```
InfoBridge/
├── app.py                      # Streamlit entrypoint
├── config/
│   └── settings.py             # Central configuration
├── frontend/
│   ├── app_shell.py            # Top-level UI orchestration
│   ├── chat.py                 # Chat rendering and response flow
│   ├── i18n.py                # CSV-based UI translations
│   ├── state.py               # Streamlit session state helpers
│   ├── styles.py              # Theme and layout styles
│   ├── ui_components.py       # Sidebar/header/source UI
│   └── translations_en_hi.csv # English-Hindi label mapping
├── src/
│   ├── data_processing/
│   │   ├── pdf_extractor.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── llm/
│   │   ├── client.py          # Gemini / Groq client
│   │   ├── citations.py       # [n] markers → page citations
│   │   ├── prompts.py
│   │   └── generator.py
│   ├── memory/
│   │   └── chat_memory.py
│   ├── retrieval/
│   │   ├── retriever.py       # hybrid search, RRF, rerank, per-PDF cap
│   │   ├── bm25.py
│   │   ├── reranker.py
│   │   └── query_processor.py # follow-up rewrite, translation, routing
│   ├── guardrails.py
│   └── vectorstore/
│       └── store.py
├── scripts/
│   └── build_index.py
├── data/
│   ├── passport/
│   ├── voter/
│   ├── driving/
│   ├── tax/
│   └── health/
│  
├── utility
│   ├── language.py   
│ 
├── vectorstore_data/
└── requirements.txt
```

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Copy `.env.example` to `.env` and paste your Gemini key (from https://aistudio.google.com/apikey):

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
MODEL=gemini-3.6-flash
FALLBACK_MODELS=gemini-3.8-flash,gemini-3.5-flash-lite
```

`.env` is git-ignored. Never commit real keys.

### 4. Add PDFs

Place government PDFs in the appropriate service folders:

```
data/
├── passport/    # Passport instruction booklets, FAQs
├── voter/       # Voter registration forms, guides
├── driving/     # DL application guides, Parivahan docs
├── tax/         # ITR form instructions, tax guides
└── health/      # Ayushman Bharat guidelines, PM-JAY docs
```

### 5. Build the FAISS index

```bash
python scripts/build_index.py
```

### 6. Run the app

```bash
streamlit run app.py
```

## Technology Stack

| Component       | Technology                           |
|-----------------|---------------------------------------|
| Frontend        | Streamlit                             |
| Backend         | Python                                |
| Embeddings      | intfloat/multilingual-e5-small        |
| Keyword search  | BM25 (scikit-learn + scipy)           |
| Reranker        | cross-encoder/ms-marco-MiniLM-L-6-v2  |
| Vector Store    | FAISS (faiss-cpu)                     |
| LLM             | Google Gemini (REST)                  |
| PDF Parsing     | PyMuPDF (+ optional Tesseract OCR)    |
| Text Splitting  | LangChain Text Splitters              |

## UI Notes

- The sidebar is collapsible and leaves a narrow left strip when closed.
- Dark mode and light mode are persisted in session state.
- The language selector is English/Hindi only, and the labels are translated from the CSV file.
- The service filter is localized to the selected UI language.



## Evaluation

The `eval/` folder generates its own test data from the PDFs and measures retrieval (Hit@k, MRR, nDCG), answer faithfulness, relevance and correctness (LLM judge), refusal of out-of-scope questions, Hindi and follow-up retrieval, and latency. Each phase has a preset, so the results form an ablation table. See [eval/README.md](eval/README.md).

## Troubleshooting

- If the UI text looks stale, restart Streamlit and hard refresh the browser.
- If the app cannot answer, rebuild the index and confirm `GEMINI_API_KEY` is set in `.env`.
- "The index was built with ... but EMBEDDING_MODEL is ...": rebuild with `python scripts/build_index.py`, or set `EMBEDDING_MODEL` back to the model that built the index.
- 429 errors: the Gemini free tier limits requests per minute. Wait a minute, or add another model to `FALLBACK_MODELS`.
