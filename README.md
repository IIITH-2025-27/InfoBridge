# 🏛️ InfoBridge: Smart Access to Public Services

**SMAI Assignment 3 — T10.7: Multi-Service Government RAG Chatbot**

InfoBridge is a Retrieval-Augmented Generation (RAG) chatbot for Indian government services. It retrieves information from official PDFs, formats citations, and generates grounded answers in a Streamlit interface.

## Features

- RAG pipeline over five service areas: Passport, Voter ID / EPIC, Driving Licence & RC, Income Tax, and Ayushman Bharat.
- Groq-powered LLM responses through `src/llm/client.py` with `MODEL` and `FALLBACK_MODELS` support.
- Source citations for every answer.
- Conversation memory for follow-up questions.
- English/Hindi UI support through the CSV translation layer in `frontend/translations_en_hi.csv`.
- Sidebar with a collapsible left strip, theme persistence, service filtering, and a localized service selector.
- Streamlit chat interface with reusable styling in `frontend/styles.py`.

## Architecture

```
PDFs -> PDF extraction -> chunking -> embeddings -> FAISS index
                                         |
                                         v
User query -> retrieval -> context + citations -> Groq LLM -> answer + sources
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
│   │   ├── client.py          # Groq client
│   │   ├── prompts.py
│   │   └── generator.py
│   ├── memory/
│   │   └── chat_memory.py
│   ├── retrieval/
│   │   └── retriever.py
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

Create a `.env` file in the project root and add:

```bash
API_KEY=your_groq_api_key_here
MODEL=llama3-8b-8192
FALLBACK_MODELS=llama3-70b-8192,mixtral-8x7b-32768
```

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
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2|
| Vector Store    | FAISS (faiss-cpu)                     |
| LLM             | Groq chat completions               |
| PDF Parsing     | PyPDF2                                |
| Text Splitting  | LangChain Text Splitters              |

## UI Notes

- The sidebar is collapsible and leaves a narrow left strip when closed.
- Dark mode and light mode are persisted in session state.
- The language selector is English/Hindi only, and the labels are translated from the CSV file.
- The service filter is localized to the selected UI language.



## Evaluation

- Responses are grounded in retrieved context.
- Every assistant answer includes source citations.
- The system supports follow-up questions through conversation memory.

## Troubleshooting

- If the UI text looks stale, restart Streamlit and hard refresh the browser.
- If the app cannot answer, rebuild the index and confirm `API_KEY` is set in `.env`.
