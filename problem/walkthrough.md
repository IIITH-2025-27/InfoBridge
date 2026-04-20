# InfoBridge Walkthrough

## Overview

InfoBridge is a Streamlit-based RAG chatbot for Indian government services. It retrieves relevant chunks from indexed PDFs, sends the formatted context to a Groq-backed LLM client, and renders the answer with source citations.

## How the App Works

### 1. Session State Setup

`frontend/state.py` initializes the session state for:

- chat messages
- conversation memory
- current language
- current service filter
- theme mode
- sidebar open/closed state
- response generator cache

### 2. Sidebar Rendering

`frontend/ui_components.py` renders the sidebar with:

- a collapsible left strip
- a light/dark theme button
- English/Hindi language selection
- a localized service filter
- system status and document counts
- a clear chat button

### 3. Styling

`frontend/styles.py` defines the app look and feel for both dark and light themes. The styling now covers:

- the collapsible sidebar strip
- the theme toggle button
- the language pills
- chat message foreground colors
- the chat composer background in light mode

### 4. Chat Flow

`frontend/chat.py` handles the user interaction loop:

1. Show previous messages.
2. Read the next prompt from `st.chat_input()`.
3. Retrieve documents for the query.
4. Generate an answer using the LLM client.
5. Render the answer and source citations.
6. Store the exchange in session memory.

### 5. Retrieval and Generation

The back end uses:

- `src/data_processing/pdf_extractor.py` for PDF text extraction
- `src/data_processing/chunker.py` for chunking
- `src/embeddings/embedder.py` for MiniLM embeddings
- `src/vectorstore/store.py` for FAISS persistence and search
- `src/retrieval/retriever.py` for query retrieval and structured citations
- `src/llm/client.py` and `src/llm/generator.py` for Groq-based response generation

## Important Current Behavior

- The app supports English and Hindi UI labels.
- The language selector uses translated labels from `frontend/translations_en_hi.csv`.
- The selected theme is persisted even when the sidebar is collapsed.
- The service filter is localized to the selected UI language.
- Assistant answers include source references from the retrieved documents.

## Build and Run

```bash
python scripts/build_index.py
streamlit run app.py
```

## Notes

- The documents should be placed under the matching folders in `data/`.
- The app expects `API_KEY` to be available in the environment.
- If the UI appears stale after edits, restart Streamlit and hard refresh the browser.
