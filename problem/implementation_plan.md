# InfoBridge Implementation Plan

## Current State

InfoBridge is implemented as a Streamlit RAG chatbot for Indian government services. The codebase now includes:

- `app.py` as the Streamlit entrypoint.
- `frontend/` modules for sidebar, header, chat, styling, state, and translation handling.
- `src/` modules for PDF extraction, chunking, embeddings, retrieval, FAISS storage, LLM client, prompts, and memory.
- `scripts/build_index.py` for one-time index creation from PDFs stored in `data/`.

## Implemented User Experience

- Collapsible sidebar with a narrow left strip when collapsed.
- Theme persistence for light and dark mode.
- English/Hindi language selection through the sidebar.
- Localized service filter and sidebar labels.
- Chat interface with source citations under each assistant response.

## Implemented RAG Flow

1. PDFs are extracted from the service folders in `data/`.
2. Extracted text is chunked with overlap for retrieval quality.
3. Chunks are encoded with `sentence-transformers/all-MiniLM-L6-v2`.
4. Vectors and metadata are stored in FAISS under `vectorstore_data/`.
5. User queries are embedded and searched against the vector store.
6. Retrieved context and citations are passed to the Groq LLM client.
7. The final response is rendered with source references.

## Current Configuration

- `config/settings.py` contains service categories, chunk sizes, top-k retrieval settings, and API/model configuration.
- `API_KEY` is read from the environment.
- `MODEL` defaults to `llama3-8b-8192`.
- `FALLBACK_MODELS` can be configured for retry behavior.
- The UI translation table lives in `frontend/translations_en_hi.csv`.

## Validation Notes

- The FAISS build script now expects the embedding function to return both embeddings and the filtered chunk list.
- The retriever now returns structured source citations instead of plain strings.
- The sidebar UI now preserves theme and language state across reruns.

## Remaining Operating Steps

- Add or update PDFs in the `data/` folders.
- Rebuild the index with `python scripts/build_index.py`.
- Launch the app with `streamlit run app.py`.
