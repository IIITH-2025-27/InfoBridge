"""Pipeline loading and caching utilities for the frontend layer."""

import logging

import streamlit as st

from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading AI models and vector store...")
def load_pipeline() -> tuple[dict | None, str | None]:
    """Load and cache the retriever and vector store."""
    try:
        vector_store = VectorStore()
        loaded = vector_store.load()

        if not loaded:
            return None, "Vector store not found. Please run: python scripts/build_index.py"

        embedder = Embedder()
        retriever = Retriever(vector_store=vector_store, embedder=embedder)

        return {
            "retriever": retriever,
            "vector_store": vector_store,
        }, None

    except Exception as exc:
        logger.error("Failed to load pipeline: %s", exc)
        return None, str(exc)
