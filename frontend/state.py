"""Session state helpers for the Streamlit app."""

import streamlit as st

from src.memory.chat_memory import ChatMemory


def init_session_state(default_language: str) -> None:
    """Initialize all required session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemory()

    if "language" not in st.session_state:
        st.session_state.language = default_language

    if "service_filter" not in st.session_state:
        st.session_state.service_filter = None

    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False

    if "generator" not in st.session_state:
        st.session_state.generator = None

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def clear_chat_state() -> None:
    """Clear chat and memory for the current session."""
    st.session_state.messages = []
    st.session_state.memory = ChatMemory()
    if st.session_state.generator:
        st.session_state.generator.clear_memory()
