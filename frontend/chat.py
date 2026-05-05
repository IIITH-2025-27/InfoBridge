"""Chat rendering and response orchestration."""

import streamlit as st
import streamlit.components.v1 as components

from frontend.i18n import t
from frontend.pipeline import load_pipeline
from frontend.ui_components import render_sources
from src.llm.generator import ResponseGenerator


def _scroll_to_chat_input() -> None:
    """Scroll viewport to the chat input after a response is rendered."""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const inputs = doc.querySelectorAll('[data-testid="stChatInput"]');
        if (inputs.length > 0) {
            inputs[inputs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            window.parent.scrollTo({ top: doc.body.scrollHeight, behavior: 'smooth' });
        }
        </script>
        """,
        height=0,
    )


def render_chat(service_categories: dict) -> None:
    """Render chat history, input, and assistant responses."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_sources(
                    sources=message["sources"],
                    service_categories=service_categories,
                    language=st.session_state.language,
                )

    lang = st.session_state.language
    placeholder = t("Ask about government services...", lang)

    pending_prompt = st.session_state.pending_prompt
    typed_prompt = st.chat_input(placeholder)

    prompt = pending_prompt or typed_prompt
    if pending_prompt:
        st.session_state.pending_prompt = None

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pipeline, error = load_pipeline()

        if not pipeline:
            error_msg = (
                f"⚠️ {t('System not ready. Please ensure the vector store is built.', lang)}\n\n"
                f"{t('Run: python scripts/build_index.py', lang)}\n\n"
                f"Error: {error}"
            )
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
            _scroll_to_chat_input()
            return

        try:
            generator = st.session_state.generator
            if generator is None or generator.retriever is not pipeline["retriever"]:
                generator = ResponseGenerator(
                    retriever=pipeline["retriever"],
                    memory=st.session_state.memory,
                )
                st.session_state.generator = generator
            else:
                generator.memory = st.session_state.memory
        except Exception as exc:
            error_msg = (
                f"⚠️ {t('LLM is not configured correctly. Please set API_KEY in .env', lang)}\n\n"
                f"Error: {exc}"
            )
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
            _scroll_to_chat_input()
            return

        spinner_text = t("Searching documents and generating response...", lang)
        with st.spinner(f"🔍 {spinner_text}"):
            response = generator.generate(
                query=prompt,
                language=lang,
                service_filter=st.session_state.service_filter,
            )

        st.markdown(response["answer"])
        render_sources(
            sources=response["sources"],
            service_categories=service_categories,
            language=lang,
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })
        st.session_state.memory = generator.memory
        _scroll_to_chat_input()
