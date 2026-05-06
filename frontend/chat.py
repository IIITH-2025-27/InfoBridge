"""Chat rendering and response orchestration."""

import streamlit as st
import streamlit.components.v1 as components

from frontend.i18n import t
from frontend.pipeline import load_pipeline
from frontend.ui_components import render_sources
from src.llm.generator import ResponseGenerator


def _scroll_to_latest_message() -> None:
    """Scroll viewport to the newest chat message after a response or question is rendered."""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const scrollToLatest = () => {
            const messages = doc.querySelectorAll('[data-testid="stChatMessage"]');
            if (messages.length > 0) {
                messages[messages.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
            } else {
                window.parent.scrollTo({ top: doc.body.scrollHeight, behavior: 'smooth' });
            }
        };

        let tries = 0;
        const tick = () => {
            scrollToLatest();
            tries += 1;
            if (tries < 10) {
                window.setTimeout(tick, 50);
            }
        };

        tick();
        </script>
        """,
        height=0,
    )


def render_chat(service_categories: dict) -> None:
    """Render chat history, input, and assistant responses."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if (
                message["role"] == "assistant"
                and "sources" in message
                and message.get("show_sources", True)
            ):
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
    _scroll_to_latest_message()

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
            _scroll_to_latest_message()
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
                f"⚠️ {t('LLM is not configured correctly. Please set GOOGLE_API_KEY in .env', lang)}\n\n"
                f"Error: {exc}"
            )
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
            _scroll_to_latest_message()
            return

        spinner_text = t("Searching documents and generating response...", lang)
        with st.spinner(f"🔍 {spinner_text}"):
            response = generator.generate(
                query=prompt,
                language=lang,
                service_filter=st.session_state.service_filter,
            )

        st.markdown(response["answer"])
        if response.get("show_sources", True):
            render_sources(
                sources=response["sources"],
                service_categories=service_categories,
                language=lang,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "show_sources": response.get("show_sources", True),
        })
        st.session_state.memory = generator.memory
        _scroll_to_latest_message()
