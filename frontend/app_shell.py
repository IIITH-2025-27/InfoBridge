"""Top-level frontend app flow orchestration."""

import streamlit as st

from config.settings import DEFAULT_LANGUAGE, LANGUAGES, SERVICE_CATEGORIES
from frontend.chat import render_chat
from frontend.pipeline import load_pipeline
from frontend.state import clear_chat_state, init_session_state
from frontend.styles import apply_styles
from frontend.ui_components import render_header, render_sidebar, show_welcome


def run_app() -> None:
    """Run the full Streamlit UI flow."""
    init_session_state(DEFAULT_LANGUAGE)

    selected_language, selected_service_filter, theme = render_sidebar(
        current_language=st.session_state.language,
        current_service_filter=st.session_state.service_filter,
        languages=LANGUAGES,
        service_categories=SERVICE_CATEGORIES,
        load_pipeline=load_pipeline,
        on_clear_chat=clear_chat_state,
    )
    st.session_state.language = selected_language
    st.session_state.service_filter = selected_service_filter

    apply_styles(theme)

    render_header(language=st.session_state.language, theme=theme)

    selected_example = show_welcome(
        language=st.session_state.language,
        has_messages=bool(st.session_state.messages),
        theme=theme,
    )
    if selected_example:
        st.session_state.pending_prompt = selected_example

    render_chat(service_categories=SERVICE_CATEGORIES)
