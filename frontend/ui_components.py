"""Reusable Streamlit UI components for InfoBridge."""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from frontend.i18n import t


def render_sidebar(
    *,
    current_language: str,
    current_service_filter: Optional[str],
    languages: dict,
    service_categories: dict,
    load_pipeline: Callable[[], tuple],
    on_clear_chat: Callable[[], None],
) -> tuple[str, Optional[str]]:
    """Render sidebar controls and system status; returns updated language and service filter."""
    with st.sidebar:
        st.markdown(f"### 🏛️ {t('InfoBridge Controls', current_language)}")
        st.divider()

        st.markdown(f"##### 🌐 {t('Language / भाषा', current_language)}")
        lang_options = {
            code: f"{info['flag']} {info['name']}"
            for code, info in languages.items()
        }
        selected_lang = st.radio(
            "Select language",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=list(lang_options.keys()).index(current_language),
            label_visibility="collapsed",
            key="lang_radio",
        )

        previous_lang = st.session_state.get("_prev_sidebar_lang")
        if previous_lang is not None and previous_lang != selected_lang:
            # Force selectbox label to refresh with new language mapping.
            st.session_state.pop("service_select", None)
        st.session_state["_prev_sidebar_lang"] = selected_lang

        st.divider()

        st.markdown(f"##### 🗂️ {t('Filter by Service', selected_lang)}")
        service_options = {"all": f"📋 {t('All Services', selected_lang)}"}
        for key, info in service_categories.items():
            localized_name = t(info["name"], selected_lang)
            service_options[key] = f"{info['icon']} {localized_name}"

        selected_key = current_service_filter or "all"
        selected_service = st.selectbox(
            "Filter by service",
            options=list(service_options.keys()),
            format_func=lambda x: service_options[x],
            index=list(service_options.keys()).index(selected_key),
            label_visibility="collapsed",
            key="service_select",
        )

        st.divider()

        if st.button(f"🗑️ {t('Clear Chat', selected_lang)}", use_container_width=True, type="secondary"):
            on_clear_chat()
            st.rerun()

        st.divider()

        st.markdown(f"##### ℹ️ {t('System Info', selected_lang)}")
        pipeline, error = load_pipeline()
        if pipeline:
            vs = pipeline["vector_store"]
            stats = vs.get_service_stats()
            st.markdown(
                f'<div class="status-indicator status-ready">🟢 {t("System Ready", selected_lang)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"📄 **{vs.size}** {t('indexed chunks', selected_lang)}")
            if stats:
                st.caption(f"**{t('Chunks per service:', selected_lang)}**")
                for svc, count in sorted(stats.items()):
                    icon = service_categories.get(svc, {}).get("icon", "📄")
                    name = service_categories.get(svc, {}).get("name", svc.title())
                    st.caption(f"  {icon} {t(name, selected_lang)}: {count}")
        else:
            st.markdown(
                f'<div class="status-indicator status-error">🔴 {t("Not Ready", selected_lang)}</div>',
                unsafe_allow_html=True,
            )
            if error:
                st.error(error)

        st.divider()
        st.caption(
            f"{t('Built with using RAG pipeline.', selected_lang)}\n\n"
            f"**{t('Tech Stack:', selected_lang)}** MiniLM embeddings, FAISS, Gemini 2.0 Flash, Streamlit"
        )

    return selected_lang, None if selected_service == "all" else selected_service


def render_header(language: str) -> None:
    """Render the main page header."""
    subtitle = t(
        "Smart Access to Indian Government Services — Ask about Passport, Voter ID, Driving Licence, Income Tax, Ayushman Bharat",
        language,
    )
    st.markdown(
        f"""
    <div class="main-header">
        <h1>🏛️ InfoBridge</h1>
        <p>{subtitle}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], service_categories: dict, language: str) -> None:
    """Render source citations as an expandable section."""
    if not sources:
        return

    label = t("Sources", language)
    document_label = t("documents", language)
    with st.expander(f"📎 {label} ({len(sources)} {document_label})", expanded=False):
        for src in sources:
            icon = service_categories.get(src["service_type"], {}).get("icon", "📄")
            name = service_categories.get(src["service_type"], {}).get(
                "name", src["service_type"].title()
            )
            st.markdown(
                f'<div class="source-card">'
                f'<strong>{icon} {src["source_file"]}</strong><br>'
                f'Service: {t(name, language)} &nbsp;|&nbsp; '
                f'Relevance: {src["relevance_score"]:.0%}'
                f"</div>",
                unsafe_allow_html=True,
            )


def show_welcome(language: str, has_messages: bool) -> Optional[str]:
    """Show welcome card and example prompts; returns selected example prompt."""
    if has_messages:
        return None

    if language == "hi":
        welcome = """
        👋 **InfoBridge में आपका स्वागत है!**

        मैं भारतीय सरकारी सेवाओं के लिए आपका AI सहायक हूँ। आप मुझसे इन सेवाओं के बारे में पूछ सकते हैं:

        - 🛂 **पासपोर्ट** — आवेदन, नवीनीकरण, आवश्यक दस्तावेज़
        - 🗳️ **वोटर आईडी** — पंजीकरण, EPIC कार्ड
        - 🚗 **ड्राइविंग लाइसेंस** — आवेदन, RC
        - 💰 **आयकर** — ITR फॉर्म, फाइलिंग
        - 🏥 **आयुष्मान भारत** — PM-JAY पात्रता, लाभ

        नीचे अपना सवाल टाइप करें! 👇
        """
    else:
        welcome = """
        👋 **Welcome to InfoBridge!**

        I'm your AI assistant for Indian government services. Ask me about:

        - 🛂 **Passport** — Application, renewal, documents required
        - 🗳️ **Voter ID** — Registration, EPIC card
        - 🚗 **Driving Licence** — Application, RC
        - 💰 **Income Tax** — ITR forms, filing
        - 🏥 **Ayushman Bharat** — PM-JAY eligibility, benefits

        Type your question below! 👇
        """

    st.info(welcome)

    st.markdown(f"**{t('Try asking:', language)}**")
    examples = {
        "en": [
            "How do I apply for a new passport?",
            "What documents are needed for Voter ID registration?",
            "How to file income tax return online?",
            "Who is eligible for Ayushman Bharat scheme?",
            "How to apply for a driving licence?",
        ],
        "hi": [
            "नया पासपोर्ट कैसे बनवाएं?",
            "वोटर आईडी के लिए कौन से दस्तावेज़ चाहिए?",
            "ऑनलाइन इनकम टैक्स रिटर्न कैसे फाइल करें?",
            "आयुष्मान भारत के लिए कौन पात्र है?",
            "ड्राइविंग लाइसेंस कैसे बनवाएं?",
        ],
    }

    cols = st.columns(2)
    for i, example in enumerate(examples.get(language, examples["en"])):
        with cols[i % 2]:
            if st.button(f"💬 {example}", key=f"example_{language}_{i}", use_container_width=True):
                return example

    return None
