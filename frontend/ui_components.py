"""Reusable Streamlit UI components for InfoBridge."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Callable, Optional
from urllib.parse import quote

import streamlit as st

from config.settings import DATA_DIR
from frontend.i18n import t

_pdf_server = None
_pdf_server_thread = None
_pdf_server_port = None


def _ensure_pdf_server_url() -> str | None:
    global _pdf_server, _pdf_server_thread, _pdf_server_port

    if _pdf_server and _pdf_server_thread and _pdf_server_thread.is_alive():
        return f"http://localhost:{_pdf_server_port}"

    handler = partial(SimpleHTTPRequestHandler, directory=str(DATA_DIR))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError:
        return None

    _pdf_server = server
    _pdf_server_port = server.server_address[1]
    _pdf_server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    _pdf_server_thread.start()

    return f"http://localhost:{_pdf_server_port}"


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(
    *,
    current_language: str,
    current_service_filter: Optional[str],
    languages: dict,
    service_categories: dict,
    load_pipeline: Callable[[], tuple],
    on_clear_chat: Callable[[], None],
) -> tuple[str, Optional[str], str]:
    """Render sidebar controls and system status; returns updated language, service filter, and theme."""
    with st.sidebar:
        theme = st.session_state.get("theme_mode", "dark")

        toggle_label = "◀" if st.session_state.sidebar_open else "▶"
        toggle_help = "Collapse sidebar" if st.session_state.sidebar_open else "Open sidebar"
        if st.button(
            toggle_label,
            key="ib_sidebar_toggle",
            use_container_width=True,
            help=toggle_help,
        ):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()

        if not st.session_state.sidebar_open:
            st.markdown("<div class='ib-strip-caption'>InfoBridge</div>", unsafe_allow_html=True)
            return current_language, current_service_filter, theme

        # Brand header
        st.markdown(
            """
            <div style="text-align:center; padding: 0.75rem 0 1.25rem 0;">
                <div style="
                    font-size: 1.55rem;
                    font-weight: 900;
                    background: linear-gradient(135deg, #FF9933, #FFD700, #FF9933);
                    background-size: 200% auto;
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -0.5px;
                ">🏛️ InfoBridge</div>
                <div style="
                    font-size: 0.68rem;
                    color: rgba(255,255,255,0.35);
                    margin-top: 0.2rem;
                    letter-spacing: 1.5px;
                    text-transform: uppercase;
                ">Government Services AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Theme toggle
        theme_button_label = "🌙  Dark Mode: ON" if theme == "dark" else "☀️  Light Mode: ON"
        if st.button(theme_button_label, key="theme_mode_button", use_container_width=True):
            st.session_state.theme_mode = "light" if theme == "dark" else "dark"
            st.rerun()
        theme = st.session_state.theme_mode

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Language section
        st.markdown(
            "<div style='font-size:0.7rem; font-weight:700; color:rgba(255,153,51,0.8);"
            " text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.5rem;'>"
            "🌐 Language</div>",
            unsafe_allow_html=True,
        )
        lang_options = {
            code: info['name']
            for code, info in languages.items()
        }
        selected_lang = st.radio(
            "Select language",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            horizontal=True,
            index=list(lang_options.keys()).index(current_language),
            label_visibility="collapsed",
            key="lang_radio",
        )

        previous_lang = st.session_state.get("_prev_sidebar_lang")
        if previous_lang is not None and previous_lang != selected_lang:
            st.session_state.pop("service_select_en", None)
            st.session_state.pop("service_select_hi", None)
        st.session_state["_prev_sidebar_lang"] = selected_lang

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # Service filter section
        st.markdown(
            "<div style='font-size:0.7rem; font-weight:700; color:rgba(255,153,51,0.8);"
            " text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.5rem;'>"
            "🗂️ Filter by Service</div>",
            unsafe_allow_html=True,
        )
        service_options = {"all": f"📋  {t('All Services', selected_lang)}"}
        for key, info in service_categories.items():
            localized_name = t(info["name"], selected_lang)
            service_options[key] = f"{info['icon']}  {localized_name}"

        selected_key = current_service_filter or "all"
        service_select_key = f"service_select_{selected_lang}"
        selected_service = st.selectbox(
            "Filter by service",
            options=list(service_options.keys()),
            format_func=lambda x: service_options[x],
            index=list(service_options.keys()).index(selected_key),
            label_visibility="collapsed",
            key=service_select_key,
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.divider()

        # System status section
        st.markdown(
            "<div style='font-size:0.7rem; font-weight:700; color:rgba(255,153,51,0.8);"
            " text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.6rem;'>"
            "⚙️ System Status</div>",
            unsafe_allow_html=True,
        )
        pipeline, error = load_pipeline()
        if pipeline:
            vs = pipeline["vector_store"]
            stats = vs.get_service_stats()
            st.markdown(
                f'<div class="status-indicator status-ready">🟢 {t("System Ready", selected_lang)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            st.caption(f"📄 **{vs.size}** {t('indexed chunks', selected_lang)}")
            if stats:
                for svc, count in sorted(stats.items()):
                    icon = service_categories.get(svc, {}).get("icon", "📄")
                    name = service_categories.get(svc, {}).get("name", svc.title())
                    st.caption(f"{icon} {t(name, selected_lang)}: **{count}**")
        else:
            st.markdown(
                f'<div class="status-indicator status-error">🔴 {t("Not Ready", selected_lang)}</div>',
                unsafe_allow_html=True,
            )
            if error:
                st.error(error)

        st.divider()

        # Clear chat button
        if st.button(f"🗑️  {t('Clear Chat', selected_lang)}", use_container_width=True):
            on_clear_chat()
            st.rerun()

        # Footer
        st.markdown(
            "<div style='text-align:center; padding-top:1rem;'>"
            "<div style='font-size:0.65rem; color:rgba(255,255,255,0.2); letter-spacing:0.5px;'>"
            "RAG · FAISS · MiniLM · Groq"
            "</div></div>",
            unsafe_allow_html=True,
        )

    return selected_lang, None if selected_service == "all" else selected_service, theme


# ─── Header ───────────────────────────────────────────────────────────────────

def render_header(language: str, theme: str = "dark") -> None:
    """Render the main animated page header."""
    subtitle = t(
        "Smart Access to Indian Government Services — Ask about Passport, Voter ID, Driving Licence, Income Tax, Ayushman Bharat",
        language,
    )

    services_hi = "🛂 पासपोर्ट &nbsp;·&nbsp; 🗳️ वोटर आईडी &nbsp;·&nbsp; 🚗 ड्राइविंग लाइसेंस &nbsp;·&nbsp; 💰 आयकर &nbsp;·&nbsp; 🏥 आयुष्मान भारत"
    services_en = "🛂 Passport &nbsp;·&nbsp; 🗳️ Voter ID &nbsp;·&nbsp; 🚗 Driving Licence &nbsp;·&nbsp; 💰 Income Tax &nbsp;·&nbsp; 🏥 Ayushman Bharat"
    services_line = services_hi if language == "hi" else services_en

    if theme == "light":
        svc_panel_style = (
            "background:rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.1);"
            " border-radius:12px; padding:0.75rem 1rem; font-size:0.78rem;"
            " color:rgba(0,0,0,0.55); line-height:2; white-space:nowrap;"
        )
    else:
        svc_panel_style = (
            "background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);"
            " border-radius:12px; padding:0.75rem 1rem; font-size:0.78rem;"
            " color:rgba(255,255,255,0.5); line-height:2; white-space:nowrap;"
        )

    st.markdown(
        f"""
        <div class="ib-header">
            <div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
                <div style="flex:1; min-width:200px;">
                    <h1 class="ib-header-logo">🏛️ InfoBridge</h1>
                    <p class="ib-header-subtitle">{subtitle}</p>
                    <div style="margin-top:0.8rem;">
                        <span class="ib-pill ib-pill-saffron">🤖 AI Powered</span>
                        <span class="ib-pill ib-pill-green">📚 RAG Pipeline</span>
                        <span class="ib-pill ib-pill-blue">⚡ Groq LLM</span>
                    </div>
                </div>
                <div style="{svc_panel_style}">{services_line}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Sources ──────────────────────────────────────────────────────────────────

def render_sources(sources: list[dict], service_categories: dict, language: str) -> None:
    """Render source citations as an expandable section."""
    if not sources:
        return

    label = t("Sources", language)
    document_label = t("documents", language)

    with st.expander(f"📎  {label} ({len(sources)} {document_label})", expanded=False):
        for src in sources:
            if isinstance(src, str):
                st.markdown(
                    f'<div class="ib-source-card"><strong>📄 {src}</strong></div>',
                    unsafe_allow_html=True,
                )
                continue

            service_type = src.get("service_type", "unknown")
            icon = service_categories.get(service_type, {}).get("icon", "📄")
            name = service_categories.get(service_type, {}).get("name", service_type.title())
            source_file = src.get("source_file", "Unknown")
            source_path = src.get("source_path", "")
            relevance_score = src.get("relevance_score")

            relevance_text = (
                f"{relevance_score:.0%}" if isinstance(relevance_score, (int, float)) else "N/A"
            )

            resolved_path = None

            if source_path:
                pdf_path = Path(source_path)
                if pdf_path.exists() and pdf_path.is_file():
                    resolved_path = pdf_path

            if resolved_path is None:
                fallback_path = DATA_DIR / service_type / source_file
                if fallback_path.exists() and fallback_path.is_file():
                    resolved_path = fallback_path

            title_html = f"{icon} {source_file}"
            if resolved_path is not None:
                server_url = _ensure_pdf_server_url()
                if server_url:
                    relative_path = f"{service_type}/{source_file}"
                    file_url = f"{server_url}/{quote(relative_path, safe='/')}"
                    title_html = (
                        f'<a href="{file_url}" target="_blank" rel="noopener noreferrer" '
                        f'style="color:#FFD700; text-decoration:none;">'
                        f"{icon} {source_file}"
                        "</a>"
                    )

            with st.container():
                st.markdown(
                    f'<div class="ib-source-card">'
                    f'<strong>{title_html}</strong><br>'
                    f'<span class="ib-source-meta">'
                    f'{t(name, language)} &nbsp;·&nbsp; '
                    f'{t("Relevance", language)}: {relevance_text}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )


# ─── Welcome Screen ───────────────────────────────────────────────────────────

def show_welcome(language: str, has_messages: bool, theme: str = "dark") -> Optional[str]:
    """Show welcome card and example prompts; returns selected example prompt."""
    if has_messages:
        return None

    if language == "hi":
        heading = "👋 नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूँ?"
        sub = "किसी भी सरकारी सेवा के बारे में पूछें — हिंदी या अंग्रेज़ी में"
    else:
        heading = "👋 Welcome! How can I help you today?"
        sub = "Ask about any Indian government service — in English or Hindi"

    st.markdown(
        f"""
        <div style="margin-bottom: 0.5rem;">
            <div class="ib-welcome-heading">{heading}</div>
            <div class="ib-welcome-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Service cards grid
    cards = [
        ("🛂", "Passport" if language == "en" else "पासपोर्ट"),
        ("🗳️", "Voter ID" if language == "en" else "वोटर आईडी"),
        ("🚗", "Driving Licence" if language == "en" else "ड्राइविंग"),
        ("💰", "Income Tax" if language == "en" else "आयकर"),
        ("🏥", "Ayushman Bharat" if language == "en" else "आयुष्मान"),
    ]

    cards_html = '<div class="ib-service-grid">'
    for icon, name in cards:
        cards_html += (
            f'<div class="ib-service-card">'
            f'<span class="svc-icon">{icon}</span>'
            f'<span class="svc-name">{name}</span>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Example prompts
    st.markdown(
        f"<div class='ib-examples-label'>💬 {t('Try asking:', language)}</div>",
        unsafe_allow_html=True,
    )

    examples = {
        "en": [
            "How do I apply for a new passport?",
            "What documents are needed for Voter ID?",
            "How to file income tax return online?",
            "Who is eligible for Ayushman Bharat?",
            "How to apply for a driving licence?",
            "What is the Tatkaal passport scheme?",
        ],
        "hi": [
            "नया पासपोर्ट कैसे बनवाएं?",
            "वोटर आईडी के लिए कौन से दस्तावेज़ चाहिए?",
            "ऑनलाइन इनकम टैक्स रिटर्न कैसे फाइल करें?",
            "आयुष्मान भारत के लिए कौन पात्र है?",
            "ड्राइविंग लाइसेंस कैसे बनवाएं?",
            "तत्काल पासपोर्ट योजना क्या है?",
        ],
    }

    cols = st.columns(2)
    for i, example in enumerate(examples.get(language, examples["en"])):
        with cols[i % 2]:
            if st.button(example, key=f"ex_{language}_{i}", use_container_width=True):
                return example

    return None
