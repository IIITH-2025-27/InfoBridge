"""Top-level frontend app flow orchestration."""

import base64
from pathlib import Path
from urllib.parse import unquote

import streamlit as st
import streamlit.components.v1 as components

from config.settings import DATA_DIR, DEFAULT_LANGUAGE, LANGUAGES, SERVICE_CATEGORIES
from frontend.chat import render_chat
from frontend.pipeline import load_pipeline
from frontend.state import clear_chat_state, init_session_state
from frontend.styles import apply_styles
from frontend.ui_components import render_header, render_sidebar, show_welcome


def _render_pdf_viewer() -> bool:
    """Render a standalone PDF viewer page when a pdf query param is present."""
    pdf_param = st.query_params.get("pdf")
    if not pdf_param:
        return False

    if isinstance(pdf_param, list):
        pdf_param = pdf_param[0]

    rel_path = unquote(str(pdf_param)).lstrip("/")
    if not rel_path:
        return False

    data_dir = Path(DATA_DIR).resolve()
    candidate = (data_dir / rel_path).resolve()

    try:
        candidate.relative_to(data_dir)
    except ValueError:
        st.error("Invalid PDF path.")
        return True

    if not candidate.exists() or not candidate.is_file():
        st.error("PDF not found.")
        return True

    if st.button("Back to app"):
        st.query_params.clear()
        st.rerun()

    st.markdown(f"### {candidate.name}")

    file_bytes = candidate.read_bytes()
    st.download_button(
        label="Download PDF",
        data=file_bytes,
        file_name=candidate.name,
        mime="application/pdf",
        use_container_width=True,
    )

    encoded_pdf = base64.b64encode(file_bytes).decode("ascii")
    pdf_js_version = "4.2.67"
    pdf_viewer_html = f"""
    <!doctype html>
    <html>
        <head>
            <meta charset="utf-8" />
            <style>
                body {{ margin:0; background:#0d1117; color:#ffffff; }}
                #viewer {{ padding:16px; }}
                canvas {{ box-shadow:0 2px 10px rgba(0,0,0,0.4); border-radius:6px; }}
            </style>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/{pdf_js_version}/pdf.min.js"></script>
        </head>
        <body>
            <div id="viewer"></div>
            <script>
                const pdfData = atob("{encoded_pdf}");
                const pdfBytes = new Uint8Array(pdfData.length);
                for (let i = 0; i < pdfData.length; i++) {{
                    pdfBytes[i] = pdfData.charCodeAt(i);
                }}

                pdfjsLib.GlobalWorkerOptions.workerSrc =
                    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/{pdf_js_version}/pdf.worker.min.js";

                const loadingTask = pdfjsLib.getDocument({{ data: pdfBytes }});
                loadingTask.promise.then(pdf => {{
                    const viewer = document.getElementById("viewer");
                    const scale = 1.2;

                    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {{
                        pdf.getPage(pageNum).then(page => {{
                            const viewport = page.getViewport({{ scale }});
                            const canvas = document.createElement("canvas");
                            const context = canvas.getContext("2d");
                            canvas.height = viewport.height;
                            canvas.width = viewport.width;
                            canvas.style.display = "block";
                            canvas.style.margin = "0 auto 16px";
                            viewer.appendChild(canvas);
                            page.render({{ canvasContext: context, viewport }});
                        }});
                    }}
                }}).catch(() => {{
                    const viewer = document.getElementById("viewer");
                    viewer.textContent = "Failed to load PDF.";
                }});
            </script>
        </body>
    </html>
    """

    components.html(
        pdf_viewer_html,
        height=920,
        scrolling=True,
    )
    return True


def run_app() -> None:
    """Run the full Streamlit UI flow."""
    if _render_pdf_viewer():
        return

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
