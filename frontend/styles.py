"""App-wide Streamlit CSS styles."""

import streamlit as st

APP_CSS = """
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Source citation styling */
    .source-card {
        background: #f0f4ff;
        border-left: 4px solid #1a237e;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .source-card strong {
        color: #1a237e;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafafa 0%, #f0f0f0 100%);
        color: #1f2937;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h5,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h6,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1a237e;
        font-size: 1rem;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    section[data-testid="stSidebar"] .stSelectbox svg {
        fill: #334155 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #93adc7 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stButton > button:focus {
        outline: 2px solid #93c5fd !important;
        outline-offset: 1px;
    }

    /* Chat message improvements */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-ready {
        background: #e8f5e9;
        color: #2e7d32;
    }
    .status-error {
        background: #ffebee;
        color: #c62828;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""


def apply_styles() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
