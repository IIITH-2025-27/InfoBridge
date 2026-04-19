"""
InfoBridge: Smart Access to Public Services
Main Streamlit Application - RAG Chatbot for Indian Government Services

T10.7 — Multi-service unified chatbot with Hindi support
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# ─── Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Ensure project root is in path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SERVICE_CATEGORIES, LANGUAGES, DEFAULT_LANGUAGE
from src.vectorstore.store import VectorStore
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.llm.generator import ResponseGenerator
from src.memory.chat_memory import ChatMemory


# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfoBridge — Smart Access to Public Services",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
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

    /* Service badge styling */
    .service-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
        background: rgba(255,255,255,0.15);
        color: white;
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
        background: #f8fafc !important;
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
""", unsafe_allow_html=True)


# ─── Initialize Session State ────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemory()

    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE

    if "service_filter" not in st.session_state:
        st.session_state.service_filter = None

    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False

    if "generator" not in st.session_state:
        st.session_state.generator = None

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


@st.cache_resource(show_spinner="Loading AI models and vector store...")
def load_pipeline():
    """Load and cache the full RAG pipeline."""
    try:
        # Load vector store
        vector_store = VectorStore()
        loaded = vector_store.load()

        if not loaded:
            return None, "Vector store not found. Please run: `python scripts/build_index.py`"

        # Initialize embedder
        embedder = Embedder()

        # Initialize retriever
        retriever = Retriever(vector_store=vector_store, embedder=embedder)

        # Initialize generator (without memory — memory is per-session)
        return {
            "retriever": retriever,
            "vector_store": vector_store,
        }, None

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None, str(e)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with controls and information."""
    with st.sidebar:
        st.markdown("### 🏛️ InfoBridge Controls")
        st.divider()

        # Language toggle
        st.markdown("##### 🌐 Language / भाषा")
        lang_options = {code: f"{info['flag']} {info['name']}" for code, info in LANGUAGES.items()}
        selected_lang = st.radio(
            "Select language",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=list(lang_options.keys()).index(st.session_state.language),
            label_visibility="collapsed",
            key="lang_radio",
        )
        st.session_state.language = selected_lang

        st.divider()

        # Service filter
        st.markdown("##### 🗂️ Filter by Service")
        service_options = {"all": "📋 All Services"}
        for key, info in SERVICE_CATEGORIES.items():
            service_options[key] = f"{info['icon']} {info['name']}"

        selected_service = st.selectbox(
            "Filter by service",
            options=list(service_options.keys()),
            format_func=lambda x: service_options[x],
            label_visibility="collapsed",
            key="service_select",
        )
        st.session_state.service_filter = None if selected_service == "all" else selected_service

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.memory = ChatMemory()
            if st.session_state.generator:
                st.session_state.generator.clear_memory()
            st.rerun()

        st.divider()

        # System info
        st.markdown("##### ℹ️ System Info")
        pipeline, error = load_pipeline()
        if pipeline:
            vs = pipeline["vector_store"]
            stats = vs.get_service_stats()
            st.markdown(
                f'<div class="status-indicator status-ready">🟢 System Ready</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"📄 **{vs.size}** indexed chunks")
            if stats:
                st.caption("**Chunks per service:**")
                for svc, count in sorted(stats.items()):
                    icon = SERVICE_CATEGORIES.get(svc, {}).get("icon", "📄")
                    name = SERVICE_CATEGORIES.get(svc, {}).get("name", svc.title())
                    st.caption(f"  {icon} {name}: {count}")
        else:
            st.markdown(
                f'<div class="status-indicator status-error">🔴 Not Ready</div>',
                unsafe_allow_html=True,
            )
            if error:
                st.error(error)

        st.divider()
        st.caption(
            "Built with ❤️ using RAG pipeline.\n\n"
            "**Tech Stack:** MiniLM embeddings, FAISS, Gemini 2.0 Flash, Streamlit"
        )


# ─── Main Chat Area ─────────────────────────────────────────────────────────
def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ InfoBridge</h1>
        <p>Smart Access to Indian Government Services — Ask about Passport, Voter ID, Driving Licence, Income Tax, Ayushman Bharat</p>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: list[dict]):
    """Render source citations as expandable section."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)} documents)", expanded=False):
        for src in sources:
            icon = SERVICE_CATEGORIES.get(src["service_type"], {}).get("icon", "📄")
            name = SERVICE_CATEGORIES.get(src["service_type"], {}).get("name", src["service_type"].title())
            st.markdown(
                f'<div class="source-card">'
                f'<strong>{icon} {src["source_file"]}</strong><br>'
                f'Service: {name} &nbsp;|&nbsp; '
                f'Relevance: {src["relevance_score"]:.0%}'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_chat():
    """Render the chat interface."""
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])

    # Chat input
    lang = st.session_state.language
    placeholder = (
        "अपना सवाल यहाँ लिखें..." if lang == "hi"
        else "Ask about government services..."
    )

    prompt = st.session_state.pending_prompt
    if prompt:
        st.session_state.pending_prompt = None
    else:
        prompt = st.chat_input(placeholder)

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            pipeline, error = load_pipeline()

            if not pipeline:
                error_msg = (
                    "⚠️ System not ready. Please ensure the vector store is built.\n\n"
                    f"Run: `python scripts/build_index.py`\n\nError: {error}"
                )
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
                return

            # Create or reuse generator with session memory
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
            except Exception as e:
                error_msg = (
                    "⚠️ LLM is not configured correctly. Please set GOOGLE_API_KEY in .env\n\n"
                    f"Error: {str(e)}"
                )
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
                return

            with st.spinner("🔍 Searching documents and generating response..." if lang == "en"
                           else "🔍 दस्तावेज़ खोज रहे हैं और उत्तर तैयार कर रहे हैं..."):
                response = generator.generate(
                    query=prompt,
                    language=lang,
                    service_filter=st.session_state.service_filter,
                )

            # Display answer
            st.markdown(response["answer"])
            render_sources(response["sources"])

            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
            })

            # Update memory reference
            st.session_state.memory = generator.memory


# ─── Welcome Message ─────────────────────────────────────────────────────────
def show_welcome():
    """Show welcome message if no chat history."""
    if not st.session_state.messages:
        lang = st.session_state.language

        if lang == "hi":
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

        # Example queries
        st.markdown("**Try asking:**")
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
        for i, example in enumerate(examples.get(lang, examples["en"])):
            col = cols[i % 2]
            with col:
                if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_prompt = example
                    st.rerun()


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_header()
    show_welcome()
    render_chat()


if __name__ == "__main__":
    main()
