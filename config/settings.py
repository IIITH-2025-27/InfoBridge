"""
InfoBridge Configuration Settings
Central configuration for all components of the RAG pipeline.
"""

import os
from pathlib import Path

# ─── Safe dotenv loading (no crash if not installed) ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore_data"

# Ensure vectorstore directory exists (critical fix)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Service Categories ──────────────────────────────────────────────────────
SERVICE_CATEGORIES = {
    "passport": {
        "name": "Passport Services",
        "data_dir": DATA_DIR / "passport",
        "icon": "🛂",
        "description": "Passport application, renewal, and related services",
    },
    "voter": {
        "name": "Voter ID / EPIC",
        "data_dir": DATA_DIR / "voter",
        "icon": "🗳️",
        "description": "Voter registration, EPIC card, and election services",
    },
    "driving": {
        "name": "Driving Licence & RC",
        "data_dir": DATA_DIR / "driving",
        "icon": "🚗",
        "description": "Driving licence, vehicle registration, and Parivahan services",
    },
    "tax": {
        "name": "Income Tax",
        "data_dir": DATA_DIR / "tax",
        "icon": "💰",
        "description": "Income tax filing, ITR forms, and tax-related queries",
    },
    "health": {
        "name": "Ayushman Bharat",
        "data_dir": DATA_DIR / "health",
        "icon": "🏥",
        "description": "PM-JAY health insurance, eligibility, and benefits",
    },
}

# ─── Embedding Configuration ────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── Chunking Configuration (FIXED) ─────────────────────────────────────────
# Smaller chunks = better retrieval accuracy
CHUNK_SIZE = 500        # ~80–120 words
CHUNK_OVERLAP = 100     # preserves context across chunks

# ─── Retrieval Configuration ────────────────────────────────────────────────
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3  # used in FAISS filtering
# If the best matching chunk score is below this, treat as no relevant documents
MIN_ANSWER_SCORE = 0.45

# ─── FAISS Configuration ────────────────────────────────────────────────────
FAISS_INDEX_FILE = VECTORSTORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTORSTORE_DIR / "metadata.pkl"

# ─── Gemini LLM Configuration ───────────────────────────────────────────────

# API key (generic)
API_KEY = os.getenv("API_KEY", "")

# Primary model
MODEL = os.getenv("MODEL", "llama3-8b-8192")

# Fallback models (string → list)
FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "FALLBACK_MODELS",
        "llama3-70b-8192,mixtral-8x7b-32768"
    ).split(",")
    if model.strip()
]

TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 2048

# ─── Conversation Memory ────────────────────────────────────────────────────
MAX_MEMORY_TURNS = 10

# ─── Supported Languages ────────────────────────────────────────────────────
LANGUAGES = {
    "en": {"name": "English (अंग्रेज़ी)", "flag": "🇬🇧"},
    "hi": {"name": "हिन्दी (Hindi)", "flag": "🇮🇳"},
}

DEFAULT_LANGUAGE = "en"