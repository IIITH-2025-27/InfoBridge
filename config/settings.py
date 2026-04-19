"""
InfoBridge Configuration Settings
Central configuration for all components of the RAG pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore_data"

# ─── Service Categories ─────────────────────────────────────────────────────
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

# ─── Chunking Configuration ─────────────────────────────────────────────────
CHUNK_SIZE = 2000  # characters (~300-350 words)
CHUNK_OVERLAP = 300  # characters (~50 words)

# ─── Retrieval Configuration ────────────────────────────────────────────────
TOP_K = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to include

# ─── FAISS Configuration ────────────────────────────────────────────────────
FAISS_INDEX_FILE = VECTORSTORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTORSTORE_DIR / "metadata.pkl"

# ─── Gemini LLM Configuration ───────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-2.0-flash-lite,gemini-1.5-flash"
    ).split(",")
    if model.strip()
]
GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_OUTPUT_TOKENS = 2048

# ─── Conversation Memory ────────────────────────────────────────────────────
MAX_MEMORY_TURNS = 10  # Number of conversation turns to retain

# ─── Supported Languages ────────────────────────────────────────────────────
LANGUAGES = {
    "en": {"name": "English", "flag": "🇬🇧"},
    "hi": {"name": "हिन्दी (Hindi)", "flag": "🇮🇳"},
}
DEFAULT_LANGUAGE = "en"
