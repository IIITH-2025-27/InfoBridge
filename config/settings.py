"""
InfoBridge Configuration Settings
Central configuration for all components of the RAG pipeline.

Every tunable value can be overridden from .env (or the shell environment), so the
eval harness can switch features on/off for ablations without code changes:
    python eval/run_eval.py --name no_reranker --set USE_RERANKER=false
"""

import os
from pathlib import Path

# ─── Safe dotenv loading (no crash if not installed) ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# Overridable so the eval can keep the old baseline index next to the new one,
# e.g. VECTORSTORE_DIR=vectorstore_baseline
VECTORSTORE_DIR = BASE_DIR / _env_str("VECTORSTORE_DIR", "vectorstore_data")
EVAL_DIR = BASE_DIR / "eval"

# Ensure vectorstore directory exists (critical fix)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Service Categories ──────────────────────────────────────────────────────
# "keywords" drive the cheap keyword-based service router (Phase 4).
SERVICE_CATEGORIES = {
    "passport": {
        "name": "Passport Services",
        "data_dir": DATA_DIR / "passport",
        "icon": "🛂",
        "description": "Passport application, renewal, and related services",
        "keywords": [
            "passport", "tatkal", "tatkaal", "psk", "popsk", "passport seva",
            "police verification", "ecnr", "ecr", "visa", "pcc", "annexure",
            "पासपोर्ट", "तत्काल",
        ],
    },
    "voter": {
        "name": "Voter ID / EPIC",
        "data_dir": DATA_DIR / "voter",
        "icon": "🗳️",
        "description": "Voter registration, EPIC card, and election services",
        "keywords": [
            "voter", "epic", "electoral", "election", "form 6", "form-6", "form 8",
            "form-8", "form 7", "blo", "polling", "e-epic", "vote",
            "वोटर", "मतदाता", "चुनाव",
        ],
    },
    "driving": {
        "name": "Driving Licence & RC",
        "data_dir": DATA_DIR / "driving",
        "icon": "🚗",
        "description": "Driving licence, vehicle registration, and Parivahan services",
        "keywords": [
            "driving", "licence", "license", "learner", "dl", "rc", "vehicle",
            "registration certificate", "parivahan", "rto", "sarathi", "vahan",
            "motor", "puc", "ड्राइविंग", "लाइसेंस", "वाहन",
        ],
    },
    "tax": {
        "name": "Income Tax",
        "data_dir": DATA_DIR / "tax",
        "icon": "💰",
        "description": "Income tax filing, ITR forms, and tax-related queries",
        "keywords": [
            "tax", "itr", "itr-1", "itr-2", "itr-3", "itr-4", "tds", "pan",
            "assessment year", "deduction", "80c", "refund", "e-filing",
            "income", "regime", "form 16", "आयकर", "टैक्स", "रिटर्न",
        ],
    },
    "health": {
        "name": "Ayushman Bharat",
        "data_dir": DATA_DIR / "health",
        "icon": "🏥",
        "description": "PM-JAY health insurance, eligibility, and benefits",
        "keywords": [
            "ayushman", "pm-jay", "pmjay", "jay", "health", "hospital",
            "insurance", "treatment", "golden card", "abha", "empanelled",
            "आयुष्मान", "अस्पताल", "स्वास्थ्य",
        ],
    },
}

# ─── Embedding Configuration ────────────────────────────────────────────────
# Phase 4: multilingual model so Hindi text embeds properly. e5-small has the same
# 384 dimensions as the old MiniLM model but ~0.5 GB RAM, which suits an 8 GB Mac.
# Changing this requires rebuilding the index (python scripts/build_index.py).
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

_KNOWN_EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-m3": 1024,
}
EMBEDDING_DIMENSION = _env_int(
    "EMBEDDING_DIMENSION", _KNOWN_EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 384)
)

# auto = cuda → mps (Apple GPU) → cpu
EMBEDDING_DEVICE = _env_str("EMBEDDING_DEVICE", "auto")

# ─── Chunking Configuration ─────────────────────────────────────────────────
CHUNK_SIZE = _env_int("CHUNK_SIZE", 500)        # ~80–120 words
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)  # preserves context across chunks
# Phase 2: was a hard-coded 30, which threw away most of the short form PDFs.
MIN_CHUNK_WORDS = _env_int("MIN_CHUNK_WORDS", 5)

# ─── Retrieval Configuration ────────────────────────────────────────────────
TOP_K = _env_int("TOP_K", 5)

# Phase 2: allow several chunks from the same PDF (was effectively 1).
MAX_CHUNKS_PER_SOURCE = _env_int("MAX_CHUNKS_PER_SOURCE", 3)
# Phase 2: was 2000, which cut off chunks 4–5 before the LLM ever saw them.
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 6000)

# Phase 3: "dense" (FAISS only), "bm25" (keywords only) or "hybrid" (both, fused with RRF)
RETRIEVAL_MODE = _env_str("RETRIEVAL_MODE", "hybrid").lower()
CANDIDATE_POOL = _env_int("CANDIDATE_POOL", 30)   # candidates per retriever before reranking
RRF_K = _env_int("RRF_K", 60)                     # standard Reciprocal Rank Fusion constant

# Phase 3: cross-encoder reranker (small English model; queries are translated to
# English before retrieval, and the documents are English).
USE_RERANKER = _env_bool("USE_RERANKER", True)
RERANKER_MODEL = _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Phase 3: also send the chunk before/after each hit (same page) to the LLM.
EXPAND_NEIGHBORS = _env_bool("EXPAND_NEIGHBORS", False)

# Dense-only floor applied inside FAISS search (only in RETRIEVAL_MODE=dense).
SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.3)

# ─── Answer / abstention thresholds (Phase 5: tune with eval/tune_threshold.py) ──
# If the best chunk's confidence is below the threshold, the bot says it has no info.
# With the reranker on, confidence = sigmoid(reranker score) in 0..1.
# 0.30 chosen with eval/tune_threshold.py (blocks ~74% of out-of-scope questions,
# wrongly blocks ~2% of answerable ones on the synthetic set).
MIN_RERANK_SCORE = _env_float("MIN_RERANK_SCORE", 0.30)
# With the reranker off, confidence = dense cosine similarity. e5 models give
# compressed, high cosine scores (~0.7–0.9), so they need a higher cut-off.
MIN_ANSWER_SCORE = _env_float(
    "MIN_ANSWER_SCORE", 0.80 if "e5" in EMBEDDING_MODEL.lower() else 0.45
)

# ─── Query Understanding (Phase 4) ──────────────────────────────────────────
ENABLE_QUERY_REWRITE = _env_bool("ENABLE_QUERY_REWRITE", True)
ENABLE_SERVICE_ROUTING = _env_bool("ENABLE_SERVICE_ROUTING", True)
QUERY_REWRITE_HISTORY_TURNS = _env_int("QUERY_REWRITE_HISTORY_TURNS", 3)

# ─── Answer Quality & Trust (Phase 5) ───────────────────────────────────────
ENABLE_INLINE_CITATIONS = _env_bool("ENABLE_INLINE_CITATIONS", True)
# Share of a sentence's content words that must appear in its cited chunk(s).
CITATION_SUPPORT_THRESHOLD = _env_float("CITATION_SUPPORT_THRESHOLD", 0.5)

PREFER_RECENT_DOCS = _env_bool("PREFER_RECENT_DOCS", True)
# Only applied between yearly editions of the same document (see Retriever._recency_bonus).
RECENCY_WEIGHT = _env_float("RECENCY_WEIGHT", 0.05)

ENABLE_PII_REDACTION = _env_bool("ENABLE_PII_REDACTION", True)
ENABLE_INJECTION_GUARD = _env_bool("ENABLE_INJECTION_GUARD", True)

# ─── FAISS Configuration ────────────────────────────────────────────────────
FAISS_INDEX_FILE = VECTORSTORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTORSTORE_DIR / "metadata.pkl"

# ─── LLM Configuration ──────────────────────────────────────────────────────
# "gemini" (default) or "groq"
LLM_PROVIDER = _env_str("LLM_PROVIDER", "gemini").lower()

# GEMINI_API_KEY is preferred; GOOGLE_API_KEY and the old generic API_KEY also work.
API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("API_KEY", "")
).strip()

_DEFAULT_MODELS = {
    "gemini": ("gemini-3.6-flash", "gemini-3.8-flash,gemini-3.5-flash-lite"),
    "groq": ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
}
_default_model, _default_fallbacks = _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["gemini"])

# Primary model
MODEL = _env_str("MODEL", _default_model)

# Fallback models (string → list), tried in order when the primary fails
FALLBACK_MODELS = _env_list("FALLBACK_MODELS", _default_fallbacks)

TEMPERATURE = _env_float("TEMPERATURE", 0.3)
# Gemini "thinking" tokens count against this limit, so keep it generous.
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 4096)
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 60)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
LLM_MAX_RETRY_WAIT_SECONDS = _env_int("LLM_MAX_RETRY_WAIT_SECONDS", 10)
# Optional: "minimal" / "low" / "medium" / "high". Empty = model default.
GEMINI_THINKING_LEVEL = _env_str("GEMINI_THINKING_LEVEL", "")

# ─── Conversation Memory ────────────────────────────────────────────────────
MAX_MEMORY_TURNS = 10

# ─── Supported Languages ────────────────────────────────────────────────────
LANGUAGES = {
    "en": {"name": "English (अंग्रेज़ी)", "flag": "🇬🇧"},
    "hi": {"name": "हिन्दी (Hindi)", "flag": "🇮🇳"},
}

DEFAULT_LANGUAGE = "en"
