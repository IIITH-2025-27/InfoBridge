"""
Shared helpers for the eval scripts.

IMPORTANT: config/settings.py reads the environment at import time, so
apply_overrides() must run before anything from config/ or src/ is imported.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA_DIR = ROOT / "eval" / "data"
EVAL_RESULTS_DIR = ROOT / "eval" / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFUSAL_MARKERS = (
    "don't have sufficient information",
    "do not have sufficient information",
    "not have enough information",
    "पर्याप्त जानकारी नहीं",
)


def apply_overrides(preset: Optional[str], pairs: Optional[list[str]]) -> dict[str, str]:
    """Set environment variables from a preset and KEY=VALUE pairs (pairs win)."""
    from eval.presets import PRESETS

    overrides: dict[str, str] = {}
    if preset:
        if preset not in PRESETS:
            raise SystemExit(f"Unknown preset '{preset}'. Choose from: {', '.join(PRESETS)}")
        overrides.update(PRESETS[preset])

    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got '{pair}'")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()

    os.environ.update(overrides)
    return overrides


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_json(raw: str) -> Optional[dict]:
    """Parse a JSON object from an LLM reply (tolerates ``` fences / extra text)."""
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"items": data}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


class Pacer:
    """Keeps LLM calls under the free-tier requests-per-minute limit."""

    def __init__(self, seconds: Optional[float] = None):
        self.seconds = float(os.getenv("EVAL_LLM_DELAY", "4")) if seconds is None else seconds
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.seconds:
            time.sleep(self.seconds - elapsed)
        self._last = time.monotonic()


def is_refusal(answer: str) -> bool:
    lower = (answer or "").lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def page_key(meta: dict) -> tuple[str, str]:
    return str(meta.get("source_file")), str(meta.get("page"))


def config_snapshot() -> dict:
    """The settings that matter for comparing runs."""
    from config import settings

    keys = [
        "LLM_PROVIDER", "MODEL", "EMBEDDING_MODEL", "VECTORSTORE_DIR", "RETRIEVAL_MODE",
        "USE_RERANKER", "RERANKER_MODEL", "EXPAND_NEIGHBORS", "CANDIDATE_POOL",
        "MAX_CHUNKS_PER_SOURCE", "MAX_CONTEXT_CHARS", "TOP_K", "MIN_ANSWER_SCORE",
        "MIN_RERANK_SCORE", "ENABLE_QUERY_REWRITE", "ENABLE_SERVICE_ROUTING",
        "ENABLE_INLINE_CITATIONS", "PREFER_RECENT_DOCS", "ENABLE_PII_REDACTION",
        "ENABLE_INJECTION_GUARD",
    ]
    return {k: str(getattr(settings, k, None)) for k in keys}


def load_pipeline():
    """(vector_store, retriever, generator) built the same way as the app."""
    from src.vectorstore.store import VectorStore
    from src.retrieval.retriever import Retriever
    from src.llm.generator import ResponseGenerator
    from src.memory.chat_memory import ChatMemory

    store = VectorStore()
    if not store.load():
        raise SystemExit("Vector store not found. Run: python scripts/build_index.py")

    retriever = Retriever(vector_store=store)
    generator = ResponseGenerator(retriever=retriever, memory=ChatMemory())
    return store, retriever, generator
