"""
Named configurations for the ablation table.

Each preset is a set of environment overrides applied before the project's
settings are imported. Run one with:
    python eval/run_eval.py --preset baseline

Index note: "baseline" … "p3_rerank" use the ORIGINAL index (MiniLM, old chunker).
Before rebuilding, copy it once:  cp -r vectorstore_data vectorstore_baseline
"p4_query" and "full" use the rebuilt index (e5 + new chunker) in vectorstore_data.
"""

_OLD_INDEX = {
    "VECTORSTORE_DIR": "vectorstore_baseline",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "MIN_ANSWER_SCORE": "0.45",
}

_ALL_NEW_FEATURES_OFF = {
    "RETRIEVAL_MODE": "dense",
    "USE_RERANKER": "false",
    "EXPAND_NEIGHBORS": "false",
    "ENABLE_QUERY_REWRITE": "false",
    "ENABLE_SERVICE_ROUTING": "false",
    "ENABLE_INLINE_CITATIONS": "false",
    "PREFER_RECENT_DOCS": "false",
    "ENABLE_PII_REDACTION": "false",
    "ENABLE_INJECTION_GUARD": "false",
}

PRESETS: dict[str, dict[str, str]] = {
    # The project as it was: dense only, 1 chunk per PDF, 2000-char context
    "baseline": {
        **_OLD_INDEX,
        **_ALL_NEW_FEATURES_OFF,
        "MAX_CHUNKS_PER_SOURCE": "1",
        "MAX_CONTEXT_CHARS": "2000",
    },
    # Phase 2: 3 chunks per PDF + 6000-char context (same old index)
    "p2_context": {
        **_OLD_INDEX,
        **_ALL_NEW_FEATURES_OFF,
        "MAX_CHUNKS_PER_SOURCE": "3",
        "MAX_CONTEXT_CHARS": "6000",
    },
    # Phase 3a: + BM25 hybrid search with RRF
    "p3_hybrid": {
        **_OLD_INDEX,
        **_ALL_NEW_FEATURES_OFF,
        "RETRIEVAL_MODE": "hybrid",
    },
    # Phase 3b: + cross-encoder reranker
    "p3_rerank": {
        **_OLD_INDEX,
        **_ALL_NEW_FEATURES_OFF,
        "RETRIEVAL_MODE": "hybrid",
        "USE_RERANKER": "true",
    },
    # Phase 3c (optional): + neighbour chunks
    "p3_neighbors": {
        **_OLD_INDEX,
        **_ALL_NEW_FEATURES_OFF,
        "RETRIEVAL_MODE": "hybrid",
        "USE_RERANKER": "true",
        "EXPAND_NEIGHBORS": "true",
    },
    # Phase 4: rebuilt index (e5 + chunker fixes) + query rewriting + routing
    "p4_query": {
        **_ALL_NEW_FEATURES_OFF,
        "RETRIEVAL_MODE": "hybrid",
        "USE_RERANKER": "true",
        "ENABLE_QUERY_REWRITE": "true",
        "ENABLE_SERVICE_ROUTING": "true",
    },
    # Phase 5: everything on (= the app's defaults)
    "full": {},
}
