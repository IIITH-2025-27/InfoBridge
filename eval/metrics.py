"""
Retrieval and summary metrics.

Each synthetic question has exactly one gold location (the page its chunk came
from), so the metrics use a single relevant item:
    Hit@k  = gold found in the top k
    MRR    = 1 / rank of the gold item (0 if missing)
    nDCG@k = 1 / log2(rank + 1) if rank <= k (ideal DCG is 1)
"""

import math
from typing import Optional

import numpy as np


def hit_at_k(rank: Optional[int], k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(rank: Optional[int], k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def first_rank(retrieved_keys: list, gold_key) -> Optional[int]:
    for i, key in enumerate(retrieved_keys, start=1):
        if key == gold_key:
            return i
    return None


def summarize_ranks(ranks: list[Optional[int]], ks=(1, 3, 5, 10)) -> dict:
    if not ranks:
        return {}
    summary = {f"hit@{k}": float(np.mean([hit_at_k(r, k) for r in ranks])) for k in ks}
    summary["mrr"] = float(np.mean([reciprocal_rank(r) for r in ranks]))
    summary["ndcg@10"] = float(np.mean([ndcg_at_k(r, 10) for r in ranks]))
    summary["n"] = len(ranks)
    return summary


def mean(values: list[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def percentile(values: list[float], q: float) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def jaccard(a: set, b: set) -> Optional[float]:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)
