#!/usr/bin/env python3
"""
Choose the "should I answer?" threshold from data instead of by hand (Phase 5).

Uses a saved run's top-1 retrieval confidence for
    answerable questions   (synthetic set, "retrieval" section)  → should answer
    unanswerable questions ("gate" section)                       → should refuse
and sweeps thresholds, reporting refusal of unanswerable questions vs false
refusals of answerable ones. Prints the setting to put in .env.

Usage:
    python eval/tune_threshold.py full
    python eval/tune_threshold.py full --max-false-refusal 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import EVAL_RESULTS_DIR, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run name in eval/results")
    parser.add_argument(
        "--max-false-refusal", type=float, default=0.10,
        help="highest acceptable share of answerable questions refused",
    )
    args = parser.parse_args()

    records = read_jsonl(EVAL_RESULTS_DIR / args.run / "per_question.jsonl")
    positives = [r["top_confidence"] for r in records if r.get("section") == "retrieval"]
    negatives = [r["top_confidence"] for r in records if r.get("section") == "gate"]
    kinds = {r.get("confidence_type") for r in records if r.get("section") in ("retrieval", "gate")}

    if not positives or not negatives:
        raise SystemExit("The run needs both the 'retrieval' and 'gate' sections.")

    setting = "MIN_RERANK_SCORE" if "rerank" in kinds else "MIN_ANSWER_SCORE"
    positives_arr, negatives_arr = np.array(positives), np.array(negatives)

    candidates = np.unique(np.round(np.concatenate([positives_arr, negatives_arr]), 3))
    rows = []
    for threshold in candidates:
        false_refusal = float(np.mean(positives_arr < threshold))
        correct_refusal = float(np.mean(negatives_arr < threshold))
        # Balanced accuracy: the two error types weigh equally
        balanced = (correct_refusal + (1 - false_refusal)) / 2
        rows.append((float(threshold), correct_refusal, false_refusal, balanced))

    best_balanced = max(rows, key=lambda r: r[3])
    within_budget = [r for r in rows if r[2] <= args.max_false_refusal]
    best_safe = max(within_budget, key=lambda r: r[1]) if within_budget else None

    print(f"Confidence type: {', '.join(sorted(k for k in kinds if k))}  →  setting {setting}")
    print(f"Answerable: {len(positives)}  Unanswerable: {len(negatives)}\n")
    print(f"{'threshold':>10} {'refuses unans.':>15} {'false refusal':>14} {'balanced acc.':>14}")
    step = max(1, len(rows) // 15)
    for threshold, correct, false, balanced in rows[::step]:
        print(f"{threshold:>10.3f} {correct:>15.1%} {false:>14.1%} {balanced:>14.3f}")

    print(f"\nBest balanced accuracy: {setting}={best_balanced[0]:.3f} "
          f"(refuses {best_balanced[1]:.0%} unanswerable, {best_balanced[2]:.0%} false refusals)")
    if best_safe:
        print(f"Best with ≤{args.max_false_refusal:.0%} false refusals: {setting}={best_safe[0]:.3f} "
              f"(refuses {best_safe[1]:.0%} unanswerable)")
    print("\nNote: synthetic questions are easier than real ones; prefer the lower of the two.")


if __name__ == "__main__":
    main()
