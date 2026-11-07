#!/usr/bin/env python3
"""
Build the ablation table from saved runs, and optionally judge two runs head-to-head.

Usage:
    python eval/compare.py                                   # all runs in eval/results
    python eval/compare.py baseline p2_context p3_hybrid full
    python eval/compare.py --pairwise baseline full --limit 30
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import EVAL_RESULTS_DIR, Pacer, read_jsonl

# (column title, path inside summary.json)
COLUMNS = [
    ("Hit@1", ("retrieval", "page_level", "hit@1")),
    ("Hit@5", ("retrieval", "page_level", "hit@5")),
    ("MRR", ("retrieval", "page_level", "mrr")),
    ("nDCG@10", ("retrieval", "page_level", "ndcg@10")),
    ("Hindi Hit@5", ("hindi", "page_level", "hit@5")),
    ("Follow-up Hit@5", ("followup", "after_rewrite", "hit@5")),
    ("Gate refusal (unans.)", ("gate", "refusal_rate_gate")),
    ("Faithful", ("e2e", "faithfulness")),
    ("Relevant", ("e2e", "answer_relevance")),
    ("Correct", ("e2e", "correctness")),
    ("False refusal", ("e2e", "false_refusal_rate")),
    ("Refusal (unans.)", ("e2e", "refusal_rate_unanswerable")),
    ("Retrieval p50 ms", ("retrieval", "latency_ms_p50")),
    ("Answer p50 ms", ("e2e", "latency_total_ms_p50")),
]


def _get(summary: dict, path: tuple):
    value = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _fmt(value, title: str) -> str:
    if value is None:
        return "–"
    if "ms" in title:
        return f"{value:.0f}"
    return f"{value:.3f}"


def table(names: list[str]) -> str:
    runs = []
    for name in names:
        path = EVAL_RESULTS_DIR / name / "summary.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
        else:
            print(f"(skipping {name}: no summary.json)")

    header = "| Run | " + " | ".join(title for title, _ in COLUMNS) + " |"
    divider = "|" + "---|" * (len(COLUMNS) + 1)
    rows = [
        f"| {run['name']} | " + " | ".join(_fmt(_get(run, path), title) for title, path in COLUMNS) + " |"
        for run in runs
    ]
    return "\n".join([header, divider, *rows])


def pairwise(name_1: str, name_2: str, limit: int) -> None:
    from eval.judge import judge_pairwise
    from src.llm.client import llmClient

    def answers(name: str) -> dict:
        return {
            r["id"]: r for r in read_jsonl(EVAL_RESULTS_DIR / name / "per_question.jsonl")
            if r.get("section") == "e2e"
        }

    first, second = answers(name_1), answers(name_2)
    shared = sorted(set(first) & set(second))[: limit or None]
    if not shared:
        raise SystemExit("No shared e2e questions. Run both with the same --seed and --limit-e2e.")

    llm, pacer = llmClient(), Pacer()
    tally = {"1": 0, "2": 0, "tie": 0}
    for qid in shared:
        pacer.wait()
        verdict = judge_pairwise(llm, first[qid]["question"], first[qid]["answer"], second[qid]["answer"])
        if verdict:
            tally[verdict] += 1

    total = sum(tally.values()) or 1
    print(f"\n{name_2} vs {name_1} on {total} questions:")
    print(f"  {name_2} wins: {tally['2'] / total:.1%}")
    print(f"  {name_1} wins: {tally['1'] / total:.1%}")
    print(f"  ties:          {tally['tie'] / total:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run names (default: all)")
    parser.add_argument("--pairwise", nargs=2, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", type=Path, help="also write the table to this markdown file")
    args = parser.parse_args()

    if args.pairwise:
        pairwise(*args.pairwise, args.limit)
        return

    names = args.runs or sorted(p.parent.name for p in EVAL_RESULTS_DIR.glob("*/summary.json"))
    markdown = table(names)
    print(markdown)
    if args.out:
        args.out.write_text(markdown + "\n")


if __name__ == "__main__":
    main()
