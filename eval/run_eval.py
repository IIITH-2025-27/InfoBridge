#!/usr/bin/env python3
"""
Run the evaluation for one configuration and save the results.

Sections (each can be limited or skipped):
  1. retrieval     synthetic questions → Hit@k, MRR, nDCG@10 (page- and chunk-level), latency
  2. hindi         Hindi versions → same metrics + overlap with the English results
  3. followup      follow-up questions with history → Hit@5 with and without rewriting
  4. gate          unanswerable questions → share refused by the confidence threshold
  5. e2e           full answers → LLM-judge faithfulness / relevance / correctness,
                   false-refusal rate, refusal rate on unanswerable, citation checks

Sections 1 and 4 need no LLM calls. 2 and 3 need one call per question (rewrite /
translation). 5 needs ~2–4 calls per question, paced by EVAL_LLM_DELAY seconds.

Usage:
    python eval/run_eval.py --preset baseline
    python eval/run_eval.py --preset full --limit-e2e 40
    python eval/run_eval.py --name my_test --set USE_RERANKER=false --skip e2e
Results: eval/results/<name>/summary.json and per_question.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import (
    EVAL_DATA_DIR,
    EVAL_RESULTS_DIR,
    Pacer,
    apply_overrides,
    config_snapshot,
    is_refusal,
    load_pipeline,
    page_key,
    read_jsonl,
    write_jsonl,
)
from eval.metrics import first_rank, jaccard, mean, percentile, summarize_ranks

METRIC_DEPTH = 10  # retrieve this many chunks when measuring ranks


def _ranks(results: list[dict], item: dict) -> tuple:
    pages = [page_key(r["metadata"]) for r in results]
    chunks = [r["metadata"].get("chunk_id") for r in results]
    gold_page = (str(item["source_file"]), str(item["page"]))
    return first_rank(pages, gold_page), first_rank(chunks, item["chunk_id"]), pages


def eval_retrieval(retriever, items: list[dict], records: list[dict]) -> dict:
    page_ranks, chunk_ranks, latencies, confidences = [], [], [], []

    for item in items:
        t0 = time.perf_counter()
        results = retriever.retrieve(item["question"], top_k=METRIC_DEPTH)
        latency = (time.perf_counter() - t0) * 1000

        page_rank, chunk_rank, pages = _ranks(results, item)
        top_conf = max((r.get("confidence", 0.0) for r in results), default=0.0)
        page_ranks.append(page_rank)
        chunk_ranks.append(chunk_rank)
        latencies.append(latency)
        confidences.append(top_conf)

        records.append({
            "section": "retrieval",
            "id": item["id"],
            "question": item["question"],
            "service": item["service"],
            "page_rank": page_rank,
            "chunk_rank": chunk_rank,
            "top_confidence": top_conf,
            "confidence_type": results[0].get("confidence_type") if results else None,
            "answerable_gate": retriever.is_answerable(results),
            "retrieved_pages": pages[:5],
            "latency_ms": latency,
        })

    by_service: dict[str, list] = {}
    for item, rank in zip(items, page_ranks):
        by_service.setdefault(item["service"], []).append(rank)

    return {
        "page_level": summarize_ranks(page_ranks),
        "chunk_level": summarize_ranks(chunk_ranks),
        "page_level_by_service": {s: summarize_ranks(r) for s, r in sorted(by_service.items())},
        "false_refusal_rate_gate": mean([0.0 if r["answerable_gate"] else 1.0 for r in records if r["section"] == "retrieval"]),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
    }


def eval_hindi(retriever, generator, items: list[dict], english_pages: dict, records: list[dict], pacer: Pacer) -> dict:
    ranks, overlaps = [], []
    for item in items:
        pacer.wait()
        plan = generator.query_processor.process(item["question"], [])
        results = retriever.retrieve(plan.retrieval_query, top_k=METRIC_DEPTH)
        page_rank, _, pages = _ranks(results, item)
        ranks.append(page_rank)

        english = english_pages.get(item["source_id"])
        overlap = jaccard(set(pages[:5]), set(map(tuple, english))) if english is not None else None
        overlaps.append(overlap)

        records.append({
            "section": "hindi",
            "id": item["id"],
            "question": item["question"],
            "retrieval_query": plan.retrieval_query,
            "page_rank": page_rank,
            "top5_overlap_with_english": overlap,
        })

    return {"page_level": summarize_ranks(ranks), "top5_overlap_with_english": mean(overlaps)}


def eval_followup(retriever, generator, items: list[dict], records: list[dict], pacer: Pacer) -> dict:
    raw_ranks, rewritten_ranks, standalone_ranks = [], [], []
    for item in items:
        history = [
            {"role": "user", "content": item["first"]},
            {"role": "assistant", "content": "(answer about the first question)"},
        ]
        pacer.wait()
        plan = generator.query_processor.process(item["followup"], history)

        raw_rank, _, _ = _ranks(retriever.retrieve(item["followup"], top_k=METRIC_DEPTH), item)
        rewritten_rank, _, _ = _ranks(retriever.retrieve(plan.retrieval_query, top_k=METRIC_DEPTH), item)
        standalone_rank, _, _ = _ranks(retriever.retrieve(item["standalone"], top_k=METRIC_DEPTH), item)

        raw_ranks.append(raw_rank)
        rewritten_ranks.append(rewritten_rank)
        standalone_ranks.append(standalone_rank)
        records.append({
            "section": "followup",
            "id": item["id"],
            "followup": item["followup"],
            "rewritten_query": plan.retrieval_query,
            "rewritten": plan.rewritten,
            "raw_rank": raw_rank,
            "rewritten_rank": rewritten_rank,
            "standalone_rank": standalone_rank,
        })

    return {
        "as_typed": summarize_ranks(raw_ranks),
        "after_rewrite": summarize_ranks(rewritten_ranks),
        "standalone_upper_bound": summarize_ranks(standalone_ranks),
    }


def eval_gate(retriever, items: list[dict], records: list[dict]) -> dict:
    refused = []
    for item in items:
        results = retriever.retrieve(item["question"], top_k=5)
        answerable = retriever.is_answerable(results)
        refused.append(0.0 if answerable else 1.0)
        records.append({
            "section": "gate",
            "id": item["id"],
            "question": item["question"],
            "top_confidence": max((r.get("confidence", 0.0) for r in results), default=0.0),
            "confidence_type": results[0].get("confidence_type") if results else None,
            "refused_by_gate": not answerable,
        })
    return {"refusal_rate_gate": mean(refused), "n": len(items)}


def eval_e2e(generator, answerable: list[dict], unanswerable: list[dict], records: list[dict], pacer: Pacer, judge: bool) -> dict:
    from eval.judge import judge_answer
    from src.llm.client import is_llm_error

    scores: dict[str, list] = {k: [] for k in ("faithfulness", "answer_relevance", "context_relevance", "correctness")}
    false_refusals, llm_errors, totals, unsupported, cited = [], 0, [], [], []

    for item in answerable:
        generator.clear_memory()
        pacer.wait()
        response = generator.generate(item["question"], language="en")
        answer = response["answer"]
        totals.append(response.get("timings", {}).get("total_ms"))

        if response.get("used_retrieval_fallback") or is_llm_error(answer):
            llm_errors += 1
            continue

        refused = is_refusal(answer)
        false_refusals.append(1.0 if refused else 0.0)
        cited.append(1.0 if "[" in answer else 0.0)
        unsupported.append(len(response.get("unsupported_sentences", [])))

        record = {
            "section": "e2e",
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "refused": refused,
            "sources": [(s["source_file"], s["page"]) for s in response["sources"]],
            "unsupported_sentences": response.get("unsupported_sentences", []),
            "timings": response.get("timings"),
        }

        if judge:
            pacer.wait()
            verdict = judge_answer(
                generator.llm,
                item["question"],
                answer,
                response.get("contexts", []),
                reference=item.get("reference_answer"),
            )
            if verdict:
                record.update(verdict)
                for key in scores:
                    if verdict.get(key) is not None:
                        scores[key].append(verdict[key])
        records.append(record)

    refusals = []
    for item in unanswerable:
        generator.clear_memory()
        pacer.wait()
        response = generator.generate(item["question"], language="en")
        if response.get("used_retrieval_fallback") or is_llm_error(response["answer"]):
            llm_errors += 1
            continue
        refused = is_refusal(response["answer"]) or not response.get("answerable", True)
        refusals.append(1.0 if refused else 0.0)
        records.append({
            "section": "e2e_unanswerable",
            "id": item["id"],
            "question": item["question"],
            "answer": response["answer"],
            "refused": refused,
        })

    summary = {key: mean(values) for key, values in scores.items()}
    summary.update({
        "false_refusal_rate": mean(false_refusals),
        "refusal_rate_unanswerable": mean(refusals),
        "answers_with_citation_markers": mean(cited),
        "avg_unsupported_sentences": mean(unsupported),
        "latency_total_ms_p50": percentile(totals, 50),
        "latency_total_ms_p95": percentile(totals, 95),
        "llm_errors_skipped": llm_errors,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", help="named config from eval/presets.py")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra setting override")
    parser.add_argument("--name", help="results folder name (default: preset name)")
    parser.add_argument("--skip", action="append", default=[], choices=["retrieval", "hindi", "followup", "gate", "e2e"])
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judge scoring in e2e")
    parser.add_argument("--limit-retrieval", type=int, default=0, help="0 = all synthetic questions")
    parser.add_argument("--limit-hindi", type=int, default=30)
    parser.add_argument("--limit-followup", type=int, default=30)
    parser.add_argument("--limit-e2e", type=int, default=30, help="answerable questions for e2e")
    parser.add_argument("--limit-e2e-unanswerable", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    overrides = apply_overrides(args.preset, args.set)
    name = args.name or args.preset or "run"
    out_dir = EVAL_RESULTS_DIR / name
    rng = random.Random(args.seed)

    synthetic = read_jsonl(EVAL_DATA_DIR / "synthetic.jsonl")
    unanswerable = read_jsonl(EVAL_DATA_DIR / "unanswerable.jsonl")
    hindi = read_jsonl(EVAL_DATA_DIR / "hindi.jsonl")
    followup = read_jsonl(EVAL_DATA_DIR / "followup.jsonl")
    if not synthetic:
        raise SystemExit("No eval data. Run: python eval/generate_datasets.py")

    _, retriever, generator = load_pipeline()
    pacer = Pacer()
    records: list[dict] = []
    summary: dict = {
        "name": name,
        "preset": args.preset,
        "overrides": overrides,
        "config": config_snapshot(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def sample(items: list[dict], limit: int) -> list[dict]:
        if not limit or limit >= len(items):
            return items
        return rng.sample(items, limit)

    if "retrieval" not in args.skip:
        print("→ retrieval")
        summary["retrieval"] = eval_retrieval(retriever, sample(synthetic, args.limit_retrieval), records)

    if "hindi" not in args.skip and hindi:
        print("→ hindi")
        english_pages = {r["id"]: r["retrieved_pages"] for r in records if r["section"] == "retrieval"}
        summary["hindi"] = eval_hindi(retriever, generator, sample(hindi, args.limit_hindi), english_pages, records, pacer)

    if "followup" not in args.skip and followup:
        print("→ followup")
        summary["followup"] = eval_followup(retriever, generator, sample(followup, args.limit_followup), records, pacer)

    if "gate" not in args.skip and unanswerable:
        print("→ gate")
        summary["gate"] = eval_gate(retriever, unanswerable, records)

    if "e2e" not in args.skip:
        print("→ e2e")
        summary["e2e"] = eval_e2e(
            generator,
            sample(synthetic, args.limit_e2e),
            sample(unanswerable, args.limit_e2e_unanswerable),
            records,
            pacer,
            judge=not args.no_judge,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_jsonl(out_dir / "per_question.jsonl", records)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "overrides")}, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
