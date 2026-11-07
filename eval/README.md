# InfoBridge evaluation

There's no hand-labelled gold set. Every dataset is generated from the indexed PDFs, and each metric has a known correct answer or needs none.

| Dataset | Built from | Measures |
|---|---|---|
| `data/synthetic.jsonl` | an LLM writes a question for a random chunk | Hit@k, MRR and nDCG@10 (the gold answer is that chunk's page); answer correctness against the chunk's answer |
| `data/unanswerable.jsonl` | 20 seed questions plus LLM-generated out-of-scope and false-premise ones | refusal rate |
| `data/hindi.jsonl` | Hindi translations of synthetic questions | Hindi retrieval and overlap with the English results |
| `data/followup.jsonl` | (first question, short follow-up) pairs | how much query rewriting helps |

Answer quality is scored by an LLM judge without reference answers: faithfulness (hallucination), answer relevance and context relevance.

## Run order

```bash
# 0. Keep the ORIGINAL index for the baseline (once, before rebuilding)
cp -r vectorstore_data vectorstore_baseline

# 1. Rebuild the index with the new chunker + multilingual embedder
python scripts/build_index.py
python scripts/check_unindexed.py        # which PDFs are still missing, and why

# 2. Generate datasets (resumable; ~4 s between LLM calls for the free tier)
python eval/generate_datasets.py

# 3. One run per phase (same data and seed, so they are comparable)
python eval/run_eval.py --preset baseline
python eval/run_eval.py --preset p2_context
python eval/run_eval.py --preset p3_hybrid
python eval/run_eval.py --preset p3_rerank
python eval/run_eval.py --preset p4_query
python eval/run_eval.py --preset full

# 4. Ablation table, head-to-head A/B, threshold tuning
python eval/compare.py --out eval/results/ablation.md
python eval/compare.py --pairwise baseline full
python eval/tune_threshold.py full
```

## Useful options
- `--skip e2e` runs retrieval only. It needs no LLM calls, so it's fast and free.
- `--set KEY=VALUE` overrides any setting from `config/settings.py`, for example `--set CANDIDATE_POOL=50`.
- `EVAL_LLM_DELAY=6` adds more seconds between LLM calls if you hit 429 rate-limit errors.
- `EVAL_JUDGE_MODEL=gemini-3.8-flash` judges with a different model than the one that wrote the answers.

## Caveats
- Synthetic questions are easier than real ones, because they're worded close to their chunk. Compare runs against each other; don't treat the numbers as absolute.
- `baseline` to `p3_*` use the old index. `p4_query` and `full` use the rebuilt one, so the jump at `p4_query` includes both the new chunker and the new embedder.
- LLM-judge scores are approximate. Rating about 30 answers yourself and checking how often you agree with the judge makes them credible.
