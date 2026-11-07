#!/usr/bin/env python3
"""
Build the eval datasets automatically (no hand labelling).

    synthetic.jsonl     LLM writes a question answered by a sampled chunk; that
                        chunk's page is the gold answer for retrieval metrics.
    unanswerable.jsonl  Out-of-scope / false-premise questions the bot must refuse.
    hindi.jsonl         Hindi translations of synthetic questions (same gold page).
    followup.jsonl      (first question, short follow-up) pairs for query rewriting.

Resumable: re-running skips items already written (useful when the free tier
runs out of quota halfway).

Usage:
    python eval/generate_datasets.py                       # everything, defaults
    python eval/generate_datasets.py --only synthetic --per-service 40
"""

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import EVAL_DATA_DIR, Pacer, append_jsonl, apply_overrides, parse_json, read_jsonl

# Hand-written seeds; the LLM adds more of the same kinds.
UNANSWERABLE_SEEDS = [
    "How do I apply for a new ration card?",
    "How can I book a Tatkal railway ticket on IRCTC?",
    "What is the process to get a birth certificate from the municipal corporation?",
    "How do I register my business for GST?",
    "How can I withdraw my EPF balance online?",
    "How do I update the address in my Aadhaar card?",
    "What is the procedure to get a caste certificate in Maharashtra?",
    "How do I apply for an LPG gas connection under Ujjwala Yojana?",
    "How can I register a property sale deed?",
    "How do I apply for a PM Kisan Samman Nidhi installment?",
    "What documents are needed for a marriage certificate?",
    "How do I get a domicile certificate in Uttar Pradesh?",
    "What is the passport fee for travelling to Mars?",
    "Can I vote in Indian elections from the moon?",
    "What is the income tax rate for assessment year 2040-41?",
    "How do I get a driving licence for a spaceship?",
    "What is the stock price of Reliance today?",
    "Who won the cricket world cup in 2011?",
    "Write me a poem about the monsoon.",
    "What is the best smartphone under 20000 rupees?",
]

_GOOD_CHUNK_MIN_WORDS = 40


def _usable(text: str) -> bool:
    words = text.split()
    if len(words) < _GOOD_CHUNK_MIN_WORDS:
        return False
    # Skip tables of contents / indexes: mostly numbers and dots
    digits = sum(ch.isdigit() for ch in text)
    dots = text.count("....")
    return digits / max(len(text), 1) < 0.25 and dots < 3


def gen_synthetic(llm, store, per_service: int, seed: int, pacer: Pacer) -> None:
    path = EVAL_DATA_DIR / "synthetic.jsonl"
    done = {r["chunk_id"] for r in read_jsonl(path)}
    skipped_path = EVAL_DATA_DIR / ".synthetic_skipped.txt"
    skipped = set(skipped_path.read_text().split("\n")) if skipped_path.exists() else set()

    rng = random.Random(seed)
    by_service: dict[str, list[int]] = {}
    for i, meta in enumerate(store.metadata):
        if _usable(store.texts[i]):
            by_service.setdefault(meta.get("service_type", "unknown"), []).append(i)

    for service, positions in sorted(by_service.items()):
        rng.shuffle(positions)
        have = sum(1 for r in read_jsonl(path) if r["service"] == service)
        print(f"[synthetic] {service}: {have}/{per_service}")

        for position in positions:
            if have >= per_service:
                break
            meta = store.metadata[position]
            chunk_id = meta.get("chunk_id")
            if chunk_id in done or chunk_id in skipped:
                continue

            pacer.wait()
            raw = llm.generate(
                prompt=f"""Passage from the Indian government document "{meta.get('source_file')}":
\"\"\"{store.texts[position]}\"\"\"

Write ONE question a citizen might ask that this passage answers.
- Paraphrase: do not copy long phrases from the passage.
- Make it self-contained (mention the service, form or scheme it is about).
- It must be answerable from this passage alone. Under 25 words.
- Also give a short answer taken from the passage.
If the passage has no useful citizen-facing information (headers, index, table of
contents, signatures), return {{"skip": true}}.

Return JSON: {{"question": "...", "answer": "..."}}""",
                system_instruction="You create evaluation data. Output only JSON.",
                json_mode=True,
                temperature=0.7,
                max_tokens=1024,
            )
            data = parse_json(raw)
            if not data or data.get("skip") or not data.get("question"):
                skipped.add(chunk_id)
                skipped_path.write_text("\n".join(sorted(skipped)))
                continue

            append_jsonl(path, {
                "id": f"syn-{len(done) + 1:04d}",
                "question": data["question"].strip(),
                "reference_answer": str(data.get("answer", "")).strip(),
                "service": service,
                "source_file": meta.get("source_file"),
                "page": meta.get("page"),
                "chunk_id": chunk_id,
            })
            done.add(chunk_id)
            have += 1


def gen_unanswerable(llm, total: int, pacer: Pacer) -> None:
    path = EVAL_DATA_DIR / "unanswerable.jsonl"
    existing = read_jsonl(path)
    questions = [r["question"] for r in existing]

    for seed_question in UNANSWERABLE_SEEDS:
        if len(questions) >= total:
            break
        if seed_question not in questions:
            append_jsonl(path, {"id": f"una-{len(questions) + 1:04d}", "question": seed_question, "kind": "seed"})
            questions.append(seed_question)

    if len(questions) >= total:
        print(f"[unanswerable] {len(questions)}/{total}")
        return

    pacer.wait()
    raw = llm.generate(
        prompt=f"""Write {total - len(questions)} realistic questions an Indian citizen might ask a
government-services chatbot that only knows about: Passport, Voter ID, Driving Licence /
vehicle RC, Income Tax and Ayushman Bharat (PM-JAY).

The questions must NOT be answerable from documents about those five services:
- about half on OTHER government services (ration card, pension, scholarships, land records, ...)
- a few with a false premise or impossible detail about the five services
- a few off-topic (general knowledge, opinions)
Mix English and a few Hinglish ones. No duplicates of: {questions[:10]}

Return JSON: {{"questions": ["...", "..."]}}""",
        system_instruction="You create evaluation data. Output only JSON.",
        json_mode=True,
        temperature=0.9,
        max_tokens=2048,
    )
    data = parse_json(raw) or {}
    for question in data.get("questions", []):
        question = str(question).strip()
        if question and question not in questions and len(questions) < total:
            append_jsonl(path, {"id": f"una-{len(questions) + 1:04d}", "question": question, "kind": "generated"})
            questions.append(question)

    print(f"[unanswerable] {len(questions)}/{total}")


def gen_hindi(llm, total: int, pacer: Pacer) -> None:
    path = EVAL_DATA_DIR / "hindi.jsonl"
    done = {r["source_id"] for r in read_jsonl(path)}
    synthetic = read_jsonl(EVAL_DATA_DIR / "synthetic.jsonl")
    if not synthetic:
        print("[hindi] needs synthetic.jsonl first")
        return

    # Spread across services
    rng = random.Random(7)
    rng.shuffle(synthetic)
    for item in synthetic:
        if len(done) >= total:
            break
        if item["id"] in done:
            continue
        pacer.wait()
        raw = llm.generate(
            prompt=f"""Translate this question into natural, everyday Hindi (Devanagari) as a citizen
would type it. Keep form names, numbers and years unchanged.

Question: {item['question']}

Return JSON: {{"hindi": "..."}}""",
            system_instruction="You translate questions. Output only JSON.",
            json_mode=True,
            temperature=0.3,
            max_tokens=512,
        )
        data = parse_json(raw) or {}
        hindi = str(data.get("hindi", "")).strip()
        if not re.search(r"[ऀ-ॿ]", hindi):
            continue
        append_jsonl(path, {
            "id": f"hin-{len(done) + 1:04d}",
            "source_id": item["id"],
            "question": hindi,
            "english_question": item["question"],
            "service": item["service"],
            "source_file": item["source_file"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
        })
        done.add(item["id"])
    print(f"[hindi] {len(done)}/{total}")


def gen_followup(llm, total: int, pacer: Pacer) -> None:
    path = EVAL_DATA_DIR / "followup.jsonl"
    done = {r["source_id"] for r in read_jsonl(path)}
    synthetic = read_jsonl(EVAL_DATA_DIR / "synthetic.jsonl")
    if not synthetic:
        print("[followup] needs synthetic.jsonl first")
        return

    rng = random.Random(11)
    rng.shuffle(synthetic)
    for item in synthetic:
        if len(done) >= total:
            break
        if item["id"] in done:
            continue
        pacer.wait()
        raw = llm.generate(
            prompt=f"""Target question: {item['question']}

Split it into a two-turn conversation:
1. "first": a broader question about the same service/topic that a citizen asks first.
2. "followup": a SHORT follow-up (under 10 words) that only makes sense after the first
   question, e.g. "what about for minors?" or "and the fee?". Together they must mean the
   target question. Do not repeat the service name in the follow-up.

Return JSON: {{"first": "...", "followup": "..."}}""",
            system_instruction="You create evaluation data. Output only JSON.",
            json_mode=True,
            temperature=0.5,
            max_tokens=512,
        )
        data = parse_json(raw) or {}
        if not data.get("first") or not data.get("followup"):
            continue
        append_jsonl(path, {
            "id": f"fup-{len(done) + 1:04d}",
            "source_id": item["id"],
            "first": data["first"].strip(),
            "followup": data["followup"].strip(),
            "standalone": item["question"],
            "service": item["service"],
            "source_file": item["source_file"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
        })
        done.add(item["id"])
    print(f"[followup] {len(done)}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["synthetic", "unanswerable", "hindi", "followup"])
    parser.add_argument("--per-service", type=int, default=40, help="synthetic questions per service")
    parser.add_argument("--unanswerable", type=int, default=50)
    parser.add_argument("--hindi", type=int, default=50)
    parser.add_argument("--followup", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", help="index to sample chunks from (default: current index)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    apply_overrides(args.preset, args.set)

    from src.llm.client import llmClient
    from src.vectorstore.store import VectorStore

    llm = llmClient()
    pacer = Pacer()

    if args.only in (None, "synthetic"):
        store = VectorStore()
        if not store.load():
            raise SystemExit("Vector store not found. Run: python scripts/build_index.py")
        gen_synthetic(llm, store, args.per_service, args.seed, pacer)
    if args.only in (None, "unanswerable"):
        gen_unanswerable(llm, args.unanswerable, pacer)
    if args.only in (None, "hindi"):
        gen_hindi(llm, args.hindi, pacer)
    if args.only in (None, "followup"):
        gen_followup(llm, args.followup, pacer)


if __name__ == "__main__":
    main()
