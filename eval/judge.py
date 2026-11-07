"""
LLM-as-judge (reference-free RAG metrics + optional correctness).

One call scores all metrics for an answer, to save free-tier quota:
    faithfulness       every claim is supported by the retrieved context (hallucination)
    answer_relevance   the answer addresses the question
    context_relevance  the retrieved context contains what is needed
    correctness        agrees with the reference answer (synthetic set only)

Scores are 1–5 and reported normalised to 0–1. Set EVAL_JUDGE_MODEL to judge with a
different model than the one that wrote the answers (reduces self-preference bias).
"""

import os
import random
from typing import Optional

from eval.common import parse_json

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of a question-answering system for Indian "
    "government services. Judge only from the material given. Output only JSON."
)

_JUDGE_PROMPT = """QUESTION:
{question}

RETRIEVED CONTEXT (what the system could see):
{context}

SYSTEM ANSWER:
{answer}
{reference_block}
Score each item from 1 (worst) to 5 (best):
- faithfulness: are ALL factual claims in the answer supported by the context? 5 = fully
  supported, 1 = mostly unsupported or contradicted. A refusal ("I don't have sufficient
  information") counts as 5.
- answer_relevance: does the answer directly address the question? A refusal scores 1
  unless the context truly lacks the answer, then 3.
- context_relevance: does the context contain the information needed to answer?
{correctness_rule}
Return JSON: {{"faithfulness": n, "answer_relevance": n, "context_relevance": n, "correctness": {correctness_type}, "reason": "<one short sentence>"}}"""

_PAIRWISE_PROMPT = """QUESTION:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Which answer is more accurate, complete and useful for a citizen? Penalise invented facts.
Return JSON: {{"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}}"""


def _judge_model() -> Optional[str]:
    return os.getenv("EVAL_JUDGE_MODEL") or None


def _normalise(value) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, (score - 1.0) / 4.0))


def judge_answer(
    llm,
    question: str,
    answer: str,
    contexts: list[str],
    reference: Optional[str] = None,
) -> Optional[dict]:
    context = "\n\n---\n\n".join(contexts)[:8000] or "(no context retrieved)"
    if reference:
        reference_block = f"\nREFERENCE ANSWER (written from the source passage):\n{reference}\n"
        correctness_rule = "- correctness: does the answer agree with the reference answer on the key facts?"
        correctness_type = "n"
    else:
        reference_block = ""
        correctness_rule = ""
        correctness_type = "null"

    raw = llm.generate(
        prompt=_JUDGE_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
            reference_block=reference_block,
            correctness_rule=correctness_rule,
            correctness_type=correctness_type,
        ),
        system_instruction=JUDGE_SYSTEM,
        json_mode=True,
        temperature=0.0,
        max_tokens=1024,
        model=_judge_model(),
    )
    data = parse_json(raw)
    if not data:
        return None

    return {
        "faithfulness": _normalise(data.get("faithfulness")),
        "answer_relevance": _normalise(data.get("answer_relevance")),
        "context_relevance": _normalise(data.get("context_relevance")),
        "correctness": _normalise(data.get("correctness")) if reference else None,
        "judge_reason": data.get("reason"),
    }


def judge_pairwise(llm, question: str, answer_1: str, answer_2: str) -> Optional[str]:
    """
    Returns "1", "2" or "tie". The two answers are shown in random order so the
    judge's preference for position A or B cancels out over many questions.
    """
    swapped = random.random() < 0.5
    answer_a, answer_b = (answer_2, answer_1) if swapped else (answer_1, answer_2)

    raw = llm.generate(
        prompt=_PAIRWISE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b),
        system_instruction=JUDGE_SYSTEM,
        json_mode=True,
        temperature=0.0,
        max_tokens=512,
        model=_judge_model(),
    )
    data = parse_json(raw)
    if not data:
        return None

    winner = str(data.get("winner", "")).strip().upper()
    if winner == "TIE":
        return "tie"
    if winner not in ("A", "B"):
        return None
    if swapped:
        return "2" if winner == "A" else "1"
    return "1" if winner == "A" else "2"
