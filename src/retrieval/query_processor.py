"""
Query understanding (Phase 4)

Turns what the user typed into what the retriever should search for:

1. Follow-up rewriting: "what about minors?" after a passport question becomes
   "passport renewal process for minors".
2. Translation: Hindi / Hinglish → English (the documents are English, and BM25
   and the English reranker need English words).
3. Service routing: which of the five services the question is about.

Plain English questions that are not follow-ups need no LLM call at all. When the
LLM is needed, rewrite + translation + routing happen in ONE JSON call, which
matters on Gemini's free tier (requests per minute are limited).
"""

import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from config.settings import (
    ENABLE_QUERY_REWRITE,
    ENABLE_SERVICE_ROUTING,
    QUERY_REWRITE_HISTORY_TURNS,
    SERVICE_CATEGORIES,
)
from src.llm.client import is_llm_error, llmClient
from utility.language import has_devanagari, looks_hinglish, normalize_query_llm_cached

logger = logging.getLogger(__name__)

_FOLLOWUP_START = re.compile(
    r"^\s*(and|also|what about|how about|then|so|but|same|for|in that case|what if|"
    r"aur|or phir|phir|और|तो|फिर)\b",
    re.IGNORECASE,
)
_FOLLOWUP_REFERENCE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|there|same|above|mentioned|"
    r"iske|uske|iska|uska|ye|yeh|wo|woh|isme|usme)\b|"
    r"(इसके|उसके|इसका|उसका|इसमें|उसमें|यह|वह|ये|वो)",
    re.IGNORECASE,
)

_ROUTER_MIN_SHARE = 0.6  # top service must own this share of keyword hits


@dataclass
class QueryPlan:
    original: str
    retrieval_query: str
    service: Optional[str] = None
    rewritten: bool = False
    translated: bool = False
    llm_used: bool = False


class QueryProcessor:

    def __init__(self, llm: Optional[llmClient] = None):
        self._llm = llm
        self._cache: "OrderedDict[tuple, dict]" = OrderedDict()

    @property
    def llm(self) -> llmClient:
        if self._llm is None:
            self._llm = llmClient()
        return self._llm

    # ─── Public ──────────────────────────────────────────────────────────────

    def process(self, query: str, history: Optional[list[dict]] = None) -> QueryPlan:
        query = (query or "").strip()
        history = history or []
        plan = QueryPlan(original=query, retrieval_query=query)

        if not query:
            return plan

        needs_translation = has_devanagari(query) or looks_hinglish(query)
        is_followup = ENABLE_QUERY_REWRITE and self.is_followup(query, history)

        llm_service = None
        if is_followup or (ENABLE_QUERY_REWRITE and needs_translation):
            parsed = self._llm_rewrite(query, history if is_followup else [])
            if parsed:
                plan.retrieval_query = parsed["query"]
                plan.rewritten = is_followup
                plan.translated = needs_translation
                plan.llm_used = True
                llm_service = parsed.get("service")

        if not plan.llm_used and needs_translation:
            # Rewrite disabled or the JSON call failed: plain translation.
            translated = normalize_query_llm_cached(query)
            plan.translated = translated != query
            plan.retrieval_query = translated

        if ENABLE_SERVICE_ROUTING:
            plan.service = self.route(f"{plan.retrieval_query} {query}") or llm_service

        logger.info(
            f"Query plan: '{query[:60]}' → '{plan.retrieval_query[:80]}' "
            f"(service={plan.service}, rewritten={plan.rewritten}, translated={plan.translated})"
        )
        return plan

    @staticmethod
    def is_followup(query: str, history: list[dict]) -> bool:
        if not any(turn.get("role") == "user" for turn in history):
            return False
        # A question that already names a service stands on its own.
        if QueryProcessor.route(query):
            return bool(_FOLLOWUP_START.search(query))
        words = query.split()
        return len(words) <= 4 or bool(_FOLLOWUP_START.search(query) or _FOLLOWUP_REFERENCE.search(query))

    @staticmethod
    def route(text: str) -> Optional[str]:
        """Keyword router: returns a service key when one service clearly dominates."""
        lower = (text or "").lower()
        hits: dict[str, int] = {}
        for key, info in SERVICE_CATEGORIES.items():
            count = 0
            for keyword in info.get("keywords", []):
                if has_devanagari(keyword):
                    count += keyword in lower
                else:
                    count += bool(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", lower))
            if count:
                hits[key] = count

        if not hits:
            return None
        best = max(hits, key=hits.get)
        if hits[best] / sum(hits.values()) >= _ROUTER_MIN_SHARE:
            return best
        return None

    # ─── LLM rewrite ─────────────────────────────────────────────────────────

    def _llm_rewrite(self, query: str, history: list[dict]) -> Optional[dict]:
        recent = [
            turn for turn in history if turn.get("role") in ("user", "assistant")
        ][-QUERY_REWRITE_HISTORY_TURNS * 2:]
        history_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'][:300]}"
            for t in recent
        )

        cache_key = (query, history_text)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        services = ", ".join(f'"{k}" ({v["name"]})' for k, v in SERVICE_CATEGORIES.items())
        # Ask for a complete natural question, not keywords: the reranker was trained
        # on real questions and scored keyword fragments ("passport exemption
        # categories") noticeably worse in the Hindi eval.
        prompt = f"""Rewrite the user's latest message as ONE complete, natural English question
that makes sense on its own.

Conversation so far:
{history_text or "(none)"}

Latest message: {query}

Rules:
- Write a full question as a citizen would ask it (e.g. "Who is allowed to enter the polling
  booth besides voters?"), not a list of keywords.
- Resolve references like "it", "that", "what about ..." using the conversation.
- Translate Hindi or Hinglish into English faithfully; keep every detail and do not add new ones.
- Keep form names, section numbers and years exactly (e.g. "Form 6", "ITR-4", "AY 2024-25").
- Do not answer the question.

Return JSON: {{"query": "<complete English question>", "service": "<one of {services} or none>"}}"""

        raw = self.llm.generate(
            prompt=prompt,
            system_instruction="You rewrite search queries. Output only the JSON object.",
            json_mode=True,
            temperature=0.0,
            max_tokens=1024,
        )
        parsed = self._parse(raw, query)

        self._cache[cache_key] = parsed
        if len(self._cache) > 500:
            self._cache.popitem(last=False)
        return parsed

    @staticmethod
    def _parse(raw: str, query: str) -> Optional[dict]:
        if is_llm_error(raw):
            return None
        text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        rewritten = str(data.get("query", "")).strip()
        if not rewritten or has_devanagari(rewritten) or len(rewritten) > max(300, len(query) * 6):
            return None

        service = str(data.get("service", "")).strip().lower()
        return {"query": rewritten, "service": service if service in SERVICE_CATEGORIES else None}
