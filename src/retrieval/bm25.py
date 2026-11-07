"""
BM25 keyword index (Phase 3).

Dense embeddings are weak on exact identifiers such as "Form 6", "ITR-4", "80C" or
"Tatkaal"; BM25 matches them exactly. Built in memory from the stored chunk texts
when the retriever starts (a couple of seconds for ~20k chunks), so it never goes
stale relative to the FAISS index and needs no extra files or packages.
"""

import logging
import re
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

# Latin letters/digits plus the whole Devanagari block (matras included), keeping
# hyphenated / slashed identifiers like "itr-4" or "12/2023" together.
_TOKEN_PATTERN = re.compile(r"[a-z0-9ऀ-ॿ]+(?:[-/][a-z0-9]+)*")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "be", "by", "with", "as", "at", "from", "it", "this", "that", "which", "i", "my",
    "me", "we", "you", "your", "do", "does", "can", "how", "what", "when", "where",
    "who", "will", "shall", "should", "has", "have", "if", "any", "all", "not", "no",
    "there", "their", "its", "into", "about", "under", "such", "may", "per",
}


def tokenize(text: str) -> list[str]:
    """
    Lowercase tokens with identifier variants, so "Form-6", "form 6" and "form6"
    all share the token "form6".
    """
    raw = _TOKEN_PATTERN.findall((text or "").lower())
    tokens: list[str] = []

    for i, tok in enumerate(raw):
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)

        if "-" in tok or "/" in tok:
            parts = [p for p in re.split(r"[-/]", tok) if p]
            tokens.extend(p for p in parts if p not in _STOPWORDS)
            tokens.append("".join(parts))

        # "itr 4" / "form 6" / "section 80c" → "itr4", "form6", ...
        if i + 1 < len(raw) and tok.isalpha() and len(raw[i + 1]) <= 4 and raw[i + 1][0].isdigit():
            tokens.append(tok + raw[i + 1])

    return tokens


class BM25Index:
    """Okapi BM25 over a fixed list of documents (positions match the vector store)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.vectorizer = CountVectorizer(analyzer=tokenize)
        tf = self.vectorizer.fit_transform(texts).astype(np.float32).tocsr()

        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = float(doc_len.mean()) or 1.0

        doc_freq = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)

        # Precompute the full BM25 weight for every (doc, term) pair once, so a
        # query is just a column sum over its terms.
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        length_norm = k1 * (1.0 - b + b * doc_len[rows] / avg_len)
        weights = tf.data * (k1 + 1.0) / (tf.data + length_norm) * idf[tf.indices]

        self.weights = sparse.csr_matrix((weights, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        self.vocabulary = self.vectorizer.vocabulary_
        self.size = n_docs

        logger.info(f"BM25 index built: {n_docs} docs, {len(self.vocabulary)} terms")

    def scores(self, query: str) -> np.ndarray:
        term_ids = sorted({self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary})
        if not term_ids:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel()

    def search(self, query: str, top_k: int, mask: Optional[np.ndarray] = None) -> list[tuple[int, float]]:
        """Return [(position, score)] for the best `top_k` documents with score > 0."""
        scores = self.scores(query)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)

        top_k = min(top_k, self.size)
        if top_k <= 0:
            return []

        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked = candidates[np.argsort(-scores[candidates])]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]
