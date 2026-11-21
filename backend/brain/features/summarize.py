"""
Extractive summarisation without an LLM.

Maximal Marginal Relevance over embeddings: repeatedly pick the text that is
most central (cosine to the centroid of all texts) while being least redundant
with what was already picked:

    pick argmax_i  LAMBDA * cos(v_i, c) - (1 - LAMBDA) * max_{j in S} cos(v_i, v_j)

Selected texts are returned in their original order. When embeddings are
unavailable, centrality falls back to TF-IDF term overlap.
"""
import math
from collections import Counter

import numpy as np

from backend.brain.memory_ranking.importance_ranker import content_terms

LAMBDA = 0.7


def _tfidf_vectors(texts):
    docs = [Counter(content_terms(t)) for t in texts]
    df = Counter(term for d in docs for term in d)
    n = len(texts)
    vocab = {t: i for i, t in enumerate(df)}
    m = np.zeros((n, len(vocab) or 1), dtype=np.float32)
    for row, d in enumerate(docs):
        for term, tf in d.items():
            m[row, vocab[term]] = tf * math.log(1 + n / df[term])
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def select(texts, embed_fn=None, k=None):
    """Indices of the k most representative, non-redundant texts (original order)."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    k = k or max(1, math.ceil(math.sqrt(len(texts))))
    vecs = None
    if embed_fn is not None:
        embedded = [embed_fn(t) for t in texts]
        if all(v is not None for v in embedded):
            vecs = np.vstack(embedded).astype(np.float32)
    if vecs is None:
        vecs = _tfidf_vectors(texts)

    centroid = vecs.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) or 1.0)
    centrality = vecs @ centroid
    sims = vecs @ vecs.T

    chosen = []
    while len(chosen) < min(k, len(texts)):
        best, best_score = None, -np.inf
        for i in range(len(texts)):
            if i in chosen:
                continue
            redundancy = max((sims[i, j] for j in chosen), default=0.0)
            score = LAMBDA * centrality[i] - (1 - LAMBDA) * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
    return sorted(chosen)


def extractive_summary(texts, embed_fn=None, k=None, sep=" "):
    texts = [t for t in texts if t and t.strip()]
    return sep.join(texts[i] for i in select(texts, embed_fn, k))
