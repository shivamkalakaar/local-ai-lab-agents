"""Problem 3 (bm25): BM25 over an inverted index, from scratch.

Score (handout formula, Lucene IDF):
    score(q, d) = sum over distinct query terms t of
        IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
    IDF(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)

Layout (CSR, the bm25s idea): every posting's full weight is computed once at
build time, so a search only gathers the postings of its terms and adds them
up. A search never allocates anything the size of the corpus, which is what
keeps a rare-term query in microseconds even over 200,000 documents.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    """Immutable after build, so it is safe to share across threads."""

    def __init__(self, docs: list[str], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.n_docs = len(docs)

        postings: dict[str, tuple[list[int], list[int]]] = {}
        lengths = np.zeros(self.n_docs, dtype=np.float64)
        for doc_id, text in enumerate(docs):
            tokens = tokenize(text or "")
            lengths[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                entry = postings.get(term)
                if entry is None:
                    postings[term] = ([doc_id], [tf])
                else:
                    entry[0].append(doc_id)
                    entry[1].append(tf)

        self.doc_lengths = lengths
        self.avgdl = float(lengths.mean()) if self.n_docs else 0.0
        norm = k1 * (1.0 - b + b * lengths / (self.avgdl or 1.0))  # per-document denominator part

        # Concatenate every term's postings into three flat arrays (CSR).
        self.term_ids: dict[str, int] = {}
        offsets = [0]
        id_chunks, weight_chunks = [], []
        n = self.n_docs
        for tid, (term, (ids, tfs)) in enumerate(postings.items()):
            self.term_ids[term] = tid
            ids_arr = np.asarray(ids, dtype=np.int64)
            tf_arr = np.asarray(tfs, dtype=np.float64)
            idf = math.log((n - len(ids) + 0.5) / (len(ids) + 0.5) + 1.0)
            id_chunks.append(ids_arr)
            weight_chunks.append(idf * tf_arr * (k1 + 1.0) / (tf_arr + norm[ids_arr]))
            offsets.append(offsets[-1] + len(ids))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.post_ids = np.concatenate(id_chunks) if id_chunks else np.zeros(0, dtype=np.int64)
        self.post_weights = np.concatenate(weight_chunks) if weight_chunks else np.zeros(0)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        tids = []
        for term in dict.fromkeys(tokenize(query or "")):  # distinct, in order: a repeated term counts once
            tid = self.term_ids.get(term)
            if tid is not None:
                tids.append(tid)
        if not tids:
            return []

        if len(tids) == 1:
            lo, hi = self.offsets[tids[0]], self.offsets[tids[0] + 1]
            cand, scores = self.post_ids[lo:hi], self.post_weights[lo:hi]
        else:
            ids = np.concatenate([self.post_ids[self.offsets[t]:self.offsets[t + 1]] for t in tids])
            ws = np.concatenate([self.post_weights[self.offsets[t]:self.offsets[t + 1]] for t in tids])
            cand, inverse = np.unique(ids, return_inverse=True)  # sorted doc ids
            scores = np.bincount(inverse, weights=ws, minlength=len(cand))

        # Best score first; equal scores -> lower doc id first.
        order = np.lexsort((cand, -scores))[:k]
        return [(int(cand[i]), float(scores[i])) for i in order]


def build_index(docs: list[str]) -> BM25Index:
    return BM25Index(docs)
