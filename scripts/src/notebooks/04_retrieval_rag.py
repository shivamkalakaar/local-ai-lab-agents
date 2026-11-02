# %% [markdown]
# # 04 · Retrieval (RAG): hybrid search with checked citations
# **Local AI Lab** · CS329Z Lecture 3 · [Module 04](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m4)
#
# The default that works: split on structure, add a short context note to each chunk, search two ways (keywords and meaning), merge with reciprocal rank fusion, and check every citation in code. Everything runs locally, including the embedding model.

# %%
# EXTRA: rank-bm25
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from lal.data import contract, sections, ids, labels
from lal.llm import structured
from lal.checks import quote_is_in_doc, check

# %% [markdown]
# ## 1. Chunk on structure, with context
# Each chunk is one section of one contract. We prepend a short note saying where it came from (Anthropic's "contextual retrieval"), which helps both search methods.

# %%
chunks = []
for cid in ids():
    for s in sections(contract(cid)):
        chunks.append({"id": f"{cid}#{s['id']}", "contract": cid, "section": s["id"],
                       "context": f"From {cid}, section {s['id']} ({s['title']}).", "text": s["text"]})
print(len(chunks), "chunks. Example:", chunks[6])

# %% [markdown]
# ## 2. Two searches, merged with RRF

# %%
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")      # small, local embedding model

class HybridIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        docs = [f"{c['context']}\n{c['text']}" for c in chunks]
        self.bm25 = BM25Okapi([d.lower().split() for d in docs])
        self.vecs = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False)

    def keyword(self, q, n):
        return list(np.argsort(-self.bm25.get_scores(q.lower().split()))[:n])

    def semantic(self, q, n):
        qv = embedder.encode([q], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.vecs @ qv))[:n])

    def search(self, q, k=5, pool=50, mode="hybrid", rrf_k=60):
        if mode == "keyword": return [self.chunks[i] for i in self.keyword(q, k)]
        if mode == "semantic": return [self.chunks[i] for i in self.semantic(q, k)]
        scores = {}
        for ranking in (self.keyword(q, pool), self.semantic(q, pool)):
            for r, i in enumerate(ranking):
                scores[i] = scores.get(i, 0) + 1 / (rrf_k + r + 1)
        return [self.chunks[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]

index = HybridIndex(chunks)
for c in index.search("msa_15 cap on damages", k=3):
    print(c["id"], "|", c["text"][:80])

# %% [markdown]
# ## 3. Measure retrieval on its own: recall@k
# Did the right chunk come back? We test with questions that use different words from the contract ("cap on damages" vs. "limitation of liability"), and questions with exact IDs.

# %%
SECTION_FOR = {"liability": "7", "law": "10", "renewal": "4", "breach": "9"}
QUERIES = []
for cid in ids()[:10]:
    QUERIES += [(f"{cid} cap on damages", f"{cid}#7"), (f"which state's law applies in {cid}", f"{cid}#10"),
                (f"{cid} how do we stop it renewing", f"{cid}#4")]

def recall_at_k(mode, k=3):
    return sum(any(c["id"] == gold for c in index.search(q, k=k, mode=mode)) for q, gold in QUERIES) / len(QUERIES)

for mode in ("keyword", "semantic", "hybrid"):
    print(f"{mode:9} recall@3 = {recall_at_k(mode):.0%}")

# %% [markdown]
# ## 4. Answer from retrieved chunks, with citations checked in code

# %%
from pydantic import BaseModel
class Answer(BaseModel):
    answer: str
    chunk_id: str
    quote: str

def rag(question, k=4):
    hits = index.search(question, k=k)
    ctx = "\n\n".join(f"[{c['id']}] {c['text']}" for c in hits)
    a = structured([{"role": "system", "content": "Answer only from the sources. Give the source id and an exact quote."},
                    {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"}], Answer)
    src = next((c for c in hits if c["id"] == a.chunk_id), None)
    a_ok = src is not None and quote_is_in_doc(a.quote, src["text"])
    return a, a_ok

a, ok = rag("What is the liability cap in msa_20?")
print(a.model_dump(), "\ncitation verified:", ok)

# %%
check("hybrid recall@3 is at least as good as keyword-only", recall_at_k("hybrid") >= recall_at_k("keyword"))
check("RAG answer cites a real quote", ok)

# %% [markdown]
# ## Your turn
# 1. Remove the context note from each chunk and rerun recall@k. What drops?
# 2. Add a reranker: `from sentence_transformers import CrossEncoder; CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`, score the top 20, keep 4.
#
# **Next:** 05 · Tools
