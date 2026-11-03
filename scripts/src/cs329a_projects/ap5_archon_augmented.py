# %% [markdown]
# # ap5 · ARCHON Augmented, rebuilt small: plan, search, generate, critique, rank, fuse, verify
# **Local AI Lab** · CS329A past project (Winter 2025) · stacking inference-time techniques, plus a planner and a search tool
#
# One model call is often not your best answer. Archon stacks simple steps (generate several answers, critique them,
# rank them, fuse the best, verify the result) into a pipeline. This project added a **planner** at the front and a
# **web search** step. You will build the whole pipeline with a local search engine over a small set of made-up web
# pages, and measure each part against a single call.

# %% [markdown]
# ## The original project
# **ARCHON Augmented: Planning and Web-Enhanced Components** by Megan Mou, Sherry Xie, Emily Zhang and Andrew Park
# (Stanford CS329A, Winter 2025; the poster is titled *Multi-Component ARCHON*).
# [Paper](https://cs329a.stanford.edu/winter2025projects/archon.pdf) · [Poster](https://cs329a.stanford.edu/winter2025projects/archonposter.pdf) ·
# code: github.com/49emily/cs329a-archon
#
# **Background.** Archon (Saad-Falcon et al. 2024, *Archon: An Architecture Search Framework for Inference-Time
# Techniques*) has seven building blocks: Generator, Fuser, Critic, Ranker, Verifier, Unit Test Generator, Unit Test
# Evaluator. It searches for the best way to stack them. Its strong results relied on models of about 70B parameters,
# and it had no tool use.
#
# **Method.** Three new blocks:
# - **Planner** (first): rephrases the request and writes a step-by-step plan that is passed to every later block.
# - **Expander** (after each generator): adds context and detail to each candidate so critics and rankers can judge it.
# - **Web Search Tool Call** (after the planner): writes q search queries from the plan, calls the Google Search API,
#   and passes the top 10 results per query to the generators.
#
# **Setup.** A closed-source configuration (GPT-4o generating and ranking, Qwen2-72B as critic and fuser) and an
# open-source one (an ensemble including Qwen2-72B, WizardLM-2 8x22B, Llama-3 70B, Mixtral 8x22B and others).
# Benchmarks: a 500-question subset of Humanity's Last Exam (HLE; 453 scored) and AlpacaEval (length-controlled win rate).
#
# **Key results (as reported).**
#
# | Architecture | HLE closed | HLE open | AlpacaEval LC closed | AlpacaEval LC open/mixed |
# |---|---|---|---|---|
# | ARCHON | 3.3% (15/453) | 3.3% (15/453) | 67.73% | 57.01% |
# | Planner + ARCHON | 3.3% (15/453) | 2.6% (12/453) | 69.62% | 67.06% |
# | Planner + Expander + ARCHON | **4.2%** (19/453) | 3.8% (17/453) | 66.01% | 68.30% |
# | Planner + Tool Call + Expander + ARCHON | 3.3% (15/453) | 2.9% (13/453) | **73.40%** | **73.92%** |
#
# Zero-shot GPT-4o scored 3.1% on HLE. On AlpacaEval the new blocks, especially the Planner, lifted the open-source
# setup from more than 10 points behind to slightly ahead of closed-source. On HLE, web search **hurt**: results were
# often off-topic or full of page clutter, which confused the models.
#
# **Limitations they state.** Reasoning stays the bottleneck; the Expander's value depends on the benchmark; the
# planner is not yet query-adaptive. (Our note: on HLE the differences are a few questions out of 453, so they are
# within noise.)
#
# **Our mini version.** Questions about made-up towns that no model can know, so search is required. The "web" is
# 16 local pages, including look-alike decoy pages, searched with BM25 (a classic keyword ranking). We skip the
# Expander to save calls. Generators mix `llama3.1:8b` and `llama3.2:3b`, a tiny "mixed-source" ensemble.

# %%
# MODELS: llama3.2:3b
import math, random, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List
from pydantic import BaseModel, Field
from lal.llm import ask, structured
from lal.checks import check

# %% [markdown]
# ## 1. A small made-up web
# Eight towns, each with a founding year, a population, a river and a bridge. Three decoy pages describe "New ..."
# towns with different numbers, and five pages are pure clutter. All generated with a seed.

# %%
rng = random.Random(5)
NAMES = ["Velmora", "Ostrand", "Brenhollow", "Quillmarsh", "Taversk", "Lunegate", "Corrowind", "Ashby Vale"]
RIVERS = ["Senn", "Ilver", "Draw", "Morrow", "Kesh", "Tull", "Avin", "Pellam"]
BRIDGES = ["Kestrel Bridge", "Lantern Bridge", "Old Mill Bridge", "Heron Bridge", "Copper Bridge", "Saint Ide Bridge", "Gull Bridge", "Iron Fox Bridge"]
TOWNS = {}
for n, r, b in zip(NAMES, RIVERS, BRIDGES):
    y = rng.randint(1100, 1800)
    TOWNS[n] = dict(founded=y, pop=rng.randint(3000, 95000), river=r, bridge=b, bridge_year=y + rng.randint(30, 300))

FILLER = ["The weekly market sells cheese, wool and honey.", "Tourists often visit the old clock tower.",
          "The local football club plays in the regional league.", "Summers are mild and winters are wet.",
          "A folk festival takes place every autumn.", "The town hall was renovated recently."]

def town_page(n, t):
    return (f"{n} is a town on the {t['river']} river. It was founded in {t['founded']}. Today it has a population of "
            f"{t['pop']:,}. The {t['bridge']} in {n} was built in {t['bridge_year']}. " + " ".join(rng.sample(FILLER, 3)))

DOCS = [{"title": n, "text": town_page(n, t)} for n, t in TOWNS.items()]
for n in rng.sample(NAMES, 3):                                    # decoys: similar names, different facts
    fake = dict(TOWNS[n], founded=TOWNS[n]["founded"] + 400, pop=TOWNS[n]["pop"] // 3 + 1000)
    DOCS.append({"title": f"New {n}", "text": f"New {n} is a newer settlement, not to be confused with {n}. It was founded in "
                 f"{fake['founded']} and has a population of {fake['pop']:,}. " + " ".join(rng.sample(FILLER, 2))})
for i in range(5):
    DOCS.append({"title": f"Travel blog {i + 1}", "text": " ".join(rng.sample(FILLER, 4)) + f" We loved our trip to {rng.choice(NAMES)}!"})
print(len(DOCS), "pages. Example:\n", DOCS[0]["text"])

# %% [markdown]
# Six questions. Most need two pages, and some need arithmetic on top.

# %%
A, B, C, D = rng.sample(NAMES, 4)
QS = [
    (f"How many years passed between the founding of {A} and the building of the {TOWNS[B]['bridge']}?",
     abs(TOWNS[B]["bridge_year"] - TOWNS[A]["founded"])),
    (f"What is the combined population of {A} and {C}?", TOWNS[A]["pop"] + TOWNS[C]["pop"]),
    (f"Which town has more people, {B} or {D}?", max((B, D), key=lambda n: TOWNS[n]["pop"])),
    (f"On which river is the town that has the {TOWNS[C]['bridge']}?", TOWNS[C]["river"]),
    (f"How many more people live in {max((C, D), key=lambda n: TOWNS[n]['pop'])} than in {min((C, D), key=lambda n: TOWNS[n]['pop'])}?",
     abs(TOWNS[C]["pop"] - TOWNS[D]["pop"])),
    (f"Which was founded first, {A} or {D}, and by how many years? Answer with the number of years.",
     abs(TOWNS[A]["founded"] - TOWNS[D]["founded"])),
]
for q, a in QS:
    print(a, "←", q)

def answer_of(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "", re.I)
    return m[-1].strip() if m else ((text or "").strip().splitlines() or [""])[-1]

def is_right(got, gold):
    got = (got or "").replace(",", "")
    if isinstance(gold, int):
        nums = re.findall(r"-?\d+", got)
        return bool(nums) and (int(nums[0]) == gold or int(nums[-1]) == gold)
    pool = NAMES if gold in NAMES else RIVERS            # the FIRST name mentioned is the answer ("Velmora (52,000 vs Ostrand's 30,000)")
    hits = sorted((got.lower().find(n.lower()), n) for n in pool if n.lower() in got.lower())
    return bool(hits) and hits[0][1] == gold

# %% [markdown]
# ## 2. The search tool: BM25 in 15 lines
# BM25 scores a page by how often it contains the query words, with rare words counting more and long pages counting
# less (Robertson & Zaragoza 2009). It is what many search engines started from.

# %%
def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokens(d["title"] + " " + d["text"])) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for t in self.tf for w in t)
        self.idf = {w: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def search(self, query, k=3):
        q = tokens(query)
        def score(i):
            return sum(self.idf.get(w, 0) * self.tf[i][w] * (self.k1 + 1) /
                       (self.tf[i][w] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)) for w in q)
        return [self.docs[i] for i in sorted(range(len(self.docs)), key=score, reverse=True)[:k]]

WEB = BM25(DOCS)
print([d["title"] for d in WEB.search(f"{A} founded")])

def as_context(pages):
    return "\n\n".join(f"[{d['title']}] {d['text']}" for d in pages)

# %% [markdown]
# ## 3. Two simple baselines
# - **Single call, no search:** the model has to guess. Expect near 0, which proves the facts are really unknown.
# - **Single call + search on the raw question:** classic retrieval-augmented generation (RAG).

# %%
FMT = "\nEnd with one line: ANSWER: <short answer>"

def single(q):
    return {"answer": answer_of(ask(q + FMT, max_tokens=300)), "calls": 1}

def single_rag(q):
    ctx = as_context(WEB.search(q, k=3))
    return {"answer": answer_of(ask(f"Web results:\n{ctx}\n\nQuestion: {q}\nUse only the web results." + FMT, max_tokens=400)), "calls": 1}

# %% [markdown]
# ## 4. The augmented Archon pipeline
# ```
# question → Planner (plan + search queries) → Web search (BM25) → 3 Generators (8B, 8B, 3B)
#          → Critic + Ranker → Fuser → Verifier → answer          (7 model calls)
# ```
# If the verifier says the fused answer is not supported by the pages, we fall back to the top-ranked candidate.

# %%
class PlanOut(BaseModel):
    rephrased: str
    steps: List[str] = Field(max_length=5)
    queries: List[str] = Field(description="1 to 3 short search queries", max_length=3)

class Critique(BaseModel):
    notes: List[str] = Field(description="one short critique per candidate, in order")
    ranking: List[int] = Field(description="candidate numbers, best first")

class Verdict(BaseModel):
    supported: bool
    reason: str

def plan_and_search(q):
    try:
        p = structured(f"Question: {q}\nRephrase the question, write a short plan, and write 1-3 web search queries "
                       "that will find every fact the plan needs.", PlanOut, max_tokens=400)
    except Exception:
        p = PlanOut(rephrased=q, steps=["answer the question"], queries=[q])
    pages, seen = [], set()
    for query in (p.queries or [q])[:3]:
        for d in WEB.search(query, k=3):
            if d["title"] not in seen:
                seen.add(d["title"]); pages.append(d)
    return p, pages[:6]

def generate(q, plan, pages, model, temperature=0.7):
    prompt = (f"Web results:\n{as_context(pages)}\n\nQuestion: {q}\nPlan:\n" + "\n".join(f"- {s}" for s in plan.steps) +
              "\nFollow the plan. Use only the web results. Watch out for pages about similarly named places." + FMT)
    try:
        return ask(prompt, model_name=model, temperature=temperature, max_tokens=500)
    except Exception:
        return ""

GENERATORS = [MODEL, MODEL, "llama3.2:3b"]          # MODEL is llama3.1:8b on a GPU runtime

def archon(q):
    plan, pages = plan_and_search(q)
    cands = [generate(q, plan, pages, m) for m in GENERATORS]
    listing = "\n\n".join(f"=== Candidate {i + 1} ===\n{c[-1200:]}" for i, c in enumerate(cands))
    try:
        crit = structured(f"Web results:\n{as_context(pages)}\n\nQuestion: {q}\n\n{listing}\n\nCritique each candidate "
                          "(facts used, arithmetic) and rank them, best first.", Critique, max_tokens=500)
        order = [i - 1 for i in crit.ranking if 1 <= i <= len(cands)]
        order += [i for i in range(len(cands)) if i not in order]
        notes = crit.notes
    except Exception:
        order, notes = list(range(len(cands))), []
    top = order[:2]
    try:
        fused = ask(f"Web results:\n{as_context(pages)}\n\nQuestion: {q}\n\nBest candidates:\n" +
                    "\n\n".join(cands[i][-1200:] for i in top) + "\n\nCritiques:\n" + "\n".join(notes[:3]) +
                    "\n\nWrite one final, correct answer. Fix any mistakes." + FMT, max_tokens=500)
    except Exception:
        fused = cands[order[0]]
    final = answer_of(fused)
    try:
        v = structured(f"Web results:\n{as_context(pages)}\n\nQuestion: {q}\nProposed answer: {final}\n"
                       "Is this answer fully supported by the web results, with correct arithmetic?", Verdict, max_tokens=200)
        ok = v.supported
    except Exception:
        ok = True
    used = final if ok else answer_of(cands[order[0]])
    return {"answer": used, "calls": 7, "plan": plan, "pages": [d["title"] for d in pages],
            "cands": [answer_of(c) for c in cands], "order": order, "fused": final, "verified": ok}

def plan_search_single(q):
    """Ablation: planner + search, then one generator. Isolates the value of the generate/critique/fuse layers."""
    plan, pages = plan_and_search(q)
    return {"answer": answer_of(generate(q, plan, pages, MODEL, temperature=0)), "calls": 2}

# %% [markdown]
# ## 5. Run everything
# At most 6 + 6 + 12 + 42 = 66 model calls.

# %%
def run(method):
    with ThreadPoolExecutor(2) as ex:
        return list(ex.map(method, [q for q, _ in QS]))

RES = {"single call": run(single), "single + search": run(single_rag),
       "planner + search + 1 gen": run(plan_search_single), "Archon augmented": run(archon)}
print(f"{'method':26s} accuracy  calls/question")
ACC = {}
for name, rs in RES.items():
    ACC[name] = sum(is_right(r["answer"], gold) for r, (_, gold) in zip(rs, QS)) / len(QS)
    print(f"{name:26s} {ACC[name]:6.0%}    {sum(r['calls'] for r in rs) / len(rs):.0f}")

for r, (q, gold) in zip(RES["Archon augmented"], QS):
    print(f"\nQ: {q}\n  gold {gold} | final {r['answer']} | candidates {r['cands']} | rank {r['order']} | verified {r['verified']}")
    print("  queries:", r["plan"].queries, "→ pages:", r["pages"])

# %% [markdown]
# **What to notice.** Search is what makes these questions answerable at all; the gap between the first two rows
# shows it. Planner-written queries usually find both pages a two-hop question needs, which a single raw-question
# search can miss. Whether the full stack beats "planner + search + 1 generator" is the interesting question: the
# extra 5 calls only pay off if the candidates disagree and the critic can tell which one is right. Look at the decoy
# pages in the retrieved lists: if a "New ..." page sneaks in, that is the HLE failure in miniature.

# %%
check("no-search baseline is near zero (facts are really unknown)", ACC["single call"] <= 2 / 6)
check("search helps: single + search beats single call", ACC["single + search"] > ACC["single call"])
check("Archon pipeline answers at least 3 of 6", ACC["Archon augmented"] >= 3 / 6)
check("BM25 finds the right town page first", WEB.search(f"{A} founded population", k=1)[0]["title"] == A)

# %% [markdown]
# ## Your turn
# 1. **Is the fuser + verifier worth 2 calls?** Using the saved results, score the pipeline as if it stopped after
#    ranking (answer = top-ranked candidate). No new calls.
# 2. **Which generator is best?** Score each generator's candidates on its own (8B #1, 8B #2, 3B). Does the ensemble
#    beat its best member? No new calls.
# 3. **A noisier web.** Add 10 more decoy pages ("Old Velmora", "Velmora Station", ...) with wrong numbers, rebuild
#    the index, and rerun Archon on 2 questions. Does search start to hurt, as it did on HLE?

# %% [markdown]
# ## Solutions

# %%
# 1. Stop after the ranker. Notice: if this matches the full pipeline, the fuser and verifier are cost without value
#    here; that is exactly the kind of ablation the paper ran for each new block.
rank_only = sum(is_right(r["cands"][r["order"][0]], gold) for r, (_, gold) in zip(RES["Archon augmented"], QS)) / len(QS)
print(f"rank only {rank_only:.0%}   full pipeline {ACC['Archon augmented']:.0%}")

# %%
# 2. Each generator alone, and the best-of-3 oracle. The ensemble can only beat its best member if the
#    ranker picks well; the oracle shows the ceiling.
for i, m in enumerate(GENERATORS):
    acc_i = sum(is_right(r["cands"][i], gold) for r, (_, gold) in zip(RES["Archon augmented"], QS)) / len(QS)
    print(f"generator {i + 1} ({m}): {acc_i:.0%}")
oracle = sum(any(is_right(c, gold) for c in r["cands"]) for r, (_, gold) in zip(RES["Archon augmented"], QS)) / len(QS)
print(f"any candidate right (oracle): {oracle:.0%}")

# %%
# 3. A noisier web (14 calls). Notice whether the planner's queries now return decoys first.
noise = []
for n in NAMES[:5]:
    for prefix in ("Old", "Upper"):
        t = TOWNS[n]
        noise.append({"title": f"{prefix} {n}", "text": f"{prefix} {n} is near {n}. It was founded in {t['founded'] - 50} and has a "
                      f"population of {t['pop'] + 777:,}. The {t['bridge']} replica in {prefix} {n} was built in {t['bridge_year'] + 9}."})
clean_web = WEB
WEB = BM25(DOCS + noise)
noisy = [archon(q) for q, _ in QS[:2]]
WEB = clean_web
for r, (q, gold) in zip(noisy, QS[:2]):
    print(f"gold {gold} | noisy-web answer {r['answer']} | pages {r['pages']}")
