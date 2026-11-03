# %% [markdown]
# # a07 · Search: sample, filter, cluster, and search while you think
# **Local AI Lab** · CS329A Lecture 8 (Oct 17) · AlphaCode-style sample-and-select, and Search-o1-style search in the middle of reasoning
#
# Two kinds of "search" make an agent better without training it. **Search over answers:** draw many programs, throw out the ones that fail the examples, group the rest by behavior, and submit one per group (AlphaCode). **Search for knowledge:** let the model stop mid-thought, run a search, boil the results down, and keep reasoning (Search-o1). We build both at small scale and grade them with code.

# %%
# EXTRA: rank-bm25
import json, math, os, random, re, subprocess, sys, tempfile, textwrap
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from lal.llm import ask
from lal.data import contract, labels, ids, sections
from lal.checks import pass_at_k, check

# %% [markdown]
# ## Part A · AlphaCode at laptop scale
# AlphaCode (Li et al. 2022, *Competition-Level Code Generation with AlphaCode*) drew up to millions of programs per problem, then cut them down to 10 submissions in two steps:
#
# 1. **Filter:** run each program on the example tests in the problem statement. This removed about 99% of samples.
# 2. **Cluster:** run the survivors on *new* test inputs (made by a separate model) and group programs that give the same outputs. Submit one program per group, biggest group first.
#
# Why biggest first? The paper's guess: there are many ways to be wrong, but correct programs tend to agree, so they pile up in the largest group. AlphaCode 2 (2023) kept the same pipeline and added a scoring model to pick the best program inside each of the 10 biggest groups.
#
# Our version: 5 small tasks, 6 samples each. Each task has **example tests** (the model sees them), **hidden tests** (used only for grading) and an **input generator** (random inputs with no answers, used only for clustering).
#
# ### A1. The tasks
# The reference solutions live here, in trusted code. They compute every expected output, so no answer is typed by hand.

# %%
REF_CODE = """
import re

def second_largest(nums):
    d = sorted(set(nums))
    return d[-2] if len(d) >= 2 else None

def merge_intervals(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def rle_decode(s):
    return "".join(ch * int(n) for ch, n in re.findall(r"([a-zA-Z])(\\d+)", s))

def max_subarray(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

def to_roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, sym in vals:
        while n >= v:
            out += sym
            n -= v
    return out
"""
REF = {}
exec(REF_CODE, REF)          # trusted code we wrote, so it is fine to run it here

rng = random.Random(0)
def rand_list(lo, hi, nmin, nmax):
    return [rng.randint(lo, hi) for _ in range(rng.randint(nmin, nmax))]
def rand_intervals():
    out = []
    for _ in range(rng.randint(0, 5)):
        s = rng.randint(0, 12); out.append([s, s + rng.randint(0, 4)])
    return out
def rand_rle():
    return "".join(rng.choice("abcxyz") + str(rng.randint(1, 14)) for _ in range(rng.randint(0, 4)))

# Each input is a list of positional arguments.
TASKS = [
    dict(fn="second_largest", ref=REF["second_largest"],
         spec="Return the second largest DISTINCT value in the list `nums`. If there are fewer than two distinct values, return None.",
         examples=[[[3, 1, 2]], [[5, 5, 4]]],
         hidden=[[[7, 7]], [[]], [[-1, -5, -3]], [[1, 2, 2, 3, 3]], [[10]], [[4, 4, 4, 9]]],
         gen=lambda: [rand_list(-4, 4, 0, 6)]),
    dict(fn="merge_intervals", ref=REF["merge_intervals"],
         spec="Merge overlapping intervals. Each interval is a list [start, end] with start <= end. Intervals that only touch "
              "(one ends where the next starts) also merge. Return a new list of merged intervals sorted by start.",
         examples=[[[[1, 3], [2, 6], [8, 10]]], [[[1, 2]]]],
         hidden=[[[]], [[[5, 6], [1, 2]]], [[[1, 2], [2, 3]]], [[[1, 10], [2, 3]]], [[[4, 5], [1, 4], [0, 0]]]],
         gen=lambda: [rand_intervals()]),
    dict(fn="rle_decode", ref=REF["rle_decode"],
         spec="Decode a run-length string. Each letter is followed by a positive whole number (it may have several digits) "
              "saying how many times to repeat it. Example: 'a3b1' -> 'aaab'. The empty string decodes to ''.",
         examples=[["a3b2"], ["x1"]],
         hidden=[["a12"], [""], ["z10y2"], ["b1c1d1"], ["q20"]],
         gen=lambda: [rand_rle()]),
    dict(fn="max_subarray", ref=REF["max_subarray"],
         spec="Return the largest possible sum of a NON-EMPTY contiguous slice of the list `nums`. "
              "`nums` has at least one element, and all elements may be negative.",
         examples=[[[1, -2, 3, 4, -1]], [[2, 3]]],
         hidden=[[[-3, -1, -2]], [[-5]], [[5, -9, 6]], [[0, 0]], [[-2, 1, -3, 4, -1, 2, 1, -5, 4]]],
         gen=lambda: [rand_list(-6, 6, 1, 7)]),
    dict(fn="to_roman", ref=REF["to_roman"],
         spec="Convert an integer 1 <= n <= 3999 to a Roman numeral string, using the subtractive forms IV, IX, XL, XC, CD and CM.",
         examples=[[3], [58]],
         hidden=[[4], [9], [14], [40], [90], [400], [1994], [3999]],
         gen=lambda: [rng.randint(1, 3999)]),
]
N_GEN = 12                                                       # generated inputs per task, for clustering only
for t in TASKS:
    t["gen_inputs"] = [t["gen"]() for _ in range(N_GEN)]
print(len(TASKS), "tasks. Example generated inputs for", TASKS[0]["fn"], ":", TASKS[0]["gen_inputs"][:4])

# %% [markdown]
# ### A2. A safe runner
# Model code runs **only** in a separate Python process with a timeout. The runner calls the function on a list of inputs and prints the outputs as JSON. A crash on one input is recorded as `"ERROR: <type>"` so it still forms part of the program's behavior.

# %%
RUNNER = r'''
import json, sys, copy
src, fn, inputs_path = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
inputs = json.load(open(inputs_path))
ns = {}
try:
    exec(compile(src, "candidate", "exec"), ns)
    f = ns[fn]
except Exception as e:
    print(json.dumps(["ERROR: " + type(e).__name__] * len(inputs))); sys.exit(0)
outs = []
for args in inputs:
    try:
        outs.append(f(*copy.deepcopy(args)))
    except Exception as e:
        outs.append("ERROR: " + type(e).__name__)
print(json.dumps(outs, default=repr))
'''
WORK = tempfile.mkdtemp(prefix="a07_")
RUNNER_PATH = os.path.join(WORK, "runner.py")
open(RUNNER_PATH, "w").write(RUNNER)

def run_outputs(code, fn, inputs, timeout=10):
    """Run `code` in a subprocess and return one JSON-normalized output per input."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=WORK, delete=False) as f:
        f.write(code); src = f.name
    inp = src + ".json"
    open(inp, "w").write(json.dumps(inputs))
    try:
        out = subprocess.run([sys.executable, "-I", RUNNER_PATH, src, fn, inp], capture_output=True, text=True, timeout=timeout)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        if not isinstance(res, list) or len(res) != len(inputs):
            raise ValueError("bad output")
    except subprocess.TimeoutExpired:
        res = ["TIMEOUT"] * len(inputs)
    except Exception:
        res = ["ERROR: runner"] * len(inputs)
    return [json.dumps(r, sort_keys=True) for r in res]       # strings, so outputs compare and hash cleanly

def expected(task, inputs):
    return [json.dumps(json.loads(json.dumps(task["ref"](*json.loads(json.dumps(a))), default=repr)), sort_keys=True) for a in inputs]

# Test the runner and the task set with the reference solutions themselves.
ref_ok = all(run_outputs(REF_CODE, t["fn"], t["hidden"]) == expected(t, t["hidden"]) for t in TASKS)
print("reference solutions pass their own hidden tests through the runner:", ref_ok)
print("a broken program:", run_outputs("def to_roman(n):\n    return 1/0", "to_roman", [[1], [2]]))
print("an endless loop:", run_outputs("def to_roman(n):\n    while True: pass", "to_roman", [[1]], timeout=3))

# %% [markdown]
# ### A3. Why clustering works: a planted wrong answer
# A common wrong `max_subarray` resets the running sum to 0, so it returns 0 when every number is negative. It **passes both example tests**. Filtering can't catch it, but its outputs on generated inputs differ from the correct program, so it lands in a different cluster.

# %%
CORRECT_MS = "def max_subarray(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n"
WRONG_MS = "def max_subarray(nums):\n    best = cur = 0\n    for x in nums:\n        cur = max(0, cur + x)\n        best = max(best, cur)\n    return best\n"
ms = TASKS[3]
for name, code in [("correct", CORRECT_MS), ("reset-to-0", WRONG_MS)]:
    print(f"{name:10} passes examples: {run_outputs(code, 'max_subarray', ms['examples']) == expected(ms, ms['examples'])} | "
          f"behavior on generated inputs: {run_outputs(code, 'max_subarray', ms['gen_inputs'])[:6]}")
planted_split = run_outputs(CORRECT_MS, "max_subarray", ms["gen_inputs"]) != run_outputs(WRONG_MS, "max_subarray", ms["gen_inputs"])

# %% [markdown]
# ### A4. Sample many programs
# For diversity, AlphaCode sampled half in Python and half in C++, put random problem tags and ratings in the prompt, and used a fairly high temperature. AlphaCode 2 also randomized the temperature per sample. We use a temperature of 0.8 and 6 samples per task (30 calls), 2 at a time.

# %%
N_SAMPLES = 6

def code_prompt(t):
    ex = "\n".join(f"{t['fn']}({', '.join(map(repr, a))}) == {t['ref'](*json.loads(json.dumps(a)))!r}" for a in t["examples"])
    return (f"Write a Python function `{t['fn']}`.\n{t['spec']}\n\nExamples:\n{ex}\n\n"
            "Return only the code, in one ```python block. No explanation.")

def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.S)
    return max(blocks, key=len) if blocks else (text or "")

def sample(t):
    return extract_code(ask(code_prompt(t), temperature=0.8, max_tokens=600))

jobs = [(ti, k) for ti in range(len(TASKS)) for k in range(N_SAMPLES)]
with ThreadPoolExecutor(2) as ex:
    codes = list(ex.map(lambda j: sample(TASKS[j[0]]), jobs))
SAMPLES = defaultdict(list)
for (ti, _), c in zip(jobs, codes):
    SAMPLES[ti].append(c)
print(SAMPLES[0][0][:400])

# %% [markdown]
# ### A5. Filter, cluster, select
# For each sample we record three things: does it pass the examples, its behavior signature on generated inputs, and (for grading only) does it pass the hidden tests.

# %%
def analyse(ti):
    t = TASKS[ti]
    rows = []
    for code in SAMPLES[ti]:
        rows.append(dict(code=code,
                         ex_ok=run_outputs(code, t["fn"], t["examples"]) == expected(t, t["examples"]),
                         sig=tuple(run_outputs(code, t["fn"], t["gen_inputs"])),
                         hidden_ok=run_outputs(code, t["fn"], t["hidden"]) == expected(t, t["hidden"])))
    return rows

def clusters_of(rows):
    """Group the rows that passed the examples by behavior signature, biggest group first."""
    groups = defaultdict(list)
    for r in rows:
        if r["ex_ok"]:
            groups[r["sig"]].append(r)
    return sorted(groups.values(), key=len, reverse=True)

RESULTS = {ti: analyse(ti) for ti in range(len(TASKS))}
for ti, rows in RESULTS.items():
    cl = clusters_of(rows)
    print(f"{TASKS[ti]['fn']:16} samples {len(rows)} | pass examples {sum(r['ex_ok'] for r in rows)} | "
          f"clusters {[len(c) for c in cl]} | cluster correct? {[c[0]['hidden_ok'] for c in cl]} | "
          f"any sample correct: {any(r['hidden_ok'] for r in rows)}")

# %% [markdown]
# ### A6. Compare selection methods
# We have a budget of `K` submissions per task. A task counts as solved if any submission passes all hidden tests.
# - **random:** pick K samples at random (expected value, using pass@k).
# - **filter + random:** pick K at random from the samples that pass the examples.
# - **filter + cluster:** one sample from each of the K biggest clusters.
# - **oracle:** any sample is correct (the ceiling, also called *coverage*).

# %%
def solve_rates(K):
    out = defaultdict(float)
    for ti, rows in RESULTS.items():
        n, c = len(rows), sum(r["hidden_ok"] for r in rows)
        f = [r for r in rows if r["ex_ok"]]
        fc = sum(r["hidden_ok"] for r in f)
        out["random"] += pass_at_k(n, c, min(K, n)) if n else 0
        out["filter + random"] += pass_at_k(len(f), fc, min(K, len(f))) if f else 0
        out["filter + cluster"] += float(any(cl[0]["hidden_ok"] for cl in clusters_of(rows)[:K]))
        out["oracle (coverage)"] += float(c > 0)
    return {k: v / len(RESULTS) for k, v in out.items()}

TABLE = {K: solve_rates(K) for K in (1, 2)}
print(f"{'method':20}" + "".join(f"{'K=' + str(K):>8}" for K in TABLE))
for m in TABLE[1]:
    print(f"{m:20}" + "".join(f"{TABLE[K][m]:>8.0%}" for K in TABLE))

# %% [markdown]
# What to notice: filtering helps because example tests are a cheap, exact verifier. Clustering helps most when many samples pass the examples but disagree with each other. The gap to the oracle row is the *selection* problem, which AlphaCode 2 attacked with a learned scoring model.

# %% [markdown]
# ## Part B · Search-o1: search in the middle of reasoning
# Search-o1 (Li et al. 2025, *Search-o1: Agentic Search-Enhanced Large Reasoning Models*) lets a reasoning model write a search query between special tags whenever it is unsure. The system pauses, retrieves documents, and a separate **Reason-in-Documents** step condenses them into a few useful facts. Only those facts go back into the reasoning, so long noisy pages don't derail it.
#
# We compare three ways to answer questions about our company's contracts and policies:
# 1. **No search:** the model alone. It has never seen our data, so it should fail.
# 2. **Naive RAG:** retrieve once with the question, then answer.
# 3. **Search-o1 style:** the model decides what to search, possibly several times, and each result is condensed first.
#
# ### B1. The corpus: 20 contracts plus a few internal policies
# The questions need **two hops**: one fact from a contract, and one rule from a policy.

# %%
POLICIES = {
    "Procurement Policy PP-4": [
        "Purpose. This policy sets who must sign off on recurring vendor spending.",
        "Approval tiers. The approver depends on the monthly fee in the vendor agreement. A monthly fee under $5,000 is approved by the Department Head. "
        "A monthly fee from $5,000 up to and including $15,000 is approved by the Finance Director. A monthly fee above $15,000 is approved by the Chief Financial Officer.",
        "Records. Keep the signed approval with the contract file for seven years.",
    ],
    "Legal Policy LP-2": [
        "Outside counsel. Disputes are handled by the outside counsel firm assigned to the state whose law governs the agreement.",
        "Assignments. Delaware law: Hale & Ross LLP. California law: Pacific Ridge Law Group. Texas law: Lone Star Counsel. "
        "Florida law: Bayfront Legal. New York law: Hudson Street Partners.",
        "Budget. Outside counsel spending above $50,000 per matter needs General Counsel approval.",
    ],
    "Contracts Desk Policy CD-1": [
        "Non-renewal letters. To stop an agreement from renewing, the Contracts Desk sends the non-renewal letter early. "
        "The letter must go out a number of days before the renewal date equal to the agreement's notice period plus a 15-day internal buffer.",
        "Tracking. Every agreement's renewal date is entered in the contracts calendar on signature.",
    ],
    "Travel Policy TR-3": [
        "Approval. Trips above $3,000 are approved by the Department Head before booking.",
        "Per diem. Meals are reimbursed up to $75 per day in the United States.",
    ],
    "IT Equipment Policy IT-7": [
        "Laptops. Standard laptops are replaced every 36 months. Exceptions are approved by the IT Director.",
        "Security. Lost devices must be reported within 24 hours.",
    ],
}

def parties(cid):
    m = re.search(r'entered into by (.+?) \("Customer"\) and (.+?) \("Vendor"\)', contract(cid))
    return (m.group(1), m.group(2)) if m else ("?", "?")

DOCS = []
for cid in ids():
    cust, vend = parties(cid)
    for s in sections(contract(cid)):
        DOCS.append({"id": f"{cid}#{s['id']}", "text": f"Agreement between {cust} (Customer) and {vend} (Vendor), "
                                                       f"section {s['id']} {s['title']}: {s['text']}"})
for title, paras in POLICIES.items():
    for i, p in enumerate(paras):
        DOCS.append({"id": f"{title}#{i + 1}", "text": f"{title}. {p}"})

tok = lambda s: re.findall(r"\w+", s.lower())
BM25 = BM25Okapi([tok(d["text"]) for d in DOCS])

def search(query, k=3):
    """The local search tool: BM25 over contract sections and policy paragraphs."""
    scores = BM25.get_scores(tok(query))
    return [DOCS[i] for i in sorted(range(len(DOCS)), key=lambda i: -scores[i])[:k]]

for d in search("Bayview Health Partners Brightline Data Corp monthly fee"):
    print(d["id"], "|", d["text"][:110])
bm25_ok = search(f"{' '.join(parties('msa_01'))} fees", k=3)[0]["id"].startswith("msa_01")

# %% [markdown]
# ### B2. Questions with known answers
# Built from the labels with a seed. We only use customer/vendor pairs that appear in exactly one contract, so each question has one right answer.

# %%
def approver(fee):
    return "Department Head" if fee < 5000 else ("Finance Director" if fee <= 15000 else "Chief Financial Officer")
COUNSEL = {"Delaware": "Hale & Ross", "California": "Pacific Ridge", "Texas": "Lone Star", "Florida": "Bayfront", "New York": "Hudson Street"}

L = labels()
pair_count = defaultdict(int)
for cid in ids():
    pair_count[parties(cid)] += 1
unique = [cid for cid in ids() if pair_count[parties(cid)] == 1]
qrng = random.Random(7)
picked = qrng.sample(unique, 6)
QUESTIONS = []
for i, cid in enumerate(picked):
    cust, vend = parties(cid); terms = L[cid]["terms"]
    if i % 3 == 0:
        QUESTIONS.append((f"Under our procurement policy, who must approve the agreement between {cust} and {vend}?", approver(terms["monthly_fee"])))
    elif i % 3 == 1:
        QUESTIONS.append((f"Which outside counsel firm handles disputes under the agreement between {cust} and {vend}?", COUNSEL[terms["governing_law"]]))
    else:
        QUESTIONS.append((f"How many days before the renewal date must the Contracts Desk send the non-renewal letter for the agreement between {cust} and {vend}?",
                          str(terms["notice_days"] + 15)))
for q, a in QUESTIONS:
    print(f"- {q}  -> gold: {a}")

def words(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def is_correct(answer, gold):
    """Lenient grading: a number must appear; for names, every word of the gold name must appear."""
    if gold.isdigit():
        return gold in re.findall(r"\d+", (answer or "").replace(",", ""))
    return all(w in words(answer) for w in words(gold))

def final_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    return m[-1].strip() if m else (text or "").strip()[-200:]

# %% [markdown]
# ### B3. Baselines: no search, and naive RAG

# %%
def no_search(q):
    return final_answer(ask(f"{q}\nIf you are not sure, give your best guess. End with a line 'ANSWER: <short answer>'.", max_tokens=300))

def naive_rag(q, k=4):
    ctx = "\n\n".join(f"[{d['id']}] {d['text']}" for d in search(q, k=k))
    return final_answer(ask(f"Sources:\n{ctx}\n\nQuestion: {q}\nThink briefly, then end with a line 'ANSWER: <short answer>'.", max_tokens=400))

# %% [markdown]
# ### B4. The Search-o1 loop
# 1. The model reasons. When it needs a fact it writes `<search>query</search>`. We pass `stop=["</search>"]` so generation pauses right there.
# 2. We run the search tool.
# 3. **Reason-in-Documents:** a separate call reads the raw pages plus the reasoning so far and writes 1–3 sentences of useful facts.
# 4. We paste those facts back inside `<result>...</result>` and let the model continue. At most 3 searches.

# %%
S1_SYSTEM = ("You answer questions about our company's vendor agreements and internal policies. You do NOT know these facts yourself. "
             "Think step by step. Whenever you need a fact, write a search query like this: <search>your query</search> and then stop. "
             "Search for one fact at a time, using the names in the question. The results will come back as <result>...</result>. "
             "When you know enough, finish with a line 'ANSWER: <short answer>'.")

from lal.llm import chat
def ask_msgs(msgs, **kw):
    return chat(msgs, max_tokens=400, **kw).content or ""

def reason_in_documents(reasoning, query, docs):
    pages = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    return ask(f"Reasoning so far:\n{reasoning[-1500:]}\n\nSearch query: {query}\n\nSearched pages:\n{pages}\n\n"
               "Read the pages and write only the facts that help with the search query, in 1-3 sentences. "
               "Copy names and numbers exactly. If nothing helps, write 'No helpful information found.'", max_tokens=200).strip()

def search_o1(q, max_searches=3, condense=True, verbose=False):
    base = [{"role": "system", "content": S1_SYSTEM}, {"role": "user", "content": q}]
    transcript, n_search = "", 0
    while True:
        msgs = base + ([{"role": "assistant", "content": transcript},
                        {"role": "user", "content": "Continue your reasoning from where you stopped."}] if transcript else [])
        out = ask_msgs(msgs, stop=["</search>"])
        out = out.split("</search>")[0]                    # in case the server ignores `stop`
        i_s, i_a = out.find("<search>"), out.find("ANSWER:")
        transcript += ("\n" if transcript else "") + out
        if i_a != -1 and (i_s == -1 or i_a < i_s):         # answered before (or without) asking for a search
            break
        if i_s == -1 or n_search >= max_searches:          # no search and no answer, or out of budget: ask for the answer
            transcript += "\n" + ask_msgs(base + [{"role": "assistant", "content": transcript},
                                                  {"role": "user", "content": "Give your final answer now as 'ANSWER: <short answer>'."}])
            break
        query = out[i_s + len("<search>"):].strip()[:200]
        docs = search(query)
        info = reason_in_documents(transcript, query, docs) if condense else "\n".join(d["text"] for d in docs)
        transcript += f"</search>\n<result>{info}</result>"
        n_search += 1
        if verbose:
            print(f"  search {n_search}: {query!r}\n    -> {info[:160]}")
    return final_answer(transcript), n_search

# %% [markdown]
# ### B5. Score all three

# %%
rows = []
for qi, (q, gold) in enumerate(QUESTIONS):
    a_none = no_search(q)
    a_rag = naive_rag(q)
    a_s1, n = search_o1(q, verbose=(qi == 0))               # print the searches for the first question
    rows.append((is_correct(a_none, gold), is_correct(a_rag, gold), is_correct(a_s1, gold), n))
    print(f"gold {gold!r:24} | none {a_none[:30]!r:34} | rag {a_rag[:30]!r:34} | search-o1 ({n} searches) {a_s1[:30]!r}")
acc = {name: sum(r[i] for r in rows) / len(rows) for i, name in enumerate(["no search", "naive RAG", "Search-o1 style"])}
print({k: f"{v:.0%}" for k, v in acc.items()})

# %% [markdown]
# What to notice: one retrieval with the whole question often brings back the contract **or** the policy, but not both. The agentic loop can search for each hop separately. The paper reports the same pattern on multi-hop QA: agentic search beat one-shot RAG, and adding Reason-in-Documents helped further. On a small model the loop can also fail by never searching, or by searching with vague words. Read the transcripts.

# %%
check("reference solutions pass their own hidden tests (the task set is sound)", ref_ok)
check("a planted wrong program that passes the examples lands in a different cluster", planted_split)
check("filter + cluster (K=1) is at least as good as random choice (K=1)", TABLE[1]["filter + cluster"] >= TABLE[1]["random"] - 1e-9)
check("BM25 finds the right contract from party names", bm25_ok)
check("Search-o1 style beats no search on our questions", acc["Search-o1 style"] > acc["no search"])

# %% [markdown]
# ## Your turn
# 1. **Model-made test inputs.** AlphaCode trained a separate model to write test inputs. Ask the model for 8 new inputs for `merge_intervals` (as a JSON list of argument lists), and cluster with those instead of our random generator. Do the clusters change?
# 2. **Turn off Reason-in-Documents.** Run `search_o1(q, condense=False)`, which pastes the raw pages back. Compare accuracy and the number of searches.
# 3. **Scale the samples.** Plot the filter + cluster solve rate (K=1) using the first 2, 4 and 6 samples per task. Does it rise as AlphaCode saw (roughly log-linear)?
#
# **Next:** a08 · Agentic frameworks for software engineering

# %% [markdown]
# ## Solutions

# %%
# Solution 1: model-written test inputs for clustering (1 call)
t = TASKS[1]
raw = ask(f"Write 8 varied test inputs for this Python function, including edge cases.\n{t['spec']}\n"
          f"The function is called as {t['fn']}(intervals). Reply with only a JSON list where each item is a list of arguments, "
          "for example [[[[1,2],[3,4]]], [[]]].", temperature=0.3, max_tokens=400)
try:
    model_inputs = json.loads(re.search(r"\[.*\]", raw, flags=re.S).group(0))
    model_inputs = [a for a in model_inputs if isinstance(a, list) and len(a) == 1][:8]
except Exception:
    model_inputs = []
print("model inputs:", model_inputs)
if model_inputs:
    groups = defaultdict(list)
    for r in RESULTS[1]:
        if r["ex_ok"]:
            groups[tuple(run_outputs(r["code"], t["fn"], model_inputs))].append(r)
    print("cluster sizes with random inputs:", [len(c) for c in clusters_of(RESULTS[1])],
          "| with model inputs:", sorted(map(len, groups.values()), reverse=True))
# What to notice: as AlphaCode says, inputs don't need answers, or even to be valid, to separate programs.
# Edge-case inputs (empty list, touching intervals) are what split right from wrong.

# %%
# Solution 2: raw pages instead of condensed facts (about 3-4 calls per question)
raw_rows = [is_correct(search_o1(q, condense=False)[0], gold) for q, gold in QUESTIONS[:3]]
cond_rows = [r[2] for r in rows[:3]]
print(f"first 3 questions | condensed: {sum(cond_rows)}/3 | raw pages: {sum(raw_rows)}/3")
# What to notice: with three short pages the difference may be small. Search-o1's motivation is long, noisy web pages,
# where raw text pushes the reasoning off track. Try search(query, k=8) to see the effect grow.

# %%
# Solution 3: solve rate as the sample budget grows (no new model calls; we reuse the samples)
for n in (2, 4, 6):
    solved = 0
    for ti, rows_t in RESULTS.items():
        cl = clusters_of(rows_t[:n])
        solved += bool(cl) and cl[0][0]["hidden_ok"]
    print(f"{n} samples per task -> filter + cluster K=1 solves {solved}/{len(RESULTS)}")
# What to notice: more samples raise the chance a correct program exists AND make the correct cluster bigger.
# AlphaCode needed millions of samples; AlphaCode 2 reached the same level with about 100, using a better model.
