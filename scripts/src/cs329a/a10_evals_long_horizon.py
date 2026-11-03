# %% [markdown]
# # a10 · Agentic evaluations and long-horizon tasks
# **Local AI Lab** · CS329A Lecture 17 (Nov 17) · measure how long a task your agent can do, grade real deliverables, and score a research report
#
# "78% on a benchmark" does not tell a customer what an agent can do for them. This lecture's three papers measure agents in human terms. **METR** asks: how long (in human minutes) is a task the agent finishes half the time? **GDPval** asks: would an expert prefer the agent's work product over another expert's? **DeepScholar-Bench** scores a written research summary with automatic metrics for coverage and citations. We build a small version of each on our contracts.

# %%
# MODELS: llama3.2:3b
import json, math, random, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from lal.llm import ask
from lal.data import contract, labels, playbook, ids
from lal.tasks import extract_int
from lal.checks import check

LAB, PB = labels(), playbook()

# %% [markdown]
# ## 1. METR's time horizon, on a toy
# METR (Kwa et al. 2025, *Measuring AI Ability to Complete Long Tasks*) timed skilled people on 170 software tasks, from seconds to many hours. For each model they fit a **logistic curve**: chance of success against the log of the human time. The **50% time horizon** is the human task length where that curve crosses 50%. They report it doubled about every 7 months since 2019 (Claude 3.7 Sonnet: about 50 minutes in the abstract).
#
# Our toy: multi-step contract checks. A task with N steps asks the agent to walk through N facts from a contract fact sheet, keeping a running total. Each extra step is one more lookup and one more chance to slip. We **assume** a person needs 1 minute to set up plus 1.5 minutes per step. That is our made-up estimate; METR timed real people.

# %%
FACTS = {c: LAB[c]["terms"] for c in ids()}
SHEET = "id | monthly fee ($) | notice days | renewal years | liability cap (months) | governing law\n" + "\n".join(
    f"{c} | {t['monthly_fee']} | {t['notice_days']} | {t['renewal_years']} | {t['liability_cap_months'] if t['liability_cap_months'] else 'none'} | {t['governing_law']}"
    for c, t in FACTS.items())
CAPPED = [c for c, t in FACTS.items() if t["liability_cap_months"]]

def make_task(n_steps, seed):
    """A chain of n_steps lookups with an exact answer."""
    rng = random.Random(f"metr-toy-{n_steps}-{seed}")
    c0 = rng.choice(ids())
    total = FACTS[c0]["monthly_fee"] // 100
    steps = [f"Start with the monthly fee of {c0} divided by 100."]
    for _ in range(n_steps - 1):
        kind = rng.randrange(4)
        if kind == 0:
            c = rng.choice(ids()); total += FACTS[c]["notice_days"]
            steps.append(f"Add the notice days of {c}.")
        elif kind == 1:
            c = rng.choice(ids()); total -= FACTS[c]["renewal_years"]
            steps.append(f"Subtract the renewal years of {c}.")
        elif kind == 2:
            c = rng.choice(CAPPED); total += FACTS[c]["liability_cap_months"]
            steps.append(f"Add the liability cap in months of {c}.")
        else:
            c = rng.choice(ids()); ok = FACTS[c]["governing_law"] in PB["allowed_law"]
            total += 10 if ok else -10
            steps.append(f"Add 10 if the governing law of {c} is California or Delaware; otherwise subtract 10.")
    return {"n": n_steps, "minutes": 1 + 1.5 * n_steps, "steps": steps, "answer": total}

STEP_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]     # 15 tasks, 2.5 to 49 "human minutes"
TASKS = [make_task(n, seed=0) for n in STEP_COUNTS]
print(TASKS[2])

def run_task(task, temperature=0.5):
    prompt = (f"Contract fact sheet:\n{SHEET}\n\nDo these steps in order. Write the running total after each step, "
              "then end with 'ANSWER: <number>'.\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(task["steps"])))
    return extract_int(ask(prompt, temperature=temperature, max_tokens=1500)) == task["answer"]

# %% [markdown]
# Run each task twice (30 calls). METR ran each agent 8 times per task; we keep it small.

# %%
N_RUNS = 2
t0 = time.time()
jobs = [t for t in TASKS for _ in range(N_RUNS)]
with ThreadPoolExecutor(2) as ex:
    outcomes = list(ex.map(run_task, jobs))
minutes = np.array([t["minutes"] for t in jobs])
success = np.array(outcomes, dtype=float)
print(f"{len(jobs)} runs in {time.time()-t0:.0f}s")
for t in TASKS:
    s = [o for j, o in zip(jobs, outcomes) if j is t]
    print(f"  {t['n']:2} steps ~{t['minutes']:4.1f} min: {sum(s)}/{len(s)}")

# %% [markdown]
# ### Fit the logistic curve
# Model: P(success) = sigmoid(a + b · log2(minutes)). We fit a and b by maximum likelihood with Newton's method (a few exact steps instead of many small ones), plus a tiny penalty so the fit stays finite when the data separate perfectly. The 50% horizon is where a + b·x = 0, so x = -a/b and horizon = 2^x minutes. The 80% horizon solves a + b·x = log(0.8/0.2).

# %%
def fit_logistic(minutes, success, l2=1e-2, iters=50):
    x = np.log2(np.asarray(minutes, float)); y = np.asarray(success, float)
    X = np.stack([np.ones_like(x), x], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) + l2 * np.array([0, w[1]])
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.diag([1e-6, 1.0])
        w = w - np.linalg.solve(H, grad)
    return w

def horizon(w, level=0.5):
    a, b = w
    if b >= 0:                       # success does not fall with length: no horizon in this range
        return float("inf")
    return float(2 ** ((math.log(level / (1 - level)) - a) / b))

# Test the fit on fake data with a known answer: true 50% horizon = 16 minutes.
rng = np.random.default_rng(0)
fake_m = np.repeat(2 ** np.linspace(0, 7, 15), 40)
fake_y = (rng.random(fake_m.size) < 1 / (1 + np.exp(-(-1.2 * (np.log2(fake_m) - 4))))).astype(float)
w_fake = fit_logistic(fake_m, fake_y)
print(f"fake data: recovered 50% horizon {horizon(w_fake):.1f} min (true 16.0)")

w = fit_logistic(minutes, success)
h50, h80 = horizon(w, 0.5), horizon(w, 0.8)
print(f"our model: slope {w[1]:.2f}, 50% horizon {h50:.1f} min, 80% horizon {h80:.1f} min")

# Bootstrap over tasks for a rough interval (METR resampled families, tasks and runs).
task_idx = np.repeat(np.arange(len(TASKS)), N_RUNS)
boots = []
for _ in range(500):
    pick = rng.integers(0, len(TASKS), len(TASKS))
    mask = np.concatenate([np.where(task_idx == k)[0] for k in pick])
    boots.append(horizon(fit_logistic(minutes[mask], success[mask])))
finite = np.array([b for b in boots if np.isfinite(b)])
if finite.size:
    print(f"bootstrap 50% horizon: median {np.median(finite):.1f} min, 10th-90th percentile {np.percentile(finite, 10):.1f}-{np.percentile(finite, 90):.1f}")

# %% [markdown]
# **How to read this.** If the 50% horizon is, say, 12 minutes, the model finishes about half of tasks that would take a person 12 minutes, and fewer of the longer ones. The 80% horizon is always shorter: METR found 80% horizons about 4-6x shorter than 50% ones. For a business, the 80% (or higher) number is the one that matters, since a review step must catch every miss. Three caveats from the paper apply here too: the human times drive everything (ours are guesses); tasks within one family are related (METR down-weighted large families); and success on tidy tasks may not carry over to messy real work.
#
# If the horizon prints as `inf` or is beyond 49 minutes, the model never dropped below 50% in our range: make longer tasks. If every run failed, make shorter ones.

# %% [markdown]
# ## 2. GDPval: would an expert prefer this deliverable?
# GDPval (Patwardhan et al. 2025, *GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks*) has 1,320 tasks across 44 occupations, written by professionals with about 14 years of experience on average. Each task asks for a real work product (a memo, a spreadsheet, a slide deck). Expert graders compare the model's deliverable against a human expert's, **blind and pairwise**, and the score is the **win-or-tie rate**. The best model on their gold set, Claude Opus 4.1, had 47.6% of deliverables rated as good as or better than the expert's. Their experimental automatic grader agreed with human experts 65.7% of the time, while human experts agreed with each other 70.8% of the time.
#
# Our version: three client emails about contracts that break the playbook. We write the expert reference from the labels. An LLM grader compares the model's email with the expert's, in both orders.

# %%
RULE_TEXT = PB["rules"]
TASK_CIDS = ["msa_08", "msa_10", "msa_17"]
EXPERT = {
    "msa_08": "Subject: msa_08 - liability cap too high\n\nThe liability cap in clause 7.1 is fees paid in the prior 36 months. Our playbook allows at most 12 months. "
              "Recommendation: revise 7.1 to cap each party's liability at the fees paid in the 12 months preceding the claim. The rest of the agreement meets the playbook.",
    "msa_10": "Subject: msa_10 - governing law\n\nClause 10.1 chooses Florida law. Our playbook requires California or Delaware law. "
              "Recommendation: change 10.1 to Delaware (or California) law. The liability cap (12 months), one-year renewals and 30-day notice all meet the playbook.",
    "msa_17": "Subject: msa_17 - renewal and notice terms\n\nClause 4.1 has two problems. It renews automatically for five-year periods; the playbook allows one year at most. "
              "It also needs only 15 days' notice of non-renewal; the playbook requires at least 30 days. Recommendation: revise 4.1 to one-year renewals with at least 30 days' notice.",
}
TASK = ("Write a short email (under 120 words) to the client's general counsel about contract {cid}. Say which playbook rules it breaks, "
        "cite the clause numbers, and recommend a specific fix.\n\nPlaybook rules:\n{rules}\n\nContract:\n{doc}")

def deliverable(cid):
    return ask(TASK.format(cid=cid, rules=json.dumps(RULE_TEXT, indent=1), doc=contract(cid)), max_tokens=300)

GRADER = """You are a senior contracts lawyer grading two work products for the same task. You cannot see who wrote them.
Task: {task}
Playbook rules: {rules}
Contract:
{doc}

Email A:
{a}

Email B:
{b}

Judge correctness first (right rules, right clause numbers, workable fix), then clarity. Length is not quality.
Reply with one word: A, B, or TIE."""

def grade_once(cid, a, b):
    r = ask(GRADER.format(task=TASK.split("\n")[0].format(cid=cid), rules=json.dumps(RULE_TEXT), doc=contract(cid), a=a, b=b), max_tokens=5)
    m = re.search(r"\b(A|B|TIE)\b", r.upper())
    return m.group(1) if m else "TIE"

def pairwise(cid, model_text, expert_text):
    """Model vs expert, graded twice with the order swapped. Disagreement after the swap counts as a tie."""
    first = grade_once(cid, model_text, expert_text)     # model is A
    second = grade_once(cid, expert_text, model_text)    # model is B
    if first == "A" and second == "B": return "win"
    if first == "B" and second == "A": return "loss"
    return "tie"

with ThreadPoolExecutor(2) as ex:
    drafts = dict(zip(TASK_CIDS, ex.map(deliverable, TASK_CIDS)))         # 3 calls
verdicts = {c: pairwise(c, drafts[c], EXPERT[c]) for c in TASK_CIDS}      # 6 calls
for c in TASK_CIDS:
    print(f"--- {c}: {verdicts[c]}\n{drafts[c][:300]}\n")
win_or_tie = sum(v in ("win", "tie") for v in verdicts.values()) / len(verdicts)
print(f"win-or-tie rate vs expert: {win_or_tie:.0%}  ({verdicts})")

# Sanity check of the grader: a confidently wrong email should lose to the expert.
WRONG = "Subject: msa_17 looks good\n\nI reviewed msa_17. It meets every playbook rule, so the client can sign it as is."
sanity = pairwise("msa_17", WRONG, EXPERT["msa_17"])                    # 2 calls
print("wrong email vs expert:", sanity)

# %% [markdown]
# **What to notice:** the swap matters. Small graders often pick the same position both times, and we count that as a tie rather than trusting it. GDPval also found that review time eats most of the speed gain: for GPT-5, the raw ratio was about 90x faster than an expert, but once an expert reviews the output and redoes it when needed ("try once, then fix"), the gain was about 1.12x faster and 1.18x cheaper. Report both numbers to a customer.

# %% [markdown]
# ## 3. DeepScholar-Bench: score a research summary automatically
# DeepScholar-Bench (Patel et al. 2025, *DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis*) asks systems to write the related-work section of a recent arXiv paper, and scores three things: **knowledge synthesis** (e.g. *nugget coverage*: how many key facts from the human-written section appear), **retrieval quality** (e.g. *reference coverage*: how many important references were found) and **verifiability** (*citation precision*: does each cited source support its sentence; *claim coverage*: is every sentence backed by a citation). No system passed a geometric mean of 31% across their metrics.
#
# Our version uses code only, so it is cheap and repeatable. We give the model six sources: three on agent evaluation (relevant) and three on memory and serving (distractors). It writes a short cited summary about evaluating agents.
#
# One simplification: the paper uses an LLM judge to decide whether a source supports a sentence. We use a word-overlap rule instead (a cited source "supports" a sentence if they share at least 3 content words). It is crude but deterministic.

# %%
SOURCES = {
    "S1": "METR measures AI ability to complete long tasks with the 50% task-completion time horizon: the length of tasks, timed on skilled humans, that a model completes with 50% success. The horizon has doubled about every seven months since 2019.",
    "S2": "GDPval evaluates models on real-world economically valuable tasks from 44 occupations, written by industry experts. Experts compare model deliverables with expert deliverables in blind pairwise grading and report win or tie rates.",
    "S3": "DeepScholar-Bench is a live benchmark for generative research synthesis. It scores generated related-work sections on knowledge synthesis, retrieval quality and verifiability, and no system exceeds a geometric mean of 31 percent.",
    "S4": "MemGPT manages memory tiers for language models like an operating system, paging data between main context and archival storage through function calls.",
    "S5": "CacheBlend reuses precomputed KV caches of retrieved text chunks and recomputes a small subset of tokens, reducing time to first token.",
    "S6": "Cartridges train a small KV cache offline for a corpus using self-study and context distillation, cutting serving memory.",
}
GOLD = {"S1", "S2", "S3"}
NUGGETS = {   # key facts a good summary should contain: (description, regex)
    "50% time horizon": r"50\s*%|half|time horizon",
    "doubling about every 7 months": r"(7|seven)\s*months|doubl",
    "44 occupations / real work": r"44|occupation",
    "blind pairwise expert grading": r"pairwise|blind|win",
    "three axes: synthesis, retrieval, verifiability": r"verifiab|retrieval quality|synthesis",
    "benchmarks far from saturated (31%)": r"31|saturat",
}
STOP = set("the a an of and or to for in on with by is are that this it as from at be its their they about each than model models".split())

def content_words(s):
    return {w for w in re.findall(r"[a-z0-9%]+", s.lower()) if w not in STOP and len(w) > 2}

def split_sentences(text):
    text = re.sub(r"([.!?])\s*((?:\[[^\]]+\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)   # "x. [S1]" -> "x [S1]."
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 3]

def citations(sentence):
    return {f"S{n}" for group in re.findall(r"\[([^\]]+)\]", sentence) for n in re.findall(r"S\s*(\d+)", group)}

def scholar_metrics(text, sources=SOURCES, gold=GOLD, nuggets=NUGGETS):
    sents = split_sentences(text)
    cited = set().union(*[citations(s) for s in sents]) & set(sources) if sents else set()
    pairs = [(s, c) for s in sents for c in citations(s) if c in sources]
    supported = [len(content_words(re.sub(r"\[[^\]]+\]", "", s)) & content_words(sources[c])) >= 3 for s, c in pairs]
    sent_ok = [any(len(content_words(re.sub(r"\[[^\]]+\]", "", s)) & content_words(sources[c])) >= 3 for c in citations(s) if c in sources) for s in sents]
    m = {
        "nugget_coverage": sum(bool(re.search(p, text, re.I)) for p in nuggets.values()) / len(nuggets),
        "reference_coverage": len(cited & gold) / len(gold),
        "retrieval_precision": len(cited & gold) / len(cited) if cited else 0.0,
        "citation_precision": sum(supported) / len(supported) if supported else 0.0,
        "claim_coverage": sum(sent_ok) / len(sent_ok) if sent_ok else 0.0,
    }
    vals = list(m.values())
    m["geometric_mean"] = float(np.exp(np.mean(np.log(np.maximum(vals, 1e-3)))))
    return m

ORACLE = ("METR reports a 50% task-completion time horizon, the length of human-timed tasks a model completes with 50% success, which has doubled about every seven months since 2019 [S1]. "
          "GDPval evaluates real-world economically valuable tasks from 44 occupations, using blind pairwise grading by experts against expert deliverables [S2]. "
          "DeepScholar-Bench scores generated related-work sections on knowledge synthesis, retrieval quality and verifiability, and no system exceeds a geometric mean of 31 percent [S3].")
BAD = ("Agents are evaluated in many ways. Memory systems page data between main context and archival storage [S1]. "
       "Caching reduces time to first token [S5]. Benchmarks are important.")
for name, txt in [("oracle", ORACLE), ("bad", BAD)]:
    print(f"{name:7}", {k: round(v, 2) for k, v in scholar_metrics(txt).items()})

# %%
SRC_BLOCK = "\n".join(f"[{k}] {v}" for k, v in SOURCES.items())
summary = ask(f"Sources:\n{SRC_BLOCK}\n\nWrite a 4-sentence related-work paragraph on how to evaluate AI agents on long, realistic tasks. "
              "Use only sources that are relevant. Cite sources inline like [S1] at the end of each sentence.", max_tokens=350)
print(summary, "\n")
model_m = scholar_metrics(summary)
print({k: round(v, 2) for k, v in model_m.items()})

# %% [markdown]
# **What to notice:** each metric catches a different failure. Citing a distractor (MemGPT, CacheBlend) lowers retrieval precision. A sentence with no citation lowers claim coverage. Citing a real source for a claim it does not make lowers citation precision. A single "quality" score would hide which one went wrong. DeepScholar-Bench's results show the same split: OpenAI DeepResearch led on the geometric mean (0.309), but a Claude-based search agent had much higher citation precision (0.701) and claim coverage (0.760).

# %%
check("logistic fit recovers a known horizon (within 30%)", abs(horizon(w_fake) - 16) / 16 < 0.3)
check("80% horizon is shorter than 50% horizon on fake data", horizon(w_fake, 0.8) < horizon(w_fake))
check("the agent solves the shortest task at least once", any(o for j, o in zip(jobs, outcomes) if j["n"] <= 2))
check("success on long tasks is no better than on short tasks", success[minutes >= 25].mean() <= success[minutes <= 8].mean())
check("grader does not prefer the wrong email over the expert", sanity != "win")
check("oracle summary beats bad summary on every metric",
      all(scholar_metrics(ORACLE)[k] > scholar_metrics(BAD)[k] for k in scholar_metrics(ORACLE)))
check("model summary cites at least 2 of the 3 relevant sources", model_m["reference_coverage"] >= 2 / 3)

# %% [markdown]
# ## Your turn
# 1. Make the toy tasks harder per step (for example, ask for the fee in dollars instead of fee / 100, so the numbers are bigger). Refit. How much does the 50% horizon move? What does that say about how much the human-time estimates matter?
# 2. Run the GDPval comparison with `llama3.2:3b` as the writer and `MODEL` as the grader. Is the win-or-tie rate lower?
# 3. Write a summary that games our DeepScholar metrics (high scores, but useless to a reader). Which metric would the paper's LLM-judge version catch it with?
#
# **Next:** a11 · Guest lectures and reasoning frontiers

# %% [markdown]
# ## Solutions

# %%
# Solution 1: bigger numbers per step (30 calls). We keep the same human-minute estimates.
def run_task_dollars(task, temperature=0.5):
    t = dict(task)
    first = t["steps"][0].replace(" divided by 100", "")
    c0 = re.search(r"msa_\d+", first).group(0)
    t["steps"] = [first] + t["steps"][1:]
    t["answer"] = task["answer"] - FACTS[c0]["monthly_fee"] // 100 + FACTS[c0]["monthly_fee"]
    return run_task(t, temperature)

with ThreadPoolExecutor(2) as ex:
    out2 = np.array(list(ex.map(run_task_dollars, jobs)), dtype=float)
print(f"fee/100: 50% horizon {h50:.1f} min | dollars: {horizon(fit_logistic(minutes, out2)):.1f} min")
# Notice: the same 'human minutes' can hide very different difficulty. METR's number is only as good as its
# human baselines, which is why they timed real people instead of guessing.

# %%
# Solution 2: a smaller writer (3 + 6 calls)
with ThreadPoolExecutor(2) as ex:
    small = dict(zip(TASK_CIDS, ex.map(lambda c: ask(TASK.format(cid=c, rules=json.dumps(RULE_TEXT, indent=1), doc=contract(c)),
                                                       model_name="llama3.2:3b", max_tokens=300), TASK_CIDS)))
v_small = {c: pairwise(c, small[c], EXPERT[c]) for c in TASK_CIDS}
print("3B writer:", v_small, "| 8B writer:", verdicts)
# Notice: with only 3 tasks the rate moves in steps of 33%. GDPval used 220 gold tasks and three graders per sample
# for a reason. Never decide between models on a handful of comparisons.

# %%
# Solution 3: a summary that games word-overlap metrics
GAMED = " ".join(f"{s.rstrip('.')} [{k}]." for k in sorted(GOLD) for s in split_sentences(SOURCES[k]))   # copy the sources word for word
print({k: round(v, 2) for k, v in scholar_metrics(GAMED).items()})
# Notice: copying the sources scores near-perfect here. The paper's organization metric (an LLM judge comparing
# against the human-written section, pairwise) would catch that it is not a synthesis, just a list of abstracts.
