# %% [markdown]
# # a06 · Open-ended evolution: let the model rewrite code, agents and experiments
# **Local AI Lab** · CS329A Lecture 7 · keep an archive, let an LLM propose changes, keep what a checker says is better
#
# Three systems from the lecture share one loop: **propose, evaluate automatically, keep the good ones in an archive, repeat.**
# You will build small versions of all three: a mini **AlphaEvolve** that evolves a bin-packing function, a mini **ADAS** meta-agent that invents agent designs, and an **AI Scientist**-style loop that goes from idea to experiment to write-up to review.
# All model-written code runs in a separate process with a timeout. About 70 model calls, 10-15 minutes on a T4.

# %%
# MODELS: llama3.2:3b
import json, os, random, re, subprocess, sys, tempfile, textwrap, time
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from lal.llm import ask, structured
from lal.checks import check

WORKER = "llama3.2:3b"     # the cheap model that runs inside agent designs (ADAS)
RUN_LONG = False           # True = more generations and a held-out test for ADAS
REPO = os.getcwd()

# %% [markdown]
# ## 1. The sandbox: never run model code in your own process
# Model-written code can loop forever, eat memory, or delete files. We write it to a temp folder and run it in a **separate Python process** with:
# - a wall-clock timeout,
# - a CPU-time and (on Linux) memory limit,
# - `python -I` (isolated mode: ignores environment variables and the user's site folder),
# - the temp folder as working directory.
#
# This is a speed bump, not a real sandbox. It does not block the network or reading files. For real use, run it in a container with no network. The AI Scientist paper reports why: in some runs their system edited its own code to raise its timeout, and relaunched itself until a human had to stop it (Lu et al. 2024).

# %%
def run_sandboxed(code, timeout=10, mem_mb=1024):
    """Run code in a fresh Python process. The program reports by printing one line: RESULT <json>."""
    # The child sets its own CPU and memory limits first. (Setting them from the parent with preexec_fn
    # is unsafe in a Jupyter kernel, which runs threads.)
    header = ("try:\n    import resource, sys as _s\n"
              f"    resource.setrlimit(resource.RLIMIT_CPU, ({timeout + 1}, {timeout + 1}))\n"
              f"    if _s.platform.startswith('linux'): resource.setrlimit(resource.RLIMIT_AS, ({mem_mb} * 2**20, {mem_mb} * 2**20))\n"
              "except Exception:\n    pass\n")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(header + code)
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout}s", "seconds": timeout}
        secs = round(time.time() - t0, 2)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not lines:
            return {"ok": False, "error": (p.stderr or p.stdout or "no RESULT line")[-500:], "seconds": secs}
        try:
            return {"ok": True, "result": json.loads(lines[-1][7:]), "seconds": secs}
        except Exception as e:
            return {"ok": False, "error": f"bad RESULT json: {e}", "seconds": secs}

def code_block(text):
    """Pull the first ```python block out of a reply (or the whole reply if there is none)."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip()

print(run_sandboxed("print('RESULT', 1+1)"))
print(run_sandboxed("while True: pass", timeout=2))

# %% [markdown]
# ## 2. Mini AlphaEvolve: evolve a bin-packing function
# AlphaEvolve (Novikov et al. 2025, Google DeepMind) keeps a **database of programs**. Each step it picks a parent, shows the LLM the parent plus a few good "inspiration" programs with their scores, asks for an improved version, runs the evaluator, and stores the child. Among its results: a way to multiply two 4x4 complex matrices with 48 scalar multiplications (Strassen's method, from 1969, needs 49), and a scheduling heuristic that recovers on average 0.7% of Google's fleet-wide compute.
#
# Our problem: **bin packing.** Put items of different sizes into as few bins of size 100 as possible. The program to evolve is `pack(items, capacity)`. The evaluator:
# 1. **checks validity** (every item used exactly once, no bin over capacity); an invalid packing scores 0,
# 2. scores quality as `lower bound / bins used`, where the lower bound is `ceil(total size / capacity)`. 1.0 would be perfect.
#
# It runs as an **evaluation cascade**, an idea from the paper: a tiny test first, and only programs that pass it get the full test.

# %%
SEED_PROGRAM = '''def pack(items, capacity):
    """Next fit: keep one open bin; start a new one when the item does not fit."""
    bins, current, room = [], [], capacity
    for x in items:
        if x > room:
            bins.append(current)
            current, room = [], capacity
        current.append(x)
        room -= x
    if current:
        bins.append(current)
    return bins
'''

BP_HARNESS = '''
import json, math, random
from collections import Counter

{code}

def run_stage(seed, n_inst, n_items, cap=100):
    rng = random.Random(seed)
    scores = []
    for _ in range(n_inst):
        items = [rng.randint(20, 70) for _ in range(n_items)]
        bins = pack(list(items), cap)
        flat = [x for b in bins for x in b]
        if Counter(flat) != Counter(items) or any(sum(b) > cap for b in bins) or any(len(b) == 0 for b in bins):
            return None
        scores.append(math.ceil(sum(items) / cap) / len(bins))
    return sum(scores) / len(scores)

try:
    quick = run_stage(seed=1, n_inst=2, n_items=10)          # stage 1: tiny and fast
    full = run_stage(seed=2, n_inst=20, n_items=60) if quick is not None else None   # stage 2: the real test
    print("RESULT " + json.dumps({{"score": full or 0.0, "valid": full is not None,
                                  "stage": 2 if quick is not None else 1}}))
except Exception as e:
    print("RESULT " + json.dumps({{"score": 0.0, "valid": False, "error": repr(e)[:200]}}))
'''

def evaluate_packer(code):
    r = run_sandboxed(BP_HARNESS.format(code=code), timeout=10)
    if not r["ok"]:
        return 0.0, r["error"][-200:]
    res = r["result"]
    return float(res.get("score", 0.0)), res.get("error") or ("ok" if res.get("valid") else f"invalid at stage {res.get('stage')}")

seed_score, note = evaluate_packer(SEED_PROGRAM)
print(f"seed program (next fit): score {seed_score:.3f} ({note})")

# %% [markdown]
# The mutation step. We ask for the whole function back. (AlphaEvolve asks for **diffs** in a SEARCH/REPLACE format and marks the parts that may change with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`. Whole functions are easier for an 8B model.)

# %%
def mutate(parent, inspirations):
    insp = "\n\n".join(f"# score {p['score']:.3f}\n{p['code']}" for p in inspirations)
    prompt = textwrap.dedent(f"""\
        You are improving a Python function for bin packing. Higher score is better (1.0 is a perfect packing).
        A packing that loses, duplicates or overfills items scores 0.

        Current program (score {parent['score']:.3f}):
        ```python
        {{parent}}
        ```

        Other programs from the archive, for inspiration:
        {{insp}}

        Write an improved version of `pack(items, capacity)`. It must return a list of bins, each a list of the
        item sizes in it, using every item exactly once. Use only the Python standard library. It must run in
        well under a second for 60 items. Think about which known heuristics pack tighter.
        Reply with one ```python code block containing the full function.""").replace("{parent}", parent["code"]).replace("{insp}", insp or "(none yet)")
    try:
        return code_block(ask(prompt, temperature=0.8, max_tokens=900))
    except Exception as e:
        return f"# model error: {e}"

archive = [{"code": SEED_PROGRAM, "score": seed_score, "gen": 0, "note": note}]
rng = random.Random(0)
GENERATIONS, CHILDREN = (10, 2) if RUN_LONG else (6, 2)           # 12 model calls by default
best_by_gen = [seed_score]
t0 = time.time()
for gen in range(1, GENERATIONS + 1):
    ranked = sorted(archive, key=lambda p: -p["score"])
    parents = [ranked[0] if rng.random() < 0.5 else rng.choice(ranked[:4]) for _ in range(CHILDREN)]
    with ThreadPoolExecutor(2) as ex:
        kids = list(ex.map(lambda par: mutate(par, [p for p in ranked[:3] if p is not par][:2]), parents))
    for code in kids:
        if "def pack(" not in code:
            archive.append({"code": code, "score": 0.0, "gen": gen, "note": "no pack() function"})
            continue
        s, note = evaluate_packer(code)
        archive.append({"code": code, "score": s, "gen": gen, "note": note})
    best_by_gen.append(max(p["score"] for p in archive))
    print(f"gen {gen}: children scored {[round(a['score'], 3) for a in archive[-CHILDREN:]]}  best so far {best_by_gen[-1]:.3f}")
print(f"[{time.time() - t0:.0f}s]")

best_prog = max(archive, key=lambda p: p["score"])
print(f"\nseed {seed_score:.3f} -> best {best_prog['score']:.3f} (found in generation {best_prog['gen']})\n")
print(best_prog["code"])

# %% [markdown]
# Things to notice:
# - **The archive only gets better or stays the same,** because we never throw away the best program. Children that crash or cheat get a 0 and are simply never picked as parents.
# - **The evaluator does the real work.** The LLM is just a source of guesses. AlphaEvolve's results came from problems where a program can score the answer. A typical win here is "first fit decreasing": sort items from big to small, put each one in the first bin where it fits.

# %% [markdown]
# ## 3. Mini ADAS: a meta-agent that invents agent designs
# ADAS (Hu, Lu & Clune 2024, *Automated Design of Agentic Systems*) writes agents as **code**: a `forward()` function that may call the model several times. A **meta agent** reads an archive of earlier designs with their scores and writes a new one. In the paper (GPT-4 as meta agent, GPT-3.5 running the agents), discovered agents beat hand-designed ones by 13.6 F1 points on DROP and 14.4% on MGSM, and kept an edge when moved to other domains and models.
#
# > The course page links this lecture item to arXiv 2505.22954, which is the follow-up paper, the **Darwin Gödel Machine** (Zhang et al. 2025). It applies the same archive idea to a coding agent that edits its own code: SWE-bench 20.0% -> 50.0%.
#
# Our version: agents are `forward(question, ask)` functions. `ask(prompt, temperature=0.0)` calls the small 3B model, at most **2 calls per question**. The tasks are seeded word problems with exact answers. Each design runs in the sandbox.

# %%
def word_problems(n, seed):
    rng, out = random.Random(seed), []
    for _ in range(n):
        a, b, c, d = rng.randint(20, 90), rng.randint(3, 9), rng.randint(4, 12), rng.randint(5, 30)
        name, thing = rng.choice(["Mia", "Ravi", "Ana", "Tom", "Lena"]), rng.choice(["stickers", "apples", "cards", "coins"])
        kind = rng.randint(0, 2)
        if kind == 0:
            q, ans = f"{name} has {a} {thing}. {name} buys {b} packs with {c} {thing} each, then gives away {d}. How many {thing} does {name} have now?", a + b * c - d
        elif kind == 1:
            q, ans = f"A box holds {c} {thing}. {name} fills {b} boxes and has {d} {thing} left over, then loses {a % 10} of the left-over ones. How many {thing} did {name} have in total at the end?", b * c + d - a % 10
        else:
            q, ans = f"{name} had {a} {thing}, sold {d} of them, and then doubled what was left. After that {name} received {b * c} more. How many {thing} now?", 2 * (a - d) + b * c
        out.append({"question": q, "answer": ans})
    return out

def extract_int(text):
    m = re.findall(r"Answer:\s*\**\s*(-?\d[\d,]*)", text or "", re.I) or re.findall(r"-?\d[\d,]*", text or "")
    try:
        return int(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None

AGENT_HARNESS = '''
import json, sys
sys.path.insert(0, {repo!r})
from lal.llm import ask as _ask
LIMIT, CALLS = 2, {{"n": 0}}

def ask(prompt, temperature=0.0):
    if CALLS["n"] >= LIMIT:
        raise RuntimeError("call budget exceeded")
    CALLS["n"] += 1
    return _ask(str(prompt), model_name={worker!r}, temperature=float(temperature), max_tokens=300)

{code}

outs = []
for q in {questions!r}:
    CALLS["n"] = 0
    try:
        outs.append(str(forward(q, ask)))
    except Exception as e:
        outs.append("ERROR: " + repr(e)[:200])
print("RESULT " + json.dumps(outs))
'''

def evaluate_agent(code, tasks):
    prog = AGENT_HARNESS.format(repo=REPO, worker=WORKER, code=code, questions=[t["question"] for t in tasks])
    r = run_sandboxed(prog, timeout=240, mem_mb=None)        # no memory cap: the model client needs room
    if not r["ok"]:
        return 0.0, "crashed: " + r["error"][-200:]
    outs = r["result"]
    right = sum(extract_int(o) == t["answer"] for o, t in zip(outs, tasks))
    errors = sum(o.startswith("ERROR") for o in outs)
    return right / len(tasks), f"{errors}/{len(tasks)} questions errored"

SEEDS = [
    {"name": "Direct", "idea": "Ask for the number only.",
     "code": "def forward(question, ask):\n    return ask(question + '\\nReply with only the final number.')\n"},
    {"name": "Chain of thought", "idea": "Reason step by step, then give the answer.",
     "code": "def forward(question, ask):\n    return ask(question + \"\\nThink step by step, then end with 'Answer: <number>'.\")\n"},
]
adas_tasks = word_problems(6, seed=7)
t0 = time.time()
agent_archive = []
for s in SEEDS:                                                  # 2 designs x 6 questions = 12 calls
    acc, note = evaluate_agent(s["code"], adas_tasks)
    agent_archive.append({**s, "score": acc, "note": note, "gen": 0})
    print(f"seed {s['name']:<18} accuracy {acc:.0%} ({note})")
print(f"[{time.time() - t0:.0f}s]")

# %% [markdown]
# The meta agent. ADAS adds two self-reflection rounds to check that each idea is new, and up to five repair rounds when a design crashes. We skip both to keep the call count small.

# %%
def propose_design():
    shown = "\n\n".join(f"Name: {a['name']}\nIdea: {a['idea']}\nAccuracy: {a['score']:.0%}\n```python\n{a['code']}```"
                        for a in sorted(agent_archive, key=lambda a: -a["score"]))
    prompt = textwrap.dedent("""\
        You design agents that solve short math word problems. An agent is a Python function
        `forward(question, ask)` that returns a string containing the final answer. `ask(prompt, temperature=0.0)`
        calls a small language model and returns its text reply. You may call `ask` AT MOST 2 times per question.
        Use only plain Python (re is allowed via `import re` inside the function).

        Archive of designs tried so far, best first:
        {shown}

        Propose a NEW design that is meaningfully different from the archive and likely to score higher
        (ideas: plan then compute, write the arithmetic as one expression and check it, verify and fix, ...).
        Reply in exactly this format:
        Name: <short name>
        Idea: <one sentence>
        ```python
        def forward(question, ask):
            ...
        ```""").replace("{shown}", shown)
    try:
        reply = ask(prompt, temperature=0.8, max_tokens=700)
    except Exception as e:
        reply = f"Name: error\nIdea: {e}\n"
    name = (re.search(r"Name:\s*(.+)", reply) or [None, "unnamed"])[1].strip()[:40]
    idea = (re.search(r"Idea:\s*(.+)", reply) or [None, ""])[1].strip()[:200]
    return {"name": name, "idea": idea, "code": code_block(reply)}

ITERS = 6 if RUN_LONG else 3                                     # 3 meta calls + up to 3 x 6 x 2 = 36 worker calls
t0 = time.time()
for it in range(1, ITERS + 1):
    d = propose_design()
    if "def forward(" not in d["code"]:
        acc, note = 0.0, "no forward() function"
    else:
        acc, note = evaluate_agent(d["code"], adas_tasks)
    agent_archive.append({**d, "score": acc, "note": note, "gen": it})
    print(f"iter {it}: {d['name']:<30} accuracy {acc:.0%} ({note})")
print(f"[{time.time() - t0:.0f}s]")

best_agent = max(agent_archive, key=lambda a: a["score"])
print(f"\nbest design: {best_agent['name']} ({best_agent['score']:.0%})\n{best_agent['code']}")

# %% [markdown]
# Be honest about what 6 questions can tell you: one question is about 17 points. A design that "wins" here may just be lucky. ADAS reports scores on a held-out test set; with `RUN_LONG = True` we do the same below.

# %%
if RUN_LONG:
    held = word_problems(12, seed=8)
    for a in [agent_archive[1], best_agent]:
        print(f"held-out: {a['name']:<30} {evaluate_agent(a['code'], held)[0]:.0%}")

# %% [markdown]
# ## 4. An "AI Scientist" loop in miniature
# The AI Scientist (Lu et al. 2024) chains the whole research process: **idea -> experiment code -> run -> write-up -> automated review**. The authors report under $15 per paper, and an automated reviewer (GPT-4o) with 0.65 balanced accuracy at accept/reject on 500 ICLR 2022 papers, against 0.66 for humans.
#
# Ours runs on bin packing, so it is cheap to test. Five calls at most. We add one check the model cannot fake: **do the numbers in the write-up actually appear in the results?**

# %%
class Idea(BaseModel):
    title: str
    hypothesis: str = Field(description="one testable sentence")
    plan: str = Field(description="what to compare and how to measure it")

class Review(BaseModel):
    soundness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=10)
    main_weakness: str

t0 = time.time()
try:
    idea = structured("Propose one small, testable research idea about bin-packing heuristics "
                      "(for example: how item-size distribution affects first-fit vs best-fit). "
                      "It must be testable in pure Python on random instances in under 10 seconds.", Idea, temperature=0.7)
except Exception as e:
    idea = Idea(title="First fit vs best fit", hypothesis="Best fit uses fewer bins than first fit on small items.",
                plan=f"Compare both on random instances (fallback idea; model error: {e})")
print("IDEA:", idea.title, "|", idea.hypothesis)

exp_prompt = (f"Write a self-contained Python script (standard library only) that tests this hypothesis:\n"
              f"{idea.hypothesis}\nPlan: {idea.plan}\n"
              "Use random.seed(0). Keep it under 5 seconds. At the very end print exactly one line:\n"
              "print('RESULT', json.dumps(results)) where results is a flat dict of names to numbers.\n"
              "Reply with one ```python code block.")
experiment = {"ok": False, "error": "not run"}
code = ""
for attempt in range(2):                                         # one repair try, like the paper's retry loop
    try:
        code = code_block(ask(exp_prompt if attempt == 0 else
                              exp_prompt + f"\n\nYour last script failed with:\n{experiment.get('error', '')[-400:]}\nFix it.",
                              temperature=0.2, max_tokens=1200))
    except Exception as e:
        code = f"raise SystemExit('model error: {e}')"
    experiment = run_sandboxed(code, timeout=20)
    print(f"experiment attempt {attempt + 1}: {'ran' if experiment['ok'] else 'failed: ' + experiment['error'][-150:]}")
    if experiment["ok"]:
        break
results = experiment.get("result", {}) if experiment["ok"] else {}
print("RESULTS:", json.dumps(results)[:400])

# %%
try:
    writeup = ask(f"Write a 120-word results note for this experiment.\nHypothesis: {idea.hypothesis}\n"
                  f"Results (JSON): {json.dumps(results)}\nReport the actual numbers, say whether the hypothesis "
                  "held, and name one limitation. Do not invent numbers that are not in the results.",
                  temperature=0.2, max_tokens=400)
except Exception as e:
    writeup = f"(write-up failed: {e})"
print(writeup)

def grounded_fraction(text, res):
    """Share of numbers in the write-up that match a result value (to 2 decimals) or appear in the results text."""
    vals = [v for v in res.values() if isinstance(v, (int, float))] if isinstance(res, dict) else []
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", text.replace(",", ""))]
    nums = [n for n in nums if n not in (0, 1, 2)]               # small counts like "two heuristics" are noise
    if not nums:
        return None
    raw = json.dumps(res)
    ok = sum(any(abs(n - v) < 0.005 or abs(n - round(v * 100, 1)) < 0.06 for v in vals) or str(n).rstrip("0").rstrip(".") in raw
             for n in nums)
    return ok / len(nums)

try:
    review = structured(f"You are a strict reviewer. Review this short research note.\nHypothesis: {idea.hypothesis}\n"
                        f"Plan: {idea.plan}\nRaw results: {json.dumps(results)}\nNote:\n{writeup}\n"
                        "Score soundness and clarity 1-5 and overall 1-10. Punish any claim not supported by the raw results.",
                        Review)
except Exception as e:
    review = None
    print("review failed:", e)
g = grounded_fraction(writeup, results)
print(f"\nREVIEW: {review.model_dump() if review else None}")
print(f"numbers in the write-up found in the results: {'n/a' if g is None else f'{g:.0%}'}   [{time.time() - t0:.0f}s]")

# %% [markdown]
# What this shows, and what it does not:
# - Each step is checkable except the idea itself. The experiment either ran or it did not; the numbers are either in the results or they are not.
# - The LLM reviewer is the weakest link. It is one model grading another, and it can be fooled by confident writing. That is why we added the grounding check, which is plain code.
# - The real AI Scientist uses much bigger models, a literature search to check novelty, and full LaTeX papers. The loop is the same.

# %%
check("sandbox stops an infinite loop", not run_sandboxed("while True: pass", timeout=2)["ok"])
check("AlphaEvolve loop beat the seed program", best_prog["score"] > seed_score)
ran_ok = [a for a in agent_archive if a["gen"] > 0 and a["note"].endswith("errored") and not a["note"].startswith(f"{len(adas_tasks)}/")]
check("ADAS: at least one new design ran on the tasks", len(ran_ok) > 0)
check("AI Scientist: the experiment ran and the reviewer returned a score", experiment["ok"] and review is not None and 1 <= review.overall <= 10)

# %% [markdown]
# ## Your turn
# 1. **Reward hacking.** Remove the validity check from the bin-packing evaluator (score only `lower bound / bins used`). Write the program a search would find first. What does this teach about evaluators?
# 2. **Overfitting the archive.** Take the best ADAS design and the chain-of-thought seed and score both on 12 fresh questions (a new seed). Does the winner still win?
# 3. **Diversity.** Our AlphaEvolve parent choice favors the best program. Change it to pick parents uniformly from the whole archive. Is progress slower, faster, or just noisier?
#
# **Next:** a07 · Self-improvement with search and deep research agents

# %% [markdown]
# ## Solutions

# %%
# Solution 1: a cheating "packer" under a weak evaluator. No model calls needed.
CHEAT = "def pack(items, capacity):\n    return [items]          # everything in one bin\n"
VALIDITY = "        if Counter(flat) != Counter(items) or any(sum(b) > cap for b in bins) or any(len(b) == 0 for b in bins):\n            return None\n"
assert VALIDITY in BP_HARNESS
WEAK = BP_HARNESS.replace(VALIDITY, "")
weak = run_sandboxed(WEAK.format(code=CHEAT))
print("weak evaluator score for the cheat:", weak.get("result"), "| real evaluator:", evaluate_packer(CHEAT))
# Notice: one overfilled bin gives "lower bound / 1 bin" = a score far above 1.0. An LLM search optimizes
# whatever you measure, cheats included. Every evaluator needs a validity check that cannot be skipped.

# %%
# Solution 2: held-out check of the ADAS winner (about 24 small-model calls).
RUN_HELDOUT = False
if RUN_HELDOUT:
    fresh = word_problems(12, seed=123)
    for a in [agent_archive[1], best_agent]:
        print(f"{a['name']:<30} search set {a['score']:.0%} | fresh set {evaluate_agent(a['code'], fresh)[0]:.0%}")
# Notice: with 6 search questions, the gap often shrinks or flips on fresh data. Use more search
# questions, or keep a held-out set and only trust gains that show up there.

# %%
# Solution 3: uniform parent choice (12 more model calls).
RUN_UNIFORM = False
if RUN_UNIFORM:
    arch2 = [{"code": SEED_PROGRAM, "score": seed_score, "gen": 0}]
    for gen in range(1, 7):
        for par in [rng.choice(arch2) for _ in range(2)]:
            c = mutate(par, sorted(arch2, key=lambda p: -p["score"])[:2])
            arch2.append({"code": c, "score": evaluate_packer(c)[0] if "def pack(" in c else 0.0, "gen": gen})
        print(f"gen {gen}: best {max(p['score'] for p in arch2):.3f}")
# Notice: uniform choice explores more but wastes calls on weak parents. AlphaEvolve balances the two
# with islands and MAP-Elites, so several different lines of programs keep improving side by side.
