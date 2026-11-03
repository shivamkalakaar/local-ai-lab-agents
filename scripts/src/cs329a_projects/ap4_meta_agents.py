# %% [markdown]
# # ap4 · Meta-agents, rebuilt small: an agent that designs agents, and whether it pays off
# **Local AI Lab** · CS329A past project (Winter 2025) · sample → evaluate → archive → repeat, then do the accounting
#
# A meta-agent writes new agent designs, tests each on a small training set, keeps them in an archive, and uses the
# archive to write the next design. It sounds like free progress. This project asked three hard questions: does the
# meta-agent really learn from its archive, are the designs different enough to combine, and does the design cost
# ever pay for itself? You will build the loop and answer all three at small scale.

# %% [markdown]
# ## The original project
# Listed on the course page as **Towards Effective Meta-Agents: Designing Agents that Design Agents**. The linked paper
# is titled **Inefficiencies of Meta Agents for Agent Design**, by Batu El, Mert Yuksekgonul and James Zou (Stanford);
# the poster is titled *Exploring Two Open Questions in Meta-Agent Design*. Stanford CS329A, Winter 2025.
# [Paper](https://cs329a.stanford.edu/winter2025projects/batu.pdf) · [Poster](https://cs329a.stanford.edu/winter2025projects/batuposter.pdf)
#
# **Problem.** Meta-agents such as ADAS (Hu et al. 2024, *Automated Design of Agentic Systems*) follow a
# sample–evaluate–iterate loop. Is that loop efficient?
#
# **Method.** An agent is code that takes a question and calls a model. The archive starts with 7 hand-written agents:
# Chain-of-Thought, Majority Voting, Refinement from Feedback, LLM Debate, Quality-Diversity, Routing, and Step-back.
# Each round, the meta-agent (gpt-4o) writes a new agent conditioned on a chosen part of the archive; agents run on
# gpt-3.5. Three ways to choose that context:
# - **Cumulative:** show all previous agents (what ADAS does).
# - **Parallel:** show only the 7 starting agents, so every sample is independent.
# - **Evolutionary:** show the 7 best agents so far (the "parents").
#
# **Setup.** MGSM, DROP, MMLU (128 train / 200 test each) and GPQA (32 train / 160 test), 30 iterations, 3 runs.
#
# **Key results (as reported).**
# - Best agent's train score, averaged over datasets: cumulative 55.0, parallel 60.0, evolutionary 60.3. "Cumulative
#   context curation does not outperform parallel", that is, showing all past designs was worse than showing none.
#   Evolutionary gave up to +10 points over cumulative on MGSM (41.4 → 56.5 train).
# - Test score of the best-on-train agent, average: initial library 49.0, cumulative 52.2, parallel 55.9, evolutionary 55.2.
# - Diversity: designed agents get many of the same questions right. Evolutionary agents are the most similar;
#   parallel gives the most coverage (questions solved by at least one agent): 95.9% on average vs. 94.2 cumulative, 94.4 evolutionary.
# - Cost: designed agents cost more per query than the starting agents (cumulative > evolutionary > parallel > initial).
#   Counting the design cost, the designed agent is cheaper *per correct answer* only after about **15,000 queries**,
#   and only on MMLU and DROP with parallel context. On the other datasets it never breaks even.
#
# **Limitations they state.** Only this one class of meta-agent; accuracy/F1 only; cost analysis fits tasks with
# strong verifiers; cosine similarity favors high-scoring agents; only 3 runs because each run is expensive.
#
# **Our mini version.** Designs are *data* (a system prompt plus one of four fixed procedures), not free-form code, so
# nothing the model writes is ever executed. `llama3.1:8b` plays the meta-agent and `llama3.2:3b` runs the agents,
# mirroring their strong-designer / cheap-runner split. We use evolutionary context, measure diversity and coverage,
# and compute the break-even point with the paper's formula: total cost = C0 + n·Cj.

# %%
# MODELS: llama3.2:3b
import json, math, random, re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from pydantic import BaseModel, Field
from lal.llm import ask, structured
from lal.checks import check

RUN_LONG = False                     # True: also run the "parallel" context strategy for comparison (~40 more calls)
DESIGNER, RUNNER = MODEL, "llama3.2:3b"          # MODEL is llama3.1:8b on a GPU runtime
T = 3                                # meta-agent iterations (the paper used 30)

# %% [markdown]
# ## 1. A small, mixed task set with exact checkers
# Six kinds of questions where small models stumble in different ways: counting letters, reversing words, sorting,
# weekday arithmetic, word problems, and comparing decimals. Mixed kinds give designs a chance to specialize.

# %%
WORDS = ["strawberry", "banana", "committee", "bookkeeper", "mississippi", "parallel", "assessment", "balloon", "occurrence"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
KINDS = ["count", "reverse", "sort", "weekday", "math", "compare"]

def make_task(rng, kind):
    if kind == "count":
        w = rng.choice(WORDS); ch = rng.choice(sorted(set(w)))
        return dict(kind=kind, q=f"How many times does the letter '{ch}' appear in the word '{w}'?", a=str(w.count(ch)))
    if kind == "reverse":
        w = rng.choice(WORDS)
        return dict(kind=kind, q=f"Write the word '{w}' backwards (its letters in reverse order).", a=w[::-1])
    if kind == "sort":
        nums = rng.sample(range(1, 99), 6)
        return dict(kind=kind, q="Sort these numbers from smallest to largest: " + ", ".join(map(str, nums)), a=",".join(map(str, sorted(nums))))
    if kind == "weekday":
        s, k = rng.randrange(7), rng.randint(9, 60)
        return dict(kind=kind, q=f"Today is {DAYS[s]}. What day of the week will it be in {k} days?", a=DAYS[(s + k) % 7].lower())
    if kind == "math":
        a, b = rng.randint(6, 24), rng.randint(3, 12); c = rng.randint(5, a * b - 1)
        return dict(kind=kind, q=f"A crate holds {a} bottles. A shop gets {b} crates and sells {c} bottles. How many bottles are left?", a=str(a * b - c))
    x = rng.randint(1, 9); p, q = f"{x}.{rng.randint(10, 99)}", f"{x}.{rng.randint(1, 9)}"
    return dict(kind=kind, q=f"Which number is larger: {p} or {q}? Answer with the number.", a=max(p, q, key=float))

rng = random.Random(42)
TRAIN = [make_task(rng, k) for k in KINDS]
TEST = [make_task(rng, k) for k in KINDS[:4]]      # 4 test tasks keeps the notebook within ~80 model calls

def final_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "", re.I)
    if m:
        return m[-1].strip()
    lines = (text or "").strip().splitlines()
    return lines[-1] if lines else ""

def correct(task, text):
    got = final_answer(text or "")
    if task["kind"] == "sort":
        return re.findall(r"\d+", got) == task["a"].split(",")
    if task["kind"] in ("reverse", "weekday"):
        return re.sub(r"[^a-z]", "", got.lower()) == task["a"]
    nums = re.findall(r"-?\d+(?:\.\d+)?", got)
    return bool(nums) and abs(float(nums[0]) - float(task["a"])) < 1e-9

for t in TRAIN:
    print(f"{t['kind']:8s} {t['a']:>20s}  ← {t['q']}")
print(correct(TRAIN[0], f"so... ANSWER: {TRAIN[0]['a']}"), correct(TRAIN[2], "ANSWER: 1, 2"))

# %% [markdown]
# ## 2. What an "agent design" is here, and the interpreter that runs it
# A design is a name, a one-line idea, a system prompt, and a **procedure** from a fixed menu. The interpreter
# counts calls and characters, so we can do cost accounting later (tokens ≈ characters / 4).

# %%
class Design(BaseModel):
    name: str
    idea: str = Field(description="one or two sentences: why this design should work")
    system_prompt: str = Field(description="instructions for the agent, under 120 words")
    procedure: Literal["direct", "think_step_by_step", "solve_then_check", "plan_then_solve"]

FMT = "\nEnd with one line: ANSWER: <your final answer>"

def run_agent(d: Design, q: str):
    """Returns (text, calls, chars)."""
    sp = d.system_prompt[:1200]
    chars, calls = 0, 0
    def call(prompt):
        nonlocal chars, calls
        try:
            out = ask(prompt, system=sp, model_name=RUNNER, temperature=0, max_tokens=600)
        except Exception as e:
            out = ""
        chars += len(sp) + len(prompt) + len(out); calls += 1
        return out
    if d.procedure == "direct":
        out = call(q + FMT)
    elif d.procedure == "think_step_by_step":
        out = call(q + "\nThink step by step." + FMT)
    elif d.procedure == "solve_then_check":
        first = call(q + "\nThink step by step." + FMT)
        out = call(f"Question: {q}\nA first attempt:\n{first[-1500:]}\n\nCheck this attempt carefully, step by step. "
                   "If it is wrong, fix it." + FMT)
    else:  # plan_then_solve
        plan = call(f"Question: {q}\nWrite a short numbered plan for solving it. Do not solve it yet.")
        out = call(f"Question: {q}\nPlan:\n{plan[-1200:]}\nNow follow the plan." + FMT)
    return out, calls, chars

def evaluate(d: Design, tasks):
    with ThreadPoolExecutor(2) as ex:
        runs = list(ex.map(lambda t: run_agent(d, t["q"]), tasks))
    vec = [1.0 if correct(t, out) else 0.0 for t, (out, _, _) in zip(tasks, runs)]
    return {"vec": vec, "score": sum(vec) / len(vec), "calls": sum(r[1] for r in runs), "chars": sum(r[2] for r in runs),
            "fails": [(t["q"], final_answer(out)[:60], t["a"]) for t, (out, _, _), v in zip(tasks, runs, vec) if not v]}

# %% [markdown]
# ## 3. The starting library: three hand-written agents
# Like the paper's initial library (Chain-of-Thought, self-refine, and so on), but only three, to save calls.

# %%
LIBRARY = [
    Design(name="Direct", idea="Just answer.", system_prompt="You are a careful assistant.", procedure="direct"),
    Design(name="Chain-of-Thought", idea="Reason before answering (Wei et al. 2022).",
           system_prompt="You are a careful assistant. Reason step by step before you answer.", procedure="think_step_by_step"),
    Design(name="Self-Refine", idea="Answer, then check and fix the answer (Madaan et al. 2023).",
           system_prompt="You are a careful assistant who double-checks work.", procedure="solve_then_check"),
]
ARCHIVE = []
for d in LIBRARY:
    r = evaluate(d, TRAIN)
    ARCHIVE.append({"design": d, "origin": "initial", **r})
    print(f"{d.name:18s} train {r['score']:.0%}  calls {r['calls']}")

# %% [markdown]
# ## 4. The meta-agent loop: sample → evaluate → add to archive
# `select_context` is the paper's φ. **Evolutionary** shows the best 3 designs so far with their scores and a few of
# their mistakes. **Parallel** shows only the starting library. **Cumulative** shows everything.

# %%
def select_context(archive, phi):
    if phi == "parallel":
        return [a for a in archive if a["origin"] == "initial"]
    if phi == "cumulative":
        return list(archive)
    return sorted(archive, key=lambda a: (a["score"], -a["chars"]), reverse=True)[:3]      # evolutionary: top-k parents

def describe(a):
    d = a["design"]
    fails = "; ".join(f"Q: {q[:70]} | gave: {g} | correct: {c}" for q, g, c in a["fails"][:3]) or "none"
    return (f"- name: {d.name}\n  idea: {d.idea}\n  procedure: {d.procedure}\n  system_prompt: {d.system_prompt}\n"
            f"  train score: {a['score']:.0%}\n  mistakes: {fails}")

META = ("You design AI agents. An agent is a system prompt plus one procedure:\n"
        "- direct: one call, answer immediately\n- think_step_by_step: one call with step-by-step reasoning\n"
        "- solve_then_check: two calls, solve then verify and fix\n- plan_then_solve: two calls, write a plan then follow it\n\n"
        "The agents answer short puzzles: counting letters, reversing words, sorting numbers, weekday arithmetic, "
        "word problems, comparing decimals.\n\nHere are existing agents and how they did:\n{ctx}\n\n"
        "Design ONE new agent that will score higher. Learn from the mistakes. Give concrete, task-specific tactics "
        "in the system prompt (for example: spell the word letter by letter). Use a new name.")

def meta_search(archive, phi, T):
    archive = list(archive)
    for t in range(T):
        ctx = "\n".join(describe(a) for a in select_context(archive, phi))
        try:
            d = structured(META.format(ctx=ctx), Design, model_name=DESIGNER, temperature=0.8, max_tokens=700)
        except Exception as e:
            print("meta-agent output was invalid:", e); continue
        r = evaluate(d, TRAIN)
        archive.append({"design": d, "origin": f"{phi} #{t + 1}", "meta_chars": len(ctx) + len(META) + len(d.model_dump_json()), **r})
        print(f"[{phi} {t + 1}] {d.name[:30]:30s} {d.procedure:18s} train {r['score']:.0%}")
    return archive

ARCHIVE = meta_search(ARCHIVE, "evolutionary", T)

# %% [markdown]
# ## 5. Question 1: did search beat the starting library?
# Pick the best design on the *training* set, then run it once on the held-out *test* set, next to the best starting agent.

# %%
best_init = max([a for a in ARCHIVE if a["origin"] == "initial"], key=lambda a: a["score"])
designed = [a for a in ARCHIVE if a["origin"] != "initial"]
best_new = max(designed, key=lambda a: a["score"]) if designed else best_init
test_init = evaluate(best_init["design"], TEST)
test_new = evaluate(best_new["design"], TEST) if best_new is not best_init else test_init
print(f"best starting agent : {best_init['design'].name:30s} train {best_init['score']:.0%}  test {test_init['score']:.0%}")
print(f"best designed agent : {best_new['design'].name:30s} train {best_new['score']:.0%}  test {test_new['score']:.0%}")
print("\nwinning design:\n", best_new["design"].model_dump_json(indent=2))

# %% [markdown]
# ## 6. Question 2: are the agents different enough to combine?
# Each agent's row of right/wrong answers is its "embedding". Cosine similarity near 1 means two agents solve the same
# questions. **Coverage** is the share of questions that at least one agent solves: the ceiling for a perfect router.

# %%
def cosine(u, v):
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    return sum(x * y for x, y in zip(u, v)) / (nu * nv) if nu and nv else 0.0

vecs = [a["vec"] for a in ARCHIVE if any(a["vec"])]
pairs = [cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
coverage = sum(any(a["vec"][i] for a in ARCHIVE) for i in range(len(TRAIN))) / len(TRAIN)
print("agent  " + "  ".join(t["kind"][:5] for t in TRAIN))
for a in ARCHIVE:
    print(f"{a['design'].name[:22]:22s} " + "  ".join("  ✓  " if x else "  ·  " for x in a["vec"]))
print(f"mean pairwise similarity {sum(pairs) / max(len(pairs), 1):.2f}   coverage {coverage:.0%}   best single agent {max(a['score'] for a in ARCHIVE):.0%}")

# %% [markdown]
# ## 7. Question 3: does the design cost ever pay off?
# Cost in tokens (≈ characters / 4). **C0** = everything spent to find the new agent: meta-agent prompts plus
# evaluating every designed agent on the training set. **Cj** = cost per query when deployed. After n queries:
# cost per correct answer = (C0 + n·Cj) / (n·accuracy). The starting agent has C0 = 0. The break-even n solves
# C0/(n·p_new) + C_new/p_new = C_init/p_init.

# %%
tok = lambda chars: chars / 4
C0 = sum(tok(a["chars"] + a.get("meta_chars", 0)) for a in designed)
c_init, c_new = tok(test_init["chars"]) / len(TEST), tok(test_new["chars"]) / len(TEST)
p_init, p_new = max(test_init["score"], 1e-9), max(test_new["score"], 1e-9)
gap = c_init / p_init - c_new / p_new
breakeven = C0 / (p_new * gap) if gap > 0 and best_new is not best_init else float("inf")
print(f"design cost C0 ≈ {C0:,.0f} tokens")
print(f"per query: starting agent {c_init:,.0f} tokens at {p_init:.0%}; designed agent {c_new:,.0f} tokens at {p_new:.0%}")
print("break-even after", "never (the designed agent is not cheaper per correct answer)" if math.isinf(breakeven) else f"{breakeven:,.0f} queries")

# %% [markdown]
# **What to notice.** On a handful of tasks, a lucky design can look great on train and slip on test: that's the
# paper's warning about noisy meta-evaluation, and why they averaged 3 runs. Two-call designs often win on accuracy but
# double the per-query cost, so the break-even point can be "never". Before you build a meta-agent, do this arithmetic.

# %%
check("archive holds the 3 starting agents plus designed ones", len(ARCHIVE) >= 3 + 1)
check("the best designed agent is no worse than the best starting agent minus one task", best_new["score"] >= best_init["score"] - 1 / len(TRAIN))
check("the best starting agent solves at least 2 of 6 training tasks", best_init["score"] >= 2 / 6)

# %% [markdown]
# ## Your turn
# 1. **Parallel vs. evolutionary.** Set `RUN_LONG = True` and run `meta_search` with `phi="parallel"`. The paper found
#    parallel about as good as evolutionary and more diverse. What do you see? (Three iterations is very noisy.)
# 2. **A router from the archive.** Each task has a kind. For each kind, pick the agent that did best on that kind in
#    training. What is this router's training score, compared with the best single agent and with coverage? (No new calls.)
# 3. **Your own prices.** Redo the break-even with real numbers: a designer token costs 10× a runner token. Does that change the answer?

# %% [markdown]
# ## Solutions

# %%
# 1. Parallel context (RUN_LONG only, ~3 meta calls + up to 36 agent calls).
if RUN_LONG:
    par = meta_search([a for a in ARCHIVE if a["origin"] == "initial"], "parallel", T)
    par_new = [a for a in par if a["origin"] != "initial"]
    pv = [a["vec"] for a in par_new if any(a["vec"])]
    pp = [cosine(pv[i], pv[j]) for i in range(len(pv)) for j in range(i + 1, len(pv))]
    print("parallel: best train", max(a["score"] for a in par_new), " mean similarity", round(sum(pp) / max(len(pp), 1), 2))
else:
    print("Set RUN_LONG = True to compare with parallel context.")

# %%
# 2. A per-kind router. Notice: on training data the router can match coverage, but with one task per kind it is
#    "choosing on the test set". A real router needs its own held-out data, which is why agent selection is hard.
router = {}                                          # kind -> index of the chosen agent in ARCHIVE
for i, t in enumerate(TRAIN):
    router[t["kind"]] = max(range(len(ARCHIVE)), key=lambda j: (ARCHIVE[j]["vec"][i], -ARCHIVE[j]["chars"]))
router_score = sum(ARCHIVE[router[t["kind"]]]["vec"][i] for i, t in enumerate(TRAIN)) / len(TRAIN)
print({k: ARCHIVE[j]["design"].name for k, j in router.items()})
print(f"router {router_score:.0%} · best single {max(a['score'] for a in ARCHIVE):.0%} · coverage {coverage:.0%}")

# %%
# 3. Designer tokens cost 10x. Only the meta-agent part of C0 gets more expensive.
meta_part = sum(tok(a.get("meta_chars", 0)) for a in designed)
C0_priced = C0 + 9 * meta_part
be = C0_priced / (p_new * gap) if gap > 0 and best_new is not best_init else float("inf")
print("break-even with 10x designer price:", "never" if math.isinf(be) else f"{be:,.0f} queries")
