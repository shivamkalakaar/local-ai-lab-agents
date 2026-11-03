# %% [markdown]
# # a04 · Multi-step reasoning and planning: search, decompose, parallelize
# **Local AI Lab** · CS329A Lecture 5 (Oct 6) · When one pass is not enough: explore a tree, split hard tasks only when needed, and run independent steps at once.
#
# We use a small number puzzle with an exact checker: combine four numbers with + - * / to hit a target. A plain greedy chain
# commits to its first idea. We then add **tree search with backtracking** (LATS), **decompose only on failure** (ADaPT),
# **wider vs deeper** budget choices (AB-MCTS), **parallel independent steps** (SPRINT), and finally turn our logged runs into
# **step-level training data** (SWiRL). No training here, just the inference-time ideas and the data they produce.

# %%
import ast, json, math, random, re, time
from fractions import Fraction
from functools import lru_cache
from typing import Literal
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from lal.llm import ask, structured
from lal.data import contract, sections, labels, playbook
from lal.checks import check

CALLS = [0]                                   # this notebook aims for about 70-80 model calls in total
def llm(prompt, **kw):
    CALLS[0] += 1
    try:
        return ask(prompt, **kw)
    except Exception as e:
        return f"ERROR: {e}"

def llm_struct(prompt, schema, **kw):
    CALLS[0] += 1
    try:
        return structured(prompt, schema, **kw)
    except Exception:                         # bad JSON after retries: treat as "no answer"
        return None

# %% [markdown]
# ## 1. The puzzle and its checker
# Rules: use each number exactly once; + - * /; every intermediate result must be a whole number that is not negative.
# The puzzles are generated with a fixed seed, and the generator only keeps targets that a brute-force solver can reach,
# so every puzzle has an answer. The solver is our ground truth; the model never sees it.

# %%
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b if a >= b else None,
       "*": lambda a, b: a * b, "/": lambda a, b: a // b if b and a % b == 0 else None}

@lru_cache(maxsize=None)
def solve(values: tuple, target: int):
    """Brute force: return an expression that reaches target, or None. values is a sorted tuple."""
    if len(values) == 1:
        return str(values[0]) if values[0] == target else None
    vals = list(values)
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i == j:
                continue
            rest = [v for k, v in enumerate(vals) if k not in (i, j)]
            for op, f in OPS.items():
                r = f(vals[i], vals[j])
                if r is None:
                    continue
                sub = solve(tuple(sorted(rest + [r])), target)
                if sub is not None:
                    return f"[{vals[i]} {op} {vals[j]} = {r}] then {sub}"
    return None

@lru_cache(maxsize=None)
def reachable(values: tuple) -> frozenset:
    if len(values) == 1:
        return frozenset(values)
    out = set()
    for i in range(len(values)):
        for j in range(len(values)):
            if i != j:
                rest = [v for k, v in enumerate(values) if k not in (i, j)]
                for f in OPS.values():
                    r = f(values[i], values[j])
                    if r is not None:
                        out |= reachable(tuple(sorted(rest + [r])))
    return frozenset(out)

def good_first_steps(nums, target):
    """How many legal first steps keep the target reachable. More = a friendlier puzzle."""
    n = 0
    for i in range(4):
        for j in range(4):
            if i != j:
                rest = [v for k, v in enumerate(nums) if k not in (i, j)]
                for f in OPS.values():
                    r = f(nums[i], nums[j])
                    n += r is not None and target in reachable(tuple(sorted(rest + [r])))
    return n

def make_puzzles(n, seed=7):
    """Random numbers 1-9; the target is one of the friendliest reachable targets (several paths work, but not all)."""
    rng, out = random.Random(seed), []
    while len(out) < n:
        nums = sorted(rng.randint(1, 9) for _ in range(4))
        cands = [t for t in reachable(tuple(nums)) if 12 <= t <= 40 and t != sum(nums)]   # not just "add everything"
        if not cands:
            continue
        ranked = sorted(cands, key=lambda t: -good_first_steps(nums, t))
        out.append({"numbers": nums, "target": rng.choice(ranked[:3])})
    return out

def check_expr(expr, numbers, target):
    """Exact check of a full expression. Returns (ok, value_or_None, message)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return False, None, "not a valid arithmetic expression"
    used = []
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            used.append(n.value); return Fraction(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add): return a + b
            if isinstance(n.op, ast.Sub): return a - b
            if isinstance(n.op, ast.Mult): return a * b
            if b == 0: raise ZeroDivisionError
            return a / b
        raise ValueError("only numbers and + - * / are allowed")
    try:
        val = ev(tree)
    except Exception as e:
        return False, None, f"could not evaluate: {e}"
    if sorted(used) != sorted(numbers):
        return False, val, f"must use exactly the numbers {sorted(numbers)}, but used {sorted(used)}"
    if val != target:
        return False, val, f"it equals {float(val):g}, not {target}"
    return True, val, "correct"

def clean_expr(text):
    """Model replies often add words or '= 24'. Keep the last line that looks like arithmetic."""
    text = (text or "").replace("×", "*").replace("÷", "/").replace("x", "*").replace("`", "")
    lines = [l.split("=")[0] for l in text.splitlines() if re.search(r"\d\s*[-+*/]\s*\(?\d", l)]
    return re.sub(r"[^\d+\-*/() ]", "", lines[-1]).strip() if lines else ""

PUZZLES = make_puzzles(6)
P = PUZZLES[:3]                               # 3 puzzles keep us inside the call budget
for p in PUZZLES:
    print(p, "| solver:", solve(tuple(p["numbers"]), p["target"]))
print(check_expr("(9 - 1) * 3 + 2", [1, 2, 3, 9], 26), check_expr("9 * 3", [1, 2, 3, 9], 27)[2])

# %% [markdown]
# ## 2. A step, a state, and a greedy chain
# A **state** is the list of numbers still left (each remembers the expression that made it). A **step** combines two of them.
# Four numbers need three steps. When two numbers are left, the last step is easy to check in code, like running a unit test,
# so every method below gets that check for free. The greedy chain asks for one step at a time and never looks back.

# %%
class Step(BaseModel):
    a: int
    op: Literal["+", "-", "*", "/"]
    b: int

class Proposals(BaseModel):
    steps: list[Step]

def start(p):
    return [(n, str(n)) for n in p["numbers"]]

def apply(state, s: Step):
    """Return the new state, or None if the step is not allowed."""
    vals = [v for v, _ in state]
    if s.a not in vals:
        return None
    i = vals.index(s.a)
    rest = state[:i] + state[i + 1:]
    j = next((k for k, (v, _) in enumerate(rest) if v == s.b), None)
    if j is None:
        return None
    ea, eb = state[i][1], rest[j][1]
    r = OPS[s.op](s.a, s.b)
    if r is None:
        return None
    return rest[:j] + rest[j + 1:] + [(r, f"({ea} {s.op} {eb})")]

def finish(state, target):
    """Two numbers left: try every last step in code. Returns the full expression or None."""
    (a, ea), (b, eb) = state
    for (x, ex), (y, ey) in (((a, ea), (b, eb)), ((b, eb), (a, ea))):
        for op, f in OPS.items():
            if f(x, y) == target:
                return f"{ex} {op} {ey}"
    return None

def values(state):
    return [v for v, _ in state]

def propose(state, target, k=1, memory="", temperature=0.0):
    prompt = (f"Number puzzle. Numbers left: {values(state)}. Target: {target}.\n"
              "Pick TWO of the numbers and one operation (+, -, *, /) to combine them into one new number. "
              "Results must be whole numbers and not negative. Choose steps that keep it possible to reach the target "
              f"with the numbers that remain.\n{memory}Give {k} different option(s), best first, as a, op, b.")
    out = llm_struct(prompt, Proposals, temperature=temperature, max_tokens=300)
    return (out.steps if out else [])[:k]

def greedy(p):
    state, traj = start(p), []
    for _ in range(2):                                   # 4 numbers -> 3 -> 2
        steps = [s for s in propose(state, p["target"], k=1) if apply(state, s)]
        if not steps:
            return {"ok": False, "traj": traj, "expr": None}
        nxt = apply(state, steps[0])
        traj.append({"state": values(state), "action": steps[0].model_dump(), "next": values(nxt)})
        state = nxt
    expr = finish(state, p["target"])
    return {"ok": expr is not None, "traj": traj, "expr": expr}

greedy_runs = [greedy(p) for p in P]
for p, r in zip(P, greedy_runs):
    print(p["numbers"], "->", p["target"], "| greedy:", "SOLVED " + r["expr"] if r["ok"] else "failed",
          "| path:", [t["next"] for t in r["traj"]])
print(CALLS[0], "calls so far")

# %% [markdown]
# ## 3. Tree search with backtracking (LATS, simplified)
# **LATS** (Zhou et al. 2023, *Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models*) runs
# Monte Carlo Tree Search over agent steps: **select** a promising node, **expand** it with n sampled actions (n = 5 in the paper),
# **evaluate** children with a value V = λ·(LM score) + (1-λ)·(self-consistency score), **backpropagate**, and store a
# **reflection** after a failed path. It reached 92.7% pass@1 on HumanEval with GPT-4; on Game of 24 it scored 0.44 against
# 0.20 for Tree-of-Thoughts.
#
# Our version is a small best-first search, which is the "select the highest value and backtrack" core of MCTS without the
# visit-count bonus:
# - **expand**: one call returns k=3 candidate steps; code drops illegal ones;
# - **evaluate**: one call rates the new states 1-10; self-consistency = how many proposals led to the same state;
# - **terminal**: with two numbers left, code checks the last step (the environment's reward);
# - **reflection**: dead ends are written into the next prompt (LATS asks the LM to write them; we let code write them, which is free).

# %%
class Scores(BaseModel):
    scores: list[int]

def rate(states, target):
    lists = "\n".join(f"{i+1}. {values(s)}" for i, s in enumerate(states))
    out = llm_struct(f"Target: {target}. For each list of numbers below, rate from 1 (impossible) to 10 (surely possible) "
                     f"how likely it is to reach the target using each number once with + - * /.\n{lists}\n"
                     f"Return exactly {len(states)} scores in order.", Scores, max_tokens=150)
    s = (out.scores if out else [])
    return [min(10, max(1, s[i])) / 10 if i < len(s) else 0.5 for i in range(len(states))]

def tree_search(p, k=3, max_expansions=4, lam=0.5):
    target, root = p["target"], start(p)
    frontier = [(1.0, 0, root, [])]                     # (value, tie-break id, state, trajectory)
    dead, expansions, uid, last = [], 0, 1, []
    while frontier and expansions < max_expansions:
        frontier.sort(key=lambda x: -x[0])
        v, _, state, traj = frontier.pop(0)             # SELECT the best open node
        memory = ("Dead ends already tried (do not repeat): " + "; ".join(dead[-4:]) + "\n") if dead else ""
        steps = propose(state, target, k=k, memory=memory, temperature=0.7)     # EXPAND
        expansions += 1
        kids = {}
        for s in steps:
            nxt = apply(state, s)
            if nxt is None:
                continue
            key = tuple(sorted(values(nxt)))
            kids.setdefault(key, [nxt, s, 0])[2] += 1     # count duplicates for self-consistency
        if not kids:
            continue
        open_kids = []
        for key, (nxt, s, count) in kids.items():
            t2 = traj + [{"state": values(state), "action": s.model_dump(), "next": values(nxt)}]
            last = t2                                     # keep the latest path, for the SWiRL section
            if len(nxt) == 2:                             # TERMINAL: code checks the last step
                expr = finish(nxt, target)
                if expr:
                    return {"ok": True, "traj": t2, "expr": expr, "expansions": expansions}
                dead.append(str(list(key)))               # REFLECTION (code-written)
            else:
                open_kids.append((key, nxt, s, count, t2))
        if open_kids:
            lm = rate([nk[1] for nk in open_kids], target)                       # EVALUATE
            for (key, nxt, s, count, t2), score in zip(open_kids, lm):
                value = lam * score + (1 - lam) * count / max(1, len(steps))
                frontier.append((value, uid, nxt, t2)); uid += 1
    return {"ok": False, "traj": last, "expr": None, "expansions": expansions, "dead": dead}

t0 = time.time()
tree_runs = [tree_search(p) for p in P]
for p, r in zip(P, tree_runs):
    print(p["numbers"], "->", p["target"], "| tree:", "SOLVED " + r["expr"] if r["ok"] else "failed",
          f"after {r['expansions']} expansions")
print(f"greedy solved {sum(r['ok'] for r in greedy_runs)}/{len(P)}, tree solved {sum(r['ok'] for r in tree_runs)}/{len(P)} "
      f"| {time.time()-t0:.0f}s, {CALLS[0]} calls so far")

# %% [markdown]
# The tree spends more calls, but on a dead end it can go back instead of failing. LATS's own limits apply here too: it costs
# more than a single chain, and it needs states you can **go back to**. A puzzle can; an agent that already sent an email cannot.

# %% [markdown]
# ## 4. ADaPT: decompose only when needed
# **ADaPT** (Prasad et al. 2024, *ADaPT: As-Needed Decomposition and Planning with Language Models*) first lets an executor try
# the whole task. Only if it fails does a planner split the task, and then ADaPT recurses on each piece, up to a max depth.
# With GPT-3.5 it beat ReAct, Plan-and-Execute and Reflexion: 71.6% vs 43.3% for ReAct on ALFWorld, 52% vs 19% on TextCraft.
#
# Here the executor writes a whole expression in one go. If the checker rejects it, the planner picks one step, and we recurse
# on the smaller puzzle. (ADaPT's executor judged its own success; our checker is stricter and more honest.)

# %%
def direct(state, target, temperature=0.0, extra=""):
    nums = values(state)
    reply = llm(f"Use each of the numbers {nums} exactly once, with + - * / and parentheses, to make {target}. "
                f"{extra}Reply with only the expression, for example (8 - 2) * (3 + 1).", temperature=temperature, max_tokens=80)
    return clean_expr(reply)

def substitute(expr, state):
    """Rewrite an expression over the state's values into one over the ORIGINAL numbers."""
    pools = {}
    for v, e in state:
        pools.setdefault(v, []).append(e)
    try:
        tree = ast.parse(expr, mode="eval")
        class Sub(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.parse(pools[node.value].pop(0), mode="eval").body
        return ast.unparse(Sub().visit(tree))
    except Exception:
        return None

def adapt(state, p, depth=0, d_max=2, log=None):
    log = [] if log is None else log
    expr = direct(state, p["target"])                                # 1. try directly
    full = substitute(expr, state) if expr else None
    ok = bool(full) and check_expr(full, p["numbers"], p["target"])[0]
    log.append({"depth": depth, "numbers": values(state), "try": expr, "ok": ok})
    if ok:
        return full, log
    if depth >= d_max or len(state) <= 2:
        return None, log
    steps = [s for s in propose(state, p["target"], k=1) if apply(state, s)]    # 2. only now: plan one sub-step
    if not steps:
        return None, log
    return adapt(apply(state, steps[0]), p, depth + 1, d_max, log)              # 3. recurse on the smaller task

adapt_runs = []
for p in P:
    expr, log = adapt(start(p), p)
    adapt_runs.append({"ok": expr is not None, "expr": expr, "log": log})
    print(p["numbers"], "->", p["target"], "|", " -> ".join(f"d{l['depth']}:{'OK' if l['ok'] else 'x'}" for l in log),
          "|", expr or "failed")
direct_only = sum(r["log"][0]["ok"] for r in adapt_runs)
print(f"direct only: {direct_only}/{len(P)} | ADaPT: {sum(r['ok'] for r in adapt_runs)}/{len(P)} | {CALLS[0]} calls so far")

# %% [markdown]
# The point is cost: easy puzzles stop after one call; only the hard ones pay for planning.

# %% [markdown]
# ## 5. Wider or deeper? (AB-MCTS)
# With a fixed budget of calls, you can sample **new** answers (wider) or **refine** the best one using feedback (deeper).
# **AB-MCTS** (Inoue et al. 2025, *Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search*)
# decides at every node, using Thompson sampling on the scores so far. With 128 calls of GPT-4o on CodeContests, pure
# refinement scored 30.1%, repeated sampling 37.9%, and AB-MCTS 40.6%. Pure "deeper" can be worse than pure "wider".
#
# We give each strategy the same budget of 3 calls per puzzle and score answers with the checker (1.0 if correct, partial
# credit if it uses the right numbers and lands close).

# %%
BUDGET = 3

def score(expr, p):
    ok, val, msg = check_expr(expr, p["numbers"], p["target"]) if expr else (False, None, "empty answer")
    if ok:
        return 1.0, msg
    if val is None or "must use" in msg:
        return 0.0, msg
    return 0.5 * max(0.0, 1 - abs(float(val) - p["target"]) / p["target"]), msg

def wider(p):
    with ThreadPoolExecutor(2) as ex:
        exprs = list(ex.map(lambda _: direct(start(p), p["target"], temperature=0.9), range(BUDGET)))
    return max(score(e, p)[0] for e in exprs)

def deeper(p):
    best = direct(start(p), p["target"], temperature=0.9)
    best_s, msg = score(best, p)
    for _ in range(BUDGET - 1):
        if best_s == 1.0:
            break
        e = direct(start(p), p["target"], temperature=0.7,
                   extra=f"Your last try was {best or '(empty)'}: {msg}. Fix it. ")
        s, m = score(e, p)
        if s >= best_s:
            best, best_s, msg = e, s, m
    return best_s

wd = {"wider": [wider(p) for p in P], "deeper": [deeper(p) for p in P]}
for name, ss in wd.items():
    print(f"{name:7s} solved {sum(s == 1.0 for s in ss)}/{len(P)} | mean best score {sum(ss)/len(ss):.2f}")
print(CALLS[0], "calls so far")

# %% [markdown]
# Three puzzles and three calls each is far too little to pick a winner; the paper needed up to 128 calls per problem. What
# you should take away is the knob itself: when feedback is informative ("it equals 30, not 24"), deeper helps; when the first
# idea is wrong at the root, only wider escapes it. AB-MCTS learns which one is paying off.

# %% [markdown]
# ## 6. SPRINT: run independent steps at the same time
# Reasoning models usually think in one long sequence. **SPRINT** (Biju et al. 2025, *SPRINT: Enabling Interleaved Planning and
# Parallelized Execution in Reasoning Models*) fine-tunes a model to alternate a **plan** (a list of independent subtasks) with
# **parallel execution** of those subtasks. It matched its baselines on math while using up to 39% fewer sequential tokens on
# long problems (and up to 45% on GPQA, 65% on Countdown).
#
# The same idea works at the agent level. A contract review plan has four independent rule checks, then one summary that needs
# all four. Independent checks can run together.

# %%
PB = playbook()
WORD = {"liability_cap": "Liability", "governing_law": "Governing", "auto_renewal": "Renewal", "termination_notice": "Renewal"}
PLAN = {"liability_cap": [], "governing_law": [], "auto_renewal": [], "termination_notice": [],
        "summary": ["liability_cap", "governing_law", "auto_renewal", "termination_notice"]}

def stages(dag):
    """Group steps into rounds: a step runs once everything it depends on is done."""
    done, out = set(), []
    while len(done) < len(dag):
        ready = [s for s, deps in dag.items() if s not in done and all(d in done for d in deps)]
        if not ready:
            raise ValueError("cycle in plan")
        out.append(ready); done |= set(ready)
    return out

def check_rule(cid, rule):
    sec = next((s for s in sections(contract(cid)) if WORD[rule] in s["title"]), None)
    v = llm(f"Playbook rule: {PB['rules'][rule]}\n\nClause: {sec['text'] if sec else contract(cid)}\n\n"
            "Does the clause follow the rule? Answer PASS or FAIL only.", max_tokens=5).strip().upper()
    return rule, "FAIL" if "FAIL" in v else "PASS"

print("plan rounds:", stages(PLAN))
cid = "msa_15"
t0 = time.time(); seq = [check_rule(cid, r) for r in stages(PLAN)[0]]; t_seq = time.time() - t0
t0 = time.time()
with ThreadPoolExecutor(2) as ex:
    par = list(ex.map(lambda r: check_rule(cid, r), stages(PLAN)[0]))
t_par = time.time() - t0
summary = "Cannot sign as written: " + ", ".join(r for r, v in par if v == "FAIL") if any(v == "FAIL" for _, v in par) else "OK to sign"
print(f"sequential {t_seq:.1f}s | parallel(2) {t_par:.1f}s | {summary}")
print("truth issues:", [i["rule"] for i in labels()[cid]["issues"]], "|", CALLS[0], "calls so far")

# %% [markdown]
# With 2 parallel slots, 4 checks take about 2 rounds instead of 4. The planner's job is to find the **independent** steps;
# here the plan was written by hand, SPRINT trains the model to write it.

# %% [markdown]
# ## 7. SWiRL: turn logged runs into step-level training data
# **SWiRL** (Goldie et al. 2025, *Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use*) generates multi-step
# trajectories, cuts each one with k actions into k **sub-trajectories** (context so far → next action), filters them, and trains
# with RL where a model judges each step. Two findings worth knowing: with Gemma 2 27B it beat baselines by 21.5% (GSM8K) and
# 12.3% (HotPotQA) relative accuracy; and **process-only filtering** (keep steps a judge calls reasonable, even when the final
# answer was wrong) worked better for RL than keeping only trajectories with correct answers.
#
# We already logged trajectories in sections 2 and 3. Our "process judge" is the solver: a step is good if the target is
# still reachable after it. (SWiRL used Gemini 1.5 Pro as the judge; a code judge is cheaper when you have one.)

# %%
trajectories = []
for p, g, t in zip(P, greedy_runs, tree_runs):
    for method, r in (("greedy", g), ("tree", t)):
        if r["traj"]:
            trajectories.append({"puzzle": p, "method": method, "steps": r["traj"], "outcome_ok": r["ok"]})

examples = []
for tr in trajectories:
    p = tr["puzzle"]
    for i, st in enumerate(tr["steps"]):                       # one example per action
        context = f"Numbers: {p['numbers']}. Target: {p['target']}. Steps so far: " + \
                  "; ".join(f"{s['action']['a']} {s['action']['op']} {s['action']['b']}" for s in tr["steps"][:i])
        step_ok = p["target"] in reachable(tuple(sorted(st["next"])))     # process label from the solver
        examples.append({"context": context, "action": st["action"], "process_ok": step_ok, "outcome_ok": tr["outcome_ok"]})

filters = {"no filter": lambda e: True, "outcome only": lambda e: e["outcome_ok"],
           "process only": lambda e: e["process_ok"], "process + outcome": lambda e: e["process_ok"] and e["outcome_ok"]}
for name, f in filters.items():
    kept = [e for e in examples if f(e)]
    print(f"{name:18s} keeps {len(kept):2d} of {len(examples)} step examples")
good_steps_failed_runs = sum(e["process_ok"] and not e["outcome_ok"] for e in examples)
print("good steps that live inside FAILED runs:", good_steps_failed_runs, "(an outcome filter throws these away)")
if examples:
    print("example:", json.dumps(examples[0]))

# %% [markdown]
# That last number is SWiRL's point in miniature: a failed run can still contain good steps, and step-level labels let you keep them.

# %%
check("puzzle checker accepts a right answer and rejects a wrong one",
      check_expr("(9 - 1) * 3 + 2", [1, 2, 3, 9], 26)[0] and not check_expr("9 * 3", [1, 2, 3, 9], 27)[0])
check("tree search is not worse than greedy (allowing one puzzle of noise)",
      sum(r["ok"] for r in tree_runs) >= sum(r["ok"] for r in greedy_runs) - 1)
check("some method solves at least one puzzle",
      any(r["ok"] for r in greedy_runs + tree_runs + adapt_runs) or any(s == 1.0 for ss in wd.values() for s in ss))
check("ADaPT solves at least one puzzle", any(r["ok"] for r in adapt_runs))
check("SPRINT: parallel run mostly gives the same verdicts as sequential", len(set(par) & set(seq)) >= 3)
check("SWiRL: built step-level examples from logged runs", len(examples) >= 3)

# %% [markdown]
# ## Your turn
# 1. **Adaptive branching.** Write `adaptive(p)`: at each call, Thompson-sample between GEN (a new answer) and REFINE (fix the best
#    answer), using Beta posteriors built from the scores each choice has produced. Same budget of 3. Compare with wider and deeper.
# 2. **What does the filter keep?** Print the step examples that the process filter keeps but the outcome filter drops. Do they look
#    like good steps?
# 3. **Plan rounds.** A bigger review plan: extract parties and fees (no deps), check each of the 4 rules (each needs the
#    extraction), compute the dollar cap (needs fees and the liability check), write the summary (needs everything). Use
#    `stages()` to count rounds, and compute the best possible speedup = steps / rounds with unlimited parallel slots.
#
# **Next:** a05 · Train-time scaling and RL

# %% [markdown]
# ## Solutions

# %%
# Solution 1: a tiny AB-MCTS-style choice between wider (GEN) and deeper (REFINE) at each step.
def adaptive(p, seed=0):
    rng = random.Random(seed)
    stats = {"GEN": [], "REFINE": []}
    best, best_s, msg = None, -1.0, ""
    for call in range(BUDGET):
        if best is None:
            arm = "GEN"
        else:
            draws = {a: rng.betavariate(1 + sum(s), 1 + len(s) - sum(s)) for a, s in stats.items()}
            arm = max(draws, key=draws.get)
        extra = f"Your last try was {best}: {msg}. Fix it. " if arm == "REFINE" else ""
        e = direct(start(p), p["target"], temperature=0.9 if arm == "GEN" else 0.7, extra=extra)
        s, m = score(e, p)
        stats[arm].append(s)
        if s > best_s:
            best, best_s, msg = e, s, m
        if best_s == 1.0:
            break
    return best_s, {a: len(s) for a, s in stats.items()}

for p in P:
    s, used = adaptive(p)
    print(p["numbers"], "->", p["target"], f"| adaptive best score {s:.2f} | calls per arm {used}")
# Notice: the first call is always GEN (there is nothing to refine). With a budget of 3 the posterior barely moves;
# AB-MCTS shines at budgets of 32-128+ where it can learn which arm pays off on each problem.

# %%
# Solution 2: steps the process filter keeps but the outcome filter drops.
for e in examples:
    if e["process_ok"] and not e["outcome_ok"]:
        print(e["context"], "->", e["action"])
# Notice: these are legal steps after which the target is still reachable. The run failed LATER. Throwing them away wastes signal.

# %%
# Solution 3: rounds for a bigger plan.
BIG = {"extract": [], "liability_cap": ["extract"], "governing_law": ["extract"], "auto_renewal": ["extract"],
       "termination_notice": ["extract"], "dollar_cap": ["extract", "liability_cap"],
       "summary": ["liability_cap", "governing_law", "auto_renewal", "termination_notice", "dollar_cap"]}
rounds = stages(BIG)
print(rounds)
print(f"{len(BIG)} steps in {len(rounds)} rounds -> best speedup {len(BIG)/len(rounds):.2f}x")
# Notice: the longest chain of dependencies (extract -> liability -> dollar_cap -> summary) sets the floor on time,
# no matter how many parallel slots you have. This "critical path" is what SPRINT's sequential-token count measures.
print("total model calls in this notebook:", CALLS[0])
