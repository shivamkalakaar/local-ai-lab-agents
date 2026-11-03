# %% [markdown]
# # ap2 · PRIME, rebuilt small: plan, pick a strategy, execute, reflect, re-plan
# **Local AI Lab** · CS329A past project (Winter 2025) · a tree search over subgoals *and* reasoning strategies
#
# Most agent tricks (reflection, debate, retries) get applied the same way to every problem. PRIME asks a better
# question: for *this* subgoal, which trick should I use? You will build a small version on the Game of 24 and
# compare it, at the same call budget, with a single shot and with plain retry-with-feedback.

# %% [markdown]
# ## The original project
# **PRIME: Planning with Reflective, Iterative, Multi-agentic Exploration** by Chelsea Zou, Samuel Liu and Jui Khankari
# (the poster lists Chelsea Zou, Jui Khankari and Miriam Cheng). Stanford CS329A, Winter 2025.
# [Paper](https://cs329a.stanford.edu/winter2025projects/prime.pdf) · [Poster](https://cs329a.stanford.edu/winter2025projects/primeposter.pdf)
#
# **Problem.** Top-down planning with LLMs is unsolved. Reflection, debate and voting are used "ad hoc", with no
# principled way to pick the right one for a given step.
#
# **Method.** Treat each reasoning technique as an *option* (a reusable sub-policy, from reinforcement learning) and
# search with Monte Carlo Tree Search (MCTS). A planner splits the problem into N subgoals (tree children). At each
# node it chooses **Drill down** (split further), **Solve** (call the best-suited self-improvement tool: self-reflection,
# self-refinement or debate) or **Backtrack** (drop a useless subgoal). Results flow back up the tree with a running
# mean, V(s) ← (V·N + r)/(N+1), and the next node is chosen with UCT = w/n + C·√(ln N / n). Leaves are judged by
# ground truth when it exists, otherwise by V(s) = λ·LM(s) + (1−λ)·SC(s), a mix of a model's score and self-consistency.
# It builds on LATS (Zhou et al. 2024, *Language Agent Tree Search*).
#
# **Setup.** GPT-4o-mini inside PRIME vs. LATS (4o-mini), plain 4o-mini, and one-shot GPT-o1, on 60 sampled
# questions each from PlanBench, WebShop and Game of 24. Plus a "reverse ablation": upgrade only the planner, or only
# the executor, to a stronger model.
#
# **Key results (as reported, Table 1, success rate).**
#
# | Method | PlanBench | WebShop | Game of 24 |
# |---|---|---|---|
# | PRIME (4o-mini) | 36.2% | 40% | 75% |
# | LATS (4o-mini) | 2.2% | 38% | 44% |
# | GPT-4o-mini | 0% | 0% | 0% |
# | GPT-o1 | 100% | 100% | 100% |
# | PRIME, upgraded planner | 95% | 94% | 100% |
# | PRIME, upgraded executor | 96% | 92% | 100% |
#
# The text gives some different numbers for the same comparisons: PRIME is "a 2.2% improvement over LATS" on
# PlanBench, and o1 scores 37.3% on PlanBench and 80% on WebShop. We report both and can't tell which is right.
# The authors estimate PRIME costs about 4× less than o1, despite more API calls.
#
# **Limitations they state.** PRIME wanders when the value of an action is unclear (obfuscated PlanBench);
# on WebShop, reflections were generic and the agent got stuck in local minima, so subgoals helped little.
#
# **Our mini version.** Game of 24 only, because a checker can grade it exactly. The planner proposes first steps
# with a confidence (the LM value). Code rejects impossible steps (backtrack). A small bandit-style MCTS picks a
# (subgoal, tool) pair with UCT, where the tools are *solve*, *reflect* and *drill down*. The checker is the reward.
# Tool statistics carry over from puzzle to puzzle, a tiny version of "learning which option works".

# %%
# MODELS: llama3.2:3b
import ast, itertools, math, random, re
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import List
from pydantic import BaseModel, Field
from lal.llm import ask, chat, structured
from lal.checks import check

# %% [markdown]
# ## 1. The task and an exact checker
# Use each of four numbers once with + − × ÷ to make 24. The checker parses the expression with Python's `ast`
# module (no `eval`), checks the numbers used, and computes with exact fractions.

# %%
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}

def evaluate(node, used):
    if isinstance(node, ast.Expression):
        return evaluate(node.body, used)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        used.append(node.value); return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left, used), evaluate(node.right, used))
    raise ValueError("only whole numbers, + - * / and brackets are allowed")

def check24(expr: str, nums) -> tuple:
    """(ok, message). The message is what a reflecting agent gets to see."""
    try:
        used = []
        val = evaluate(ast.parse(expr, mode="eval"), used)
    except ZeroDivisionError:
        return False, "it divides by zero"
    except Exception as e:
        return False, f"it is not a valid expression ({e})"
    if sorted(used) != sorted(nums):
        return False, f"it uses the numbers {sorted(used)} but must use exactly {sorted(nums)}"
    if val != 24:
        return False, f"it equals {val}, not 24"
    return True, "correct"

def extract_expr(text: str) -> str:
    text = (text or "").replace("×", "*").replace("÷", "/").replace("−", "-")
    m = re.findall(r"ANSWER:\s*(.+)", text, re.I)
    cand = m[-1] if m else (text.strip().splitlines() or [""])[-1]
    cand = re.sub(r"(?<=[\d\)])\s*[xX]\s*(?=[\d\(])", "*", cand)
    parts = [p for p in cand.split("=") if re.search(r"\d", p)]
    if len(parts) > 1 and re.fullmatch(r"\s*24\s*\.?\s*", parts[-1]):
        parts = parts[:-1]
    cand = max(parts, key=lambda p: len(re.findall(r"\d+", p))) if parts else ""
    return re.sub(r"[^\d\s\+\-\*/\(\)]", "", cand).strip()

def solve24(nums):
    """Brute force: returns one solution or None. Used only to pick solvable puzzles, never by the agents."""
    items = [(Fraction(n), str(n)) for n in nums]
    def rec(items):
        if len(items) == 1:
            return items[0][1] if items[0][0] == 24 else None
        for i, j in itertools.permutations(range(len(items)), 2):
            (a, ea), (b, eb) = items[i], items[j]
            rest = [items[k] for k in range(len(items)) if k not in (i, j)]
            for v, e in [(a + b, f"({ea}+{eb})"), (a - b, f"({ea}-{eb})"), (a * b, f"({ea}*{eb})")] + \
                        ([(a / b, f"({ea}/{eb})")] if b != 0 else []):
                r = rec(rest + [(v, e)])
                if r:
                    return r
        return None
    return rec(items)

rng = random.Random(329)
PUZZLES = []
while len(PUZZLES) < 6:
    nums = sorted(rng.randint(1, 13) for _ in range(4))
    if nums not in PUZZLES and solve24(nums):
        PUZZLES.append(nums)
print(PUZZLES)
print(check24("(8-2)*(3+1)", [1, 2, 3, 8]), check24("8*3", [1, 2, 3, 8]), extract_expr("So: (8 − 2) × (3 + 1) = 24\nANSWER: (8-2)*(3+1) = 24"))

# %% [markdown]
# ## 2. Baselines: one shot, and retry with feedback (same budget as PRIME)
# The retry baseline is a plain reflection loop: try, get the checker's message, try again. If PRIME only wins
# because it makes more calls, this baseline will catch it.

# %%
BUDGET = 5            # model calls per puzzle for the two search methods
ASK24 = ("Use each of the numbers {nums} exactly once, with + - * / and brackets, to make exactly 24.\n"
         "Think briefly, then end with one line: ANSWER: <expression>")

def single_shot(nums):
    out = ask(ASK24.format(nums=nums), max_tokens=600)
    ok, _ = check24(extract_expr(out), nums)
    return {"ok": ok, "calls": 1}

def retry_with_feedback(nums, budget=BUDGET):
    msgs = [{"role": "user", "content": ASK24.format(nums=nums)}]
    for call in range(1, budget + 1):
        out = chat(msgs, temperature=0.7, max_tokens=600).content or ""
        expr = extract_expr(out)
        ok, why = check24(expr, nums)
        if ok:
            return {"ok": True, "calls": call}
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": f"Your expression '{expr}' is wrong: {why}. Reflect on the mistake and try a different approach."}]
    return {"ok": False, "calls": budget}

# %% [markdown]
# ## 3. PRIME-lite
# - **Plan** (1 call): propose up to 4 first steps like `8 - 2 = 6`, each with a confidence 1–10.
# - **Backtrack** for free: code drops steps that use numbers we don't have or have wrong arithmetic.
# - **Options** for each surviving subgoal: `solve` (1 call), `reflect` (solve, and if the checker says no, fix once: up to 2 calls),
#   `drill` (propose the next step, code checks it, then solve with both steps fixed: 2 calls).
# - **Select** the (subgoal, option) arm with the highest UCT score, run it, get reward 1 or 0 from the checker,
#   update the arm with the running mean from the paper. Stop on success or when the budget is used up.
# - **Learn across puzzles:** each option keeps a global success rate that becomes the starting value of new arms.

# %%
class Subgoal(BaseModel):
    step: str = Field(description="one operation on two of the available numbers, like '8 - 2 = 6'")
    confidence: int = Field(ge=1, le=10, description="how likely this step leads to 24")

class Plan(BaseModel):
    subgoals: List[Subgoal] = Field(max_length=4)

def fmt(x: Fraction):
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"

def apply_step(step: str, state):
    """Check one step against the available numbers. Returns (new_state, text) or None (= backtrack)."""
    m = re.search(r"(\d+(?:/\d+)?)\s*([\+\-\*/xX×÷])\s*(\d+(?:/\d+)?)", (step or "").replace("−", "-"))
    if not m:
        return None
    a, op, b = Fraction(m.group(1)), m.group(2), Fraction(m.group(3))
    pool = list(state)
    for v in (a, b):
        if v not in pool:
            return None
        pool.remove(v)
    op = {"x": "*", "X": "*", "×": "*", "÷": "/"}.get(op, op)
    if op == "/" and b == 0:
        return None
    val = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
    return pool + [val], f"{fmt(a)} {op} {fmt(b)} = {fmt(val)}"

SOLVE = ("Numbers: {nums}. Goal: 24, using each number exactly once with + - * / and brackets.\n"
         "Follow this plan: {steps}. After these steps the numbers left are {left}.\n"
         "Write ONE expression with the ORIGINAL numbers {nums} that follows the plan and equals 24.\n"
         "End with one line: ANSWER: <expression>")

class Step(BaseModel):
    step: str

TOOL_STATS = {t: [1, 2] for t in ("solve", "reflect", "drill")}     # [successes+1, tries+2]: a gentle prior

def run_tool(tool, nums, steps, left, budget_left):
    """Returns (ok, calls_used)."""
    prompt = SOLVE.format(nums=nums, steps="; ".join(steps), left=[fmt(x) for x in left])
    if tool == "drill":
        try:
            nxt = structured(f"Numbers left: {[fmt(x) for x in left]}. Goal: reach 24 in the end. "
                             f"Propose the single best next step on two of these numbers, like 'a op b = c'.", Step, temperature=0.7)
            res = apply_step(nxt.step, left)
        except Exception:
            res = None
        if res is None or budget_left < 2:
            return False, 1
        left, txt = res
        prompt = SOLVE.format(nums=nums, steps="; ".join(steps + [txt]), left=[fmt(x) for x in left])
        ok, _ = check24(extract_expr(ask(prompt, temperature=0.3, max_tokens=500)), nums)
        return ok, 2
    msgs = [{"role": "user", "content": prompt}]
    out = chat(msgs, temperature=0.3, max_tokens=500).content or ""
    ok, why = check24(extract_expr(out), nums)
    if ok or tool == "solve" or budget_left < 2:
        return ok, 1
    msgs += [{"role": "assistant", "content": out},
             {"role": "user", "content": f"A checker says your expression is wrong: {why}. Reflect, then give a fixed ANSWER line."}]
    ok, _ = check24(extract_expr(chat(msgs, temperature=0.3, max_tokens=500).content), nums)
    return ok, 2

def prime(nums, budget=BUDGET, C=1.0, log=False):
    calls = 1
    try:
        plan = structured(f"Numbers: {nums}. Goal: make 24 using each number once with + - * /.\n"
                          "Propose up to 4 DIFFERENT promising first steps. Each step combines two of the numbers.",
                          Plan, temperature=0.7)
        subgoals = plan.subgoals
    except Exception:
        subgoals = []
    state0 = [Fraction(n) for n in nums]
    arms = []                                            # one arm per (valid subgoal, tool)
    for sg in subgoals:
        res = apply_step(sg.step, state0)
        if res is None:
            if log: print("   backtrack (invalid step):", sg.step)
            continue
        left, txt = res
        for tool, (s, n) in TOOL_STATS.items():
            prior = 0.5 * sg.confidence / 10 + 0.5 * s / n   # V = λ·LM + (1−λ)·(what this option did before)
            arms.append({"txt": txt, "left": left, "tool": tool, "N": 1, "V": prior})
    if not arms:                                         # nothing valid: fall back to solving from scratch
        arms = [{"txt": "(no plan)", "left": state0, "tool": "reflect", "N": 1, "V": 0.5}]
    total = sum(a["N"] for a in arms)
    while calls < budget:
        uct = lambda a: a["V"] + C * math.sqrt(math.log(total) / a["N"])
        arm = max(arms, key=uct)
        steps = [] if arm["txt"] == "(no plan)" else [arm["txt"]]
        ok, used = run_tool(arm["tool"], nums, steps, arm["left"], budget - calls)
        calls += used
        r = 1.0 if ok else 0.0
        arm["V"] = (arm["V"] * arm["N"] + r) / (arm["N"] + 1); arm["N"] += 1; total += 1     # backpropagation
        TOOL_STATS[arm["tool"]][0] += r; TOOL_STATS[arm["tool"]][1] += 1
        if log: print(f"   try {arm['tool']:7s} on [{arm['txt']}] -> {'✓' if ok else '✗'} (calls {calls})")
        if ok:
            return {"ok": True, "calls": calls}
    return {"ok": False, "calls": calls}

print(apply_step("8 - 2 = 6", [Fraction(n) for n in (1, 2, 3, 8)]), apply_step("9 - 2", [Fraction(n) for n in (1, 2, 3, 8)]))

# %% [markdown]
# ## 4. Run all three methods
# Puzzles run two at a time. Total model calls: at most 6 + 30 + 30.

# %%
def run(method):
    with ThreadPoolExecutor(2) as ex:
        return list(ex.map(method, PUZZLES))

results = {"single shot": run(single_shot), "retry + feedback": run(retry_with_feedback)}
print("PRIME-lite, puzzle by puzzle:")
results["PRIME-lite"] = []
for nums in PUZZLES:                                   # sequential so tool statistics carry over in order
    print(" ", nums)
    results["PRIME-lite"].append(prime(nums, log=True))

for name, rs in results.items():
    acc = sum(r["ok"] for r in rs) / len(rs)
    calls = sum(r["calls"] for r in rs) / len(rs)
    print(f"{name:18s} success {acc:4.0%}   avg calls {calls:.1f}")
print("option statistics (successes+1, tries+2):", TOOL_STATS)

# %% [markdown]
# **What to notice.** With an 8B model the single shot is often weak, like 4o-mini's 0% in the paper. Compare PRIME-lite
# with *retry + feedback*, not just with the single shot: both spend up to 5 calls. If PRIME-lite wins, the
# plan + strategy choice is doing work. If they tie, the gain came from more tries, which is the honest result the
# paper's WebShop row also shows. Also look at the option statistics: which tool did the search learn to prefer?

# %%
acc = {k: sum(r["ok"] for r in v) / len(v) for k, v in results.items()}
check("all three methods ran on every puzzle", all(len(v) == len(PUZZLES) for v in results.values()))
check("search methods stay within the call budget", all(r["calls"] <= BUDGET for k in ("retry + feedback", "PRIME-lite") for r in results[k]))
check("checker rejects a wrong answer", not check24("8*3", [1, 2, 3, 8])[0] and check24("(8-2)*(3+1)", [1, 2, 3, 8])[0])
check("PRIME-lite solves at least one puzzle", acc["PRIME-lite"] > 0)

# %% [markdown]
# ## Your turn
# 1. **Reverse ablation.** PRIME upgraded one part at a time to see where a better model helps. Use `llama3.2:3b`
#    for the executor tools but keep `llama3.1:8b` as the planner. Does accuracy drop more than you expected?
# 2. **Turn off learning.** Reset `TOOL_STATS` before every puzzle. Does carrying option statistics across puzzles help?
# 3. **Exploration constant.** Try `C=0.2` and `C=2.0`. What happens to the number of different arms tried?

# %% [markdown]
# ## Solutions

# %%
# 1. Reverse ablation on 2 puzzles (up to 10 calls). We swap the model used by the executor tools only.
#    Notice: in the paper the executor upgrade mattered about as much as the planner upgrade.
import lal.llm as L
_ask, _chat = L.ask, L.chat
def small_ask(p, **kw): return _ask(p, model_name="llama3.2:3b", **kw)
def small_chat(m, **kw): return _chat(m, model_name="llama3.2:3b", **kw)
ask, chat = small_ask, small_chat            # run_tool looks these names up at call time
small = [prime(n) for n in PUZZLES[:2]]
ask, chat = _ask, _chat
print("small executor:", sum(r["ok"] for r in small), "/ 2   vs 8B executor:",
      sum(r["ok"] for r in results["PRIME-lite"][:2]), "/ 2")

# %%
# 2. No learning across puzzles: reset the option statistics each time (written, not run, to save calls).
#    for nums in PUZZLES:
#        TOOL_STATS.update({t: [1, 2] for t in TOOL_STATS}); prime(nums)
#    Notice: with only 6 puzzles the effect is small and noisy; this is a 30+ puzzle experiment.

# %%
# 3. Exploration: count distinct arms tried by logging. Low C sticks with the planner's favourite step;
#    high C spreads calls across subgoals. Run one puzzle each way with a budget of 3 (up to 6 calls):
for c in (0.2, 2.0):
    print("C =", c); prime(PUZZLES[0], budget=3, C=c, log=True)
