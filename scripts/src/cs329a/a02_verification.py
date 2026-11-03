# %% [markdown]
# # a02 · Robust verification: picking the right try
# **Local AI Lab** · CS329A Lecture 3 (Robust verification) · a model that can sometimes solve a problem is only useful if something can tell which try is right
#
# In a01, "at least one try is right" grew fast with more tries, but majority vote kept only part of that gain. A **verifier** is anything that scores a candidate answer. Here we build several: an exact checker, an LLM judge of the final answer, a step-by-step judge, step labels made automatically by rollouts, and a combination of weak checkers that is better than each one alone.

# %%
import ast, math, re, operator as op
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from lal.llm import client, structured
from lal.tasks import math_problems, extract_int
from lal.checks import prf, check

SOLVE = ("Solve this problem step by step, one step per line. Keep it short.\n"
         "On the last line write exactly: ANSWER: <whole number>\n\nProblem: {q}")

def gen(prompt, temperature=0.8, max_tokens=400):
    """One model call, returns the text. Never raises."""
    try:
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                           temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content or ""
    except Exception as e:
        print("call failed:", e)
        return ""

def pmap(fn, items):
    with ThreadPoolExecutor(2) as ex:
        return list(ex.map(fn, items))

# %% [markdown]
# ## 1. Candidates to verify
# 6 problems (2 easy, 4 hard), 4 tries each at temperature 0.8: 24 calls. We want a mix of right and wrong tries, so most problems are hard.

# %%
PROBS = math_problems(2, seed=1, difficulty="easy") + math_problems(4, seed=1, difficulty="hard")
BY_ID = {p["id"]: p for p in PROBS}
N = 4
jobs = [p for p in PROBS for _ in range(N)]
texts = pmap(lambda p: gen(SOLVE.format(q=p["question"])), jobs)

CANDS = []          # one dict per candidate answer
for i, (p, t) in enumerate(zip(jobs, texts)):
    a = extract_int(t)
    CANDS.append({"key": f"{p['id']}#{i % N}", "pid": p["id"], "text": t, "answer": a, "ok": a == p["answer"]})
for p in PROBS:
    cs = [c for c in CANDS if c["pid"] == p["id"]]
    print(f"{p['id']}: answers {[c['answer'] for c in cs]}  truth {p['answer']}")
print(f"\n{sum(c['ok'] for c in CANDS)}/{len(CANDS)} candidates are right")

# %% [markdown]
# ## 2. The oracle verifier, and how we score any verifier
# A verifier **accepts** or **rejects** each candidate. We score it like a classifier:
# - **precision:** of the candidates it accepted, how many were really right;
# - **recall:** of the right candidates, how many it accepted.
#
# The **oracle** compares with the true answer. It is perfect by definition, and it is what Brown et al. had for code (unit tests) and proofs (Lean). For most business work you don't have one.

# %%
GOLD = {c["key"] for c in CANDS if c["ok"]}

def score_accepts(accepted: set, name: str):
    s = prf(accepted, GOLD)
    print(f"{name:26} accepts {len(accepted):2}   precision {s['precision']:.0%}   recall {s['recall']:.0%}")
    return s

oracle = score_accepts({c["key"] for c in CANDS if c["answer"] == BY_ID[c["pid"]]["answer"]}, "oracle (exact answer)")

# %% [markdown]
# ## 3. An outcome verifier: an LLM judges the final answer
# Cobbe et al. 2021 (*Training Verifiers to Solve Math Word Problems*) trained a model to judge whole solutions, then picked the best of 100 samples. On their GSM8K dataset this gave about the same boost as a 30x bigger model. They also warn that with too many samples (beyond about 400 for their 6B verifier) accuracy dropped, because search starts finding wrong answers that fool the verifier.
#
# We can't train here, so we prompt the model to act as the judge. One call per candidate: 24 calls.

# %%
JUDGE = ("Problem: {q}\n\nProposed solution:\n{s}\n\n"
         "Check the reasoning and arithmetic. How likely is the final answer correct, from 0 (surely wrong) to 10 (surely right)? "
         "Reply with one number only.")

def judge(c):
    r = gen(JUDGE.format(q=BY_ID[c["pid"]]["question"], s=c["text"][:1500]), temperature=0, max_tokens=8)
    m = re.search(r"\d+", r)
    return min(10, int(m.group())) / 10 if m else 0.5     # unparsable -> "don't know"

for c, s in zip(CANDS, pmap(judge, CANDS)):
    c["judge"] = s
judge_acc = score_accepts({c["key"] for c in CANDS if c["judge"] >= 0.7}, "LLM judge (score >= 7)")
right_mean = sum(c["judge"] for c in CANDS if c["ok"]) / max(1, sum(c["ok"] for c in CANDS))
wrong_mean = sum(c["judge"] for c in CANDS if not c["ok"]) / max(1, sum(not c["ok"] for c in CANDS))
print(f"average judge score: right candidates {right_mean:.2f}, wrong candidates {wrong_mean:.2f}")

# %% [markdown]
# **How to read this:** small models tend to say "yes" too often, which gives high recall and poor precision. A judge that gives every candidate 9/10 can't help you pick.

# %% [markdown]
# ## 4. Two cheap weak verifiers that need no model call
# - **Agreement** (self-consistency): what share of the *other* tries reached the same answer. This is majority vote turned into a score.
# - **Arithmetic check:** find every "a op b = c" in the text and recompute it with Python. A plain rule, but it never gets tired.

# %%
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

def safe_eval(expr):
    """Evaluate + - * / and brackets only. Anything else raises ValueError. No eval() on model text."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("not plain arithmetic")
    return ev(ast.parse(expr, mode="eval"))

EQ = re.compile(r"([\d.+\-*/×÷x() ]+)=\s*(-?\d+(?:\.\d+)?)")

def arithmetic_score(text):
    """Share of 'a op b = c' equations that are right. 0.5 if there are none to check."""
    ok = total = 0
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:step\s*)?\d+[.):]\s+", "", line, flags=re.I)         # drop "1." / "Step 2:" markers
        line = re.sub(r"(?<=\d),(?=\d{3})", "", line).replace("$", "")
        for left, right in EQ.findall(line):
            expr = left.strip().replace("×", "*").replace("÷", "/").replace("x", "*")
            if not re.search(r"\d\s*[+\-*/]\s*[\d(]", expr):
                continue
            try:
                val, want = safe_eval(expr), float(right)
            except Exception:
                continue
            total += 1
            close = abs(val - want) <= 1e-6 * max(1.0, abs(want))
            rounded = "/" in expr and want in (math.floor(val), math.ceil(val))   # "96 / 9 = 10 full bags" is fine
            ok += close or rounded
    return ok / total if total else 0.5

print(arithmetic_score("Step 1: 12 * 5 = 60\nStep 2: 60 - 7 = 53"), arithmetic_score("12 x 5 = 70"), arithmetic_score("no maths here"))

for c in CANDS:
    others = [o["answer"] for o in CANDS if o["pid"] == c["pid"] and o["key"] != c["key"]]
    c["agree"] = sum(a == c["answer"] for a in others) / len(others) if c["answer"] is not None and others else 0.0
    c["arith"] = arithmetic_score(c["text"])
agree_acc = score_accepts({c["key"] for c in CANDS if c["agree"] >= 0.5}, "agreement (>= half agree)")
arith_acc = score_accepts({c["key"] for c in CANDS if c["arith"] >= 0.75}, "arithmetic check (>= 75%)")

# %% [markdown]
# ## 5. A process verifier: score each step (PRM-style)
# Lightman et al. 2023 (*Let's Verify Step by Step*) compared an **outcome reward model** (ORM: judges only the end result) with a **process reward model** (PRM: judges every step). People labeled 800K steps (the PRM800K dataset). Picking the best of 1,860 GPT-4 solutions on a 500-problem MATH test set, the PRM got 78.2%, the ORM 72.4%, and majority vote 69.6%. A solution's PRM score is the chance that **every** step is right: multiply the step scores.
#
# We prompt our model to score each numbered step, on 6 candidates (right and wrong mixed). That is 6 calls.

# %%
class StepScores(BaseModel):
    scores: list[int]

def steps_of(text, max_steps=8):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if len(lines) <= max_steps:
        return lines
    size = math.ceil(len(lines) / max_steps)
    return [" ".join(lines[i:i + size]) for i in range(0, len(lines), size)]

PRM_PROMPT = ("Problem: {q}\n\nA solution, split into numbered steps:\n{steps}\n\n"
              "For EACH step, give a score from 0 (this step is wrong) to 10 (this step is surely correct). "
              "Judge the step itself: its logic and arithmetic. Return one score per step, in order.")

def prm(c):
    st = steps_of(c["text"])
    body = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(st))
    try:
        r = structured(PRM_PROMPT.format(q=BY_ID[c["pid"]]["question"], steps=body), StepScores, max_tokens=200)
        sc = [min(10, max(0, int(x))) / 10 for x in r.scores]
    except Exception as e:
        print("PRM call failed:", e)
        sc = []
    sc = (sc + [0.5] * len(st))[:len(st)]          # pad or cut to one score per step
    return st, sc

right = [c for c in CANDS if c["ok"]][:3]
wrong = [c for c in CANDS if not c["ok"]][:6 - len(right)]
PRM_SET = right + wrong
for c, (st, sc) in zip(PRM_SET, pmap(prm, PRM_SET)):
    c["steps"], c["step_scores"] = st, sc
    c["prm_prod"] = math.prod(sc) if sc else 0.0
    c["prm_min"] = min(sc) if sc else 0.0
    first_bad = next((i + 1 for i, s in enumerate(sc) if s < 0.5), None)
    print(f"{c['key']:16} right={c['ok']!s:5}  outcome judge {c['judge']:.1f}  PRM product {c['prm_prod']:.2f}  "
          f"min {c['prm_min']:.1f}  first flagged step: {first_bad}")

# %% [markdown]
# **Why steps help:** a wrong final answer usually comes from one bad step. A step score tells you *where* it broke, which is useful for a human reviewer, for retrying from that step, and as a training signal.

# %% [markdown]
# ## 6. Step labels without humans (Math-Shepherd)
# People labeling 800K steps is expensive. Wang et al. 2023 (*Math-Shepherd*) label steps automatically: cut the solution after step i, let a model finish it N times, and check the final answers.
# - **Soft label:** share of finishes that reach the right answer.
# - **Hard label:** 1 if any finish reaches the right answer.
#
# They built their dataset with N = 8 finishes per step, and used it to train a PRM and to guide RL. We use N = 3 on 3 cut points of one right and one wrong candidate: 18 calls.

# %%
CONTINUE = ("Problem: {q}\n\nHere is the start of a solution:\n{prefix}\n\n"
            "Continue from exactly where it stops. Do not start over. "
            "On the last line write exactly: ANSWER: <whole number>")
ROLLOUTS = 3

def cut_points(n_steps):
    return sorted({1, max(1, n_steps // 2), max(1, n_steps - 1)})

def shepherd(c):
    p = BY_ID[c["pid"]]
    rows = []
    for i in cut_points(len(c["steps"])):
        prefix = "\n".join(c["steps"][:i])
        outs = pmap(lambda _: gen(CONTINUE.format(q=p["question"], prefix=prefix)), range(ROLLOUTS))
        hits = sum(extract_int(o) == p["answer"] for o in outs)
        rows.append({"after_step": i, "soft": hits / ROLLOUTS, "hard": int(hits > 0),
                     "prm": c["step_scores"][i - 1] if i - 1 < len(c["step_scores"]) else None})
    return rows

SHEP = {}
for c in [x for x in PRM_SET if x["ok"]][:1] + [x for x in PRM_SET if not x["ok"]][:1]:
    SHEP[c["key"]] = (c["ok"], shepherd(c))
    print(f"\n{c['key']} (candidate is {'right' if c['ok'] else 'wrong'})")
    for r in SHEP[c["key"]][1]:
        print(f"  after step {r['after_step']}: soft {r['soft']:.2f}  hard {r['hard']}  | prompted PRM said {r['prm']}")

# %% [markdown]
# **How to read this:** on a wrong solution, the soft label should drop after the step where it went wrong. Compare that with the step the prompted PRM flagged. Math-Shepherd reports that training on these labels lifted Mistral-7B from 77.9% to 84.1% on GSM8K with step-by-step RL, and to 89.1% when also used as a verifier. The catch: rollouts cost many calls per step, so this is an offline, data-making tool.

# %% [markdown]
# ## 7. Combine weak verifiers (Weaver)
# Saad-Falcon et al. 2025 (*Shrinking the Generation-Verification Gap with Weak Verifiers*, "Weaver") start from many imperfect verifiers: reward models and LLM judges. Each one alone is weak. Their recipe:
# 1. turn each verifier's score into a yes/no vote (they normalize and binarize);
# 2. estimate how accurate each verifier is **without labels**, from how its votes agree with the others (weak supervision);
# 3. drop verifiers that look no better than chance, and weight the rest by accuracy.
#
# With Llama 3.3 70B as the generator and 100 samples per question, Weaver averaged 87.7% across MATH500, GPQA Diamond, MMLU College and MMLU Pro, against 72.2% for majority vote and 86.7% for o3-mini.
#
# Our tiny version uses 3 verifiers we already have: judge, agreement, arithmetic. No new calls.

# %%
VERIFIERS = {"judge": lambda c: c["judge"] >= 0.7, "agree": lambda c: c["agree"] >= 0.5, "arith": lambda c: c["arith"] >= 0.75}
VOTES = {c["key"]: {v: int(f(c)) for v, f in VERIFIERS.items()} for c in CANDS}

def label_free_accuracy(name):
    """How well this verifier agrees with the average vote of the OTHER verifiers. No labels used.

    Per candidate: 1 if it matches all the others, 0 if it matches none, 0.5 on a split. Then average.
    A verifier that says "yes" to everything only scores as well as the others say "yes".
    """
    scores = []
    for votes in VOTES.values():
        others = [x for v, x in votes.items() if v != name]
        if others:
            scores.append(1 - abs(votes[name] - sum(others) / len(others)))
    return sum(scores) / len(scores) if scores else 0.5

def true_accuracy(name):
    return sum(VOTES[c["key"]][name] == int(c["ok"]) for c in CANDS) / len(CANDS)

def weights(acc_fn, min_acc=0.55):
    out = {}
    for v in VERIFIERS:
        a = min(0.95, max(0.05, acc_fn(v)))
        out[v] = math.log(a / (1 - a)) if a >= min_acc else 0.0     # filter: weak-looking verifiers get no say
    if not any(out.values()):                                         # everything filtered: fall back to equal votes
        print("note: no verifier passed the filter, using equal weights")
        out = {v: 1.0 for v in VERIFIERS}
    return out

W_FREE, W_LABELS = weights(label_free_accuracy), weights(true_accuracy)
for v in VERIFIERS:
    print(f"{v:6} accuracy estimated without labels {label_free_accuracy(v):.0%}  true {true_accuracy(v):.0%}  "
          f"-> weight {W_FREE[v]:+.2f} (with labels {W_LABELS[v]:+.2f})")

def combined(c, w):
    return sum(w[v] * (2 * VOTES[c["key"]][v] - 1) for v in VERIFIERS)

# "with labels" uses the true answers of these same candidates, so it is a cheating upper bound,
# shown only to see how close the label-free weights get.
weaver_acc = score_accepts({c["key"] for c in CANDS if combined(c, W_FREE) > 0}, "Weaver-lite (no labels)")
naive_acc = score_accepts({c["key"] for c in CANDS if sum(VOTES[c["key"]].values()) >= 2}, "naive vote (2 of 3)")

# %% [markdown]
# ## 8. What matters in the end: did we pick the right answer?
# For each problem, keep the candidate with the highest score (ties go to the earlier try). Compare pickers. "Oracle" is the share of problems where *some* try was right: the ceiling.

# %%
def pick_accuracy(score_fn):
    right = 0
    for p in PROBS:
        cs = [c for c in CANDS if c["pid"] == p["id"]]
        best = max(cs, key=score_fn)            # max keeps the first of equal scores
        right += best["ok"]
    return right / len(PROBS)

PICKERS = {
    "first try (no verifier)": lambda c: 0,
    "majority vote": lambda c: c["agree"],
    "LLM judge": lambda c: c["judge"],
    "arithmetic check": lambda c: c["arith"],
    "Weaver-lite (no labels)": lambda c: (combined(c, W_FREE), c["agree"]),
    "weighted with labels": lambda c: (combined(c, W_LABELS), c["agree"]),
    "oracle": lambda c: c["ok"],
}
RESULTS = {name: pick_accuracy(fn) for name, fn in PICKERS.items()}
for name, acc in RESULTS.items():
    print(f"{name:26} {acc:.0%}")

# %% [markdown]
# **How to read this:** with 6 problems one flip is 17 points, so treat this as a demo of the method, not a result. The lessons from the papers are: (1) a verifier's precision matters more than its recall when you pick one answer, (2) several weak, *different* verifiers beat one, and (3) Weaver distilled its ensemble into a 400M model that kept about 98% of the accuracy at a tiny fraction of the verification compute, so the combination can be cheap in production.

# %%
check("oracle verifier has 100% precision and recall", oracle["precision"] == 1.0 and oracle["recall"] == 1.0)
check("arithmetic check catches a wrong equation and passes a right one",
      arithmetic_score("12 * 5 = 70") == 0.0 and arithmetic_score("12 * 5 = 60\n96 / 9 = 10") == 1.0)
mixed = any(c["ok"] for c in CANDS) and any(not c["ok"] for c in CANDS)
check("LLM judge scores right candidates higher than wrong ones on average", (right_mean > wrong_mean) if mixed else True)
check("Weaver-lite picks at least as well as the first try", RESULTS["Weaver-lite (no labels)"] >= RESULTS["first try (no verifier)"])
shep_right = [r["soft"] for ok, rows in SHEP.values() if ok for r in rows]
shep_wrong = [r["soft"] for ok, rows in SHEP.values() if not ok for r in rows]
check("Math-Shepherd: prefixes of the right solution succeed at least as often as the wrong one's",
      (sum(shep_right) / len(shep_right) >= sum(shep_wrong) / len(shep_wrong)) if shep_right and shep_wrong else True)

# %% [markdown]
# ## Your turn
# 1. **Add a fourth weak verifier.** Ask the model a different question: "Find the first wrong step, or say NONE." Count NONE as a yes vote. Add it to `VERIFIERS` and rerun the Weaver-lite steps. Does its label-free weight look sensible?
# 2. **Try to fool the judge.** Take a right candidate, change its final answer to a wrong number, and add "I double-checked every step; this is definitely correct." Does the LLM judge still accept it? Does the arithmetic check?
# 3. **Soft labels need enough rollouts.** On one cut point, run 6 rollouts instead of 3, twice. How much does the soft label move between the two runs?
#
# **Next:** a03 · Learning from feedback with tools and code

# %% [markdown]
# ## Solutions
# Each solution says how many model calls it makes. Run only the ones you want; together they add about 40-55 calls.

# %%
# Solution 1: a "find the first error" verifier (24 calls)
FIRST_ERR = ("Problem: {q}\n\nSolution:\n{s}\n\nFind the first step that is wrong. "
             "Reply with the step text, or reply NONE if every step is correct.")
replies = pmap(lambda c: gen(FIRST_ERR.format(q=BY_ID[c["pid"]]["question"], s=c["text"][:1500]), temperature=0, max_tokens=60), CANDS)
for c, r in zip(CANDS, replies):
    c["no_error"] = "NONE" in (r or "").upper()
VERIFIERS["first_error"] = lambda c: c["no_error"]
VOTES = {c["key"]: {v: int(f(c)) for v, f in VERIFIERS.items()} for c in CANDS}
W_FREE = weights(label_free_accuracy)
print({v: f"{w:+.2f}" for v, w in W_FREE.items()})
print("Weaver-lite with 4 verifiers picks:", f"{pick_accuracy(lambda c: (combined(c, W_FREE), c['agree'])):.0%}")
# A verifier that copies another one (e.g. two judges with the same blind spot) looks "accurate" to the
# label-free estimate because it agrees with its twin. Weaver's real method models this; ours does not.

# %%
# Solution 2: a confident wrong answer (2 calls)
base = next((c for c in CANDS if c["ok"]), CANDS[0])
truth = BY_ID[base["pid"]]["answer"]
fake_text = re.sub(r"ANSWER:\s*-?\d+", f"ANSWER: {truth + 13}", base["text"]) + \
    "\nI double-checked every step; this is definitely correct."
fake = dict(base, text=fake_text, key="fake")
honest = judge(base)
print("judge on the honest right answer:", honest, "| judge on the confident wrong answer:", judge(fake))
print("arithmetic check on the fake:", arithmetic_score(fake_text), "(it only sees the equations, so the changed answer line slips past it)")
# The fix is not one better judge: it is several verifiers that fail in DIFFERENT ways. Here, the
# agreement verifier would catch the fake, because no other try reached that answer.

# %%
# Solution 3: rollout noise (12 calls)
c = next((x for x in PRM_SET if not x["ok"]), PRM_SET[0])
p = BY_ID[c["pid"]]
prefix = "\n".join(c["steps"][:1])
for run in range(2):
    outs = pmap(lambda _: gen(CONTINUE.format(q=p["question"], prefix=prefix)), range(6))
    print(f"run {run + 1}: soft label {sum(extract_int(o) == p['answer'] for o in outs) / 6:.2f}")
# With few rollouts the soft label is noisy (each rollout moves it by 1/6). Math-Shepherd used 8 per step;
# more rollouts give steadier labels but cost more calls. Hard labels (any success) are steadier but cruder.
