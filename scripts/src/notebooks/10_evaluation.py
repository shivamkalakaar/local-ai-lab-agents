# %% [markdown]
# # 10 · Evaluation: build the test set before the agent
# **Local AI Lab** · CS329Z Lecture 10 (Terminal-Bench) and the evaluation readings · [Module 10](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m10)
#
# An eval is how you know a change helped, whether a cheap local model is good enough, and what to tell a customer. Every task has four parts: **request, environment, stopping rule, scorer**. We grade with code wherever we can, test the eval itself, and measure consistency with pass^k.

# %%
import json, time, re
from pydantic import BaseModel
from lal.llm import structured, ask
from lal.data import contract, labels, playbook, ids
from lal.checks import prf, pass_at_k, pass_hat_k, bootstrap_ci, check

PB = playbook()
RULES = list(PB["rules"])

# %% [markdown]
# ## 1. The system under test
# A simple reviewer: one call returns the list of broken rules.

# %%
class Review(BaseModel):
    broken_rules: list[str]

def reviewer(cid, temperature=0.0):
    r = structured(f"Playbook rules (name: rule):\n{json.dumps(PB['rules'], indent=1)}\n\nContract:\n{contract(cid)}\n\n"
                   "List the names of the rules this contract breaks. Empty list if none.", Review, temperature=temperature)
    return {x for x in r.broken_rules if x in RULES}

print(reviewer("msa_15"), "| truth:", {i["rule"] for i in labels()["msa_15"]["issues"]})

# %% [markdown]
# ## 2. The scorer: code, precision AND recall
# Recall alone rewards flagging everything. Precision alone rewards flagging nothing. Clean contracts in the set punish false flags.

# %%
def score(pred, cid):
    gold = {i["rule"] for i in labels()[cid]["issues"]}
    s = prf(pred, gold)
    s["exact"] = pred == gold
    return s

# %% [markdown]
# ## 3. Test the eval itself before trusting it
# - **Oracle:** the true answers must score 100%.
# - **Do nothing:** must score low on recall.
# - **Flag everything:** must score low on precision.

# %%
def run_eval(system, cids):
    rows = [score(system(c), c) for c in cids]
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "exact": sum(r["exact"] for r in rows) / len(rows)}

ALL = ids()
oracle = lambda c: {i["rule"] for i in labels()[c]["issues"]}
nothing = lambda c: set()
everything = lambda c: set(RULES)
for name, fn in [("oracle", oracle), ("do nothing", nothing), ("flag everything", everything)]:
    print(f"{name:16}", {k: f"{v:.0%}" for k, v in run_eval(fn, ALL).items()})

# %% [markdown]
# ## 4. Run the real reviewer: 3 runs per contract
# Same documents, several runs at the temperature we'd use in production. Then report the average **and** pass^3 (right on all 3 runs), with a confidence interval over documents.

# %%
N_RUNS, TEMP = 3, 0.3
SAMPLE = ALL[::2]          # 10 contracts to keep it quick; use ALL for the full set
t0 = time.time()
runs = {c: [score(reviewer(c, TEMP), c)["exact"] for _ in range(N_RUNS)] for c in SAMPLE}
print(f"{len(SAMPLE)} contracts x {N_RUNS} runs in {time.time()-t0:.0f}s")

per_doc_avg = [sum(r) / N_RUNS for r in runs.values()]
per_doc_hat = [pass_hat_k(N_RUNS, sum(r), N_RUNS) for r in runs.values()]
per_doc_at = [pass_at_k(N_RUNS, sum(r), N_RUNS) for r in runs.values()]
for name, xs in [("average exact", per_doc_avg), (f"pass@{N_RUNS}", per_doc_at), (f"pass^{N_RUNS}", per_doc_hat)]:
    lo, hi = bootstrap_ci(xs)
    print(f"{name:14} {sum(xs)/len(xs):.0%}  (95% CI {lo:.0%}-{hi:.0%})")
print("\nper contract:", {c: r for c, r in runs.items()})

# %% [markdown]
# **How to read this:** pass@3 is "a person could pick a good one out of three". pass^3 is "it's right every time", which is what matters for repeated legal work. The gap between them is inconsistency.

# %% [markdown]
# ## 5. An LLM judge, done carefully
# Use a judge only for open-ended parts (like redline quality). Give it a reference answer, swap the order, and count disagreement after swapping as a tie.

# %%
JUDGE = """Two rewrites of a liability clause. The firm's rule: cap must not exceed 12 months of fees.
Reference (expert): {ref}
A: {a}
B: {b}
Which rewrite better follows the rule and keeps the meaning? Longer is not better. Answer A, B, or TIE only."""

def pairwise(ref, x, y):
    first = ask(JUDGE.format(ref=ref, a=x, b=y)).strip().upper()[:3]
    second = ask(JUDGE.format(ref=ref, a=y, b=x)).strip().upper()[:3]
    if first.startswith("A") and second.startswith("B"): return "x wins"
    if first.startswith("B") and second.startswith("A"): return "y wins"
    return "tie (judge changed its mind when order swapped)" if first[:1] != second[:1] else "tie"

ref = "Each party's total liability shall not exceed the fees paid in the twelve (12) months preceding the claim."
good = "Total liability of each party is capped at fees paid during the 12 months before the claim."
bad = "Total liability of each party is capped at fees paid during the 24 months before the claim, which is a commercially reasonable and widely accepted market standard."
print(pairwise(ref, good, bad))

# %%
check("oracle scores 100% exact", run_eval(oracle, ALL)["exact"] == 1.0)
check("flag-everything has low precision (< 50%)", run_eval(everything, ALL)["precision"] < 0.5)
check("reviewer beats do-nothing on recall", sum(sum(r) for r in runs.values()) > 0)

# %% [markdown]
# ## Your turn
# 1. Write a "cheat" reviewer that games the scorer (for example, always flag the two most common rules). Does the eval catch it?
# 2. Run all 20 contracts with 5 runs and compare pass^5 between temperature 0 and 0.7.
#
# **Next:** 11 · Optimization with GEPA
