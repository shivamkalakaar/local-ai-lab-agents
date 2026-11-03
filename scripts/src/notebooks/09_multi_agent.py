# %% [markdown]
# # 09 · Multi-agent systems: when one agent plus a checker wins
# **Local AI Lab** · CS329Z Lecture 8 · [Module 09](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m9)
#
# Across 260 setups, multi-agent systems averaged **−0.3%** versus a single agent. They helped on wide, separable work and hurt on step-by-step work. On one local GPU, every extra agent also means more waiting. Let's measure both designs on the same task.

# %%
import json, time
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from lal.llm import ask, structured
from lal.data import contract, sections, labels, playbook, ids
from lal.checks import prf, check

PB = playbook()
RULE_SECTION = {"liability_cap": "Liability", "governing_law": "Governing", "auto_renewal": "Renewal", "termination_notice": "Renewal"}

class Verdicts(BaseModel):
    liability_cap: bool
    governing_law: bool
    auto_renewal: bool
    termination_notice: bool

# %% [markdown]
# ## Design A: one agent, then one checker
# One call reads the four relevant clauses and judges all four rules (true = breaks the rule). A second step, in code, re-checks the two numeric rules directly.

# %%
import re
def single_agent(cid):
    secs = {s["title"]: s["text"] for s in sections(contract(cid))}
    relevant = "\n".join(f"{t}: {x}" for t, x in secs.items() if any(w in t for w in ("Liability", "Governing", "Renewal")))
    v = structured(f"Playbook rules:\n{json.dumps(PB['rules'], indent=1)}\n\nClauses:\n{relevant}\n\n"
                   "For each rule, return true if the contract BREAKS it, false if it follows it.", Verdicts)
    flags = {k for k, broken in v.model_dump().items() if broken}
    # checker in code: numbers we can verify directly
    renew = next(x for t, x in secs.items() if "Renewal" in t)
    years = re.search(r"\((\d+)\) year", renew); days = re.search(r"(\d+)\)? days", renew)
    if years: flags = (flags - {"auto_renewal"}) | ({"auto_renewal"} if int(years.group(1)) > 1 else set())
    if days: flags = (flags - {"termination_notice"}) | ({"termination_notice"} if int(days.group(1)) < 30 else set())
    return flags

# %% [markdown]
# ## Design B: a team of four specialist agents plus a merger agent
# Each specialist judges one rule. A fifth agent merges their reports into a final list.

# %%
def specialist(args):
    cid, rule = args
    clause = next(s for s in sections(contract(cid)) if RULE_SECTION[rule] in s["title"])["text"]
    return rule, ask(f"You are the {rule} specialist. Rule: {PB['rules'][rule]}\nClause: {clause}\n"
                     "Write a two-sentence report ending with VERDICT: BREAKS or VERDICT: FOLLOWS.")

def team(cid):
    with ThreadPoolExecutor(4) as ex:
        reports = list(ex.map(specialist, [(cid, r) for r in PB["rules"]]))
    merged = ask("You are the lead reviewer. Combine these specialist reports. List the rule names that are broken, "
                 "comma-separated, or 'none'.\n\n" + "\n\n".join(f"[{r}] {t}" for r, t in reports))
    return {r for r in PB["rules"] if r in merged}

# %% [markdown]
# ## Compare on 8 contracts: accuracy, time and model calls

# %%
sample = ["msa_01", "msa_05", "msa_07", "msa_09", "msa_12", "msa_14", "msa_15", "msa_18"]
res = {"single+checker": [], "team of 5": []}
times = {k: 0.0 for k in res}
for cid in sample:
    gold = {i["rule"] for i in labels()[cid]["issues"]}
    for name, fn in [("single+checker", single_agent), ("team of 5", team)]:
        t0 = time.time(); pred = fn(cid); times[name] += time.time() - t0
        res[name].append(prf(pred, gold))
calls = {"single+checker": 1, "team of 5": 5}
for name, rows in res.items():
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn_ = sum(r["fn"] for r in rows)
    print(f"{name:15} precision {tp/max(tp+fp,1):.0%}  recall {tp/max(tp+fn_,1):.0%}  "
          f"time {times[name]:.0f}s  model calls/contract {calls[name]}")

# %% [markdown]
# ## When a team does make sense
# Separable, wide work: 20 independent contracts at once. Each worker gets a written brief (goal, output format, limits, when to stop) and returns a short result. Here the "team" is parallel copies of the single agent.

# %%
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    batch = dict(zip(ids(), ex.map(single_agent, ids())))
print(f"reviewed {len(batch)} contracts in {time.time()-t0:.0f}s")
print({k: sorted(v) for k, v in list(batch.items())[6:10]})

# %%
s_tp = sum(r["tp"] for r in res["single+checker"]); s_fn = sum(r["fn"] for r in res["single+checker"])
check("single agent + checker finds most issues (recall >= 70%)", s_tp / max(s_tp + s_fn, 1) >= 0.7)

# %% [markdown]
# ## Your turn
# 1. Remove the code checker from Design A. How much does recall change on the renewal rules?
# 2. Give the team a vague brief ("review this") instead of a specific one. What breaks?
#
# **Next:** 10 · Evaluation

# %% [markdown]
# ## Solutions

# %%
# Solution 1: Design A without the code checker
def single_agent_no_checker(cid):
    secs = {s["title"]: s["text"] for s in sections(contract(cid))}
    relevant = "\n".join(f"{t}: {x}" for t, x in secs.items() if any(w in t for w in ("Liability", "Governing", "Renewal")))
    v = structured(f"Playbook rules:\n{json.dumps(PB['rules'], indent=1)}\n\nClauses:\n{relevant}\n\n"
                   "For each rule, return true if the contract BREAKS it, false if it follows it.", Verdicts)
    return {k for k, broken in v.model_dump().items() if broken}

for name, fn in [("with checker", single_agent), ("without checker", single_agent_no_checker)]:
    rows = [prf({r for r in fn(c) if r in ("auto_renewal", "termination_notice")},
                {i["rule"] for i in labels()[c]["issues"] if i["rule"] in ("auto_renewal", "termination_notice")}) for c in sample]
    tp = sum(r["tp"] for r in rows); fn_ = sum(r["fn"] for r in rows); fp = sum(r["fp"] for r in rows)
    print(f"{name:16} renewal rules: recall {tp/max(tp+fn_,1):.0%}, false flags {fp}")

# %%
# Solution 2: the team with a vague brief
def vague_specialist(args):
    cid, rule = args
    clause = next(s for s in sections(contract(cid)) if RULE_SECTION[rule] in s["title"])["text"]
    return rule, ask(f"Review this.\n{clause}")

def vague_team(cid):
    with ThreadPoolExecutor(4) as ex:
        reports = list(ex.map(vague_specialist, [(cid, r) for r in PB["rules"]]))
    merged = ask("Combine these reports into a list of problems.\n\n" + "\n\n".join(t for _, t in reports))
    return {r for r in PB["rules"] if r in merged}, merged

pred, merged = vague_team("msa_15")
print("found:", pred, "| truth:", {i["rule"] for i in labels()["msa_15"]["issues"]})
print(merged[:500])
# With no rule, no output format and no stopping point, the specialists comment on anything and the merger
# can't map their prose back to rule names. Most multi-agent failures are this kind of design problem.
