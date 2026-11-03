# %% [markdown]
# # 06 · Design patterns: chaining, routing, parallel, checker loops, ReAct vs code actions
# **Local AI Lab** · CS329Z Lecture 6 · [Module 06](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m6)
#
# Five workflow patterns cover most needs. Start with the simplest one that works, and add complexity only when your eval says it helps.

# %%
import re, time
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import Literal
from lal.llm import ask, structured
from lal.data import contract, sections, labels, playbook, ids
from lal.checks import quote_is_in_doc, check

PB = playbook()
def section(cid, word):
    return next(s for s in sections(contract(cid)) if word in s["title"])

# %% [markdown]
# ## 1. Chaining: fixed steps with checks between them

# %%
class Cap(BaseModel):
    months: int | None
    quote: str

def chain(cid):
    s = section(cid, "Liability")
    cap = structured(f"{s['text']}\n\nExtract the cap in months of fees (null if unlimited) and quote the sentence.", Cap)   # step 1: LLM
    if not quote_is_in_doc(cap.quote, contract(cid)):                                                                     # gate: code
        return {"status": "HOLD", "why": "quote not found in contract"}
    ok = cap.months is not None and cap.months <= PB["max_cap_months"]                                                    # step 2: rule in code
    return {"status": "OK" if ok else "ISSUE", "months": cap.months}

for cid in ["msa_01", "msa_07", "msa_16"]:
    print(cid, chain(cid), "| truth:", labels()[cid]["terms"]["liability_cap_months"])

# %% [markdown]
# ## 2. Routing: classify first, then send to the right handler
# Easy inputs can go to a cheaper path. Here a cheap regex check handles most cases and the model handles the rest.

# %%
def route_law(cid):
    text = section(cid, "Governing")["text"]
    m = re.search(r"State of (\w+(?: \w+)?)", text)
    if m:                                         # easy case: code handles it, no model call
        return m.group(1).split(",")[0], "code"
    return ask(f"{text}\nWhich US state's law governs? State name only.").strip(), "model"

for cid in ["msa_03", "msa_09", "msa_15"]:
    print(cid, route_law(cid))

# %% [markdown]
# ## 3. Parallel: independent checks at the same time
# Separate checks don't depend on each other, so run them together. Ollama runs them in parallel when `OLLAMA_NUM_PARALLEL` > 1 (our setup sets 2); otherwise they queue.

# %%
def check_rule(args):
    cid, rule = args
    word = {"liability_cap": "Liability", "governing_law": "Governing", "auto_renewal": "Renewal", "termination_notice": "Renewal"}[rule]
    verdict = ask(f"Playbook rule: {PB['rules'][rule]}\n\nClause: {section(cid, word)['text']}\n\n"
                  f"Does the clause follow the rule? Answer PASS or FAIL only.").strip().upper()
    return rule, "FAIL" if "FAIL" in verdict else "PASS"

cid = "msa_15"
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(check_rule, [(cid, r) for r in PB["rules"]]))
print(results, f"{time.time()-t0:.1f}s")
print("truth issues:", [i["rule"] for i in labels()[cid]["issues"]])

# %% [markdown]
# ## 4. Generator + checker: draft, check with real checks, revise
# The checker is code first. Asking the same model "is this good?" without an outside signal doesn't help; the course showed it can make results worse.

# %%
class Redline(BaseModel):
    new_text: str
    months: int

def draft_redline(cid, rounds=3):
    clause = section(cid, "Liability")["text"]
    feedback = ""
    for r in range(rounds):
        d = structured(f"Rewrite this clause so the liability cap meets the playbook: {PB['rules']['liability_cap']}\n"
                       f"Clause: {clause}\n{feedback}\nReturn the new clause text and the cap in months.", Redline)
        problems = []
        stated = [int(n) for n in re.findall(r"(\d+)\)? months", d.new_text)]       # check the TEXT, not the model's own claim
        if not stated: problems.append("the new text doesn't state the cap as a number of months, like 'twelve (12) months'")
        elif max(stated) > PB["max_cap_months"]: problems.append(f"the text says {max(stated)} months; the playbook max is {PB['max_cap_months']}")
        if not problems:
            return d, r + 1
        feedback = "Fix these problems: " + "; ".join(problems)
    return d, rounds

d, n = draft_redline("msa_08")
print(f"after {n} round(s):", d.new_text)

# %% [markdown]
# ## 5. ReAct vs code-as-action (CodeAct)
# Instead of one JSON tool call per turn, the model writes a small Python snippet that does several steps at once. Fewer turns when a task needs loops. We run the code in a restricted namespace (a toy sandbox; real systems use containers).

# %%
def code_act(question):
    code = ask("You can use these Python functions: contract(cid) -> str, ids() -> list of ids, "
               "sections(text) -> list of {'id','title','text'}.\n"
               f"Write Python that computes the answer and stores it in a variable named result. Question: {question}\n"
               "Reply with only the code, no explanation, no backticks.")
    code = re.sub(r"^```\w*|```$", "", code.strip(), flags=re.M).strip()
    import builtins
    safe = {k: getattr(builtins, k) for k in ("len", "range", "sorted", "any", "all", "str", "int", "list", "set", "dict",
                                              "enumerate", "zip", "min", "max", "print", "isinstance")}
    g = {"__builtins__": safe, "contract": contract, "ids": ids, "sections": sections, "re": re}   # one namespace, so generators and defs see the helpers
    try:
        exec(code, g)
        return g.get("result"), code
    except Exception as e:
        return f"ERROR {e}", code

result, code = code_act("Which contract ids are governed by Texas, Florida or New York law?")
print(code, "\n->", result)
print("truth:", [c for c, l in labels().items() if l["terms"]["governing_law"] in ("Texas", "Florida", "New York")])

# %%
check("chain flags the 24-month cap in msa_07", chain("msa_07")["status"] == "ISSUE")
check("redline text meets the playbook", bool(re.findall(r"(\d+)\)? months", d.new_text)) and max(int(n) for n in re.findall(r"(\d+)\)? months", d.new_text)) <= PB["max_cap_months"])

# %% [markdown]
# ## Your turn
# 1. Add a sixth pattern, **plan then execute**: one call writes a list of checks, then code runs each check.
# 2. Replace the code checker in step 4 with "ask the model if the draft is good". Compare how often it catches a 24-month cap.
#
# **Next:** 07 · Frameworks (DSPy and LangGraph)

# %% [markdown]
# ## Solutions

# %%
# Solution 1: plan then execute. One call writes the plan; code runs every step.
class Plan(BaseModel):
    checks: list[Literal["liability_cap", "governing_law", "auto_renewal", "termination_notice"]]

import json
plan = structured(f"You will review a vendor services agreement against these playbook rules: {json.dumps(PB['rules'])}.\n"
                  "Return the list of rule checks to run.", Plan)
print("plan:", plan.checks)
results = [check_rule(("msa_15", r)) for r in dict.fromkeys(plan.checks)]       # code executes the plan
print("results:", results, "| truth:", [i["rule"] for i in labels()["msa_15"]["issues"]])

# %%
# Solution 2: "ask the model if it's good" vs a code checker, on drafts we KNOW are bad (cap > 12 months)
bad_drafts = [
    "Each party's total liability shall not exceed the fees paid in the twenty-four (24) months preceding the claim.",
    "Liability is capped at fees paid during the 18 months before the claim, a market-standard position.",
    "Total liability shall not exceed three (3) years of fees.",
]
def llm_checker(text):
    v = ask(f"Playbook: the cap must not exceed fees paid in the prior 12 months.\nClause: {text}\n"
            "Does the clause meet the playbook? Answer YES or NO only.").strip().upper()
    return v.startswith("YES")
def code_checker(text):
    months = [int(n) for n in re.findall(r"(\d+)\)? months", text)]
    years = [int(n) * 12 for n in re.findall(r"(\d+)\)? years", text)]
    vals = months + years
    return bool(vals) and max(vals) <= 12
print("LLM checker wrongly approved:", sum(llm_checker(t) for t in bad_drafts), "of", len(bad_drafts))
print("code checker wrongly approved:", sum(code_checker(t) for t in bad_drafts), "of", len(bad_drafts))
