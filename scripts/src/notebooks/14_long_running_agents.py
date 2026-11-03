# %% [markdown]
# # 14 · Long-running agents: checklists, progress files and one step at a time
# **Local AI Lab** · Anthropic "Effective harnesses for long-running agents", SWE-agent · [Module 14](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m14)
#
# When a job is longer than one context window (a whole data room, 200 contracts), the agent's memory must be written down:
# 1. A **setup session** writes a checklist where every item starts as "not done".
# 2. Each **work session** reads the progress file, does **one** item, marks it done only with evidence, and saves.
# 3. The job can't finish until every item has evidence.

# %%
import json, shutil, subprocess
from pathlib import Path
from pydantic import BaseModel
from lal.llm import structured
from lal.data import contract, sections, ids, labels, playbook
from lal.checks import quote_is_in_doc, check

WORK = Path("matter_workspace"); shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir()
subprocess.run(["git", "init", "-q"], cwd=WORK)
subprocess.run(["git", "config", "user.email", "agent@local"], cwd=WORK); subprocess.run(["git", "config", "user.name", "agent"], cwd=WORK)
PB = playbook()

# %% [markdown]
# ## 1. Setup session: write the checklist
# Code writes it (one item per contract per rule). The agent may only flip `done` and attach evidence; it may never edit the list itself.

# %%
MATTER = ids()[12:16]           # 4 contracts x 4 rules = 16 items
items = [{"id": f"{c}:{r}", "contract": c, "rule": r, "done": False, "evidence": None}
         for c in MATTER for r in PB["rules"]]
(WORK / "checklist.json").write_text(json.dumps({"matter": "demo", "items": items}, indent=1))
(WORK / "progress.md").write_text("# Progress\n")
subprocess.run(["git", "add", "-A"], cwd=WORK); subprocess.run(["git", "commit", "-qm", "setup"], cwd=WORK)
print(len(items), "checklist items written")

# %% [markdown]
# ## 2. One work session = one item, with evidence that code verifies

# %%
WORD = {"liability_cap": "Liability", "governing_law": "Governing", "auto_renewal": "Renewal", "termination_notice": "Renewal"}

class Finding(BaseModel):
    breaks_rule: bool
    quote: str

def work_session():
    state = json.loads((WORK / "checklist.json").read_text())
    todo = next((i for i in state["items"] if not i["done"]), None)
    if todo is None:
        return "ALL DONE"
    clause = next(s for s in sections(contract(todo["contract"])) if WORD[todo["rule"]] in s["title"])["text"]
    f = structured(f"Rule: {PB['rules'][todo['rule']]}\nClause: {clause}\n"
                   "Does the clause break the rule? Quote the exact words that decide it.", Finding)
    if not quote_is_in_doc(f.quote, contract(todo["contract"])):             # evidence must be real
        with open(WORK / "progress.md", "a") as p: p.write(f"- {todo['id']}: evidence failed check, retry later\n")
        return "RETRY"
    todo.update(done=True, evidence={"breaks_rule": f.breaks_rule, "quote": f.quote})
    (WORK / "checklist.json").write_text(json.dumps(state, indent=1))
    with open(WORK / "progress.md", "a") as p: p.write(f"- {todo['id']}: {'BREAKS' if f.breaks_rule else 'ok'}\n")
    subprocess.run(["git", "add", "-A"], cwd=WORK); subprocess.run(["git", "commit", "-qm", f"check {todo['id']}"], cwd=WORK)
    return todo["id"]

for _ in range(len(items) + 4):               # a stop rule: never loop forever
    r = work_session()
    if r == "ALL DONE": break
print(open(WORK / "progress.md").read())

# %% [markdown]
# ## 3. Every step is a commit you can review and replay

# %%
print(subprocess.run(["git", "log", "--oneline", "-n", "6"], cwd=WORK, capture_output=True, text=True).stdout)

# %% [markdown]
# ## 4. Score the finished matter against the truth

# %%
state = json.loads((WORK / "checklist.json").read_text())
correct = sum((i["evidence"] or {}).get("breaks_rule") == (i["rule"] in {x["rule"] for x in labels()[i["contract"]]["issues"]})
              for i in state["items"] if i["done"])
print(f"{sum(i['done'] for i in state['items'])}/{len(items)} items done, {correct} correct")

# %%
check("every checklist item finished with verified evidence", all(i["done"] and i["evidence"] for i in state["items"]))
check("at least 12 of 16 items judged correctly", correct >= 12)

# %% [markdown]
# ## Your turn
# 1. Kill the loop halfway (interrupt the cell) and rerun it. Does it pick up where it stopped?
# 2. Add a "stop hook": the job may only print a final report when every item has evidence.
#
# **Next:** 15 · Capstone

# %% [markdown]
# ## Solutions

# %%
# Solution 1: resume after a crash. Mark 3 items as not done (as if the run died) and rerun the loop.
state = json.loads((WORK / "checklist.json").read_text())
for i in state["items"][-3:]:
    i.update(done=False, evidence=None)
(WORK / "checklist.json").write_text(json.dumps(state, indent=1))
redone = []
for _ in range(10):
    r = work_session()
    if r == "ALL DONE": break
    redone.append(r)
print("redone after restart:", redone)
# Only the unfinished items ran again. The checklist file, not the model's memory, holds the progress.

# %%
# Solution 2: a stop hook. The final report refuses to run until every item has evidence.
def final_report():
    st = json.loads((WORK / "checklist.json").read_text())
    missing = [i["id"] for i in st["items"] if not (i["done"] and i["evidence"])]
    if missing:
        raise RuntimeError(f"Cannot finish: {len(missing)} items lack evidence: {missing[:5]}")
    broken = [i["id"] for i in st["items"] if i["evidence"]["breaks_rule"]]
    return f"Matter complete. {len(st['items'])} checks, {len(broken)} issues: {broken}"
print(final_report())
