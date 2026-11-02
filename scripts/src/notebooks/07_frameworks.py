# %% [markdown]
# # 07 · Frameworks and harnesses: DSPy, LangGraph, and crash-safe steps
# **Local AI Lab** · CS329Z Lecture 5 · [Module 07](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m7)
#
# The harness is all the code around the model. For smaller models it matters as much as the model: in one study, the same small model scored 31% to 48% depending only on the harness. Here we use the two frameworks the course teaches, both pointed at our local model.

# %%
# EXTRA: dspy langgraph langchain-openai
import json, os, sqlite3, hashlib, time
import dspy
from lal.data import contract, sections, labels, ids
from lal.checks import check

# %% [markdown]
# ## 1. DSPy: declare inputs and outputs, not prompts
# A **signature** says what goes in and what comes out, with types. A **module** composes signatures with plain Python.

# %%
lm = dspy.LM(f"openai/{MODEL}", api_base="http://localhost:11434/v1", api_key="local", temperature=0)
dspy.configure(lm=lm)

class ExtractCap(dspy.Signature):
    """Read a limitation-of-liability clause and extract the cap."""
    clause: str = dspy.InputField()
    cap_months: int | None = dspy.OutputField(desc="months of fees the cap equals; None if liability is unlimited")
    favorable_to_customer: bool = dspy.OutputField()

extract = dspy.Predict(ExtractCap)
s = next(x for x in sections(contract("msa_08")) if "Liability" in x["title"])
p = extract(clause=s["text"])
print(p)
print("truth:", labels()["msa_08"]["terms"]["liability_cap_months"])

# %% [markdown]
# A module mixes model calls with Python logic. Here code decides whether the cap breaks the playbook.

# %%
class CapReview(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought(ExtractCap)
    def forward(self, cid):
        s = next(x for x in sections(contract(cid)) if "Liability" in x["title"])
        p = self.extract(clause=s["text"])
        issue = p.cap_months is None or p.cap_months > 12            # the rule lives in code
        return dspy.Prediction(cap_months=p.cap_months, issue=issue)

review = CapReview()
for cid in ["msa_02", "msa_20"]:
    print(cid, review(cid=cid))
print("\nThe exact prompt DSPy sent (always read it):")
lm.inspect_history(n=1)

# %% [markdown]
# ## 2. LangGraph: state, nodes and edges
# A graph with a fixed path plus one decision made by code: if extraction fails, skip straight to "needs human".

# %%
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=MODEL, base_url="http://localhost:11434/v1", api_key="local", temperature=0)

class State(TypedDict, total=False):
    cid: str; clause: str; law: str; status: str

def get_law_clause(state):
    s = next(x for x in sections(contract(state["cid"])) if "Governing" in x["title"])
    return {"clause": s["text"]}

def read_law(state):
    out = llm.invoke(f"{state['clause']}\nWhich US state's law governs? Reply with the state name only.").content.strip()
    return {"law": out.rstrip(".")}

def decide(state):
    return "check" if state["law"] and len(state["law"]) < 20 else "human"

def check_law(state):
    return {"status": "OK" if state["law"] in ("California", "Delaware") else "ISSUE"}

def human(state):
    return {"status": "NEEDS HUMAN"}

g = StateGraph(State)
for name, fn in [("get", get_law_clause), ("read", read_law), ("check", check_law), ("human", human)]:
    g.add_node(name, fn)
g.set_entry_point("get")
g.add_edge("get", "read")
g.add_conditional_edges("read", decide, {"check": "check", "human": "human"})
g.add_edge("check", END); g.add_edge("human", END)
app = g.compile()

for cid in ["msa_04", "msa_10"]:
    print(cid, app.invoke({"cid": cid}))

# %% [markdown]
# ## 3. Crash-safe steps (the idea behind Temporal)
# Long jobs crash: the GPU runs out of memory, the runtime disconnects. Record each finished step. On restart, reuse recorded results instead of calling the model again. The course demo used Temporal for this; here is the same idea in 20 lines with SQLite.

# %%
db = sqlite3.connect("steps.sqlite")
db.execute("create table if not exists steps (key text primary key, result text)")

def durable(step_name):
    def wrap(fn):
        def inner(*args):
            key = hashlib.sha256(json.dumps([step_name, args]).encode()).hexdigest()
            row = db.execute("select result from steps where key=?", (key,)).fetchone()
            if row:
                return json.loads(row[0])                            # already done: no repeat model call
            out = fn(*args)
            db.execute("insert into steps values (?,?)", (key, json.dumps(out))); db.commit()
            return out
        return inner
    return wrap

@durable("law")
def law_step(cid):
    return app.invoke({"cid": cid})["status"]

t0 = time.time(); first = [law_step(c) for c in ids()[:6]]; t1 = time.time() - t0
t0 = time.time(); again = [law_step(c) for c in ids()[:6]]; t2 = time.time() - t0
print(f"first run {t1:.1f}s, rerun after a 'crash' {t2:.2f}s, same results: {first == again}")

# %%
check("DSPy extraction returns an int or None", p.cap_months is None or isinstance(p.cap_months, int))
check("LangGraph flags Florida law in msa_10", app.invoke({"cid": "msa_10"})["status"] == "ISSUE")
check("rerun reused recorded steps", t2 < t1)

# %% [markdown]
# ## Your turn
# 1. Use `dspy.ReAct("question -> answer", tools=[...])` with `get_clause` from notebook 01 as a tool. One line gives you an agent.
# 2. Add a LangGraph node that drafts a redline only when status is ISSUE.
#
# **Next:** 08 · Memory
