# %% [markdown]
# # 01 · What an agent is: build the loop from scratch
# **Local AI Lab** · CS329Z Lecture 1 · [Module 01](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m1)
#
# An agent is an LLM in a loop: it picks a step, calls a tool, sees the result, and repeats until it can answer. We build one in about 40 lines with no framework, then build the same task as a fixed **workflow** and compare.

# %% [markdown]
# ## 1. Tools the agent can use

# %%
import json, re, time
# lal.llm = the OpenAI client pointed at the local Ollama server, http://localhost:11434/v1 (see notebook 00)
from lal.llm import client, ask
from lal.data import contract, clauses, sections, labels

def list_sections(contract_id: str) -> str:
    """Section headings with numbers."""
    return "\n".join(f"{s['id']}. {s['title']}" for s in sections(contract(contract_id)))

def get_clause(contract_id: str, clause: str) -> str:
    """One numbered clause, e.g. '7.1'."""
    cl = clauses(contract(contract_id))
    return cl.get(clause, f"NOT FOUND. Valid clauses: {', '.join(cl)}")

TOOLS = [
    {"type": "function", "function": {"name": "list_sections",
        "description": "List the section headings of a contract with their numbers. Call this first.",
        "parameters": {"type": "object", "properties": {"contract_id": {"type": "string"}}, "required": ["contract_id"]}}},
    {"type": "function", "function": {"name": "get_clause",
        "description": "Read one clause by number, for example '7.1' or '10.1'.",
        "parameters": {"type": "object", "properties": {"contract_id": {"type": "string"}, "clause": {"type": "string"}},
                       "required": ["contract_id", "clause"]}}},
]
IMPL = {"get_clause": get_clause, "list_sections": list_sections}
print(list_sections("msa_07"))

# %% [markdown]
# ## 2. The agent loop
# Four safety limits matter most: a **turn cap**, a **stuck detector** (the same call twice), **errors returned as results** so the model can fix its own mistakes, and a **format check**: small local models sometimes *write* a tool call as text instead of making it, so we send that back as an error.

# %%
def run_agent(task, max_turns=8, verbose=True):
    messages = [{"role": "system", "content": "You review contracts. Use the tools to read clauses before answering. Cite clause numbers."},
                {"role": "user", "content": task}]
    seen, calls = set(), 0
    for turn in range(max_turns):                                   # limit 1: turn cap
        msg = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            if '"name"' in (msg.content or "") and turn < max_turns - 1:     # limit 4: a tool call written as text
                messages.append({"role": "user", "content": "FORMAT ERROR: you wrote a tool call as text. "
                                 "Use the tool-calling interface, or give your final answer in plain words."})
                continue
            return {"answer": msg.content, "turns": turn + 1, "tool_calls": calls}
        for call in msg.tool_calls:
            calls += 1
            key = (call.function.name, call.function.arguments)
            if key in seen:                                         # limit 2: stuck detector
                result = "ERROR: you already made this exact call. Use the result you have, or answer."
            else:
                seen.add(key)
                try:
                    result = IMPL[call.function.name](**json.loads(call.function.arguments))
                except Exception as e:                              # limit 3: errors become results
                    result = f"ERROR: {type(e).__name__}: {e}. Check the tool name and arguments."
            if verbose:
                print(f"  turn {turn+1}: {call.function.name}({call.function.arguments}) -> {str(result)[:90]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return {"answer": "STOPPED: turn limit reached.", "turns": max_turns, "tool_calls": calls}

t0 = time.time()
out = run_agent("In contract msa_07, what is the liability cap? Quote the clause.")
print("\nANSWER:", out["answer"])
print(f"{out['turns']} turns, {out['tool_calls']} tool calls, {time.time()-t0:.1f}s")

# %% [markdown]
# ## 3. The same task as a workflow
# We already know where the liability clause is, so code fetches it and the model only reads one clause. Fewer decisions for the model means fewer ways to fail.

# %%
def cap_workflow(cid):
    sec = next(s for s in sections(contract(cid)) if "Liability" in s["title"])     # code finds the clause
    return ask(f"{sec['text']}\n\nWhat is the liability cap, in months of fees? "
               f"Answer with just a number, or 'none' if liability is unlimited.")   # model reads it

t0 = time.time()
print(cap_workflow("msa_07"), f"({time.time()-t0:.1f}s)")
print("Truth:", labels()["msa_07"]["terms"]["liability_cap_months"])

# %% [markdown]
# ## 4. Compare on five contracts
# This is the course's main lesson in miniature: **try the simple workflow first.**

# %%
def to_months(text):
    if not text: return "?"
    if re.search(r"\b(none|unlimited|no cap|not limited)\b", text.lower()): return None
    for word, n in {"twelve": 12, "twenty-four": 24, "thirty-six": 36}.items():
        if word in text.lower(): return n
    m = re.search(r"\b(12|24|36)\b", text)
    return int(m.group(1)) if m else "?"

rows = []
for cid in ["msa_01", "msa_07", "msa_08", "msa_15", "msa_18"]:
    truth = labels()[cid]["terms"]["liability_cap_months"]
    t0 = time.time(); a = run_agent(f"In contract {cid}, what is the liability cap in months of fees? Answer with a number or 'none'.", verbose=False); ta = time.time()-t0
    t0 = time.time(); w = cap_workflow(cid); tw = time.time()-t0
    rows.append((cid, truth, to_months(a["answer"]) == truth, round(ta, 1), to_months(w) == truth, round(tw, 1)))
print(f"{'contract':9} {'truth':>5} {'agent ok':>9} {'secs':>5} {'workflow ok':>12} {'secs':>5}")
for r in rows:
    print(f"{r[0]:9} {str(r[1]):>5} {str(r[2]):>9} {r[3]:>5} {str(r[4]):>12} {r[5]:>5}")

# %%
from lal.checks import check
check("agent loop returns an answer", bool(out["answer"]))
check("workflow correct on at least 4 of 5", sum(r[4] for r in rows) >= 4)

# %% [markdown]
# ## Your turn
# 1. Remove the stuck detector and ask something the tools can't answer ("What is the vendor's tax ID?"). What happens?
# 2. Add a `search_text(contract_id, phrase)` tool. Does the agent use it?
#
# **Next:** 02 · LLMs for builders

# %% [markdown]
# ## Solutions

# %%
# Solution 1: the same loop WITHOUT the stuck detector, on a question the tools can't answer
def run_agent_no_stuck(task, max_turns=8):
    messages = [{"role": "system", "content": "You review contracts. Use the tools to read clauses before answering. Cite clause numbers."},
                {"role": "user", "content": task}]
    for turn in range(max_turns):
        msg = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return {"answer": msg.content, "turns": turn + 1}
        for call in msg.tool_calls:
            try:
                result = IMPL[call.function.name](**json.loads(call.function.arguments))
            except Exception as e:
                result = f"ERROR: {e}"
            print(f"  turn {turn+1}: {call.function.name}({call.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return {"answer": "STOPPED: turn limit reached.", "turns": max_turns}

q = "What is the vendor's tax ID in contract msa_07?"
print("WITHOUT stuck detector:", run_agent_no_stuck(q))
print("\nWITH stuck detector:   ", run_agent(q, verbose=False))
# What to notice: without the detector, small models often re-read the same clauses until the turn cap.
# With it, the model is told to stop repeating and usually answers "not in the contract". The turn cap is the last safety net.

# %%
# Solution 2: add a search_text tool
STOP = {"of", "the", "a", "an", "to", "in", "for", "and", "or", "is", "how", "many", "what"}
def search_text(contract_id: str, phrase: str) -> str:
    """Return every clause containing ALL the meaningful words of the phrase ("days of notice" finds "days' prior written notice")."""
    words = [w for w in re.findall(r"[a-z0-9]+", phrase.lower()) if w not in STOP]
    hits = [f"{num}: {text}" for num, text in clauses(contract(contract_id)).items() if all(w in text.lower() for w in words)]
    return "\n".join(hits) or "No clause contains those words. Try one keyword."

TOOLS.append({"type": "function", "function": {"name": "search_text",
    "description": "Find clauses in one contract that contain all the given keywords, e.g. 'notice renewal' or 'governing law'. Faster than reading every clause.",
    "parameters": {"type": "object", "properties": {"contract_id": {"type": "string"}, "phrase": {"type": "string"}},
                   "required": ["contract_id", "phrase"]}}})
IMPL["search_text"] = search_text

out = run_agent("In contract msa_13, how many days of notice are needed to stop the renewal?")
print("\nANSWER:", out["answer"], "| truth:", labels()["msa_13"]["terms"]["notice_days"])
