# %% [markdown]
# # 03 · Context engineering
# **Local AI Lab** · CS329Z Lectures 2 and 6 · [Module 03](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m3)
#
# The context window is a small, expensive desk. Local models have smaller effective windows and slow down as the prompt grows. We'll measure that, then use the course's techniques: an append-only event log, observation masking, compaction, and pinned rules.

# %% [markdown]
# ## 1. Measure it: longer prompts are slower
# We ask the same question with 1, 5 and 10 contracts in the prompt.

# %%
import time
# lal.llm = the OpenAI client pointed at the local Ollama server, http://localhost:11434/v1 (see notebook 00)
from lal.llm import client, ask
from lal.data import contract, ids, labels

question = "What governing law does contract msa_09 use? Answer with the state name only."
for n in (1, 5, 10):
    docs = [cid for cid in ids() if cid != "msa_09"][: n - 1] + ["msa_09"]
    prompt = "\n\n".join(f"=== {cid} ===\n{contract(cid)}" for cid in docs) + "\n\n" + question
    t0 = time.time()
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    print(f"{n:2} contracts: {r.usage.prompt_tokens:6} prompt tokens, {time.time()-t0:5.1f}s, answer: {r.choices[0].message.content.strip()[:30]}")
print("Truth:", labels()["msa_09"]["terms"]["governing_law"])

# %% [markdown]
# ## 2. An append-only event log
# Store every step forever. Rebuild what the model sees from the log each turn. Then you can trim what the model sees without losing the record (which you need for audit and replay).

# %%
events = []   # the full truth, never edited
def log(role, content, kind="chat"):
    events.append({"role": role, "content": content, "kind": kind})

log("user", "Review contract msa_15 against the playbook.")
for sec in ["3", "4", "7", "9", "10"]:
    log("assistant", f"Reading section {sec}.")
    log("user", f"[tool result] section {sec}: " + contract("msa_15").split(f"\n{sec}. ")[1][:400], kind="tool")
print(len(events), "events,", sum(len(e["content"]) for e in events), "characters")

# %% [markdown]
# ## 3. Observation masking and pinned rules
# Old tool outputs are most of the tokens. Replace them with a one-line placeholder; keep the goal, the hard rules and the last few turns word for word. This needs no extra model call.

# %%
PINNED = """You review contracts for a law firm.
HARD RULES (never dropped by trimming):
- You never send, file or delete anything. You only write a review for a lawyer.
- Every issue must cite a clause number."""

def build_context(events, keep_recent=4):
    msgs = [{"role": "system", "content": PINNED}, events[0]]             # pinned rules + the user's goal
    for e in events[1:-keep_recent]:
        if e["kind"] == "tool":
            msgs.append({"role": e["role"], "content": f"[old tool output hidden, {len(e['content'])} chars]"})
        else:
            msgs.append(e)
    return msgs + events[-keep_recent:]

full = sum(len(e["content"]) for e in events)
trimmed = sum(len(m["content"]) for m in build_context(events))
print(f"full: {full} chars -> trimmed: {trimmed} chars ({1 - trimmed/full:.0%} smaller)")

# %% [markdown]
# ## 4. Compaction: summarize the middle when you must
# When masking isn't enough, summarize the middle of the history. Keep the pinned rules out of the summary so they can never be lost. (One real agent deleted a user's inbox after "confirm before acting" was summarized away.)

# %%
def compact(events, keep_recent=2):
    middle = "\n".join(e["content"] for e in events[1:-keep_recent])
    summary = ask("Summarize these contract-review notes in at most 5 short bullets. Keep clause numbers and numbers exact.\n\n" + middle)
    return [{"role": "system", "content": PINNED}, events[0],
            {"role": "assistant", "content": "Summary of earlier work:\n" + summary}] + events[-keep_recent:]

ctx = compact(events)
print(ctx[2]["content"])
print("\nPinned rules still present:", "never send" in ctx[0]["content"])

# %% [markdown]
# ## 5. Just-in-time loading
# Instead of pasting all 10 contracts, keep a list of IDs and load the one you need. Compare with step 1.

# %%
def answer_with_lookup(cid, q):
    return ask(f"{contract(cid)}\n\n{q}")
t0 = time.time()
print(answer_with_lookup("msa_09", question).strip(), f"({time.time()-t0:.1f}s)")

# %%
from lal.checks import check
check("masking makes the context smaller", trimmed < full)
check("pinned rules survive compaction", "never send" in ctx[0]["content"])

# %% [markdown]
# ## Your turn
# 1. Put the important question at the very start of the 10-contract prompt instead of the end. Does accuracy change?
# 2. Write a `notes.md` file the agent appends to after each section, and rebuild context from it.
#
# **Next:** 04 · Retrieval (RAG)

# %% [markdown]
# ## Solutions

# %%
# Solution 1: question at the START of a 10-contract prompt vs at the END
docs = [cid for cid in ids() if cid != "msa_09"][:9] + ["msa_09"]
body = "\n\n".join(f"=== {cid} ===\n{contract(cid)}" for cid in docs)
for where, prompt in [("question first", question + "\n\n" + body), ("question last", body + "\n\n" + question)]:
    answers = [ask(prompt).strip()[:25] for _ in range(2)]
    print(f"{where:15} -> {answers}")
print("truth:", labels()["msa_09"]["terms"]["governing_law"])
# Models often do better when the question is near the end (closest to where they start writing),
# and they can lose facts buried in the middle. Measure it on your own model rather than assuming.

# %%
# Solution 2: a notes file the agent appends to, and a context rebuilt from it
from pathlib import Path
from lal.data import sections
from lal.llm import chat
notes = Path("notes.md")
notes.write_text("# Review notes: msa_15\n")
for s in sections(contract("msa_15")):
    if s["title"] in ("Renewal", "Limitation Of Liability", "Governing Law"):
        line = ask(f"In one line, with exact numbers, what does this clause say?\n{s['text']}").strip()
        with open(notes, "a") as f:
            f.write(f"- {s['id']}. {s['title']}: {line}\n")
print(notes.read_text())
ctx = [{"role": "system", "content": PINNED},
       {"role": "user", "content": "Using only these notes, list any concerns against a playbook that requires CA or DE law "
                                   "and a cap of at most 12 months of fees:\n" + notes.read_text()}]
print(chat(ctx).content)
# The notes survive any context reset: a new session can start from notes.md instead of re-reading the whole contract.
