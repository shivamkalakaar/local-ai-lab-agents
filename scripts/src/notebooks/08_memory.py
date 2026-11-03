# %% [markdown]
# # 08 · Memory that learns from corrections without drifting
# **Local AI Lab** · CS329Z Lecture 7 · [Module 08](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m8)
#
# Keep memory simple, cited and approved. In the course's benchmarks, plain files and plain search beat dedicated memory products. For client work, an agent must never silently rewrite what it remembers.
#
# | Memory type | Example | Who may write |
# |---|---|---|
# | Episodic (what happened) | "Partner changed our flag on 10.1" | append-only log |
# | Semantic (lessons) | "This client accepts New York law" | agent proposes, person approves |
# | Procedural (how to work) | prompts, checklists | change control only |

# %%
import json, time, hashlib, shutil
from pathlib import Path
# lal.llm = the OpenAI client pointed at the local Ollama server, http://localhost:11434/v1 (see notebook 00)
from lal.llm import ask
from lal.data import contract, sections, labels
from lal.checks import check

MEM = Path("memory_demo"); shutil.rmtree(MEM, ignore_errors=True)
(MEM / "client_northwind" / "lessons").mkdir(parents=True)

# %% [markdown]
# ## 1. Episodic memory: an append-only corrections log

# %%
def log_correction(client, cid, rule, model_said, human_said, reason, by):
    rec = {"ts": time.time(), "client": client, "contract": cid, "rule": rule,
           "model": model_said, "human": human_said, "reason": reason, "by": by}
    rec["id"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:10]
    with open(MEM / client / "corrections.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec["id"]

# Suppose the agent flagged New York law three times, and the partner overrode it each time.
ev = [log_correction("client_northwind", c, "governing_law", "ISSUE", "OK",
                     "This client has an NY office; NY law is acceptable for them.", "partner_a") for c in ["msa_15", "x_101", "x_102"]]
print(open(MEM / "client_northwind" / "corrections.jsonl").read())

# %% [markdown]
# ## 2. Semantic memory: the agent proposes a lesson, citing evidence

# %%
def propose_lesson(client, text, evidence_ids):
    if not evidence_ids:
        raise ValueError("A lesson must cite the corrections that support it.")
    prop = {"client": client, "text": text, "evidence": evidence_ids, "status": "pending"}
    with open(MEM / client / "proposals.jsonl", "a") as f:
        f.write(json.dumps(prop) + "\n")
    return prop

corrections = [json.loads(l) for l in open(MEM / "client_northwind" / "corrections.jsonl")]
draft = ask("Write ONE short rule (max 20 words) a contract reviewer should follow for this client, based on these corrections:\n"
            + "\n".join(f"- {c['rule']}: model said {c['model']}, partner said {c['human']} because {c['reason']}" for c in corrections))
prop = propose_lesson("client_northwind", draft.strip(), [c["id"] for c in corrections])
print(prop)

# %% [markdown]
# ## 3. A person approves; memory is versioned, never overwritten

# %%
def approve(client, proposal, approver):
    d = MEM / client / "lessons"
    version = len(list(d.glob("v*.json"))) + 1
    current = json.loads((d / f"v{version-1}.json").read_text()) if version > 1 else []
    current.append({**proposal, "status": "approved", "by": approver})
    (d / f"v{version}.json").write_text(json.dumps(current, indent=2))
    return version

def load_lessons(client, version):
    return json.loads((MEM / client / "lessons" / f"v{version}.json").read_text())

v = approve("client_northwind", prop, "partner_a")
print("approved, memory version", v, "->", load_lessons("client_northwind", v))

# %% [markdown]
# ## 4. Use the lessons, scoped to one client
# The same contract, reviewed for two clients. Only Northwind's review uses Northwind's lesson. Each run records the memory version it used, so any decision can be replayed.

# %%
def review_law(cid, client=None, version=None):
    lessons = load_lessons(client, version) if client and version else []
    notes = "\n".join(f"- {l['text']}" for l in lessons)
    clause = next(s for s in sections(contract(cid)) if "Governing" in s["title"])["text"]
    verdict = ask(f"Playbook: governing law must be California or Delaware.\n"
                  f"{'Client-specific approved lessons:' + chr(10) + notes if notes else ''}\n\nClause: {clause}\n"
                  "Answer ISSUE or OK only.").strip().upper()
    return {"verdict": "ISSUE" if "ISSUE" in verdict else "OK", "client": client, "memory_version": version}

print("generic client:  ", review_law("msa_15"))
print("Northwind, v1:   ", review_law("msa_15", "client_northwind", v))

# %%
check("corrections log is append-only and has 3 records", len(corrections) == 3)
try:
    propose_lesson("client_northwind", "Always accept Texas law.", [])
    rejected = False
except ValueError:
    rejected = True
check("lessons without evidence are rejected", rejected)
check("each run records its memory version", review_law("msa_15", "client_northwind", v)["memory_version"] == v)

# %% [markdown]
# ## Your turn
# 1. Add a second lesson and approve it as version 2. Rerun a review with version 1 and version 2 and compare.
# 2. What should happen if a lesson from Northwind is retrieved while reviewing for another client? Add a guard.
#
# **Next:** 09 · Multi-agent systems

# %% [markdown]
# ## Solutions

# %%
# Solution 1: a second lesson as version 2, then compare reviews with v1 and v2
ev2 = [log_correction("client_northwind", c, "governing_law", "ISSUE", "OK", "Client also accepts Texas law for logistics vendors.", "partner_b")
       for c in ["msa_09", "x_201"]]
prop2 = propose_lesson("client_northwind", "Texas governing law is acceptable for this client.", ev2)
v2 = approve("client_northwind", prop2, "partner_b")
print("v1 lessons:", [l["text"] for l in load_lessons("client_northwind", 1)])
print("v2 lessons:", [l["text"] for l in load_lessons("client_northwind", v2)])
print("msa_09 with v1:", review_law("msa_09", "client_northwind", 1))
print("msa_09 with v2:", review_law("msa_09", "client_northwind", v2))
# Old runs can still be replayed with v1, so you can always explain an earlier decision.

# %%
# Solution 2: a scope guard so one client's lessons can never be used for another client
def load_lessons_for(review_client, lesson_client, version):
    if review_client != lesson_client:
        raise PermissionError(f"Lessons from {lesson_client} cannot be used when reviewing for {review_client}.")
    return load_lessons(lesson_client, version)

try:
    load_lessons_for("client_bayview", "client_northwind", v2)
except PermissionError as e:
    print("blocked:", e)
