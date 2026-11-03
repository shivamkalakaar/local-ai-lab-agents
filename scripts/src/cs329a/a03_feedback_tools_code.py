# %% [markdown]
# # a03 · Learning from feedback: tools, code execution and a constitution
# **Local AI Lab** · CS329A Lecture 4 (Oct 3) · Agents get better when the world talks back: test results, tool outputs, written rules.
#
# A model guessing alone makes the same mistakes again and again. In this notebook the model gets three kinds of outside signal:
# **unit-test results** for code it writes (the idea behind RLEF), **tool results** inside a Thought/Action/Observation loop (ReAct),
# and **a written constitution** it uses to critique and fix its own answers (Constitutional AI). Last, we turn those fixes into
# preference data, the raw material for training.

# %%
import json, random, re, time
from concurrent.futures import ThreadPoolExecutor
from lal.llm import ask
from lal.data import contract, clauses, sections, labels, playbook
from lal.codetasks import tasks, task, run_tests, extract_code, run_python
from lal.checks import check

CALLS = [0]                                   # count model calls; this notebook aims for about 60-80
def llm(prompt, **kw):
    CALLS[0] += 1
    try:
        return ask(prompt, **kw)
    except Exception as e:                    # a server hiccup should not kill the whole notebook
        return f"ERROR: {e}"

TASKS = tasks()
print(len(TASKS), "coding tasks:", [t["id"] for t in TASKS])
print("\nExample prompt:\n" + TASKS[4]["prompt"])

# %% [markdown]
# ## 1. A grader the model cannot argue with
# Each task has a reference solution and hidden unit tests (some generated with a fixed seed). `run_tests` runs code in a
# **separate Python process with a timeout**, so a bad or endless program cannot hurt the notebook. It returns
# `(passed, total, feedback)`, and the feedback is short text we can show the model.

# %%
t = task("second_largest")
print("reference:", run_tests(t["reference"], t)[:2])
buggy = "def second_largest(nums):\n    return sorted(nums)[-2]\n"
p, n, fb = run_tests(buggy, t)
print("buggy:", p, "/", n)
print(fb)
print("endless loop:", run_tests("def second_largest(nums):\n    while True: pass\n", t, timeout=2))

# %% [markdown]
# ## 2. Execution feedback, at inference time (the idea behind RLEF)
# **RLEF** (Gehring et al. 2024, *RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning*) sets code
# writing up as a conversation: the model writes code, sees the results on the **public** tests, and tries again, up to 3
# attempts. Grading uses **private** tests it never sees. Their key finding: off-the-shelf Llama 3.1 Instruct models barely use
# the feedback (8B went from 11.8% single-turn to 10.5% multi-turn on CodeContests). After RL training on this loop, the 8B
# model reached 16.0% and the 70B reached 40.1%.
#
# We can't run RL here, but we can run the same loop and measure it. Our prompt examples are too few to catch bugs, so we give
# the loop a **dev set**: the prompt examples plus half of the hidden tests. We grade on **all** tests and also report the
# **held-out** half separately. If a fix only passes the tests it was shown, the held-out number tells us.

# %%
def split(t):
    hidden = t["tests"][len(t["public_tests"]):]
    dev = dict(t, public_tests=t["public_tests"] + hidden[::2])          # what the loop may see
    held = dict(t, tests=t["public_tests"] + hidden[1::2])               # never shown to the model
    return dev, held

SYSTEM = "You write correct, simple Python. Reply with one ```python code block containing the function, and nothing else."

def solve_with_feedback(t, max_turns=3, temperature=0):
    """Turn 1 is a normal one-shot attempt. Later turns see the dev-test failures."""
    dev, held = split(t)
    history, code, log = t["prompt"], "", []
    for turn in range(1, max_turns + 1):
        reply = llm(history, system=SYSTEM, temperature=temperature, max_tokens=700)
        code = extract_code(reply)
        dp, dn, fb = run_tests(code, dev, public=True)                    # feedback signal
        ap, an, _ = run_tests(code, t)                                    # grade: all tests
        log.append({"turn": turn, "dev_ok": dp == dn, "all_ok": ap == an, "held_ok": run_tests(code, held)[0] == len(held["tests"])})
        if dp == dn:                                                      # stop when the visible tests pass (as in RLEF)
            break
        history = (f"{t['prompt']}\n\nYour previous attempt:\n```python\n{code}```\n"
                   f"Running it on tests gave:\n{fb}\nFix the code. Give it another try.")
    return code, log

t0 = time.time()
with ThreadPoolExecutor(2) as ex:
    fb_runs = dict(zip([t["id"] for t in TASKS], ex.map(solve_with_feedback, TASKS)))
print(f"{time.time()-t0:.0f}s, {CALLS[0]} calls so far")
for tid, (_, log) in fb_runs.items():
    print(f"{tid:18s}", " -> ".join(("PASS" if r["all_ok"] else "fail") for r in log))

# %% [markdown]
# Pass rate **after each turn**: a task counts as solved at turn k if its last attempt up to turn k passes all tests.

# %%
def pass_by_turn(runs, max_turns=3):
    out = []
    for k in range(1, max_turns + 1):
        solved = sum((log[min(k, len(log)) - 1]["all_ok"]) for _, log in runs.values())
        out.append(solved / len(runs))
    return out

fb_curve = pass_by_turn(fb_runs)
for k, r in enumerate(fb_curve, 1):
    print(f"after turn {k}: {r:.0%} of tasks pass all tests")
fixed = [tid for tid, (_, log) in fb_runs.items() if len(log) > 1]
print("tasks that needed feedback:", fixed)
print("of the final answers, how many pass the held-out half:",
      sum(log[-1]["held_ok"] for _, log in fb_runs.values()), "/", len(fb_runs))

# %% [markdown]
# ### The fair baseline: same budget, no feedback
# RLEF compares against **independent sampling**: just try again from scratch, at a higher temperature, without showing the
# errors. If feedback doesn't beat this, the model is not really using the feedback. We give each task that failed turn 1
# the same extra attempts, with the dev tests picking which attempt to keep.

# %%
def resample(t, extra=2):
    dev, _ = split(t)
    for i in range(extra):
        code = extract_code(llm(t["prompt"], system=SYSTEM, temperature=0.8, max_tokens=700))
        if run_tests(code, dev, public=True)[0] == len(dev["public_tests"]):
            break
    return run_tests(code, t)[0] == len(t["tests"])

failed_first = [t for t in TASKS if not fb_runs[t["id"]][1][0]["all_ok"]]
with ThreadPoolExecutor(2) as ex:
    rs = list(ex.map(resample, failed_first))
first_turn = sum(log[0]["all_ok"] for _, log in fb_runs.values())
print(f"turn 1 (shared):            {first_turn}/{len(TASKS)}")
print(f"+ 2 turns with feedback:     {sum(log[-1]['all_ok'] for _, log in fb_runs.values())}/{len(TASKS)}")
print(f"+ 2 fresh samples, no feedback: {first_turn + sum(rs)}/{len(TASKS)}")
print(CALLS[0], "calls so far")

# %% [markdown]
# On 10 small tasks the gap is noisy, so read it as a sign, not proof. RLEF's point is that the gap is small for untrained
# models and grows after RL on exactly this loop. To train, you would log these episodes: (prompt, attempts, feedback,
# final reward = all private tests pass).

# %% [markdown]
# ## 3. ReAct: think, act, observe
# **ReAct** (Yao et al. 2022, *ReAct: Synergizing Reasoning and Acting in Language Models*) has the model write a **Thought**,
# then an **Action** (a tool call), then read the **Observation** (the tool result), and repeat. On HotpotQA, chain-of-thought
# answers were wrong because of made-up facts in 56% of the failures the authors studied, against 0% for ReAct, which looks facts up.
#
# Our task needs two lookups and one multiplication: the **dollar** liability cap = cap months × monthly fee. Tools: read the
# section list, read a clause, and a calculator that runs in a subprocess.

# %%
def tool_sections(cid):
    try:
        return "; ".join(f"{s['id']}. {s['title']}" for s in sections(contract(cid)))
    except Exception:
        return f"ERROR: unknown contract id {cid!r}"

def tool_clause(cid, num):
    try:
        cl = clauses(contract(cid))
    except Exception:
        return f"ERROR: unknown contract id {cid!r}"
    return cl.get(num.strip(), f"NOT FOUND. Valid clauses: {', '.join(cl)}")

def tool_calc(expr):
    if not re.fullmatch(r"[\d\s+\-*/().,]+", expr):                         # numbers and operators only
        return "ERROR: calculator accepts only numbers and + - * / ( )"
    out, err = run_python(f"print({expr.replace(',', '')})", timeout=3)       # still in a subprocess
    return out.strip() or f"ERROR: {err.strip()[-200:]}"

REACT = """Answer the question by thinking step by step and using tools. Use EXACTLY this format, one step per reply:
Thought: <your reasoning>
Action: <one of: sections[contract_id] | clause[contract_id, clause_number] | calculate[expression] | finish[answer]>

Tools:
- sections[msa_01] lists the section headings.
- clause[msa_01, 2.1] returns the text of clause 2.1.
- calculate[12 * 4500] does arithmetic.
- finish[54000] gives the final answer (a plain number).
"""

def react(question, max_steps=5, verbose=True):
    transcript = f"{REACT}\nQuestion: {question}\n"
    for step in range(max_steps):
        reply = llm(transcript, temperature=0, max_tokens=250, stop=["Observation:"])
        m = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, flags=re.S)
        thought = (re.search(r"Thought:\s*(.*)", reply) or re.search(r"(.*)", reply)).group(1)[:120]
        if not m:
            obs = "FORMAT ERROR: end with a line 'Action: tool[arguments]'."
        else:
            name, arg = m.group(1).lower(), m.group(2)
            if name == "finish":
                if verbose: print(f"  {step+1}. finish[{arg}]")
                return arg, step + 1
            parts = [a.strip(" '\"") for a in arg.split(",", 1)]
            obs = {"sections": lambda: tool_sections(parts[0]),
                   "clause": lambda: tool_clause(parts[0], parts[1]) if len(parts) == 2 else "ERROR: use clause[contract_id, number]",
                   "calculate": lambda: tool_calc(arg)}.get(name, lambda: f"ERROR: unknown tool {name}")()
        if verbose: print(f"  {step+1}. {thought} | {m.group(0) if m else '??'} -> {str(obs)[:80]}")
        transcript += reply.strip() + f"\nObservation: {obs}\n"
    return None, max_steps

def to_number(s):
    try:
        return float(re.sub(r"[^\d.]", "", str(s).split()[0] if s else "") or "nan")
    except Exception:
        return float("nan")

L = labels()
react_ids = ["msa_07", "msa_08", "msa_20"]
react_results = {}
for cid in react_ids:
    truth = L[cid]["terms"]["liability_cap_months"] * L[cid]["terms"]["monthly_fee"]
    print(f"{cid} (truth ${truth:,})")
    ans, steps = react(f"In contract {cid}, what is the liability cap in dollars? "
                       "(The cap is a number of months of fees; multiply by the monthly fee.)")
    react_results[cid] = to_number(ans) == truth
    print("  ->", ans, "| correct:", react_results[cid])
print(CALLS[0], "calls so far")

# %% [markdown]
# Notice where each fact came from: the clause text, not the model's memory. When a step fails (wrong clause number, bad
# format), the error comes back as an observation and the model can recover. That is "learning from feedback" inside one episode.

# %% [markdown]
# ## 4. Constitutional AI: critique and revise against written rules
# **Constitutional AI** (Bai et al. 2022, *Constitutional AI: Harmlessness from AI Feedback*) replaces most human labels
# with a short list of principles. Phase 1 (supervised): the model answers, **critiques** its answer against a randomly picked
# principle, and **revises**. The paper used 16 principles and 4 critique-revision rounds per prompt, then fine-tuned on the
# revisions. Phase 2 (RLAIF): a model compares two answers using a principle; those AI preferences train a reward model.
#
# Our constitution is for contract-review answers. Each rule is also something code can partly check.

# %%
PB = playbook()
CONSTITUTION = [
    "Cite the clause number (like 7.1) for every claim about the contract.",
    "Judge the contract ONLY against the firm playbook rules given. Do not give legal advice beyond the playbook; "
    "for anything else, say the team should ask counsel.",
    "Name every playbook rule the contract breaks, and say clearly whether the contract can be signed as written.",
    "Be short and plain: at most 6 sentences, no legal jargon.",
]

def contract_question(cid):
    return (f"Firm playbook rules: {json.dumps(PB['rules'])}\n\nContract {cid}:\n{contract(cid)}\n\n"
            "Question from the business team: can we sign this contract as written? What needs to change?")

def critique_revise(cid, rng):
    q = contract_question(cid)
    draft = llm(q + "\nAnswer briefly.", temperature=0.7, max_tokens=300)           # a typical first answer
    principle = rng.choice(CONSTITUTION)                                              # CAI samples one principle per step
    critique = llm(f"{q}\n\nAnswer to review:\n{draft}\n\nPrinciple: {principle}\n"
                   "Critique the answer ONLY against this principle. List concrete problems, or say 'No problems'.",
                   max_tokens=250)
    revision = llm(f"{q}\n\nOriginal answer:\n{draft}\n\nCritique:\n{critique}\n\n"
                   f"Rewrite the answer so it follows every rule below and fixes the critique.\nRules:\n- "
                   + "\n- ".join(CONSTITUTION) + "\nReturn only the new answer.", max_tokens=300)
    return {"cid": cid, "principle": principle, "draft": draft, "critique": critique, "revision": revision}

def score(answer, cid):
    """Code checks for the constitution: real citations, issue clauses cited, a clear verdict."""
    cited = set(re.findall(r"\b(\d{1,2}\.\d)\b", answer))
    valid = set(clauses(contract(cid)))
    issue_clauses = {i["clause"] for i in L[cid]["issues"]}
    return {"cites": len(cited & valid),
            "bad_cites": len(cited - valid),
            "issue_recall": len(cited & issue_clauses) / max(1, len(issue_clauses)),
            "verdict": bool(re.search(r"\b(cannot|can't|should not|not be signed|do not sign|don't sign|not sign)\b", answer, re.I))}

cai_ids = ["msa_15", "msa_17", "msa_20"]            # each breaks two playbook rules
rng = random.Random(0)
cai = [critique_revise(cid, rng) for cid in cai_ids]
for r in cai:
    r["s_draft"], r["s_rev"] = score(r["draft"], r["cid"]), score(r["revision"], r["cid"])
    print(f"=== {r['cid']} | principle: {r['principle'][:60]}...")
    print("DRAFT:", r["draft"][:400].replace("\n", " "), "\n  ", r["s_draft"])
    print("REVISED:", r["revision"][:400].replace("\n", " "), "\n  ", r["s_rev"], "\n")
print(CALLS[0], "calls so far")

# %% [markdown]
# ## 5. From revisions to preference data (RLAIF)
# Each (draft, revision) pair is a training example: *chosen* = revision, *rejected* = draft. In CAI's phase 2, a model is shown
# two answers and a principle and picks the better one; the paper turned the model's probabilities into soft labels and found
# clamping them to 40-60% worked best when the judge reasoned step by step first. We ask our model to judge, **shuffle the
# order** so position bias can't fake agreement, and compare its pick with our code score.

# %%
def code_total(s):
    return s["cites"] - 2 * s["bad_cites"] + 3 * s["issue_recall"] + s["verdict"]

def ai_judge(r, rng):
    a, b, flipped = r["draft"], r["revision"], False
    if rng.random() < 0.5:
        a, b, flipped = b, a, True
    principle = "Which answer better follows these rules? " + " ".join(CONSTITUTION)
    v = llm(f"{contract_question(r['cid'])}\n\nAnswer (A):\n{a}\n\nAnswer (B):\n{b}\n\n{principle}\n"
            "Think in one or two sentences, then end with exactly 'Choice: A' or 'Choice: B'.", max_tokens=200)
    m = re.findall(r"Choice:\s*\(?([AB])", v)
    pick = m[-1] if m else None
    if pick is None:
        return None
    return (pick == "A") == flipped          # True if the judge prefers the revision

pairs = []
for r in cai:
    prefers_rev = ai_judge(r, rng)
    code_prefers_rev = code_total(r["s_rev"]) >= code_total(r["s_draft"])
    pairs.append({"prompt": contract_question(r["cid"]), "chosen": r["revision"], "rejected": r["draft"],
                  "principle": r["principle"], "ai_prefers_revision": prefers_rev, "code_prefers_revision": code_prefers_rev})
    print(r["cid"], "| AI judge prefers revision:", prefers_rev, "| code score prefers revision:", code_prefers_rev)
print(f"\n{len(pairs)} preference pairs built. Total model calls: {CALLS[0]}")

# %% [markdown]
# A real pipeline would keep only pairs where the AI judge and the code checks agree, then train with DPO or a reward model
# plus RL. The constitution is the only human input, which makes it cheap to change: edit a rule, regenerate the data.

# %%
n_turn1, n_final = fb_curve[0], fb_curve[-1]
check("code grader: reference passes, buggy fails", run_tests(task("second_largest")["reference"], task("second_largest"))[0] == 11 and run_tests(buggy, task("second_largest"))[0] < 11)
check("feedback loop solves at least half the tasks", n_final >= 0.5)
n_all = sum(log[-1]["all_ok"] for _, log in fb_runs.values())
n_dev = sum(log[-1]["dev_ok"] for _, log in fb_runs.values())
check("passing the dev tests mostly means passing all tests (fixes generalize)", n_all >= 0.7 * n_dev)
check("ReAct gets at least one dollar cap right with tools", sum(react_results.values()) >= 1)
check("revisions cite real clauses at least as often as drafts",
      sum(r["s_rev"]["cites"] > 0 for r in cai) >= sum(r["s_draft"]["cites"] > 0 for r in cai))

# %% [markdown]
# ## Your turn
# 1. **Is the model reading the feedback?** RLEF tested this by sending *random* feedback. Re-run the feedback loop on the tasks
#    that failed turn 1, but replace the failure details with only "Some tests failed." Does the pass rate drop?
# 2. **Act without thinking.** Change the ReAct prompt so the model must skip the `Thought:` line (the paper's "Act" baseline) and
#    run it on one contract. Does it still find both clauses?
# 3. **Data, not just demos.** Write the preference pairs to `cai_pairs.jsonl` in the `{"prompt","chosen","rejected"}` format that
#    TRL's DPO trainer reads, keeping only pairs where the AI judge and the code score agree.
#
# **Next:** a04 · Multi-step reasoning and planning

# %% [markdown]
# ## Solutions

# %%
# Solution 1: same loop, but the feedback hides WHICH tests failed and why.
def solve_blind(t, max_turns=3):
    dev, _ = split(t)
    history = t["prompt"]
    for turn in range(1, max_turns + 1):
        code = extract_code(llm(history, system=SYSTEM, temperature=0, max_tokens=700))
        if run_tests(code, dev, public=True)[0] == len(dev["public_tests"]):
            break
        history = f"{t['prompt']}\n\nYour previous attempt:\n```python\n{code}```\nSome tests failed. Give it another try."
    return run_tests(code, t)[0] == len(t["tests"])

retry = failed_first[:3]                       # small budget: at most 3 tasks x 3 calls
blind = [solve_blind(t) for t in retry]
full = [fb_runs[t["id"]][1][-1]["all_ok"] for t in retry]
print("solved with detailed feedback:", sum(full), "| with 'some tests failed':", sum(blind), "| of", len(retry))
# Notice: at temperature 0 with a vague message, the model often repeats its old code (RLEF saw this too in untrained models).

# %%
# Solution 2: Act-only, no thoughts.
ACT = REACT.replace("Thought: <your reasoning>\n", "").replace("thinking step by step and using tools", "using tools. Do not explain")
REACT_SAVED, REACT = REACT, ACT
ans, steps = react("In contract msa_08, what is the liability cap in dollars? (Cap months times the monthly fee.)")
REACT = REACT_SAVED
print("Act-only answer:", ans, "| truth:", L["msa_08"]["terms"]["liability_cap_months"] * L["msa_08"]["terms"]["monthly_fee"])
# Notice: without a thought step, small models more often finish after one lookup, or forget to multiply.

# %%
# Solution 3: keep only pairs where the AI judge and the code checks agree, then write DPO-format JSONL.
keep = [p for p in pairs if p["ai_prefers_revision"] is not None and p["ai_prefers_revision"] == p["code_prefers_revision"]]
with open("cai_pairs.jsonl", "w") as f:
    for p in keep:
        chosen, rejected = (p["chosen"], p["rejected"]) if p["ai_prefers_revision"] else (p["rejected"], p["chosen"])
        f.write(json.dumps({"prompt": p["prompt"], "chosen": chosen, "rejected": rejected}) + "\n")
print(f"kept {len(keep)} of {len(pairs)} pairs -> cai_pairs.jsonl")
# Notice: when both judges prefer the DRAFT, we flip chosen/rejected. A revision is not automatically better.
print("total model calls in this notebook:", CALLS[0])
