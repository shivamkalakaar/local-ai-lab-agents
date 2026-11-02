# %% [markdown]
# # 12 · Data and flywheels: synthetic cases, traces and corrections
# **Local AI Lab** · CS329Z Lecture 11 · [Module 12](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m12)
#
# Every run, every correction and every failure is data. Two techniques from the course:
# 1. **Synthetic cases with known answers** (SWE-smith): take a clean document, inject a known problem in code, and the grader knows the answer by construction.
# 2. **The flywheel:** log every run, turn corrections into tests at once, and into training data later.

# %%
import json, random, re, time
from pathlib import Path
from lal.llm import client
from lal.data import contract, labels, playbook
from lal.checks import check

# %% [markdown]
# ## 1. Inject faults into a clean contract

# %%
FAULTS = {
    "liability_cap":      lambda t: re.sub(r"IN THE 12 MONTHS", "IN THE FORTY-EIGHT (48) MONTHS", t),
    "governing_law":      lambda t: re.sub(r"State of (California|Delaware)", "State of Nevada", t),
    "auto_renewal":       lambda t: t.replace("successive one (1) year", "successive four (4) year"),
    "termination_notice": lambda t: re.sub(r"at least [\w\s()]+? days'", "at least seven (7) days'", t),
}

def make_case(clean_id, rng):
    text = contract(clean_id)
    picked = rng.sample(list(FAULTS), rng.choice([1, 2]))
    for f in picked:
        new = FAULTS[f](text)
        assert new != text, f"fault {f} did not apply"
        text = new
    return {"text": text, "issues": sorted(picked), "source": clean_id}

rng = random.Random(7)
clean = [c for c, l in labels().items() if not l["issues"]]
synthetic = [make_case(rng.choice(clean), rng) for _ in range(30)]
print(len(synthetic), "new labeled cases from", len(clean), "clean contracts")
print(synthetic[0]["issues"], "\n", [l for l in synthetic[0]["text"].splitlines() if "Nevada" in l or "(4) year" in l or "(48)" in l or "(7) days" in l])

# %% [markdown]
# ## 2. Log every run as a trace
# A trace is the full record: input hash, prompt, model, output, time. Traces are your audit trail and your future training data.

# %%
TRACES = Path("traces.jsonl"); TRACES.unlink(missing_ok=True)
import hashlib
def traced_call(task, prompt, **kw):
    t0 = time.time()
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, **kw)
    out = r.choices[0].message.content
    rec = {"task": task, "model": MODEL, "input_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12],
           "prompt": prompt, "output": out, "secs": round(time.time() - t0, 2), "tokens": r.usage.total_tokens}
    with open(TRACES, "a") as f: f.write(json.dumps(rec) + "\n")
    return out

PB = playbook()
def review(text):
    out = traced_call("review", f"Rules: {json.dumps(PB['rules'])}\n\nContract:\n{text}\n\n"
                                "List the names of broken rules, comma-separated, or 'none'.")
    return sorted(r for r in PB["rules"] if r in out)

results = [(c["issues"], review(c["text"])) for c in synthetic[:6]]
for gold, pred in results: print("truth", gold, "| model", pred)
print(sum(1 for _ in open(TRACES)), "traces logged")

# %% [markdown]
# ## 3. Corrections become tests immediately
# When a reviewer fixes a miss, we store input, wrong output, right output and the reason. It goes into a regression set straight away, and later into a training set (never into the held-out eval).

# %%
REGRESSION = Path("regression.jsonl"); REGRESSION.unlink(missing_ok=True)
misses = [(c, pred) for c, (gold, pred) in zip(synthetic[:6], results) if pred != gold]
for case, pred in misses:
    rec = {"text": case["text"], "model_said": pred, "correct": case["issues"], "reason": "planted fault missed or false flag"}
    with open(REGRESSION, "a") as f: f.write(json.dumps(rec) + "\n")
print(len(misses), "misses saved as regression tests")

# %% [markdown]
# ## 4. Build a fine-tuning set by rejection sampling
# Keep only outputs that pass the checker. This JSONL file is the standard chat format that TRL or Unsloth fine-tune on.

# %%
sft = []
for c in synthetic[6:16]:
    for _ in range(2):                                          # up to 2 tries per case
        pred = review(c["text"])
        if pred == c["issues"]:                                 # the checker: matches the known answer
            sft.append({"messages": [{"role": "user", "content": f"Review against the playbook:\n{c['text']}"},
                                     {"role": "assistant", "content": ", ".join(pred) or "none"}]})
            break
Path("sft_train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sft))
print(f"kept {len(sft)} of 10 cases for fine-tuning (only checker-approved outputs)")

# %%
check("30 synthetic cases built with known labels", len(synthetic) == 30)
check("traces were logged", TRACES.exists())

# %% [markdown]
# ## Your turn
# 1. Add a new fault type (for example, cure period shorter than 10 days) and a playbook rule for it.
# 2. Mix 50% synthetic and 50% real cases into one eval. Why does the course warn against synthetic-only data?
#
# **Next:** 13 · Safety
