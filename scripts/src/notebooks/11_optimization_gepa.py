# %% [markdown]
# # 11 · Optimization: tune the prompt of a small local model with GEPA
# **Local AI Lab** · CS329Z Lecture 9 · [Module 11](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m11)
#
# Improve prompts before weights. GEPA lets a bigger model read the small model's mistakes and rewrite the instructions. In the course demo this took Gemma 3 12B from 55% to 100% on a held-out set, changing only the instruction text.
#
# Here: **student** = Llama 3.2 3B (fast, weak), **reflection model** = Llama 3.1 8B. Both run locally. Budget is small so it finishes in about 10–15 minutes on a T4.

# %%
# EXTRA: dspy
# MODELS: llama3.1:8b llama3.2:3b
import random, re
import dspy
from lal.data import contract, sections, labels, ids
from lal.checks import check

student = dspy.LM("openai/llama3.2:3b", api_base="http://localhost:11434/v1", api_key="local", temperature=0)
teacher = dspy.LM("openai/llama3.1:8b", api_base="http://localhost:11434/v1", api_key="local", temperature=1.0, max_tokens=4000)
dspy.configure(lm=student)

# %% [markdown]
# ## 1. The task: pull three numbers out of the renewal and liability clauses
# Grading is exact, field by field. The metric returns **text feedback** ("expected X, got Y"), which is what GEPA learns from.

# %%
class RenewalTerms(dspy.Signature):
    """Extract terms from contract clauses."""
    clauses: str = dspy.InputField()
    renewal_years: int = dspy.OutputField()
    notice_days: int = dspy.OutputField()
    cap_months: int = dspy.OutputField(desc="0 if liability is unlimited")

def example(cid):
    secs = {s["title"]: s["text"] for s in sections(contract(cid))}
    text = "\n".join(f"{t}: {x}" for t, x in secs.items() if t in ("Renewal", "Limitation Of Liability"))
    t = labels()[cid]["terms"]
    return dspy.Example(clauses=text, renewal_years=t["renewal_years"], notice_days=t["notice_days"],
                        cap_months=t["liability_cap_months"] or 0).with_inputs("clauses")

data = [example(c) for c in ids()]
random.Random(0).shuffle(data)
train, test = data[:10], data[10:]
print(len(train), "train /", len(test), "held-out test")

def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    wrong = [f"{f}: expected {getattr(gold, f)}, got {getattr(pred, f, None)}"
             for f in ("renewal_years", "notice_days", "cap_months") if getattr(pred, f, None) != getattr(gold, f)]
    return dspy.Prediction(score=1.0 - len(wrong) / 3, feedback="All correct." if not wrong else "Wrong -> " + "; ".join(wrong))

def evaluate(program, ds):
    scores = []
    for ex in ds:
        try: scores.append(metric(ex, program(clauses=ex.clauses)).score)
        except Exception: scores.append(0.0)
    return sum(scores) / len(scores)

# %% [markdown]
# ## 2. Baseline: the small model with a plain prompt

# %%
program = dspy.Predict(RenewalTerms)
before = evaluate(program, test)
print(f"baseline on held-out test: {before:.0%}")

# %% [markdown]
# ## 3. GEPA: the reflection model rewrites the instructions
# `max_metric_calls` is the budget. Bigger budgets usually do better; 60 keeps this demo short.

# %%
gepa = dspy.GEPA(metric=metric, max_metric_calls=60, reflection_lm=teacher, track_stats=True)
optimized = gepa.compile(program, trainset=train, valset=train[:6])
after = evaluate(optimized, test)
print(f"held-out test: {before:.0%} -> {after:.0%}")

# %% [markdown]
# ## 4. What changed? Only the instructions.

# %%
for name, pred in optimized.named_predictors():
    print(f"--- {name} instructions ---\n{pred.signature.instructions}")
optimized.save("renewal_terms_gepa.json")    # freeze and version the winning prompt

# %% [markdown]
# ## 5. The ladder from here (from the lecture)
# 1. **Prompt optimization** (this notebook): cheap, tells you whether you need training at all.
# 2. **Fine-tune (SFT)** on a teacher's outputs that pass your checker.
# 3. **On-policy distillation:** the student writes, the teacher scores every token. Needs an open-weight teacher.
# 4. **RL (GRPO)** only where answers can be checked automatically.
# 5. **Re-tune prompts** on the new model.
#
# The core of GRPO in four lines: sample a group of answers to the same prompt, and push toward those that beat the group average.

# %%
import numpy as np
rewards = np.array([1.0, 0.67, 0.67, 0.33, 1.0, 0.0])        # 6 answers to one contract, scored by our metric
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
print(advantages.round(2), "<- raise the probability of the positive ones")

# %%
check("GEPA ran and produced a program", optimized is not None)
check("held-out score did not get worse", after >= before)

# %% [markdown]
# ## Your turn
# 1. Raise `max_metric_calls` to 150. Does the held-out score keep rising?
# 2. Make the feedback vague ("wrong") instead of specific. Rerun GEPA. What happens?
#
# **Next:** 12 · Data and flywheels
