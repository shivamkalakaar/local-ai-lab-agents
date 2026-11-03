# %% [markdown]
# # a01 · Test-time compute: sample more, then pick well
# **Local AI Lab** · CS329A Lecture 2 (Test-time compute scaling) · more tries at answer time beat a bigger model, if you can pick the right try
#
# You can make a model better without training it: let it try a problem many times and keep a good answer. We measure how fast "at least one try is right" grows with the number of tries, how much of that gain simple picking methods keep, what it costs in tokens, and when to spend tries in parallel or one after another.

# %%
# EXTRA: matplotlib
import math, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from lal.llm import client
from lal.tasks import math_problems, extract_int
from lal.checks import pass_at_k, check

SOLVE = ("Solve this problem step by step. Keep it short.\n"
         "On the last line write exactly: ANSWER: <whole number>\n\nProblem: {q}")

def gen(prompt, temperature=0.8, max_tokens=400, model_name=None):
    """One model call. Returns (text, completion tokens). Never raises, so one bad call can't stop a loop."""
    try:
        r = client.chat.completions.create(model=model_name or MODEL, messages=[{"role": "user", "content": prompt}],
                                           temperature=temperature, max_tokens=max_tokens)
        text = r.choices[0].message.content or ""
        toks = getattr(r.usage, "completion_tokens", None) or max(1, len(text) // 4)
    except Exception as e:
        print("call failed:", e)
        text, toks = "", 0
    return text, toks

def majority(answers):
    """Most common answer, ignoring replies we could not parse. None if there are none."""
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None

# %% [markdown]
# ## 1. A task set we can grade with code
# Test-time scaling only pays off when you can tell right from wrong. So we use word problems with exact integer answers, made by `lal.tasks.math_problems` with a fixed seed. "Easy" takes 2 steps. "Hard" takes 3 to 5 steps (percentages, rounding up, a little algebra).

# %%
EASY = math_problems(4, seed=0, difficulty="easy")
HARD = math_problems(4, seed=0, difficulty="hard")
PROBS = EASY + HARD
BY_ID = {p["id"]: p for p in PROBS}
for p in PROBS[:1] + PROBS[-1:]:
    print(p["id"], "|", p["question"], "| answer:", p["answer"])
print("extract_int test:", extract_int("...so 12 x 5 = 60 apples.\nANSWER: 60"))

# %% [markdown]
# ## 2. Repeated sampling: K tries per problem
# We ask the same question K = 6 times at temperature 0.8, so the tries differ. That is 8 problems x 6 = 48 calls, two at a time.
#
# Brown et al. 2024 (*Large Language Monkeys*) did this with up to 10,000 samples per problem. Their key word is **coverage**: the share of problems where at least one sample is right.

# %%
K = 6
jobs = [p for p in PROBS for _ in range(K)]
t0 = time.time()
with ThreadPoolExecutor(2) as ex:
    outs = list(ex.map(lambda p: gen(SOLVE.format(q=p["question"])), jobs))
print(f"{len(jobs)} calls in {time.time() - t0:.0f}s")

SAMPLES = {p["id"]: [] for p in PROBS}
for p, (text, toks) in zip(jobs, outs):
    a = extract_int(text)
    SAMPLES[p["id"]].append({"text": text, "answer": a, "ok": a == p["answer"], "tokens": toks})

for pid, ss in SAMPLES.items():
    print(f"{pid}: right {sum(s['ok'] for s in ss)}/{K}   answers {[s['answer'] for s in ss]}   truth {BY_ID[pid]['answer']}")

# %% [markdown]
# ## 3. Coverage (pass@k) grows with k
# `pass_at_k(n, c, k)` is the unbiased estimate of "at least one of k tries is right", given c right out of n. We plot it with a log scale on k. Brown et al. found coverage often rises in a straight line against log(k), over four orders of magnitude.

# %%
def coverage(probs, k):
    return sum(pass_at_k(K, sum(s["ok"] for s in SAMPLES[p["id"]]), k) for p in probs) / len(probs)

ks = list(range(1, K + 1))
cov = {name: [coverage(ps, k) for k in ks] for name, ps in [("all", PROBS), ("easy", EASY), ("hard", HARD)]}
for name, ys in cov.items():
    print(f"{name:5}", "  ".join(f"k={k}:{y:.0%}" for k, y in zip(ks, ys)))

plt.figure(figsize=(5, 3.2))
for name, ys in cov.items():
    plt.plot(ks, ys, marker="o", label=name)
plt.xscale("log"); plt.xticks(ks, [str(k) for k in ks]); plt.ylim(0, 1.05)
plt.xlabel("k (tries per problem, log scale)"); plt.ylabel("coverage (pass@k)"); plt.title("More tries, more problems solved")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# %% [markdown]
# ## 4. Coverage is an upper bound. Picking is the hard part.
# Coverage assumes an **oracle verifier**: a checker that always knows which try is right. Here we have one (the exact answer), but real work often doesn't.
#
# Without it, the simplest picker is **majority vote** (also called self-consistency): keep the most common answer. Brown et al. report that on MATH with Llama-3-8B-Instruct, coverage passed 95% while majority vote and reward models stalled near 40-41% after about 100 samples.

# %%
def majority_acc(probs, k):
    return sum(majority([s["answer"] for s in SAMPLES[p["id"]][:k]]) == p["answer"] for p in probs) / len(probs)

maj = [majority_acc(PROBS, k) for k in ks]
print(" k   oracle (coverage)   majority vote   gap")
for k, c, m in zip(ks, cov["all"], maj):
    print(f"{k:2}   {c:>14.0%}   {m:>13.0%}   {c - m:+.0%}")

# %% [markdown]
# **How to read this:** the gap between the two columns is the **generation-verification gap**. The model *could* give the right answer, but our picker can't find it. Module a02 is about closing that gap.

# %% [markdown]
# ## 5. What it costs: tokens vs accuracy
# Every try costs tokens. Plot accuracy against total tokens spent, so you can compare one big call with many small ones on the same axis.

# %%
avg_toks = sum(s["tokens"] for ss in SAMPLES.values() for s in ss) / (len(PROBS) * K)
spent = [k * avg_toks * len(PROBS) for k in ks]
print(f"average tokens per try: {avg_toks:.0f}")
for k, t, c, m in zip(ks, spent, cov["all"], maj):
    print(f"k={k}: {t:6.0f} tokens for the set   oracle {c:.0%}   majority {m:.0%}")

plt.figure(figsize=(5, 3.2))
plt.plot(spent, cov["all"], marker="o", label="oracle verifier (coverage)")
plt.plot(spent, maj, marker="s", label="majority vote")
plt.xlabel("output tokens spent on the 8 problems"); plt.ylabel("share solved"); plt.ylim(0, 1.05)
plt.legend(); plt.grid(alpha=0.3); plt.title("Accuracy per token"); plt.show()

# %% [markdown]
# Brown et al. made the same point with money: on SWE-bench Lite, 5 tries of DeepSeek-Coder-V2-Instruct solved 29.62% for $10.8, while 1 try of GPT-4o solved 24.00% for $39. Cheap model plus many tries plus a checker can beat an expensive model.

# %% [markdown]
# ## 6. Why the curve looks like a power law (no model calls)
# Schaeffer et al. 2025 (*How Do Large Language Monkeys Get Their Power (Laws)?*) explain the shape.
# - For **one** problem with single-try success p, the chance that all k tries fail is (1 - p)^k. That falls **exponentially** in k.
# - Averaged over **many** problems, the failure curve looks like a **power law** in k (a straight line on a log-log plot). That happens when a few problems are very hard (p close to 0). This is called a **heavy left tail**.
#
# We check this with exact math. Say single-try success rates follow a Beta(alpha, beta) distribution, which has a tail near 0 shaped like p^(alpha-1). Then the theory says -log(coverage) should fall like k^(-alpha).

# %%
ALPHA, BETA = 0.3, 3.0

def agg_fail(k, a=ALPHA, b=BETA):
    """Mean over problems of (1-p)^k when p ~ Beta(a, b). Exact: B(a, b+k) / B(a, b)."""
    lb = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    return math.exp(lb(a, b + k) - lb(a, b))

kk = np.unique(np.logspace(0, 4, 40).astype(int))
neglog_cov = np.array([-math.log(1 - agg_fail(k)) for k in kk])
big = kk >= 100
slope = np.polyfit(np.log(kk[big]), np.log(neglog_cov[big]), 1)[0]
print(f"fitted power-law exponent for k >= 100: {slope:.3f}   (theory: -alpha = {-ALPHA})")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for p in (0.3, 0.05, 0.01):
    fail = [(1 - p) ** k for k in kk]
    ax[0].plot(kk, [-math.log1p(-f) if 0 < f < 1 else np.nan for f in fail], label=f"one problem, p={p}")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 10)
ax[0].set_title("Each problem: falls off a cliff (exponential)"); ax[0].set_xlabel("k"); ax[0].set_ylabel("-log(pass@k)"); ax[0].legend(fontsize=7)
ax[1].plot(kk, neglog_cov, marker=".", label=f"many problems, p ~ Beta({ALPHA}, {BETA})")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_title("Average: straight line (power law)"); ax[1].set_xlabel("k"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Why a builder cares:** the exponent tells you how fast more samples pay off, so you can forecast before you spend. Schaeffer et al. estimate it from the distribution of single-try success rates and report about an order of magnitude lower error than fitting the curve directly, which they describe as roughly 2-4 orders of magnitude less inference compute.

# %% [markdown]
# ## 7. An Archon-style pipeline: generate, rank, fuse, verify
# Saad-Falcon et al. 2024 (*Archon*) treat inference-time tricks as building blocks and search for the best stack. The blocks include a **Generator** (makes candidates), a **Ranker** (picks the best ones), a **Fuser** (merges candidates into one better answer), and a **Verifier** (keeps only answers whose reasoning checks out).
#
# We build one fixed stack by hand on the 4 hard problems: reuse the 6 samples, then rank (1 call), fuse (1 call), verify the fused answer (1 call). If the verifier says NO, we fall back to majority vote. That is 12 calls.

# %%
def numbered(cands, limit=600):
    return "\n\n".join(f"Candidate {i + 1}:\n{c['text'][:limit]}" for i, c in enumerate(cands))

RANK = ("Problem: {q}\n\n{c}\n\nWhich candidate has the correct final answer? Check the arithmetic. "
        "Reply with the candidate number only.")
FUSE = ("Problem: {q}\n\nHere are several attempts. Some may be wrong.\n\n{c}\n\n"
        "Use the attempts to write one careful, correct solution. On the last line write exactly: ANSWER: <whole number>")
VERIFY = ("Problem: {q}\n\nProposed solution:\n{s}\n\nCheck every step and the arithmetic. "
          "Is the final answer correct? Reply YES or NO only.")

def archon(p):
    cands = SAMPLES[p["id"]]
    r, _ = gen(RANK.format(q=p["question"], c=numbered(cands)), temperature=0, max_tokens=10)
    digits = [int(x) for x in "".join(ch if ch.isdigit() else " " for ch in r).split()]
    idx = digits[0] - 1 if digits and 1 <= digits[0] <= len(cands) else 0
    fused, _ = gen(FUSE.format(q=p["question"], c=numbered(cands)), temperature=0, max_tokens=500)
    v, _ = gen(VERIFY.format(q=p["question"], s=fused), temperature=0, max_tokens=5)
    fused_ans = extract_int(fused)
    final = fused_ans if "YES" in v.upper() else majority([c["answer"] for c in cands])
    return {"first": cands[0]["answer"], "majority": majority([c["answer"] for c in cands]),
            "ranked": cands[idx]["answer"], "fused": fused_ans, "fused+verify": final,
            "oracle": p["answer"] if any(c["ok"] for c in cands) else None}

with ThreadPoolExecutor(2) as ex:
    ARCH = dict(zip([p["id"] for p in HARD], ex.map(archon, HARD)))
for method in ["first", "majority", "ranked", "fused", "fused+verify", "oracle"]:
    right = sum(ARCH[p["id"]][method] == p["answer"] for p in HARD)
    print(f"{method:13} {right}/{len(HARD)}")

# %% [markdown]
# With 4 problems this is noisy, so don't read too much into one run. The paper's lesson is about **combining** blocks: Archon reports that across its benchmarks the Fuser helped on every one (8.9% on average), the Ranker came within 2.7% of oracle selection on MT-Bench and Arena-Hard-Auto, and the Verifier helped most on reasoning tasks. Their searched stacks beat o1, GPT-4o and Claude 3.5 Sonnet by 15.1% on average.

# %% [markdown]
# ## 8. Spend compute by difficulty (Snell et al. 2024)
# Snell et al. 2024 (*Scaling LLM Test-Time Compute Optimally*) ask: with a fixed budget, should you sample in **parallel** (many independent tries, then vote) or **sequentially** (one try, then revise it, then revise again)? Their answer: it depends on difficulty. On easy questions, sequential revision worked best. On harder ones, a mix of both did. Picking the strategy per question beat best-of-N with up to 4x less compute.
#
# First, how hard is each problem *for this model*? Snell et al. measure difficulty as the model's own success rate over many samples, not a human label. We do the same with our 6 samples.

# %%
for p in PROBS:
    rate = sum(s["ok"] for s in SAMPLES[p["id"]]) / K
    print(f"{p['id']}  label={p['difficulty']:4}  model success rate={rate:.0%}")

# %% [markdown]
# Now a budget of 3 generations per problem, two ways:
# - **parallel:** majority vote over 3 of our existing samples (0 new calls);
# - **sequential:** take sample 1, then ask the model twice to check and fix its own work (2 new calls per problem, 16 in total).
#
# Note: Snell et al. fine-tuned a special revision model. We only prompt, so our revisions are weaker than theirs.

# %%
REVISE = ("Problem: {q}\n\nYour previous attempt:\n{prev}\n\nCheck each step of the attempt. If you find a mistake, fix it. "
          "Write the full corrected solution. On the last line write exactly: ANSWER: <whole number>")

def sequential(p, rounds=2):
    text = SAMPLES[p["id"]][0]["text"]
    for _ in range(rounds):
        new, _ = gen(REVISE.format(q=p["question"], prev=text[:1500]), temperature=0.3)
        text = new or text
    return extract_int(text)

with ThreadPoolExecutor(2) as ex:
    SEQ = dict(zip([p["id"] for p in PROBS], ex.map(sequential, PROBS)))
PAR = {p["id"]: majority([s["answer"] for s in SAMPLES[p["id"]][:3]]) for p in PROBS}

for name, ps in [("easy", EASY), ("hard", HARD)]:
    par = sum(PAR[p["id"]] == p["answer"] for p in ps)
    seq = sum(SEQ[p["id"]] == p["answer"] for p in ps)
    print(f"{name}: parallel (vote of 3) {par}/{len(ps)}   sequential (1 + 2 revisions) {seq}/{len(ps)}")

# %% [markdown]
# **How to read this:** a "compute-optimal" policy picks, for each difficulty level, whichever strategy wins there. To do that in production you need a cheap **difficulty predictor** (Snell et al. used a verifier's scores; you could use agreement between a few samples). Snell et al. also found the trade has limits: on the hardest questions, or when you serve a lot of traffic, a bigger pretrained model was the better use of compute.

# %%
check("task generator is seeded (same seed, same problems)", math_problems(5, 3, "hard") == math_problems(5, 3, "hard"))
check("coverage never goes down as k grows", all(b >= a - 1e-9 for a, b in zip(cov["all"], cov["all"][1:])))
check("repeated sampling helps: coverage@6 > pass@1 (or pass@1 is already 100%)", cov["all"][-1] > cov["all"][0] or cov["all"][0] == 1.0)
check("the model solves at least half the easy problems within 6 tries", cov["easy"][-1] >= 0.5)
check("power-law exponent matches the Beta tail (within 0.1)", abs(slope + ALPHA) < 0.1)

# %% [markdown]
# ## Your turn
# 1. **Temperature and diversity.** Rerun repeated sampling on the 4 hard problems with 4 tries each at temperature 0.2 and at 1.0. Which gives higher coverage? Which gives better majority vote?
# 2. **Small model, more tries.** Pull `llama3.2:3b` and give it 6 tries on the hard problems. Does the 3B model's coverage@6 beat the 8B model's pass@1? Compare tokens spent too. (This is a tiny version of Snell et al.'s "smaller model plus test-time compute vs bigger model".)
# 3. **Fit your own power law.** Change `ALPHA` to 0.1 and 0.8 and refit the slope. Which tail gives slower gains from more samples, and why?
#
# **Next:** a02 · Robust verification

# %% [markdown]
# ## Solutions
# Each solution says how many model calls it makes. Run only the ones you want; together they add about 40-55 calls.

# %%
# Solution 1: temperature 0.2 vs 1.0 (4 problems x 4 tries x 2 settings = 32 calls)
for temp in (0.2, 1.0):
    jobs = [p for p in HARD for _ in range(4)]
    with ThreadPoolExecutor(2) as ex:
        outs = list(ex.map(lambda p: gen(SOLVE.format(q=p["question"]), temperature=temp), jobs))
    ans = {p["id"]: [] for p in HARD}
    for p, (text, _) in zip(jobs, outs):
        ans[p["id"]].append(extract_int(text))
    cov4 = sum(any(a == p["answer"] for a in ans[p["id"]]) for p in HARD) / len(HARD)
    maj4 = sum(majority(ans[p["id"]]) == p["answer"] for p in HARD) / len(HARD)
    print(f"temperature {temp}: coverage@4 {cov4:.0%}   majority@4 {maj4:.0%}   distinct answers {[len(set(v)) for v in ans.values()]}")
# Low temperature gives near-identical tries, so extra tries add little coverage.
# High temperature adds diversity (more coverage) but more noise, which can hurt majority vote.

# %%
# Solution 2: llama3.2:3b with 6 tries vs llama3.1:8b with 1 try (24 calls plus a ~2 GB download)
from lal.runtime import pull
SMALL = "llama3.2:3b"
pull(SMALL)
jobs = [p for p in HARD for _ in range(K)]
with ThreadPoolExecutor(2) as ex:
    outs = list(ex.map(lambda p: gen(SOLVE.format(q=p["question"]), model_name=SMALL), jobs))
right = {p["id"]: 0 for p in HARD}
small_toks = 0
for p, (text, toks) in zip(jobs, outs):
    right[p["id"]] += extract_int(text) == p["answer"]
    small_toks += toks
small_cov = sum(pass_at_k(K, c, K) for c in right.values()) / len(HARD)
big_p1 = cov["hard"][0]
big_toks = sum(s["tokens"] for p in HARD for s in SAMPLES[p["id"]]) / K
print(f"3B coverage@{K}: {small_cov:.0%} using {small_toks} tokens")
print(f"8B pass@1:      {big_p1:.0%} using about {big_toks:.0f} tokens")
# A 3B token is much cheaper than an 8B token. If the small model's coverage is higher AND you have
# a checker to pick the right try, many small tries can be the better deal. Without a checker, it often isn't.

# %%
# Solution 3: the tail shape sets the exponent
for a in (0.1, 0.3, 0.8):
    y = np.array([-math.log(1 - agg_fail(k, a=a)) for k in kk])
    s = np.polyfit(np.log(kk[big]), np.log(y[big]), 1)[0]
    print(f"alpha={a}: fitted exponent {s:.2f}")
# Small alpha = many problems with success rate near 0 = a flatter line: each 10x more samples
# removes less of the failure. That is the "heavy tail" that makes some tasks expensive to brute-force.
