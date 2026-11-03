# %% [markdown]
# # ap3 · Better math reasoning, rebuilt small: distill, rerank by steps, and (optionally) GRPO
# **Local AI Lab** · CS329A past project (Winter 2025) · the three-stage recipe behind many reasoning models
#
# A strong reasoning model is often built in three stages: copy good reasoning from a bigger model (**distillation**),
# reward the model for right answers (**reinforcement learning**, here GRPO), then at test time sample several answers
# and let a **verifier** pick the best. You will build each stage in miniature with checkable word problems.

# %% [markdown]
# ## The original project
# **Enhancing Mathematical Reasoning in Large Language Models through Reasoning Distillation, GRPO, and Multi-agent PRM
# Reranking** by Abhinav Agarwal, Shree Reddy, Carlo Baronio and Shubhra Mishra (Stanford CS329A, Winter 2025).
# [Paper](https://cs329a.stanford.edu/winter2025projects/enhancing.pdf) · [Poster](https://cs329a.stanford.edu/winter2025projects/enhancingposter.pdf) ·
# code: github.com/abhinav30219/math_reasoner
#
# **Problem.** Olympiad-level math (the AIMO prize) needs long, consistent reasoning, and today's models spend a lot of
# tokens getting there.
#
# **Method.**
# 1. **Data:** 387K problems: OpenR1-Math-220K (55%), 150K newly generated (37.5%), Bespoke-Stratos-17K (7.5%).
#    A GPT-4o-mini judge plus format checks rejected about 15% of generated solutions.
# 2. **Distillation:** DeepSeek R1 as teacher, Qwen2.5 as student, loss L = α·KL(teacher‖student) + (1−α)·CE, with
#    a temperature that starts high and drops, and extra weight on reasoning tokens.
# 3. **GRPO** (Shao et al. 2024, *DeepSeekMath*): sample a group of answers per problem and use the group's average reward
#    as the baseline, so no separate value network is needed. Their key choice: **KL penalty β = 0**, so the reference
#    model can be dropped from memory. Reward = 1.0 for a correct answer + 0.2 for the right format.
# 4. **Multi-agent "PRM" reranking:** several solvers with different settings, self-verification, then a reward model
#    ranks the candidates and the best one is kept. (In this paper PRM stands for *Preference* Reward Model.)
#
# **Key results (as reported, AIME'24).**
# - 32B model after GRPO: **76.7%** with 5,048 tokens per problem, vs. 32K tokens for baselines (a "6.3× reduction").
# - With multi-agent reranking over 4 inference passes: **79.9%**.
# - Ablation: base model 50.0% → distillation only 72.1% (+22.1) → + GRPO 76.7% (+4.6) → + reranking 79.9% (+3.2).
# - Baselines: Qwen2.5-32B Instruct 26.7%, Qwen2.5-Math-72B 30.0% (40.0% with tool-integrated reasoning), R1-Distill-Qwen-32B 73.3%.
# - Scale: Qwen2.5-1.5B GRPO 10.0%, 7B GRPO 16.7%. Their 7B pipeline: 65.2% with 5,137 tokens vs. 58.7% baseline.
# - 2.1× faster inference. After about 320 GRPO steps on 7B, truncated answers hurt accuracy until they raised the
#   context window from 8,192 to 16,384 tokens.
# - Reward-model selection beat majority voting at every size (for example Qwen2.5-Math-7B: 70.0% rm@64 vs. 43.3% maj@64).
#
# **Read with care (our note).** AIME'24 has 30 problems, so scores move in steps of 3.3 points. 79.9% is not a
# multiple of 1/30; it may be an average over runs, but the paper doesn't say. Also, some baselines (such as
# R1-Distill) are known to be reasoning models given a 32K budget, so the token comparison mixes model types.
# The poster adds a neat fact: with β = 0, one gradient step per batch and normalized advantages, the GRPO loss is
# always 0 while the gradient is not.
#
# **Our mini version.** This repo does not use Qwen or DeepSeek models, so: `llama3.1:8b` is the teacher,
# `llama3.2:3b` is the student at test time, and `SmolLM2-135M-Instruct` is the model we train with GRPO (optional).
# Problems come from `lal.tasks` (generated with a seed), so every answer is exact.

# %%
# EXTRA: trl datasets
# MODELS: llama3.2:3b
import ast, json, random, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from lal.llm import ask, structured
from lal.checks import check, pass_at_k

RUN_LONG = False          # True: also run the GRPO training (about 10 minutes on a T4)
TEACHER, STUDENT = MODEL, "llama3.2:3b"           # MODEL is llama3.1:8b on a GPU runtime

# %% [markdown]
# ## 1. Problems with exact answers
# We reuse the repo's seeded word problems (`lal.tasks`): "easy" ones take 2 steps, "hard" ones 3–5 steps.
# Training (for the teacher) and test (for the student) mix both.

# %%
from lal.tasks import math_problems, extract_int
to_pairs = lambda ps: [(p["question"], p["answer"]) for p in ps]
TRAIN = to_pairs(math_problems(5, seed=31, difficulty="easy") + math_problems(5, seed=31, difficulty="hard"))
TEST = to_pairs(math_problems(3, seed=32, difficulty="easy") + math_problems(3, seed=32, difficulty="hard"))
for q, a in TRAIN[4:7]:
    print(a, "←", q)

# %% [markdown]
# A **step checker** made of code: it recomputes every `a op b = c` it finds in a step. We use it twice:
# to filter teacher data, and later as a process reward for reranking.

# %%
def safe_eval(expr):
    node = ast.parse(expr.replace("×", "*").replace("÷", "/").replace("−", "-").strip(), mode="eval")
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(n.left), ev(n.right)
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b if b else float("nan")}[type(n.op)]
        raise ValueError("unsupported")
    return ev(node)

CALC = re.compile(r"([\d\.\s\+\-\*/×÷\(\)]*\d[\d\.\s\+\-\*/×÷\(\)]*)=\s*(-?\d+(?:\.\d+)?)")

def step_scores(solution):
    """One score per step: 1.0 if every calculation in it is right, 0.0 if any is wrong."""
    scores = []
    for line in (solution or "").replace(",", "").replace("$", "").splitlines():
        if not re.match(r"\s*\**step", line, re.I):
            continue
        ok = True
        for left, right in CALC.findall(line):
            if not re.search(r"[\+\-\*/×÷]", left.strip().lstrip("-")):
                continue                                   # "x = 5" is not a calculation
            try:
                ok &= abs(safe_eval(left) - float(right)) < 0.01
            except Exception:
                pass                                       # can't parse: don't punish
        scores.append(1.0 if ok else 0.0)
    return scores

print(step_scores("Step 1: 12 * 7 = 84\nStep 2: 84 - 9 = 76\nStep 3: 76 / 4 = 19\nANSWER: 19"))   # step 2 is wrong

# %% [markdown]
# ## 2. Stage 1: reasoning distillation data
# The real recipe trains the student on the teacher's *full reasoning*. The data step is what we can do here:
# sample 2 rationales per problem from the teacher, **keep only the correct ones** (rejection sampling, like the
# paper's ~15% reject rate). "Correct" means the right final answer *and* no wrong calculation in any step: a right
# answer reached through a wrong step teaches the student bad habits. Among the correct ones keep the **shortest**. That last step pushes toward the token
# efficiency the paper reports. The output is a ready-to-train SFT file (20 calls).

# %%
STEP_PROMPT = ("Solve the problem step by step. Put each step on its own line as 'Step k: ...' and write every "
               "calculation as 'a op b = c'. End with a line 'ANSWER: <integer>'.\n\nProblem: {q}")

def sample(q, model, temperature):
    try:
        return ask(STEP_PROMPT.format(q=q), model_name=model, temperature=temperature, max_tokens=500)
    except Exception as e:
        print("call failed:", e); return ""

jobs = [(q, a) for q, a in TRAIN for _ in range(2)]
with ThreadPoolExecutor(2) as ex:
    teacher_out = list(ex.map(lambda qa: sample(qa[0], TEACHER, 0.7), jobs))

by_q = {}
for (q, a), out in zip(jobs, teacher_out):
    by_q.setdefault(q, []).append((extract_int(out) == a, out))
sft, lucky = [], 0
for q, cands in by_q.items():
    # Right answer AND no wrong calculation in any step (the step checker is defined in section 3).
    lucky += sum(ok and 0.0 in step_scores(o) for ok, o in cands)
    good = sorted((o for ok, o in cands if ok and 0.0 not in step_scores(o)), key=len)
    if good:
        sft.append({"prompt": [{"role": "user", "content": STEP_PROMPT.format(q=q)}],
                    "completion": [{"role": "assistant", "content": good[0].strip()}]})
n_ok = sum(ok for c in by_q.values() for ok, _ in c)
avg_all = sum(len(o) for o in teacher_out) / max(len(teacher_out), 1)
avg_kept = sum(len(r["completion"][0]["content"]) for r in sft) / max(len(sft), 1)
print(f"teacher correct: {n_ok}/{len(jobs)} (rejected {1 - n_ok / len(jobs):.0%})")
print(f"right answer but a wrong step inside (also rejected): {lucky}")
print(f"problems with at least one correct rationale: {len(sft)}/{len(TRAIN)}")
print(f"avg characters: all samples {avg_all:.0f} → kept (shortest correct) {avg_kept:.0f}")
DATA = Path("cs329a_projects/ap3_distill.jsonl"); DATA.parent.mkdir(exist_ok=True)
DATA.write_text("\n".join(json.dumps(r) for r in sft))
print("wrote", DATA, "\n\nexample:\n", sft[0]["completion"][0]["content"][:400] if sft else "(none)")

# %% [markdown]
# ## 3. Stage 3: sample N answers, then rerank with step scores
# The student (`llama3.2:3b`) writes N = 4 solutions per test problem (24 calls). We compare five ways to pick one:
# - **first sample**: what a single call gives you;
# - **majority vote** (self-consistency, Wang et al. 2023);
# - **code step-checker**: a process reward made by code. Every `a op b = c` in a step is recomputed; a step
#   scores 1 if all its calculations are right. A solution's score is its **weakest step** (min), as in
#   Lightman et al. 2023, *Let's Verify Step by Step*;
# - **LLM judge**: the 8B model reads all 4 solutions and names the first wrong step in each (1 call per problem);
# - **oracle** (pass@4): is any of the 4 right? This is the ceiling for any reranker.

# %%
N = 4
with ThreadPoolExecutor(2) as ex:
    cand = list(ex.map(lambda qi: sample(TEST[qi[0]][0], STUDENT, 0.8), [(i, j) for i in range(len(TEST)) for j in range(N)]))
CANDS = [cand[i * N:(i + 1) * N] for i in range(len(TEST))]

# %%
class Verdict(BaseModel):
    solution: int
    first_wrong_step: int = Field(description="number of the first wrong step, or 0 if all steps are right")

class Verdicts(BaseModel):
    verdicts: List[Verdict]

def llm_judge(q, sols):
    text = "\n\n".join(f"=== Solution {i + 1} ===\n{s[:1200]}" for i, s in enumerate(sols))
    try:
        v = structured(f"Problem: {q}\n\n{text}\n\nCheck every step of every solution. For each solution, give the "
                       "number of the first wrong step (0 if all steps are right).", Verdicts, model_name=TEACHER, max_tokens=400)
        got = {x.solution: x.first_wrong_step for x in v.verdicts}
    except Exception:
        got = {}
    return [1.0 if got.get(i + 1, 1) == 0 else 0.0 for i in range(len(sols))]

with ThreadPoolExecutor(2) as ex:
    JUDGE = list(ex.map(lambda i: llm_judge(TEST[i][0], CANDS[i]), range(len(TEST))))

def pick(answers, scores):
    """Highest score wins; ties go to the most common answer among the tied."""
    best = max(scores)
    tied = [a for a, s in zip(answers, scores) if s == best and a is not None]
    return Counter(tied).most_common(1)[0][0] if tied else answers[0]

def evaluate(cands, judge, agg=min):
    res = Counter()
    for (q, gold), sols, js in zip(TEST, cands, judge):
        ans = [extract_int(s) for s in sols]
        prm = [agg(step_scores(s)) if step_scores(s) else 0.5 for s in sols]
        votes = Counter(a for a in ans if a is not None)
        res["first sample"] += ans[0] == gold
        res["majority vote"] += (votes.most_common(1)[0][0] if votes else None) == gold
        res["code step-checker"] += pick(ans, prm) == gold
        res["LLM judge"] += pick(ans, js) == gold
        res["oracle (pass@N)"] += gold in ans
    return {k: v / len(TEST) for k, v in res.items()}

ACC = evaluate(CANDS, JUDGE)
for k, v in ACC.items():
    print(f"{k:18s} {v:.0%}")
correct = [sum(extract_int(s) == g for s in sols) for (q, g), sols in zip(TEST, CANDS)]
print("unbiased pass@1 over all samples:", f"{sum(pass_at_k(N, c, 1) for c in correct) / len(TEST):.0%}")

# %% [markdown]
# **What to notice.** The gap between *first sample* and *oracle* is the room a reranker has. Majority vote needs the
# right answer to be common; a step checker can rescue a right answer that appears only once. The paper's reranker
# added +3.2 points on top of a much stronger model; on easy problems with a weak model, the gap here can be bigger.

# %% [markdown]
# ## 4. Stage 2 (optional, `RUN_LONG`): GRPO on a tiny model
# GRPO samples a *group* of answers per prompt (here 8), scores them, and pushes up the ones above the group average.
# We use the paper's reward weights (1.0 correct + 0.2 format) and its β = 0 (no KL penalty, no reference model).
# The task is easy on purpose (two-number sums) so a 135M model can visibly learn in a few minutes. The T4 has no
# bf16, so we train in fp32. TRL API checked against the current GRPOTrainer docs.

# %%
SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
grng = random.Random(7)
def sum_prompt():
    a, b = grng.randint(10, 99), grng.randint(10, 99)
    return {"prompt": [{"role": "user", "content": f"What is {a} + {b}? Reply with only the number inside <answer></answer>."}],
            "answer": str(a + b)}

def _text(c):
    return c[0]["content"] if isinstance(c, list) else str(c)

def correctness_reward(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", _text(c))
        out.append(1.0 if m and m.group(1) == a else 0.0)
    return out

def format_reward(completions, **kwargs):
    return [0.2 if re.search(r"<answer>.*?</answer>", _text(c), re.S) else 0.0 for c in completions]

print(correctness_reward([[{"content": "<answer>42</answer>"}]], ["42"]), format_reward(["<answer>x</answer>"]))

GRPO_LOG = {}
if RUN_LONG:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    train_ds = Dataset.from_list([sum_prompt() for _ in range(400)])
    eval_rows = [sum_prompt() for _ in range(40)]
    tok = AutoTokenizer.from_pretrained(SMALL); tok.padding_side = "left"

    def mean_reward(model):
        model.eval(); total = 0.0
        for r in eval_rows:
            text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
            comp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            total += correctness_reward([comp], [r["answer"]])[0] + format_reward([comp])[0]
        return total / len(eval_rows)

    base = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")   # the config says bf16; a T4 needs fp32
    GRPO_LOG["before"] = mean_reward(base)
    args = GRPOConfig(output_dir="/tmp/ap3_grpo", per_device_train_batch_size=16, num_generations=8,
                      max_completion_length=48, learning_rate=1e-5, beta=0.0, max_steps=80, logging_steps=10,
                      fp16=False, bf16=False, gradient_checkpointing=False, report_to="none", save_strategy="no")
    trainer = GRPOTrainer(model=base, reward_funcs=[correctness_reward, format_reward], args=args,
                          train_dataset=train_ds, processing_class=tok)
    trainer.train()
    GRPO_LOG["after"] = mean_reward(trainer.model)
    GRPO_LOG["history"] = [{k: h[k] for k in h if k in ("step", "reward", "loss", "completions/mean_length")} for h in trainer.state.log_history]
    for h in GRPO_LOG["history"]:
        print(h)
    print(f"mean reward (max 1.2): before {GRPO_LOG['before']:.2f} → after {GRPO_LOG['after']:.2f}")
else:
    print("RUN_LONG is False: skipping GRPO training.")

# %%
check("distillation kept only correct rationales", all(extract_int(r["completion"][0]["content"]) == dict(TRAIN)[r["prompt"][0]["content"].split("Problem: ")[1]] for r in sft))
check("the teacher got at least 3 of 10 problems right", len(sft) >= 3)
check("code step-checker catches a wrong step", step_scores("Step 1: 2 + 2 = 5") == [0.0] and step_scores("Step 1: 2 + 2 = 4") == [1.0])
check("some reranker gets at least 2 of 6 test problems", max(ACC.values()) >= 2 / 6)
if RUN_LONG:
    check("GRPO raised the mean reward", GRPO_LOG["after"] > GRPO_LOG["before"])

# %% [markdown]
# ## Your turn
# 1. **Fewer samples.** Rerank using only the first 2 of the 4 candidates. How much of the gain is left? (No new calls.)
# 2. **Min vs. mean.** Score each solution by the *average* step score instead of the weakest step. Which picks better here, and why might min be safer?
# 3. **Close the loop (RUN_LONG).** Fine-tune `SmolLM2-135M-Instruct` on `ap3_distill.jsonl` with TRL's `SFTTrainer`,
#    then ask it one test problem. With 10 examples it won't get good at math; the point is to see the full pipeline run.

# %% [markdown]
# ## Solutions

# %%
# 1. N = 2. Notice the oracle drops too: reranking can only pick from what was sampled.
two = evaluate([c[:2] for c in CANDS], [j[:2] for j in JUDGE])
for k in ACC:
    print(f"{k:18s} N=4 {ACC[k]:.0%}   N=2 {two[k]:.0%}")

# %%
# 2. Mean aggregation. Notice: mean lets a long solution with one wrong step outscore a short correct one;
#    min treats any wrong step as fatal, which matches "one wrong step ruins the answer".
mean_acc = evaluate(CANDS, JUDGE, agg=lambda s: sum(s) / len(s))
print("code step-checker  min:", f"{ACC['code step-checker']:.0%}", "  mean:", f"{mean_acc['code step-checker']:.0%}")

# %%
# 3. SFT on the distilled data (RUN_LONG only). prompt/completion conversational rows are supported by SFTTrainer.
if RUN_LONG and sft:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    sft_trainer = SFTTrainer(model=SMALL, train_dataset=Dataset.from_list(sft),
                             args=SFTConfig(output_dir="/tmp/ap3_sft", max_steps=30, per_device_train_batch_size=2,
                                            learning_rate=2e-5, logging_steps=5, fp16=False, bf16=False,
                                            report_to="none", save_strategy="no"))
    sft_trainer.train()
    m, t = sft_trainer.model, sft_trainer.processing_class
    text = t.apply_chat_template([{"role": "user", "content": STEP_PROMPT.format(q=TEST[0][0])}], tokenize=False, add_generation_prompt=True)
    ids = t(text, return_tensors="pt").to(m.device)
    print(t.decode(m.generate(**ids, max_new_tokens=150, do_sample=False)[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    print("gold:", TEST[0][1])
else:
    print("Set RUN_LONG = True to run the SFT step.")
