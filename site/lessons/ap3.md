---
summary: How a Stanford team made a smaller model great at hard math by copying a stronger model's reasoning, training with rewards, and picking the best of several answers.
minutes: 12
---

## The big idea

Hard math problems need long, careful reasoning. Big models can do it, but they use a lot of tokens (the small chunks of text a model reads and writes). Tokens cost time and money.

Abhinav Agarwal, Shree Reddy, Carlo Baronio and Shubhra Mishra tackled this in their CS329A project (Winter 2025). They stacked three ideas on top of a Qwen2.5 model:

1. **Distillation.** A strong "teacher" model (DeepSeek R1) shows its full reasoning, and the smaller "student" learns to copy it. Like a junior lawyer learning by reading a senior partner's annotated memos.
2. **GRPO** (Group Relative Policy Optimization). The model writes several answers to the same problem, a reward scores them, and the model is nudged toward the answers that beat the group average. Like grading a team's drafts against each other.
3. **Reranking.** At test time, write several answers and let a reward model pick the best one.

The surprise was where the gain came from. Most of it came from the first step: good examples from a stronger model.

## Why it matters for us

A firm can do the same with its own contract reviews. A large local model writes careful reasoning for many clauses. A smaller, faster model learns from the best of those examples.

- Good training data matters more than clever training. Filter teacher examples with a checker, and check the steps, not just the final answer.
- A checker that scores each answer can beat a simple vote when the right answer is rare.
- Reward training mostly bought shorter answers here. Shorter answers mean faster reviews on the firm's own machines.

## How it works, step by step

1. **Build a big, clean dataset.** They gathered 387K problems. About 15% of newly generated solutions were rejected by a GPT-4o-mini judge. This is rejection sampling: generate many, keep only the good ones.

2. **Distill.** Train the student on the teacher's full chain of thought (the written-out reasoning steps).

3. **Train with GRPO.** The reward was 1.0 for a correct answer plus 0.2 for the right format. They set the KL penalty to 0. The KL penalty normally keeps the model close to a frozen copy of itself, the reference model. Setting it to 0 means no reference model at all, which saves a whole model's worth of memory.

4. **Rerank at test time.** Several solvers write answers. A reward model picks the best.

Here is the ablation (turning parts on one at a time) on AIME'24, a hard math contest:

| Stage | Score |
|---|---|
| Base model | 50.0% |
| + distillation | 72.1% (+22.1) |
| + GRPO | 76.7% (+4.6) |
| + reranking | 79.9% (+3.2) |

The 32B GRPO model used 5,048 tokens per problem, against 32K for the baselines. Reward-model picking beat majority voting at every size. For example, Qwen2.5-Math-7B scored 70.0% with the reward model over 64 answers, against 43.3% by majority vote. One warning: after about 320 GRPO steps on the 7B model, answers got cut off, and they had to double the context window (how much text the model can handle at once) from 8,192 to 16,384 tokens.

A note of care: AIME'24 has only 30 problems, so 79.9% is not a score you can get on one run. The paper doesn't explain it.

## Worked example

Picture the firm's teacher model reviewing a fee clause. It writes: "Step 1: 12 months × $10,000 = $120,000. Step 2: cap is $150,000, which is above the limit." The final verdict might be right, but if a step's math is wrong, we don't want the student to learn it.

Our notebook uses seeded word problems with exact answers, and a small code checker that recomputes every "a op b = c" in each step:

```python
print(step_scores("Step 1: 12 * 7 = 84\nStep 2: 84 - 9 = 76\nStep 3: 76 / 4 = 19\nANSWER: 19"))
```

- The input is a three-step solution.
- Step 1 is right: 12 × 7 is 84. Score 1.0.
- Step 2 is wrong: 84 − 9 is 75, not 76. Score 0.0.
- Step 3 is right as written: 76 ÷ 4 is 19. Score 1.0.
- The output is `[1.0, 0.0, 1.0]`.

We use this checker twice.

First, for distillation data. The teacher (llama3.1:8b) writes 2 answers per problem. We keep only answers with the right final number and no wrong step. Among those, we keep the shortest. The result is a ready-to-train file.

Second, for reranking. The student (llama3.2:3b) writes 4 answers per test problem. A solution's score is its weakest step, the minimum. One bad step ruins the answer, just as one wrong number ruins a fee calculation. We compare five ways to choose: the first answer, a majority vote, the code checker, a model judge that names the first wrong step, and the oracle. The oracle asks "was any of the 4 right?" It is the ceiling for any picker.

An optional part trains a tiny model (SmolLM2-135M) with GRPO, using the paper's 1.0 + 0.2 reward and a KL penalty of 0.

## Common mistakes

- **Spending effort on reward training first.** Here distillation gave +22.1 points and GRPO gave +4.6. Get good examples first.
- **Checking only the final answer.** A right answer reached through a wrong step teaches bad habits. Check every step.
- **Relying on majority vote for rare answers.** If the right answer shows up once in four, a vote loses it. A step checker can still pick it.
- **Ignoring cut-off answers during training.** Watch answer length. Raise the limit before truncation hurts accuracy.
- **Reading small benchmarks too precisely.** With 30 problems, one question moves the score by 3.3 points.

## Check your understanding

**Q: Which stage gave the biggest gain in the ablation?**
A: Distillation, which lifted the score from 50.0% to 72.1%. GRPO and reranking added smaller gains after that.

**Q: Why score a solution by its weakest step instead of the average?**
A: One wrong step usually ruins the answer. An average lets a long solution with one bad step beat a short correct one.

**Q: What does setting the KL penalty to 0 save?**
A: It removes the need for a frozen reference model during training, which saves a whole model's memory.

## Try it in Colab

Run the notebook to build the filtered distillation file and see how many teacher answers were rejected, including right answers with a wrong step. Then compare the five ways to pick among 4 student answers, and look at the gap between "first sample" and the oracle. Set `RUN_LONG = True` on a GPU to watch GRPO raise a tiny model's reward.
