---
summary: How a Stanford team made a model strong at hard math with distillation, GRPO and reranking, and which step mattered most.
minutes: 9
source: CS329A past project · Enhancing Mathematical Reasoning
slides: https://cs329a.stanford.edu/winter2025projects/enhancing.pdf
---

## Learning objective
- Build the three-stage reasoning recipe in miniature (distill, RL, rerank) and know which stage gives the most.

## The problem
- Olympiad-level math (the AIMO prize) needs long, consistent reasoning.
- Strong models get there, but spend many tokens doing it.
- Token: a small chunk of text the model reads or writes. More tokens = more time and cost.

## What the students built: a three-stage recipe
- Team: Abhinav Agarwal, Shree Reddy, Carlo Baronio, Shubhra Mishra (CS329A, Winter 2025).
- Distillation: a strong teacher (DeepSeek R1) writes full reasoning; the student (Qwen2.5) learns to copy it.
- GRPO (Group Relative Policy Optimization): sample a group of answers, reward them, push up the ones above the group average.
- Reranking: at test time, write several answers; a reward model picks the best.

## How it works
- Data: 387K problems (OpenR1-Math-220K 55%, 150K newly generated 37.5%, Bespoke-Stratos-17K 7.5%).
  - Rejection sampling: a GPT-4o-mini judge rejected about 15% of generated solutions.
- GRPO reward: 1.0 for a correct answer + 0.2 for the right format.
- KL penalty β = 0: no frozen reference model, saves a whole model's memory.
- Scaled to 32B with DeepSpeed ZeRO-3 and colocated vLLM.

## What they found (AIME'24 ablation)
| Stage | Score |
|---|---|
| Base model | 50.0% |
| + distillation | 72.1% (+22.1) |
| + GRPO | 76.7% (+4.6) |
| + reranking over 4 passes | 79.9% (+3.2) |

- Most of the gain came from distillation, not RL.
- RL mostly bought token efficiency.

## Fewer tokens, better picking
- 32B GRPO model: 5,048 tokens per problem vs. 32K for baselines.
- Reward model vs. majority vote, Qwen2.5-Math-7B over 64 answers:

| Majority vote (maj@64) | Reward model (rm@64) |
|---|---|
| 43.3% | 70.0% |

- After about 320 GRPO steps on 7B, answers got cut off; context grew from 8,192 to 16,384 tokens.
- Caution: AIME'24 has 30 problems; 79.9% is not a multiple of 1/30. The paper doesn't explain it.

## Our Colab rebuild: smaller
| Paper | Our rebuild |
|---|---|
| 387K olympiad problems | Seeded word problems from `lal.tasks`, exact answers |
| DeepSeek R1 → Qwen2.5 | Teacher llama3.1:8b; distilled file only (SFT behind `RUN_LONG`) |
| Reward model reranks | llama3.2:3b writes 4 answers; 5 ways to pick |
| GRPO on up to 32B | Optional: GRPO on SmolLM2-135M, same 1.0 + 0.2 reward, β = 0 |

## A step checker in code
```python
print(step_scores("Step 1: 12 * 7 = 84\nStep 2: 84 - 9 = 76\nStep 3: 76 / 4 = 19\nANSWER: 19"))
```
- Input: a three-step solution.
- Step 1: 12 × 7 = 84 is right → 1.0.
- Step 2: 84 − 9 is 75, not 76 → 0.0.
- Step 3: 76 ÷ 4 = 19 is right as written → 1.0.
- Output: `[1.0, 0.0, 1.0]`. Code recomputes every "a op b = c".

## Stage 1: filter the teacher's answers
- Teacher writes 2 answers per problem.
- Keep only: right final number and no wrong step.
- Among those, keep the shortest (pushes toward fewer tokens).
- Write a ready-to-train SFT file (`ap3_distill.jsonl`).

| Right answer, wrong step | Right answer, all steps right |
|---|---|
| Rejected: teaches bad habits | Kept, if shortest |

## Stage 3: five ways to pick among 4 answers
- First sample: what one call gives you.
- Majority vote: most common answer.
- Code step-checker: score = weakest step (min).
- LLM judge: the 8B model names the first wrong step.
- Oracle (pass@4): was any of the 4 right? The ceiling for any picker.

## How this applies to us
- A large local model writes careful reasoning for many clauses; keep only the checked ones.
- A smaller, faster model learns from those, on the firm's own machine.
- Check each step ("12 months × $10,000 = $120,000"), not just the final verdict.

## Common mistakes
- RL first → get good teacher data first; it gave +22.1 vs. +4.6.
- Checking only the final answer → check every step.
- Majority vote when the right answer is rare → use a step checker.
- Ignoring cut-off answers in training → watch length; raise the limit.

## Check your understanding
**Q: Which stage gave the biggest gain in the ablation?**
A: Distillation, from 50.0% to 72.1%. GRPO and reranking added smaller gains.

**Q: Why score a solution by its weakest step instead of the average?**
A: One wrong step usually ruins the answer. An average lets a long solution with one bad step beat a short correct one.

**Q: What does setting the KL penalty to 0 save?**
A: The frozen reference model is no longer needed, which saves a whole model's memory.

## Try it in Colab
- Build the distillation file; count rejected answers, including right answers with a wrong step.
- Compare the five pickers; look at the gap between first sample and oracle.
- On a GPU, set `RUN_LONG = True` and watch GRPO raise the tiny model's mean reward.
