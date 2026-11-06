---
summary: How a Stanford team made a model strong at hard math with distillation, GRPO and reranking, and which step mattered most.
minutes: 16
source: CS329A past project · Enhancing Mathematical Reasoning
slides: https://cs329a.stanford.edu/winter2025projects/enhancing.pdf
---

## Learning objective
- Build the three-stage reasoning recipe in miniature (distill, RL, rerank), know which stage gives the most, and read the reported numbers with care.

# The problem and why it matters

## The project at a glance
- Enhancing Mathematical Reasoning in LLMs through Reasoning Distillation, GRPO, and Multi-agent PRM Reranking.
- Team: Abhinav Agarwal, Carlo Baronio, Shree Reddy, Shubhra Mishra (CS329A, Winter 2025).
- Sources: the 10-page report and the poster.
- Code: github.com/abhinav30219/math_reasoner. Model: huggingface.co/abhinav302019/Math-Reasoning-Model-32B.

## The target: olympiad math
- AIMO Progress Prize: an AI contest on national-olympiad-level problems.
- Needs careful reasoning, creativity and exact computing.
- Today's LLMs are inconsistent on such problems and use a lot of compute.
- Token: a small chunk of text the model reads or writes. More tokens = more time and cost.

## Three challenges they name
- Keep reasoning coherent across many steps.
- Balance compute cost against answer quality.
- Stay consistent across very different problem types.
- The poster adds: "cold start" problems in RL for reasoning, and robust checking of solutions.

## Their five contributions
- A curated set of 387K math problems, olympiad-focused.
- A distillation method that moves reasoning from a big model to a smaller one.
- A memory-saving GRPO setup that scales to 32B models.
- Multi-agent reranking with a reward model.
- Top AIME'24 results with far fewer tokens.
- AIME'24: the 2024 American Invitational Mathematics Examination; 30 problems with integer answers.

## The recipe at a glance
| Stage | What happens | In short |
|---|---|---|
| 1. Distillation | Teacher (DeepSeek R1) writes full reasoning; student (Qwen2.5) learns it | Copy a master's worked solutions |
| 2. GRPO | Sample a group of answers, reward right ones, push above-average ones up | Practice with a grader |
| 3. Reranking | Write several answers at test time; a reward model picks the best | Pick your best draft |

# Background they build on

## Math LLMs and RL
- DeepSeek-Math, MAmmoTH: special pretraining and fine-tuning for math.
- Tool-augmented reasoning: the model calls a calculator or code.
- Plain supervised fine-tuning often copies surface patterns, not reasoning.
- PPO (Schulman et al., 2017): standard RL; heavy on compute; hard with sparse rewards (right/wrong only at the end).
- Other routes: DPO (learn from preferred vs. rejected answers), RL from language feedback.

## GRPO (Shao et al., 2024, DeepSeekMath)
- GRPO: Group Relative Policy Optimization.
- PPO needs a separate critic (value network) to estimate a baseline.
- GRPO drops it: sample a group of answers to one question; the group's mean reward is the baseline.
- Advantage: how much better one answer is than its group, normalized.
- Result: much less memory, stable updates.

## The GRPO objective (Eq. 2)
- `J(θ) = (1/G) Σ_i (1/|o_i|) Σ_t [ π_θ(o_i,t | q, o_i,<t) / [π_θ(o_i,t | q, o_i,<t)]_stopgrad ] · Â_i,t − β · KL(π_θ ‖ π_ref)`.
- G: number of answers in the group. o_i: answer i; |o_i| its length in tokens.
- π_θ: the model being trained. The ratio over its own stop-gradient copy equals 1 but passes the gradient.
- Â_i,t: the normalized advantage for token t of answer i.
- β · KL: a penalty for drifting from a frozen reference model π_ref.

## Checking answers: votes vs. verifiers
| Method | How it picks | Weak spot |
|---|---|---|
| Self-consistency (Wang et al., 2023) | Most common final answer | Fails when the right answer is rare |
| Verifiers (Lightman et al., 2023, "Let's Verify Step by Step") | A model scores each solution or step | Needs a trained scorer |
| Math-Shepherd (Wang et al., 2023) | Step-level checking, agent roles | Extra calls |

- Their reranker combines these without relying only on majority vote.

## Knowledge distillation
- Distillation: train a small student to match a big teacher.
- Response-based (Hinton et al., 2015): match the teacher's softened output probabilities.
- Sequence-level (Kim and Rush, 2016): train on the teacher's full outputs.
- Chain-of-thought distillation: copy the teacher's reasoning steps, not just answers.
- They adapt CoT distillation to math with a mixed KL + CE loss.

# The method, step by step

## Stage 0: the dataset (387K problems)
| Source | Share | What it brings |
|---|---|---|
| OpenR1-Math-220K | 55% | 2–4 reasoning traces per problem |
| Their own generated set | 37.5% | 150K problems with new solutions |
| Bespoke-Stratos-17K | 7.5% | High-quality step-by-step solutions |

- Poster: the 150K came from NuminaMath 1.5 (860K+ competition problems), minus those in OpenR1 and Bespoke.

## Cleaning the data
- Removed overlapping problems (stops test leakage, "contamination").
- Dropped K-12 school-level problems.
- Selected by validity tags and math topics.
- Balanced across difficulty levels.

## Generating and checking 150K solutions
- A faster pipeline: 10 minutes per batch → 3 minutes per batch.
- GPT-4o-mini as a judge of mathematical validity.
- Auto-filter of problems with invalid tags.
- Checks on correctness, reasoning validity and format.
- About 15% of generated solutions rejected.

## Stage 1: distillation loss (Eq. 1)
- `L = α · L_KL + (1 − α) · L_CE`.
- L_KL: gap between teacher's and student's softened token probabilities.
- L_CE: cross-entropy between the student's predictions and the ground truth.
- α: how much weight goes to copying the teacher vs. matching the answer.
- Teacher: DeepSeek R1. Student: Qwen2.5 (the poster says Qwen2.5-32B).

## Why KL + CE (their table, Section 4.1)
| Method | Reasoning transfer | Efficiency | Complexity |
|---|---|---|---|
| Pure CE | Low | High | Low |
| Pure KL | Medium | Medium | Medium |
| KL + CE (theirs) | High | Medium | Medium |

- This is a qualitative table; no measured numbers are given.

## Distillation tricks
- Adaptive temperature: start high (soft, broad patterns), end low (precise tokens, right answers).
- Temperature: how much the probabilities are softened; high = flatter.
- Weighted masking: more weight on reasoning steps than on the problem text.
- Speed: mixed precision (AMP) with gradient accumulation.

## Stage 2: GRPO with β = 0
- Their key change: set the KL penalty β to 0.
- Faster learning, especially with the format reward.
- No reference model in memory.
- No extra forward passes through a reference model.
- Poster test: on Qwen2.5-1.5B with GSM8K, a smaller KL coefficient learned better.

## A curious fact (poster)
- With β = 0, one gradient step per batch, and normalized advantages:
  - The GRPO loss is always 0.
  - Yet the gradients are not 0, so the model still learns.
- Why: the ratio is exactly 1 and normalized advantages sum to 0, but each token's gradient still points its own way.

## The reward
- `R = 1.0` if the final answer is correct.
- `R = 0.2` if the output uses the correct format.
- Format reward: keeps structured reasoning. Correctness reward: gets the answer right.

## Systems work to reach 32B
- 8 vLLM inference engines, each with its own prompts.
- vLLM: a fast serving library for LLMs.
- vLLM engines moved to CPU memory during policy updates.
- DeepSpeed ZeRO-3: splits weights, gradients and optimizer state across GPUs.
- Poster: forked OpenRLHF to put inference and training on the same GPUs.
- Later runs: rollout batch 48, curriculum (easy to hard), 16,384-token context, weighted rewards.

## Stage 3: multi-agent reranking
- Diverse generation: several agents with different settings write solutions.
- Self-reflection and verification: each solution is checked for correctness.
- PRM: here "Preference Reward Model"; ranks by correctness, clarity, fit of technique.
- Selection: the top-scored solution wins.
- A post-GRPO fine-tuning phase steadies the model for contest-style problems.

## Why rerank (Section 6.2)
- Robustness: one bad sample doesn't sink the answer.
- Diversity: several valid methods for hard problems.
- Cross-checking: solutions validate each other.
- Quality: the reward model picks the best reasoning path.

# Results

## Training observations (Section 7.1)
- Context window: after about 320 steps on 7B, hard problems got cut off and scores fell.
  - Raising the context from 8,192 to 16,384 tokens restored and improved them.
- Token use: answers first got shorter, then grew again only on harder problems.
- The 1.0 + 0.2 reward gave a useful training signal.

## Training curves (poster)
- Training reward over ~550 steps: noisy, rising overall.
- Response length: falls from near 6,000 tokens to about 3,500, then climbs again late in training.
- KL vs. no-KL run on the same setup: the no-KL run reaches a higher reward.
- These are read from plots; exact values aren't printed.

## Model scale (Section 7.2)
| Model | AIME'24 |
|---|---|
| Qwen2.5-1.5B GRPO | 10.0% |
| Qwen2.5-7B GRPO | 16.7% |
| Qwen2.5-32B Base | 50.0% |
| Qwen2.5-32B Instruct | 26.7% |
| Qwen2.5-32B GRPO | 76.7% |

- "26.7% accuracy gain over the base model" (50.0 → 76.7).
- 76.7% single pass; 79.9% with 4 inference passes.

## Ablation (Section 8.1)
| Configuration | AIME'24 | Avg. tokens |
|---|---|---|
| Full pipeline (multi-agent) | 79.9% | 5,048 × 4 |
| Without multi-agent PRM | 76.7% | 5,048 |
| Without GRPO (distillation only) | 72.1% | 31,764 |
| Without distillation (base) | 50.0% | 32,000 |

- Distillation +22.1; GRPO +4.6; reranking +3.2.
- GRPO's bigger effect: tokens fell from 31,764 to 5,048.

## The 7B pipeline
- Same relative gains, lower scores.
- 65.2% accuracy with 5,137 tokens.
- Baseline: 58.7% with 32K tokens.

## Comparison with baselines (Section 9.1)
| Model | AIME'24 | Tokens |
|---|---|---|
| Qwen 32B Base | 50.0% | 32,000 |
| Qwen 32B Instruct | 26.7% | 32,000 |
| Qwen2.5-Math-72B | 30.0% | 32,000 |
| Qwen2.5-Math-72B (TIR) | 40.0% | 32,000 |
| Qwen-R1-Distilled (32B) | 73.3% | 32,000 |
| Theirs, single | 76.7% | 5,048 |
| Theirs, multi-agent | 79.9% | 5,048 × 4 |

- TIR: tool-integrated reasoning; the model runs code to compute.

## Token efficiency (Section 8.2, B.1)
- 32,000 / 5,048 ≈ 6.3× fewer tokens.
- Simple problems: 2,184 tokens on average. Hard ones: up to 7,912.
- The model drops redundant explanation and focuses on the right technique.
- 2.1× faster inference on their test set.

## Holdout check (poster)
- 1,000 held-out tasks from OpenR1-Math-220K.

| Checkpoint | Accuracy |
|---|---|
| Base model | 0.546 |
| Checkpoint-350 | 0.585 |
| Final model | 0.631 |

- Final reasoning traces were shorter and more accurate.

## Tool use and sampling budget (Figure 1)
| Model | Standard | TIR | maj@64 | rm@64 | maj@256 | rm@256 |
|---|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | 10.0% (3/30) | 23.3% (7/30) | 30.0% | 60.0% | 30.0% | 63.3% |
| Qwen2.5-Math-7B | 16.7% (5/30) | 20.0% (6/30) | 43.3% | 70.0% | 46.7% | 70.0% |
| Qwen2.5-Math-72B | 30.0% (9/30) | 40.0% (12/30) | 46.7% | 60.0% | 53.3% | 63.3% |

- maj@64: majority vote over 64 samples. rm@64: reward model picks among 64.

## What Figure 1 shows
- TIR helps at every size; the smallest model gains most in relative terms.
- More samples help: single → maj@64/256 → rm@64/256.
- The reward model beats majority vote at every size.
  - 7B: 70.0% (rm@64) vs. 43.3% (maj@64).
- Their reading: a scorer that judges quality beats a simple consensus.

## Their trade-off summary (Section 9.2, 10.3)
- Single inference (76.7%): best balance of accuracy and cost.
- Multi-agent (79.9%): peak accuracy, 4× the compute.
- Small models (1.5B, 7B): tools remain an effective boost.
- Best of both, they suggest: a GRPO-trained model that calls tools only when needed.

# Reading the numbers with care

## Caveats we found
- AIME'24 has 30 problems; scores move in steps of 3.3 points.
  - 79.9%, 72.1% and 65.2% are not multiples of 1/30. The report doesn't say they are averages.
- The 1.5B and 7B "GRPO" scores (10.0%, 16.7%) equal the "Standard" Qwen2.5-Math scores in Figure 1.
  - So it's unclear whether they are their own GRPO runs.
- The 7B pipeline is 65.2% in Section 8.1 but "Qwen2.5-7B GRPO" is 16.7% in Section 7.2.
- The poster's bar for R1-Distill reads 70.0%; the report's table says 73.3%.
- Baselines got a 32K budget; R1-style models are built to use it, so token comparisons mix model types.

# Limitations and future work

## Limitations
- Stated: truncation hurt training until the context grew.
- Stated: multi-agent reranking costs 4× the compute.
- Our notes:
  - One benchmark of 30 problems; no error bars or repeat runs.
  - The distillation table is qualitative; α and temperatures not given.
  - Reranker details (models, prompts) are thin.

## Future work they list
- Other reasoning domains beyond math.
- Hybrids: token-efficient models plus special tools.
- Multi-agent systems with more specialized roles.
- Poster: token efficiency for longer problems; training on Chain of Draft (CoD) data.

# Our smaller Colab rebuild

## What we change
| Paper | Our rebuild |
|---|---|
| 387K olympiad problems | Seeded word problems from `lal.tasks`, exact answers |
| DeepSeek R1 → Qwen2.5 | Teacher llama3.1:8b; distilled file only (SFT behind `RUN_LONG`) |
| GPT-4o-mini judge filters data | Code step-checker plus exact answer filter |
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
- Teacher writes 2 answers per problem (20 calls).
- Keep only: right final number and no wrong step.
- Among those, keep the shortest (pushes toward fewer tokens, like the paper).
- Write a ready-to-train SFT file (`ap3_distill.jsonl`).

| Right answer, wrong step | Right answer, all steps right |
|---|---|
| Rejected: teaches bad habits | Kept, if shortest |

## Stage 3: five ways to pick among 4 answers
- First sample: what one call gives you.
- Majority vote: most common answer.
- Code step-checker: score = weakest step (min), as in "Let's Verify Step by Step".
- LLM judge: the 8B model names the first wrong step.
- Oracle (pass@4): was any of the 4 right? The ceiling for any picker.

## Stage 2 in code: the paper's reward
```python
def format_reward(completions, **kwargs):
    return [0.2 if re.search(r"<answer>.*?</answer>", _text(c), re.S) else 0.0 for c in completions]
```
- One reward per sampled answer in the group.
- 0.2 if the answer sits inside `<answer></answer>` tags, as in the paper.
- A second function gives 1.0 when the number inside is right.
- Training uses `beta=0.0`, 8 answers per prompt, fp32 on a T4.

## Solutions: three experiments
- Rerank with only 2 of the 4 samples: the oracle drops too.
- Mean vs. min step score: mean lets a long answer with one wrong step win.
- SFT on the distilled file (RUN_LONG): see the full pipeline run end to end.

# How this applies to us

## Distill on the firm's machine
- A large local model writes careful reasoning for many clauses.
- Keep only reasoning a checker accepts.
- A smaller, faster model learns from those; no data leaves the building.

## Check each step, not just the verdict
| Final-answer check | Step check |
|---|---|
| "Breach: yes" matches the label | "12 months × $10,000 = $120,000" recomputed |
| Lucky right answers slip in | Right answer with a wrong step rejected |

## Pick the right stage to invest in
- Their biggest gain was good teacher data (+22.1), not RL (+4.6).
- For a firm: spend effort on clean, checked examples first.
- Use reranking with a scorer when the right answer is rare; skip majority vote there.

## Common mistakes
- RL first → get good teacher data first; it gave +22.1 vs. +4.6.
- Checking only the final answer → check every step.
- Majority vote when the right answer is rare → use a step checker.
- Ignoring cut-off answers in training → watch length; raise the limit.
- Quoting scores without the test size → on 30 problems, one question is 3.3 points.

## Check your understanding
**Q: Which stage gave the biggest gain in the ablation?**
A: Distillation, from 50.0% to 72.1%. GRPO and reranking added smaller gains.

**Q: What did GRPO mainly buy?**
A: Token efficiency: from 31,764 to 5,048 tokens per problem, plus +4.6 points.

**Q: Why score a solution by its weakest step instead of the average?**
A: One wrong step usually ruins the answer. An average lets a long solution with one bad step beat a short correct one.

**Q: What does setting the KL penalty to 0 save?**
A: The frozen reference model is no longer needed, which saves a whole model's memory and its forward passes.

**Q: How does GRPO avoid a value network?**
A: It samples a group of answers per question and uses the group's mean reward as the baseline.

**Q: Why should you doubt the 79.9% figure a little?**
A: AIME'24 has 30 problems, so scores are multiples of 3.3 points. 79.9% isn't one, and the report doesn't explain it.

**Q: What fixed the drop after about 320 GRPO steps on 7B?**
A: Raising the context window from 8,192 to 16,384 tokens, so hard answers were no longer cut off.

## Try it in Colab
- Build the distillation file; count rejected answers, including right answers with a wrong step.
- Compare the five pickers; look at the gap between first sample and oracle.
- On a GPU, set `RUN_LONG = True` and watch GRPO raise the tiny model's mean reward.
