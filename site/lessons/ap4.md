---
summary: How a Stanford team tested agents that design agents, and why you do the cost arithmetic before building one.
minutes: 9
source: CS329A past project · Towards Effective Meta-Agents
slides: https://cs329a.stanford.edu/winter2025projects/batu.pdf
---

## Learning objective
- Run a small meta-agent loop and decide, with break-even math, whether a designed agent is worth its cost.

## The problem
- Meta-agent: an agent that writes new agents.
- ADAS (Automated Design of Agentic Systems): sample a design → evaluate → add to archive → repeat.
- Archive: every design so far, with its score.
- Three open questions:
  - Does it learn from past designs?
  - Are the designs different enough to combine?
  - Does the design cost ever pay back?

## What the students built
- Team: Batu El, Mert Yuksekgonul, James Zou (paper: "Inefficiencies of Meta Agents for Agent Design").
- Archive starts with 7 hand-written agents: Chain-of-Thought, Majority Voting, Refinement from Feedback, LLM Debate, Quality-Diversity, Routing, Step-back.
- gpt-4o designs; the agents run on gpt-3.5.
- 30 iterations, 3 runs, on MGSM, DROP, MMLU and GPQA.

## Three context strategies
| Strategy | What the designer sees |
|---|---|
| Cumulative (ADAS) | All past agents |
| Parallel | Only the 7 starting agents |
| Evolutionary | The 7 best agents so far ("parents") |

## How they measured it
- Score vector: one agent's right/wrong row over the questions.
- Cosine similarity: how alike two rows are; near 1 = same questions right.
- Coverage: share of questions at least one agent gets right.
- Total cost = C0 + n × Cj.
  - C0: one-time design cost
  - Cj: cost per question once deployed
  - n: number of questions

## What they found: scores
| | Cumulative | Parallel | Evolutionary |
|---|---|---|---|
| Best train score (avg) | 55.0 | 60.0 | 60.3 |
| Test score (avg) | 52.2 | 55.9 | 55.2 |
| Coverage (avg) | 94.2% | 95.9% | 94.4% |

- Initial library on test: 49.0.
- Showing everything did not beat showing only the start.
- Evolutionary agents were the least diverse.

## What they found: cost
- Designed agents cost more per query: cumulative > evolutionary > parallel > initial.
- Cheaper per correct answer only after about 15,000 queries.
- Only on MMLU and DROP, with parallel context.
- Elsewhere: never.

## Our Colab rebuild: smaller
| Paper | Our rebuild |
|---|---|
| Model writes agent code | Design = data: system prompt + 1 of 4 fixed procedures; no model code runs |
| gpt-4o designs, gpt-3.5 runs | llama3.1:8b designs, llama3.2:3b runs |
| 7 starting agents, 30 iterations | 3 starting agents, 3 iterations, evolutionary context |
| 4 benchmarks | 6 kinds of checkable puzzles (letters, reversing, sorting, weekdays, word problems, decimals) |

- Procedures: direct, think_step_by_step, solve_then_check, plan_then_solve.

## Break-even in code
```python
C0 = sum(tok(a["chars"] + a.get("meta_chars", 0)) for a in designed)
c_init, c_new = tok(test_init["chars"]) / len(TEST), tok(test_new["chars"]) / len(TEST)
p_init, p_new = max(test_init["score"], 1e-9), max(test_new["score"], 1e-9)
gap = c_init / p_init - c_new / p_new
breakeven = C0 / (p_new * gap) if gap > 0 and best_new is not best_init else float("inf")
```
- Line 1: C0 = designer prompts + testing every new design; `tok` ≈ characters / 4.
- Line 2: tokens per question, old agent vs. new.
- Line 3: test accuracy; the tiny number avoids dividing by zero.
- Line 4: difference in cost per correct answer.
- Line 5: questions needed to earn back C0; no gap → never.

## One call vs. two calls
| Starting agent | Designed agent |
|---|---|
| "Direct": 1 call per question | "solve_then_check": 2 calls per question |
| C0 = 0 | C0 = all design and testing |
| Lower accuracy | Often higher accuracy |

- Two calls double the per-query cost, so break-even can be "never".

## How this applies to us
- A firm might want a system that keeps inventing better contract-review agents.
- Local models lower the price per call, not the number of calls.
- Plug in real numbers: if break-even is 15,000 clauses and you review 500 a month, that is years away.

## Common mistakes
- Skipping the cost math → compute break-even first.
- Showing the designer everything → show the best few designs.
- Expecting a big "best agent per question" boost → designs overlap; push for variety.
- Trusting one run → meta-evaluation is noisy; use several runs.

## Check your understanding
**Q: Which context strategy did ADAS use, and how did it compare?**
A: Cumulative, showing all past agents. It scored 55.0 on train, below parallel (60.0) and evolutionary (60.3).

**Q: What do C0 and Cj stand for in the cost formula?**
A: C0 is the one-time cost of designing the agent. Cj is the cost per question once it is in use.

**Q: When did a designed agent pay for itself?**
A: Only after about 15,000 queries, and only on MMLU and DROP with parallel context. Elsewhere never.

## Try it in Colab
- Test the 3 starting agents, then let the meta-agent design 3 more; compare train and test.
- Read the right/wrong table, mean similarity and coverage.
- Read the break-even line; then try the 10x designer price in the solutions.
