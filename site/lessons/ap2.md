---
summary: How a Stanford team let an agent pick the reasoning trick for each step, and why you compare at equal call budgets.
minutes: 9
source: CS329A past project · PRIME
slides: https://cs329a.stanford.edu/winter2025projects/prime.pdf
---

## Learning objective
- Use tree search to pick a reasoning trick per subgoal, and compare it fairly at the same call budget.

## The problem
- Many tricks make a model reason better: reflection, refinement, debate, voting.
- Teams apply them ad hoc: one trick, used the same way on every step.
- No principled way to pick the right trick for each step of a multi-step plan.

## One trick everywhere vs. one trick per step
| Fixed strategy | PRIME |
|---|---|
| Reflect on every step | Easy step: solve once |
| Same effort on easy and hard steps | Hard step: reflect, or drill down |
| Nothing learned between tasks | Keeps stats on which trick worked |

## What the students built: PRIME
- Team: Chelsea Zou, Samuel Liu, Jui Khankari (the poster lists Chelsea Zou, Jui Khankari, Miriam Cheng).
- Option: one reasoning technique, treated as a reusable move.
- MCTS (Monte Carlo Tree Search): try promising moves, keep score, spend more on winners.
- Built on LATS (Language Agent Tree Search), an earlier tree-search agent.

## How it works
- Planner: splits the task into subgoals.
- At each node, three actions:
  - Drill down: split the subgoal further
  - Solve: call a tool (self-reflection, self-refinement, debate)
  - Backtrack: drop a useless subgoal
- UCT: score = how well a move did + a bonus for moves tried rarely.
- Running mean: a move's value = average of all its rewards so far.

## Setup
- GPT-4o-mini inside PRIME.
- Compared with LATS, plain GPT-4o-mini and one-shot GPT-o1.
- 60 sampled questions each from PlanBench, WebShop and Game of 24.
- Reverse ablation: upgrade only the planner, or only the executor, to a stronger model.

## What they found (Table 1, success rate)
| Method | PlanBench | WebShop | Game of 24 |
|---|---|---|---|
| PRIME (4o-mini) | 36.2% | 40% | 75% |
| LATS (4o-mini) | 2.2% | 38% | 44% |
| GPT-4o-mini | 0% | 0% | 0% |
| GPT-o1 | 100% | 100% | 100% |
| PRIME, upgraded planner | 95% | 94% | 100% |
| PRIME, upgraded executor | 96% | 92% | 100% |

- Caution: the paper's text gives different numbers for some cells.
- About 4x cheaper than o1, despite more calls (their estimate).

## Where it struggled
- WebShop: PRIME and LATS nearly tied; reflections were generic.
- Unclear step value (obfuscated PlanBench): the search wandered.
- Tree search shines when each step's result is clear and checkable.

## Our Colab rebuild: smaller
| PRIME | PRIME-lite |
|---|---|
| 3 benchmarks, 60 questions each | Game of 24 only, 6 solvable puzzles |
| Full tree of subgoals | Planner proposes up to 4 first steps, with a confidence |
| Tools: reflection, refinement, debate | Tools: solve, reflect, drill down |
| LM score + self-consistency | Exact checker: reward 1 or 0 |

- Game of 24: use four numbers once each, with + − × ÷, to make 24.
- Code drops impossible steps for free (backtrack).
- Tool stats carry over from puzzle to puzzle.

## The selection loop in code
```python
uct = lambda a: a["V"] + C * math.sqrt(math.log(total) / a["N"])
arm = max(arms, key=uct)
ok, used = run_tool(arm["tool"], nums, steps, arm["left"], budget - calls)
calls += used
r = 1.0 if ok else 0.0
arm["V"] = (arm["V"] * arm["N"] + r) / (arm["N"] + 1); arm["N"] += 1; total += 1
```
- Arm: one (first step, tool) pair, e.g. "8 - 2 = 6, then reflect".
- Line 1: UCT score; `V` = past success, square root = curiosity bonus, `C` = how curious.
- Line 2: pick the highest-scoring arm.
- Line 3: run the tool; get success and calls used.
- Line 4: count calls against the budget of 5.
- Lines 5–6: checker gives 1 or 0; update the running mean.

## Fair comparison: same budget
| Method | Calls per puzzle |
|---|---|
| Single shot | 1 |
| Retry with feedback (checker message, try again) | up to 5 |
| PRIME-lite | up to 5 |

- PRIME-lite beats single shot only → the extra calls did the work.
- PRIME-lite beats retry → the strategy choice did the work.

## How this applies to us
- A contract review mixes easy steps ("find governing law") and hard ones ("does this indemnity break the 12-month cap rule?").
- On the firm's own machine, calls are cheap but not free: spend them on the hard steps.
- Reverse ablation shows which step deserves the bigger local model.

## Common mistakes
- Comparing at unequal budgets → add a retry baseline with the same call count.
- Tree search without a clear reward → use it where a checker exists.
- Upgrading every part at once → upgrade one part at a time.
- Trusting one table → this paper's numbers disagree in places; rerun what matters.

## Check your understanding
**Q: What does the curiosity bonus in UCT do?**
A: It raises the score of moves tried rarely. The search doesn't lock onto its first favorite too early.

**Q: Why does the notebook include a "retry with feedback" baseline?**
A: It uses the same 5-call budget as PRIME-lite. If they tie, the gain came from extra tries, not strategy choice.

**Q: Where did PRIME struggle?**
A: On WebShop it nearly tied with LATS, with generic reflections. It also wandered when a step's value was unclear.

## Try it in Colab
- Compare single shot, retry with feedback and PRIME-lite: success rate next to average calls.
- Read the option statistics: which tool did the search learn to prefer?
- Run one puzzle with `C=0.2` and `C=2.0`; count the different arms tried.
