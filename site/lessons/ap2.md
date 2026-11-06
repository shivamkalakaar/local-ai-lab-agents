---
summary: How a Stanford team let an agent pick the reasoning trick for each step, and why you compare at equal call budgets.
minutes: 15
source: CS329A past project · PRIME
slides: https://cs329a.stanford.edu/winter2025projects/prime.pdf
---

## Learning objective
- Use tree search to pick a reasoning trick per subgoal, read its results with care, and compare it fairly at the same call budget.

# The problem and why it matters

## The project at a glance
- PRIME: Planning with Reflective, Iterative, Multi-agentic Exploration.
- Report authors: Chelsea Zou, Samuel Liu, Jui Khankari.
- The poster lists Chelsea Zou, Jui Khankari, Miriam Cheng.
- CS329A, Winter 2025. Sources: the 7-page report and the poster.

## Top-down planning is unsolved
- Many ways reach a goal, so the number of plans explodes.
- Most research instead searches step by step, bottom-up.
- Tricks that help multi-step reasoning: reflexion, debate, voting, self-feedback.
- Used alone, they are applied "ad hoc":
  - No principled way to know which trick suits which problem.
  - Vague signals about whether a step moves toward the goal.

## One trick everywhere vs. one trick per step
| Fixed strategy | PRIME |
|---|---|
| Reflect on every step | Easy step: solve once |
| Same effort on easy and hard steps | Hard step: reflect, debate, or drill down |
| Nothing learned between tasks | Keeps stats on which trick worked |

## Their idea in one line
- Treat each reasoning technique as an "option" and search over them with MCTS.
- Option: a reusable sub-policy from reinforcement learning (RL); here, one reasoning trick.
- Hypothesis: choosing tricks per subtask beats mixing them blindly, especially for small models with small budgets.
- Main claim: a structured, learnable way to combine tricks instead of trial and error.

# Background they build on

## Option discovery (Wan and Sutton, 2022)
- A new RL objective that makes planning faster.
- At each step, choose from a better subset of actions ("options") instead of every single action.
- Improved planning speed in four-room grid worlds.
- PRIME borrows the idea: search for the best reflective tool at each subgoal.

## Chain-of-thought and ReAct
| Method | What it does | Gap |
|---|---|---|
| Chain-of-Thought (Wei et al., 2022) | Write steps before the answer | No acting, no search |
| ReAct (Yao et al., 2023) | Interleave reasoning with actions in an environment | No structured search |

- CoT improved math, commonsense and symbolic logic.
- ReAct beat CoT on question answering and web navigation.

## LATS: tree search for agents (Zhou et al., 2024)
- LATS: Language Agent Tree Search.
- MCTS + self-reflection + outside feedback.
- Explores several reasoning paths, backtracks, refines.
- PRIME's critique:
  - Picks reasoning strategies by fixed heuristics.
  - Does not split problems into reusable subtasks.
- PRIME extends LATS: choose the reflective strategy per state, not one for all.

## MCTS in four moves
- MCTS (Monte Carlo Tree Search): grow a tree of choices; spend more tries on branches that pay off.
- Select: walk down by a score that mixes "worked before" and "rarely tried".
- Expand: add children (here: subgoals).
- Evaluate: score the new node (a checker, or a model's judgment).
- Backpropagate: push the result up so parents' values update.

## The three benchmarks
| Benchmark | What the agent does | Type |
|---|---|---|
| PlanBench (Valmeekam et al., 2022) | Write a plan in a block-style world with given action rules | Sequential decisions |
| WebShop (Yao et al., 2022) | Search and click in a store to buy an item that fits a request | Real-world decisions |
| Game of 24 (Yao et al., 2023) | Combine 4 numbers with + − × ÷ to make 24 | Pure reasoning |

- WebShop: 1.18 million real products, 12,000 human-written instructions.
- WebShop success: share of tasks where the chosen item meets every criterion.

# The method, step by step

## Planning framework (Figure 1)
- Root: "Problem". Three subgoal children: 1, 2, 3.
- Node 1 → "Backtrack"; its child 4 is crossed out.
- Node 2 → "Drill down" → node 5 "Solve", score 7.
- Node 3 → "Drill down" → 6 → 7 (drill down) → 8 "Solve", score 10.
- "Best path is 3 → 6 → 7 → 8!"
- At each node a self-reflective agent uses "the best tool to determine the next action".

## Step 1: planning decomposition
- A language model writes N plausible, actionable subgoals.
- Together they should reach the final goal.
- These become child nodes of a new tree.
- At each node the planner: checks its state, picks a reflection tool, picks one of three actions.
- Tools: self-reflection, self-refinement, debate.

## Step 2: three actions at each node
| Action | When | What happens |
|---|---|---|
| Drill down | Subgoal too broad or vague | Split into finer subgoals, until atomic |
| Solve | Node is a base case (a leaf) | Ask the best-suited tool to solve it |
| Backtrack | Subgoal irrelevant or unhelpful | Drop it, explore other subgoals |

- Tool choice prompt: "What is the best tool from the following set to accomplish this subgoal?"
- A leaf is decided by ground truth, or by the value function for subjective plans.

## Step 3: backpropagation
- Formula: `N(s_i) = N(s_(i−1)) + 1` and `V(s_i) = (V(s_(i−1)) · N(s_(i−1)) + r) / N(s_i)`.
- N: number of visits to the node.
- V: the node's value.
- r: reward for reaching the end state.
- In words: the new value is the running mean of all rewards so far.
- Effect: good strategies get stronger, bad ones weaker.

## Step 4: pick the next node with UCT
- Formula: `UCT(s) = w_i / n_i + C · sqrt( ln N(s) / n_i )`.
- w_i / n_i: average reward of child i (exploit what works).
- sqrt(ln N / n_i): large when child i was rarely tried (explore).
- N(s): visits to the parent. C: exploration constant, how curious to be.
- Repeat until the best plan is found.

## Step 5: the value function
- Leaves: a model judges how well-defined the subgoal is toward the goal ("self-judgement").
- Non-leaf nodes: `V(s) = λ · LM(s) + (1 − λ) · SC(s)`.
- LM(s): the language model's score for the state.
- SC(s): self-consistency score; do repeated samples agree?
- λ: a weight that balances the two. The report gives no value for λ.

# Setup

## Methods compared
- PRIME with GPT-4o-mini.
- LATS with GPT-4o-mini.
- GPT-4o-mini alone (called "brute force" in the text).
- GPT-o1, one shot.
- Reverse ablation: PRIME with only the planner upgraded, or only the executor.

## Data and run times
- 60 randomly sampled questions per benchmark ("60 runs" of Game of 24), due to limited compute.
- PlanBench: about 3 hours.
- WebShop: about 30 minutes.
- Game of 24: about 5 minutes.
- WebShop actions: search and click; observations: browser feedback plus the agent's own reflections.

## Two extra evaluations
- Counterfactual evaluation: change early plan steps on purpose; does the change carry through to the output?
  - Tests that plans really steer reasoning, not just justify it after the fact.
- Reverse ablation: upgrade one component at a time, keep the rest on 4o-mini.
  - Shows which part gains most from a better model.

# Results

## Table 1: success rates
| Method | PlanBench | WebShop | Game of 24 |
|---|---|---|---|
| PRIME | 36.2% | 40% | 75% |
| LATS | 2.2% | 38% | 44% |
| GPT-4o-mini | 0% | 0% | 0% |
| GPT-o1 | 100% | 100% | 100% |
| PRIME upgraded planner (4o) | 95% | 94% | 100% |
| PRIME upgraded execution (4o) | 96% | 92% | 100% |

- The poster shows the same table.

## What the text says (Section 4.4)
- PlanBench: PRIME 36.2%, "a 2.2% improvement over LATS"; 4o-mini 0%; o1 37.3% (cited from Valmeekam et al.).
- WebShop: PRIME 40%; o1 80%; LATS 38% (cited from the LATS paper).
- Game of 24: PRIME 75%; LATS 44% (cited); o1 100%.
- "Sometimes matching" o1 despite "approximately 10x less parameter count" for 4o-mini.
- More API calls than o1, but an estimated 4x lower cost (OpenAI prices, December 2024).
  - Footnote: the exact gap depends on tokens and steps per question.

## Caveat: the numbers disagree with each other
| Cell | Table 1 | Text |
|---|---|---|
| LATS, PlanBench | 2.2% | PRIME is "2.2% improvement over LATS" → LATS ≈ 34% |
| o1, PlanBench | 100% | 37.3% |
| o1, WebShop | 100% | 80% |
| Upgrade model | "(4o)" | "GPT-o1" |

- We can't tell which is right. Treat every cell as uncertain.

## Caveat: were the LATS rows rerun?
- The text says LATS was run with 4o-mini, but cites the LATS paper for its numbers.
- The LATS paper reports exactly 38.0% WebShop success and 0.44 on Game of 24, both with GPT-3.5.
- So the LATS rows may be copied from a different model, not rerun.
- The "10x fewer parameters" claim is also unsupported; OpenAI does not publish model sizes.

## Reading the reverse ablation
- Upgrading only the planner: 95% / 94% / 100%.
- Upgrading only the executor: 96% / 92% / 100%.
- Both jump far above PRIME's 36.2% / 40% / 75%.
- Takeaway: both parts are bottlenecks; neither alone explains the gap.
- Poster: run "to identify performance bottlenecks".

# Analysis

## PlanBench: what helped
- Planning helped a little on reasoning questions.
- Bigger gains came from search: more samples, more paths explored.
- More nodes n → consistently better, at more cost (for both LATS and PRIME).
- PRIME beat LATS by combining top-down subgoals with tree search.

## PlanBench: clear vs. unclear actions
| Unclear action outcomes ("obfuscated", Figure 2) | Clear outcomes ("non-obfuscated", Figure 3) |
|---|---|
| Actions like "paltry", "sip", "clip" with only fact rules | Same kind of world, action effects easy to follow |
| Output: "No plan found for subgoal: 1. Understand the given initial conditions." | Output: a full [PLAN] of clip / wretched / tightfisted / sip steps |
| Search wanders tree to tree, stuck in local minima | Plan found despite many steps |

- Their guess: the tree can't be "unrolled" when it's unclear where an action leads.

## WebShop: no real gain
- With k = 60 trajectories, PRIME and LATS scored about the same.
- Subgoals help less in complex, multi-input settings like shopping.
- Reflections were often generic and gave no useful feedback.
- The agent got stuck in local minima.
- (The report calls the method "PRISM" in this paragraph, likely an old name.)

## Game of 24: the clear win
- PRIME picks tools per puzzle; LATS uses one approach for all.
- Drill down: split hard number combinations into smaller steps.
- Solve: compute simple expressions directly.
- Works because every step's result is exact and checkable.

## Counterfactual analysis
- Changing early plan steps changed both the subgoals and the final outputs.
- So sub-agents really follow the planner.
- Open question: does the plan's wording mean to the model what it means to a person?

# Limitations and future work

## Limitations
- Stated: wanders when action value is unclear; generic reflections on WebShop.
- Stated: more nodes cost more compute.
- Our notes:
  - Only 60 questions per benchmark, one run, no error bars.
  - Table and text disagree; LATS rows may come from another paper and model.
  - No equal-budget comparison: PRIME uses more calls than single shots.
  - λ, C and N not reported.

## Future work they list
- Cluster problems: label training-problem types, learn the best tool set per cluster, reuse at test time.
- Better self-reflection: more tools, "meta-cognitive" reasoning.
- Adaptive value function: change how subgoals are judged as more problems are seen.

# Our smaller Colab rebuild

## What we change
| PRIME | PRIME-lite |
|---|---|
| 3 benchmarks, 60 questions each | Game of 24 only, 6 solvable puzzles |
| Full tree of subgoals | Planner proposes up to 4 first steps, with a confidence |
| Tools: reflection, refinement, debate | Tools: solve, reflect, drill down |
| LM score + self-consistency | Exact checker: reward 1 or 0 |
| GPT-4o-mini, o1 | Local llama models; 3B for the reverse ablation |

- Game of 24: use four numbers once each, with + − × ÷, to make 24.
- Code drops impossible steps for free (backtrack).
- Tool stats carry over from puzzle to puzzle.

## The exact checker
- Parses the answer with Python's `ast` module, never `eval`.
- Checks the numbers used match the puzzle exactly.
- Computes with exact fractions, so 8 / 3 stays exact.
- Returns a message the reflecting agent sees: "it equals 23, not 24".

## Our three tools
| Tool | Calls | What it does |
|---|---|---|
| solve | 1 | Follow the plan step, write one expression |
| reflect | up to 2 | Solve; if the checker says no, fix once |
| drill | 2 | Propose the next step (code checks it), then solve with both steps |

## Starting value of an arm
- `prior = 0.5 * confidence / 10 + 0.5 * successes / tries`.
- Our version of `V = λ·LM + (1−λ)·SC` with λ = 0.5.
- LM part: the planner's own confidence (1–10).
- Second part: how this tool did on earlier puzzles (instead of self-consistency).

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
- Lines 5–6: checker gives 1 or 0; update the running mean, as in the paper.

## Fair comparison: same budget
| Method | Calls per puzzle |
|---|---|
| Single shot | 1 |
| Retry with feedback (checker message, try again) | up to 5 |
| PRIME-lite | up to 5 |

- PRIME-lite beats single shot only → the extra calls did the work.
- PRIME-lite beats retry → the strategy choice did the work.

## Solutions: three experiments
- Reverse ablation: executor tools on llama3.2:3b, planner stays 8B; 2 puzzles.
- Turn off learning: reset tool stats each puzzle (left as written code; 6 puzzles is too noisy).
- Exploration: `C=0.2` sticks to the planner's favorite step; `C=2.0` spreads calls across arms.

# How this applies to us

## Easy and hard steps in a contract review
| Easy step | Hard step |
|---|---|
| "Find the governing-law clause" | "Does this indemnity break our 12-month liability cap rule?" |
| Solve once | Reflect, or drill into cap amount, carve-outs, time window |

- On the firm's own machine, calls are cheap but not free: spend them on the hard steps.

## Use search where a checker exists
- Tree search worked on Game of 24 (exact check), not on WebShop (fuzzy).
- Contract work has checkable parts: "is the cap ≤ 12 months of fees?" is a number check.
- Put the search there; use a single pass for open-ended summaries.

## Find which model deserves the upgrade
- Reverse ablation: run the planner on a bigger local model, executor on a small one, then swap.
- Pay for the bigger model only where accuracy jumps.

## Common mistakes
- Comparing at unequal budgets → add a retry baseline with the same call count.
- Tree search without a clear reward → use it where a checker exists.
- Upgrading every part at once → upgrade one part at a time.
- Trusting one table → this paper's numbers disagree in places; rerun what matters.
- Quoting a baseline from another paper as your own run → rerun it with your model.

## Check your understanding
**Q: What does the curiosity bonus in UCT do?**
A: It raises the score of moves tried rarely. The search doesn't lock onto its first favorite too early.

**Q: How does PRIME update a node's value after a try?**
A: Running mean: new value = (old value × visits + reward) / (visits + 1).

**Q: Why does the notebook include a "retry with feedback" baseline?**
A: It uses the same 5-call budget as PRIME-lite. If they tie, the gain came from extra tries, not strategy choice.

**Q: Where did PRIME struggle?**
A: On WebShop it nearly tied with LATS, with generic reflections. It also wandered when a step's value was unclear.

**Q: Name two ways PRIME's Table 1 disagrees with its text.**
A: Table 1 puts o1 at 100% on PlanBench and WebShop; the text says 37.3% and 80%. And the text calls 36.2% "a 2.2% improvement" over LATS, which the table lists at 2.2%.

**Q: What did the reverse ablation show?**
A: Upgrading only the planner or only the executor both lifted scores to 92–100%. Both parts were bottlenecks.

## Try it in Colab
- Compare single shot, retry with feedback and PRIME-lite: success rate next to average calls.
- Read the option statistics: which tool did the search learn to prefer?
- Run one puzzle with `C=0.2` and `C=2.0`; count the different arms tried.
