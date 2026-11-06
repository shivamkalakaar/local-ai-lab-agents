---
summary: How a Stanford team tested agents that design agents, and why you do the cost arithmetic before building one.
minutes: 16
source: CS329A past project · Towards Effective Meta-Agents
slides: https://cs329a.stanford.edu/winter2025projects/batu.pdf
---

## Learning objective
- Run a small meta-agent loop, measure how alike its designs are, and decide with break-even math whether a designed agent is worth its cost.

# The problem and why it matters

## The project at a glance
- Course title: "Towards Effective Meta-Agents: Designing Agents that Design Agents".
- Report title: "Inefficiencies of Meta Agents for Agent Design".
- Poster title: "Exploring Two Open Questions in Meta-Agent Design" (an earlier stage of the project).
- Authors: Batu El, Mert Yuksekgonul, James Zou (Stanford).
- Sources: the 10-page report and the poster.

## Why automate agent design?
- Agentic systems now do research, finance work and task automation.
- Until recently, researchers hand-built them with domain knowledge.
- The Bitter Lesson (Sutton, 2019): hand-made designs lose to methods that scale with search and learning.
- So recent work lets a meta-agent design agents.
- Meta-agent: an agent that writes new agents.

## The sample–evaluate–iterate loop (Figure 1)
- The meta-agent samples a new agent.
- It is evaluated on training questions.
- It is added to an archive.
- The next sample is conditioned on (part of) the archive.
- Archive: every design so far, with its scores.

## Three questions they ask
| Question | Short name |
|---|---|
| Does the meta-agent learn from earlier designs? | Meta-learning |
| Are the designs different enough to combine per question? | Diversity and complementarity |
| Does design cost ever pay back? | Economic viability |

## Their three findings in one slide
- Showing all past agents (as ADAS does) is worse than showing none of them.
  - An evolutionary context (best designs as "parents") does better.
- Designed agents behave alike, which limits combining them; most alike with evolutionary context.
- Only on two datasets, past about 15,000 queries, is designing cheaper than using a hand-made agent.

# Background they build on

## ADAS: the main reference (Hu et al., 2024)
- ADAS: Automated Design of Agentic Systems.
- Searches for agents in code space: each agent is a Python program.
- A meta-agent writes the next agent after seeing all earlier ones.
- This report tests ADAS's assumption that the meta-agent learns from that history.

## Other meta-agent work
| Approach | How it designs agents |
|---|---|
| MAS-GPT, ScoreFlow | Train a model to generate a multi-agent system per query |
| AgentSquare, Archon | Recombine fixed modules to search the design space |
| AutoFlow, Weak-for-Strong, ADAS | Sample–evaluate–iterate with a meta-agent |
| Multi-agent Supernet, Flow, AFlow | Other automated workflow designers |

- Recursive self-improvement: STOP, Promptbreeder, Gödel Agent refine their own prompts or code.
- Cost-of-pass (Erol et al., 2025): cost of producing one correct answer; the basis of their cost analysis.

## Setup definitions (Section 3)
- Agent f_i: a program that takes a question and calls a language model to answer.
- Score vector s_i = eval(f_i, D_train): one score per training question.
- Archive A: designed agents and their score vectors; starts with the initial library F.
- Π: the language model that samples a new design.
- φ: the curation step; picks which part of the archive the meta-agent sees.
- Each new design is revised for format and debugged with execution feedback.

## Algorithm 1 in words
- Start: A = the initial agents and their scores.
- For t = 1 … T:
  - Â = φ(A): choose the context.
  - f_t ~ Π(· | Â): sample a new agent, then revise and debug.
  - s_t = eval(f_t): score it on training questions.
  - Add (f_t, s_t) to A.

# The method, step by step

## Three ways to choose the context (φ)
| Strategy | What the meta-agent sees |
|---|---|
| Cumulative (φ_C, as in ADAS) | All past agents |
| Parallel (φ_P) | Only the 7 initial agents; each new sample is independent |
| Evolutionary (φ_E) | The 7 best-scoring agents so far ("parents", top-k) |

## The initial library (A.1)
| Agent | What it does |
|---|---|
| Chain-of-Thought | Reason before answering |
| Majority Voting | Pick the most common of several answers |
| Refinement from Feedback | Improve the answer from self-feedback |
| LLM Debate | Several model copies argue |
| Quality-Diversity | Generate diverse answers and combine them |
| Routing | Send the task to a model prompted as the right expert |
| Step-back | First recall the relevant principles, then answer |

- Same library as ADAS.

## Tasks and models (A.2)
| Dataset | Skill | Train | Test | Metric |
|---|---|---|---|---|
| MGSM | Multilingual grade-school math | 128 | 200 | Accuracy |
| DROP | Reading comprehension with arithmetic | 128 | 200 | F1 |
| MMLU | Multi-subject problem solving | 128 | 200 | Accuracy |
| GPQA | Graduate-level science | 32 (each run 5×) | 160 | Accuracy |

- Meta-agent: gpt-4o. Agents run on gpt-3.5. Same as ADAS.
- T = 30 iterations. Every result averaged over 3 runs.

## Measuring diversity
- Stack every agent's score vector as a row: matrix S (Figure 6).
  - White cell = right, black = wrong, gray = partial (DROP F1, GPQA repeats).
- Each row is the agent's "embedding" from the questions' point of view.
- Cosine similarity matrix C: entry (i, j) = cosine(s_i, s_j).
- Near 1: the two agents solve the same questions.
- Coverage: share of questions at least one designed agent gets right.

## Measuring cost
- Total cost of a designed agent: `C0 + n · Cj`.
- C0: fixed design cost; all sampling (line 6) plus evaluating each design on training (line 7).
- Cj: inference cost per query once deployed.
- n: number of queries served.
- Hand-made agents have C0 = 0.
- Break-even: the n where cost per correct answer drops below the best initial agent's.

# Results

## Table 1: best agent's training score
| Dataset | Cumulative | Parallel | Evolutionary |
|---|---|---|---|
| DROP | 71.4 (2.0) | 72.5 (4.2) | 74.4 (3.2) |
| MGSM | 41.4 (6.2) | 56.2 (10.5) | 56.5 (4.7) |
| MMLU | 74.7 (2.0) | 76.3 (1.6) | 76.6 (2.7) |
| GPQA | 32.3 (2.6) | 35.2 (2.8) | 33.8 (2.2) |
| Avg. | 55.0 | 60.0 | 60.3 |

- In brackets: standard deviation over 3 runs.

## Table 1: average of the top 5, 10, 15 agents (train)
| Top-k average | Cumulative | Parallel | Evolutionary |
|---|---|---|---|
| Best-5 | 50.0 | 55.9 | 57.0 |
| Best-10 | 46.7 | 53.2 | 54.8 |
| Best-15 | 44.0 | 51.2 | 53.1 |

- Averages over the four datasets.
- Evolutionary leads more as k grows: its whole top set is stronger.
- MGSM Best-15: 22.4 (cumulative) vs. 39.8 (parallel) vs. 42.7 (evolutionary).

## Table 1: test score of the best-on-train agent
| Dataset | Initial | Cumulative | Parallel | Evolutionary |
|---|---|---|---|---|
| DROP | 64.8 (1.3) | 71.9 (3.2) | 72.6 (7.8) | 73.2 (5.1) |
| MGSM | 38.4 (2.8) | 41.2 (4.8) | 51.8 (7.6) | 53.5 (2.0) |
| MMLU | 62.8 (2.3) | 66.2 (4.2) | 67.8 (0.8) | 65.8 (3.3) |
| GPQA | 30.0 (2.4) | 29.7 (2.7) | 31.3 (0.0) | 28.5 (3.1) |
| Avg. | 49.0 | 52.2 | 55.9 | 55.2 |

- Initial: best initial agent, chosen by training score.

## Finding 1: history doesn't help (Section 4.1)
- Cumulative did not beat parallel.
- So ADAS-style meta-agents gain little from past designs.
- They do worse than ignoring past designs entirely.
- Evolutionary helps: up to +10 points over cumulative on MGSM (training best 41.4 → 56.5).
- Lesson: include only high-quality past designs in context.

## Training curves (Figure 11)
- Training score of each designed agent over the 30 iterations.
- A dotted red line marks the best initial agent.
- Most designed agents hover around that line, rarely far above it.
- Frequent drops to near 0: designs that failed outright.
- No strategy shows a clear upward trend over iterations.

## Finding 2: diversity (Figure 2)
- Histograms of each agent's average similarity to all other agents, per dataset.
- Evolutionary: generally the highest similarity (least diverse).
- Cumulative: the lowest similarity overall.
- Parallel: similar scores to evolutionary, but slightly more diverse.
- GPQA: parallel gives both better and more diverse agents.
- Trade-off: exploitation (evolutionary) vs. exploration (parallel).

## Coverage (Table 2)
| Strategy | DROP | MGSM | MMLU | GPQA | Avg. |
|---|---|---|---|---|---|
| Cumulative | 96.6 | 89.1 | 99.2 | 91.9 | 94.2 |
| Parallel | 96.0 | 95.3 | 97.7 | 94.4 | 95.9 |
| Evolutionary | 93.6 | 93.0 | 99.2 | 91.9 | 94.4 |

- Over all 90 agents designed across 3 runs.
- Parallel: highest coverage, so it explores best.
- Coverage is the ceiling for a perfect per-question picker; best single agents sit far below it.

## Finding 3: cost per query (Figure 3)
- Average inference cost per test query: Cumulative > Evolutionary > Parallel > Initial.
- Same order on all four datasets in the bars.
- Best agents only (Figure 5): parallel again the cheapest designed agent.
- Every designed group still costs more than the initial library.

## Design cost over time (Figure 12)
- Cost of designing the next agent, per iteration.
- Parallel and evolutionary: stable.
- Cumulative: grows linearly, because the context keeps getting longer.
- Each bar splits into sampling, reflection and debugging; reflection is the largest part.
- Debugging adds spikes when a design fails to run.

## Break-even (Figure 4)
- Plots questions solved (solid) and attempted (dashed) against dollars spent.
- x-intercept: the fixed cost C0 (0 for the initial agent). Slope: cost per attempt or solution.
- Break-even where the red solid line (initial) crosses another solid line.
- Result: about n = 15,000 queries, only for DROP and MMLU with parallel context.
- All other datasets and strategies: gains never justify the cost, at any scale.

# The poster: an earlier version

## Different questions on the poster
- 1. Cut error and cost build-up while still learning from past attempts → Dynamic Context Management (DCM).
- 2. Agent selection, like model selection in ML → Retrieval Augmented Routing (RADAR).
- Also: Simple Abstractions (SA), a 17-line agent interface instead of ADAS's 172 lines.
- Seed agents: Chain-of-Thought, Majority Voting, Feedback and Revision, Routing, Task Decomposition.
- A bar chart shows design cost per iteration (sampling, reflection, debugging) rising over 30 iterations.

## Poster: simpler abstractions help
| Dataset | ADAS abstractions (172 lines) | Simple abstractions (17 lines) |
|---|---|---|
| MGSM | LLM Debate 47.7 ± 4.3 | LLM Debate 55.47 ± 4.3 |
| MMLU | LLM Debate 72.7 ± 3.9 | Chain of Thought 75.0 ± 3.7 |
| DROP | Self-Quality-Diversity 67.2 ± 1.2 | Majority Vote 73.93 ± 3.6 |

- Best initial agent's test accuracy ± std (poster Table 2).

## Poster: dynamic context management
| Dataset | CoT | ADAS | DCM | SA + DCM |
|---|---|---|---|---|
| MGSM | 35.9 ± 4.1 | 57.0 ± 4.3 | 66.4 ± 4.1 | 70.3 ± 3.9 |
| MMLU | 64.8 ± 4.1 | 74.2 ± 3.7 | 75.0 ± 3.7 | 75.0 ± 3.7 |
| DROP | 63.5 ± 1.0 | 67.2 ± 1.2 | 74.0 ± 1.2 | 75.3 ± 3.5 |

- Curves: with cumulative context, fitness spikes once then collapses (DROP) or trends down (MMLU).
- With DCM, fitness stays higher and trends up.

## Poster: chosen on train vs. chosen on test
| Dataset | Initial (train / test) | ADAS (train / test) | DCM (train / test) |
|---|---|---|---|
| MGSM | 47.7 / 47.7 | 48.4 / 57.0 | 64.8 / 66.4 |
| MMLU | 65.6 / 72.7 | 65.6 / 74.2 | 65.6 / 75.0 |
| DROP | 67.2 / 67.2 | 64.5 / 67.2 | 74.0 / 74.0 |

- "Selected on training" vs. "selected on test": picking the agent by test score flatters results.
- CoT baseline: 35.9 / 64.8 / 63.5.

## Poster: routing (RADAR)
| Dataset | CoT | ADAS | SA + DCM | SA + DCM + RADAR |
|---|---|---|---|---|
| MGSM | 35.9 ± 4.1 | 48.4 ± 4.3 | 63.28 ± 4.1 | 71.09 ± 3.9 |
| DROP | 63.5 ± 1.0 | 64.5 ± 1.0 | 67.68 ± 3.85 | 64.57 ± 3.9 |

- Routing helped on MGSM, hurt slightly on DROP.
- Pie charts: on DROP the router used mostly one agent; on MGSM it spread across several debate variants.
- The final report dropped DCM and RADAR and focused on the three inefficiencies.

# Limitations and future work

## Limitations they state
- Scope: only sample–evaluate–iterate meta-agents.
- Metrics: accuracy and F1; may not fit domains that need consistency.
- Cost analysis fits domains with strong verifiers.
- Cosine similarity favors agents that succeed on the same questions.
  - Strong agents look alike; always-failing agents look unrelated.
  - Checked with Hamming distance; similar trends (Figures 9, 10).

## Noise in meta-evaluation
- Five sources of randomness:
  - LM output randomness.
  - Errors compounding inside an agent's chain.
  - The meta-agent's own sampling.
  - Noisy scores for designed agents, which steer later choices.
  - Whole trajectories diverging.
- Reliable conclusions need many runs; they could afford 3.

## Safety
- Meta-agents generate and run complex code with no human check.
- Such systems are hard to audit or control before deployment.

# Our smaller Colab rebuild

## What we change
| Paper | Our rebuild |
|---|---|
| Model writes agent code | Design = data: system prompt + 1 of 4 fixed procedures; no model code runs |
| gpt-4o designs, gpt-3.5 runs | llama3.1:8b designs, llama3.2:3b runs |
| 7 starting agents, 30 iterations | 3 starting agents, 3 iterations, evolutionary context (top 3) |
| 4 benchmarks, 128 train / 200 test | 6 kinds of checkable puzzles, 6 train / 4 test |
| Dollars | Tokens ≈ characters / 4 |

- Procedures: direct, think_step_by_step, solve_then_check, plan_then_solve.
- Starting library: Direct, Chain-of-Thought, Self-Refine.

## Our six puzzle kinds
| Kind | Example |
|---|---|
| count | How many times does 'r' appear in 'strawberry'? |
| reverse | Write 'balloon' backwards |
| sort | Sort six numbers |
| weekday | Today is Monday; what day in 23 days? |
| math | Crates of bottles, some sold; how many left? |
| compare | Which is larger: 3.45 or 3.9? |

- Mixed kinds give designs a chance to specialize.

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

## Solutions: three experiments
- Parallel context (RUN_LONG): compare best score and mean similarity with evolutionary.
- Per-kind router: pick the best agent per puzzle kind; compare with best single agent and coverage.
  - With one task per kind, this is "choosing on the test set", the poster's warning.
- 10× designer price: only the meta-agent part of C0 grows.

# How this applies to us

## Should a firm build an agent designer?
- Idea: a system that keeps inventing better contract-review agents.
- Local models lower the price per call, not the number of calls.
- Do the arithmetic first: break-even = C0 / (accuracy × gap in cost per correct answer).

## Plug in real numbers
| Input | Example |
|---|---|
| Break-even | 15,000 clauses (the paper's best case) |
| Firm volume | 500 clauses a month |
| Time to pay back | 30 months |

- And that's the best case; on two of four datasets it was "never".

## If you do build one
- Show the designer the best few designs, not everything.
- Hold out test clauses; pick the agent on train only.
- Run it several times before trusting a winner.
- Keep designs as data (prompt + fixed procedure) so no model-written code runs on the firm's machine.

## Common mistakes
- Skipping the cost math → compute break-even first.
- Showing the designer everything → show the best few designs.
- Expecting a big "best agent per question" boost → designs overlap; push for variety.
- Picking the winner by test score → choose on train, report test.
- Trusting one run → meta-evaluation is noisy; use several runs.

## Check your understanding
**Q: Which context strategy did ADAS use, and how did it compare?**
A: Cumulative, showing all past agents. Its best agent scored 55.0 on train, below parallel (60.0) and evolutionary (60.3), Table 1.

**Q: What do C0 and Cj stand for in the cost formula?**
A: C0 is the one-time cost of designing the agent. Cj is the cost per question once it is in use.

**Q: When did a designed agent pay for itself?**
A: Only after about 15,000 queries, and only on MMLU and DROP with parallel context. Elsewhere never.

**Q: Which strategy made the least diverse agents?**
A: Evolutionary. Its agents had the highest similarity; parallel had the highest coverage (95.9%).

**Q: What does coverage tell you?**
A: The share of questions at least one agent solves. It is the ceiling for a perfect per-question router.

**Q: Why does cumulative design cost grow over time?**
A: Its context holds every past agent, so each new prompt is longer (Figure 12).

**Q: Why can't we trust a single meta-agent run?**
A: Randomness enters at five levels, from model outputs to whole design trajectories. The paper averaged 3 runs.

## Try it in Colab
- Test the 3 starting agents, then let the meta-agent design 3 more; compare train and test.
- Read the right/wrong table, mean similarity and coverage.
- Read the break-even line; then try the 10x designer price in the solutions.
