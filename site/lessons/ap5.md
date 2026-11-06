---
summary: How a Stanford team added a planner and web search to a pipeline of model calls, and why search helps only when it finds the right pages.
minutes: 13
source: CS329A past project · ARCHON Augmented
slides: https://cs329a.stanford.edu/winter2025projects/archon.pdf
---

## Learning objective
- Build a plan → search → generate → critique → fuse → verify pipeline, ablate each block on your own data, and spot when a comparison isn't fair.

# The problem and why it matters

## The project at a glance
- ARCHON Augmented: Planning and Web-Enhanced Components.
- Team: Megan Mou, Sherry Xie, Emily Zhang, Andrew Park (CS329A, Winter 2025).
- Poster title: "Multi-Component ARCHON".
- Sources: the 11-page report and the poster.
- Code: github.com/49emily/cs329a-archon.

## What Archon is
- Archon: a framework that stacks inference-time techniques into a pipeline of model calls.
- Inference-time technique: a way to spend more compute at answer time (sample more, critique, merge) instead of training.
- Idea: many LLMs plus many techniques can beat any one of them alone.
- It picks a good stack for each benchmark by search.

## Four gaps they saw
- Only seven building blocks; room for more.
- No web search or tool use.
- The query category is fixed in advance; no per-query adjustment.
- Strong results relied on models of about 70B parameters; small open models untested.

## Research question
- Can more inference-time components improve Archon for both closed-source and open-source models?
- Their answer, in short:
  - Expander value depends on the benchmark.
  - New blocks let open models rival closed ones.
  - Web search helps instruction following.
  - Reasoning stays hard.

# Background they build on

## Archon (Saad-Falcon et al., 2024)
| Block | What it does |
|---|---|
| Generator | Writes candidate answers |
| Fuser | Merges candidates into one answer |
| Critic | Notes each candidate's strengths and flaws |
| Ranker | Orders candidates, best first |
| Verifier | Checks the answer is supported |
| Unit Test Generator | Writes tests for a coding answer |
| Unit Test Evaluator | Runs answers against those tests |

- Blocks are stacked in layers.
- ITAS (inference-time architecture search): searches over stacks and settings.
- Its own limits section: adding new techniques is "a promising avenue".

## Archon's instruction-following stack (poster Figure 1)
```
Prompt → Generator (n=10 samples) → Critic → Ranker → Fuser → Output
```
- In the Archon paper's all-source version, every block uses Claude 3.5 Sonnet.

## Other work they draw on
| Work | Idea | Use here |
|---|---|---|
| Mixture-of-Agents, LLM-Blender | Combine many LLMs at answer time | Archon's roots; narrower scope |
| ReAct (Yao et al., 2023) | Interleave reasoning and actions | Inspired a dedicated planning layer |
| Toolformer (Schick et al., 2023) | Models learn to call tools | Inspired the web search block |
| "LLMs still can't plan" (Valmeekam et al., 2024) | Planning is hard even for o1 | Why a planner might matter most |

## The two benchmarks
| Benchmark | What it tests | How scored |
|---|---|---|
| Humanity's Last Exam (HLE) | Frontier academic knowledge and reasoning; 2,700 questions, 100+ subjects | Accuracy |
| AlpacaEval | Following real-world instructions | Win rate vs. a strong reference model, judged automatically |

- They used 500 HLE questions (compute limit). HLE is far from solved by frontier models.
- LC (length-controlled) win rate: corrects the judge's bias toward longer answers.

# The method, step by step

## Three new blocks (Figure 1)
| Block | Where | What it does |
|---|---|---|
| Planner | First | Rephrases the request, writes a step-by-step plan; the plan is prepended to later inputs |
| Web Search Tool Call | After the Planner, before the Generator | Writes q queries from the plan; top 10 Google results per query become context |
| Expander | After each Generator | Adds context, detail or clarification so Critics and Rankers can judge |

- Web search block was inspired by the course's Homework 2.

## The full pipeline (poster Figure 2, ablation 3)
```
Planner → Web Search Tool Call (queries=5) → Generator (n=10) → Expander → Critic → Ranker → Fuser
```
- The poster gives 5 queries; the report only says q.

## The Planner prompt (A.1.1)
- System: "You are an expert at planning and reasoning given a user query."
- Step 1: "Summarize what the user is asking in a clearer way."
- Step 2: "Provide a step-by-step reasoning process on what you need to do in order to fully answer the user query."
- Their hypothesis: the planner would matter most, since reasoning is crucial yet hard.

## The Expander prompt (A.1.2)
- "Only add information that is directly relevant to the original response."
- "Maintain factual accuracy and consistency with the original."
- "Focus on adding valuable context that enhances understanding."
- "Keep additions clear and well-structured."
- "Do not contradict or modify the original content."

## The search prompts (A.2)
- Query writing: given the query and the step-by-step analysis, "output {num_searches} search queries", comma-separated, nothing else.
- Generation: "Use these search results to improve your answer to the user's query".

## Blocks they considered and dropped
- Rewriter, summarizer, debugger.
- Too close to existing blocks (especially the verifier), or too costly for a small gain.

## The ablations
- 1. Planner + base Archon.
- 2. Planner + Expander + base Archon.
- 3. Planner + Tool Call + Expander + base Archon.
- Each compared with base Archon.
- Two questions: which blocks help most; how do they change open vs. closed models?

# Setup

## Two model configurations (Section 4.3)
| Config | Generation | Critic | Ranker | Fuser |
|---|---|---|---|---|
| Closed-source | GPT-4o | Qwen2-72B-Instruct | GPT-4o | Qwen2-72B-Instruct |
| Open-source | Ensemble (below) | Qwen2-72B-Instruct | Qwen2-72B-Instruct | Qwen2-72B-Instruct |

- Open ensemble (text): Qwen2-72B-Instruct, DeepSeek R1 Distill, WizardLM-2 8x22B, QwQ-32B, LLaMA-3 70B Chat, Mixtral 8x22B.
- The appendix diagram lists 7 generator models (Qwen2 twice).

## What the appendix diagrams actually show (A.3)
| Setup | Diagram |
|---|---|
| archon-closed | generator gpt-4o, critic Qwen2, ranker gpt-4o, fuser Qwen2 |
| planner-archon-closed | every block gpt-4o |
| planner-expander-archon-mixed | planner Qwen2.5, 7-model generator, rest Qwen2 |
| planner-toolcall-expander-archon-mixed | planner Qwen2.5; tools, generator, expander, ranker gpt-4o; critic and fuser Qwen2 |

- So the "mixed" setup with search is mostly GPT-4o, not open models.
- A query-adaptive design (Figure 10) is drawn, but no results are given.

# Results

## HLE results (Figure 2 table)
| Architecture | Closed source | Open source |
|---|---|---|
| ARCHON | 3.3% (15/453) | 3.3% (15/453) |
| Planner + ARCHON | 3.3% (15/453) | 2.6% (12/453) |
| Planner + Expander + ARCHON | 4.2% (19/453) | 3.8% (17/453) |
| Planner + Tool Call + Expander + ARCHON | 3.3% (15/453) | 2.9% (13/453) |

- Zero-shot GPT-4o: 3.1% (red line in Figure 4).
- 453 of the 500 questions were scored; the report doesn't say why.

## Reading the HLE table
- Best: Planner + Expander, closed, 4.2%: 4 more right answers than base Archon.
- Open version of the same: 3.8%.
- Search lowered both: back to 3.3% and down to 2.9%.
- Why search hurt: pages were often off-topic or full of unrelated text, which confused the models.
- All gaps are a handful of questions out of 453: likely noise.

## AlpacaEval LC results (Figure 3 table)
| Architecture | Closed source | Mixed source |
|---|---|---|
| ARCHON | 67.73% | 57.01% |
| Planner + ARCHON | 69.62% | 67.06% |
| Planner + Expander + ARCHON | 66.01% | 68.30% |
| Planner + Tool Call + Expander + ARCHON | 73.40% | 73.92% |

- GPT-4o reference line on the poster chart: 57.5%.
- The poster's chart labels the second column "Open Source Models".

## Reading the AlpacaEval table
- Planner: +10.05 points for mixed (57.01 → 67.06), +1.89 for closed.
- Mixed started more than 10 points below closed; with all blocks it ends slightly above (73.92% vs. 73.40%).
- Expander: hurt closed (69.62 → 66.01), helped mixed (67.06 → 68.30).
- Full stack with search: best in both columns, above 70%.

## Caveat: is "open rivals closed" fair?
- The best "mixed" row (73.92%) uses GPT-4o for tools, generation, expansion and ranking (appendix Figure 9).
- So it isn't an open-model result.
- The cleaner evidence: Planner + ARCHON lifted the mixed setup from 57.01% to 67.06%.
  - Its diagram only adds a Qwen2.5 planner to the open ensemble, so that one is a fair test.
- Closed "Planner + ARCHON" also switches critic and fuser to gpt-4o, so it changes two things at once.

# Analysis and discussion

## What they conclude (Section 6)
- Planner: consistently helps structure answers.
- Expander: limited on AlpacaEval because length control rewards short answers; helped on HLE via longer reasoning.
- Tool Call: pushed AlpacaEval above 70%, beyond every original Archon setup.
- Planner and Tool Call let open models match closed ones on instruction following and retrieval.
- Block value depends on the benchmark: tailor the stack to the task.

## Why search split the two benchmarks
| AlpacaEval | HLE |
|---|---|
| Real-world requests; outside facts help | Deep reasoning, not lookup |
| Search grounded answers, made them more relevant | Pages off-topic or cluttered |
| Score went up | Score went down |

- Still, OpenAI's Deep Research reached 26.6% on HLE, so iterative plan-and-search loops may help.

## Reasoning remains the bottleneck
- Expander gains on HLE were too small to overcome reasoning limits.
- Adding search did not help hill-climb HLE.
- Suggested next steps: hierarchical planning, multi-step verification, process reward models.

# Limitations and future work

## Limitations
- Stated: reasoning still hard; search noise on HLE.
- Our notes:
  - One run per setup; HLE gaps are 2–4 questions.
  - Configs change more than one block at a time (see appendix diagrams).
  - "Mixed" with search is mostly GPT-4o.
  - No cost or call counts reported.

## Future work they list
- Query-adaptive planner: build a whole Archon stack per prompt.
- More tool calls: outside APIs and special databases.
- Early stopping: fewer calls for the same quality.
- Model size: can these blocks make up for smaller models (e.g., 7B)?
- Stronger reasoning methods, such as process reward models.

# Our smaller Colab rebuild

## What we change
| Paper | Our rebuild |
|---|---|
| Google search over the real web | BM25 over 16 local pages |
| HLE and AlpacaEval | 6 questions about made-up towns (search is required) |
| ~70B models and GPT-4o | Generators: llama3.1:8b, llama3.1:8b, llama3.2:3b |
| Planner, Expander, Search | Planner and Search; Expander skipped to save calls |
| Closed vs. open configs | One local config; four setups at known call counts |

- BM25: keyword search; rare words count more, long pages count less.

## Our small web
- 8 made-up towns, each with founding year, population, river, bridge and bridge year.
- 3 decoy pages: "New ..." towns with different numbers.
- 5 clutter pages: travel blogs with filler.
- 6 questions; most need two pages, some need arithmetic.
- No model can know these facts, so the value of search is clear.

## BM25 scoring, in one line
- `score = Σ_words IDF(w) · tf·(k1+1) / (tf + k1·(1 − b + b·len/avglen))`.
- IDF: rare words weigh more.
- tf: how often the word is on the page.
- k1 = 1.5: caps how much repeats count.
- b = 0.75: penalty for long pages.

## The pipeline
```
question → Planner (plan + search queries) → Web search (BM25) → 3 Generators (8B, 8B, 3B)
         → Critic + Ranker → Fuser → Verifier → answer          (7 model calls)
```
- Planner: up to 3 queries, one per fact needed.
- BM25: top 3 pages per query.
- One call critiques and ranks all three candidates.
- Fuser: one answer from the top two.
- Verifier says "not supported" → fall back to the top-ranked candidate.

## Four setups, compared
| Setup | Calls per question |
|---|---|
| Single call, no search | 1 |
| Single call + search (RAG) | 1 |
| Planner + search + 1 generator | 2 |
| Full pipeline | 7 |

- RAG: retrieval-augmented generation; fetch pages, then answer.
- If the 2-call setup ties the 7-call one, use the 2-call one.

## Solutions: three free or cheap ablations
- Stop after the ranker: if it matches the full pipeline, the fuser and verifier add cost, not value.
- Score each generator alone and the best-of-3 oracle: does the ensemble beat its best member?
- Noisier web: add "Old ..." and "Upper ..." decoys; does search start to hurt, as on HLE?

# How this applies to us

## Search the firm's own files
- Contract review needs outside facts: past deals, precedent memos, the playbook.
- Run search over the firm's own documents, on the firm's machine.
- No pages leave the building; no web noise.

## Decoys are the real risk
| Search finds the right page | Search finds a look-alike |
|---|---|
| "Acme Holdings MSA: liability cap 12 months of fees" | "Acme Holdings UK MSA: cap 24 months" |
| Correct check against the playbook | Wrong flag, confidently stated |

- Test search with look-alike names before trusting it.

## Start with the cheap blocks
- A planner is one call and helped weaker models most.
- Add generators, critics and fusers only if an ablation shows they pay.
- Count calls per question next to accuracy.

## Common mistakes
- Assuming search always helps → check what it returns; on HLE it hurt.
- Adding blocks without testing → the Expander helped in one setup and hurt in another; ablate.
- Reading tiny gaps as wins → four questions out of 453 is noise.
- Changing two things at once → swap one block or one model per ablation.
- Ignoring cost → 7 calls per question; compare with the 2-call setup.

## Check your understanding
**Q: Why did web search hurt on Humanity's Last Exam?**
A: The results were often off-topic or cluttered. That confused the models instead of helping.

**Q: What does the Planner add to the pipeline?**
A: It rephrases the question and writes a plan and search queries. Every later block uses that plan.

**Q: How much did the Planner alone lift the mixed setup on AlpacaEval?**
A: From 57.01% to 67.06%, about 10 points (Figure 3 table).

**Q: Why is "open models rival closed" weaker than it looks?**
A: The best "mixed" setup uses GPT-4o for tools, generation, expansion and ranking, per the appendix diagram.

**Q: Why did the Expander help on HLE but not on AlpacaEval?**
A: AlpacaEval's length control rewards short answers; on HLE the extra detail helped reasoning a little.

**Q: Why does the notebook use questions about made-up towns?**
A: No model can know the answers, so search is truly needed. The value of search is easy to see.

## Try it in Colab
- Compare the four setups: accuracy next to calls per question.
- Read the retrieved page lists: does a "New ..." decoy sneak in?
- In the solutions, score "rank only" vs. the full pipeline, then test a noisier web.
