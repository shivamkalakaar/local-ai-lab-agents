---
summary: How a Stanford team added a planner and web search to a pipeline of model calls, and why search helps only when it finds the right pages.
minutes: 9
source: CS329A past project · ARCHON Augmented
slides: https://cs329a.stanford.edu/winter2025projects/archon.pdf
---

## Learning objective
- Build a plan → search → generate → critique → fuse → verify pipeline, and ablate each block on your own data.

## The problem
- Archon: stacks inference-time steps (generate, fuse, critique, rank, verify, unit tests) into a pipeline.
- Only seven building blocks.
- No tool use, no web search.
- Its strong results relied on models of about 70B parameters.

## Archon's blocks, one line each
- Generator: writes candidate answers.
- Critic: notes each candidate's flaws.
- Ranker: orders the candidates, best first.
- Fuser: merges the best candidates into one answer.
- Verifier: checks the answer is supported.

## What the students built: three new blocks
- Team: Megan Mou, Sherry Xie, Emily Zhang, Andrew Park (CS329A, Winter 2025).
- Planner (first): rephrases the request, writes a step-by-step plan for every later block.
- Expander (after each generator): adds detail so critics and rankers can judge.
- Web Search Tool Call: queries from the plan, top 10 Google results per query.

## Setup
- Closed-source config: GPT-4o generates and ranks; Qwen2-72B critiques and fuses.
- Open-source config: an ensemble of open models.
- HLE (Humanity's Last Exam): 500-question subset, 453 scored.
- AlpacaEval LC: instruction following, win rate adjusted for answer length.
- Ablation: add or remove one block at a time.

## What they found: AlpacaEval LC
| Setup | Closed | Mixed |
|---|---|---|
| Archon | 67.73% | 57.01% |
| + Planner | 69.62% | 67.06% |
| + Planner + Expander | 66.01% | 68.30% |
| + Planner + Search + Expander | 73.40% | 73.92% |

- The planner helped the weaker open setup catch up.

## What they found: HLE
| Setup | Closed | Open |
|---|---|---|
| Archon | 3.3% | 3.3% |
| + Planner + Expander | 4.2% (19/453) | 3.8% (17/453) |
| + Planner + Search + Expander | 3.3% | 2.9% |

- Zero-shot GPT-4o: 3.1%.
- Search hurt: results were off-topic or cluttered.
- A few questions out of 453 is noise.

## Our Colab rebuild: smaller
| Paper | Our rebuild |
|---|---|
| Google search over the real web | BM25 over 16 local pages |
| HLE and AlpacaEval | 6 questions about made-up towns (search is required) |
| ~70B models | Generators: llama3.1:8b, llama3.1:8b, llama3.2:3b |
| Planner, Expander, Search | Planner and Search; Expander skipped to save calls |

- BM25: keyword search; rare words count more, long pages count less.
- The 16 pages: 8 real towns, 3 look-alike decoys ("New ..."), 5 clutter pages.

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

## How this applies to us
- Contract review needs outside facts: past deals, precedent memos, the playbook. Search the firm's own files, locally.
- A planner is cheap and helped open models catch up.
- Decoys matter: "Acme Holdings" vs. "Acme Holdings UK" may have different contracts. Test search with look-alikes.

## Common mistakes
- Assuming search always helps → check what it returns; on HLE it hurt.
- Adding blocks without testing → the Expander helped in one setup and hurt in another; ablate.
- Reading tiny gaps as wins → four questions out of 453 is noise.
- Ignoring cost → 7 calls per question; compare with the 2-call setup.

## Check your understanding
**Q: Why did web search hurt on Humanity's Last Exam?**
A: The results were often off-topic or cluttered. That confused the models instead of helping.

**Q: What does the Planner add to the pipeline?**
A: It rephrases the question and writes a plan and search queries. Every later block uses that plan.

**Q: Why does the notebook use questions about made-up towns?**
A: No model can know the answers, so search is truly needed. The value of search is easy to see.

## Try it in Colab
- Compare the four setups: accuracy next to calls per question.
- Read the retrieved page lists: does a "New ..." decoy sneak in?
- In the solutions, score "rank only" vs. the full pipeline, then test a noisier web.
