---
summary: How a Stanford team added a planner and web search to a pipeline of model calls, and why search only helps when it finds the right pages.
minutes: 11
---

## The big idea

One model call is often not your best answer. Archon (the Archon paper, 2024) stacks simple steps into a pipeline. Several models generate answers. A critic points out flaws. A ranker orders them. A fuser merges the best into one. A verifier checks the result.

Megan Mou, Sherry Xie, Emily Zhang and Andrew Park extended Archon in their CS329A project, ARCHON Augmented (Winter 2025). Archon had only seven building blocks, no tools, and its strong results relied on models of about 70B parameters. They added three new blocks:

- A Planner at the front. It rephrases the question and writes a step-by-step plan for every later block.
- An Expander after each generator. It adds detail so the critic and ranker can judge each answer.
- A Web Search block. It writes search queries from the plan and passes the top 10 results per query to the generators.

Think of a research desk at a newspaper. An editor plans the story. Researchers pull clippings. Several writers draft. A copy editor critiques, picks the best parts, and a fact-checker signs off. Every step can help. But if the researchers pull the wrong clippings, the story gets worse, not better.

## Why it matters for us

A firm's contract review often needs outside facts: the client's past deals, the firm's precedent memos, the latest playbook. A local pipeline with search over the firm's own files is a natural fit.

- A planner is cheap and helped weaker open models catch up on following instructions. That is good news for firms running open models on their own computers.
- Search helps only when it finds the right documents. Noisy or look-alike pages can make answers worse.
- Each block costs model calls. Test each one on your own data and drop the ones that don't help.

## How it works, step by step

1. **Plan.** The Planner rephrases the request and writes a short plan and search queries.

2. **Search.** The queries go to a search engine. The results become context for the next step.

3. **Generate.** Several models write candidate answers, using the plan and the pages.

4. **Expand, critique, rank.** The Expander fills in detail. The critic notes flaws. The ranker orders the candidates.

5. **Fuse and verify.** The fuser writes one answer from the best candidates. The verifier checks it is supported.

6. **Ablate.** Ablation means removing one block at a time to see what it adds. The team did this on two benchmarks.

Results on AlpacaEval (an instruction-following test scored by win rate against a reference model, adjusted for answer length), closed / mixed setups:

| Setup | Closed | Mixed |
|---|---|---|
| Archon | 67.73% | 57.01% |
| + Planner | 69.62% | 67.06% |
| + Planner + Expander | 66.01% | 68.30% |
| + Planner + Search + Expander | 73.40% | 73.92% |

On Humanity's Last Exam (HLE, a very hard quiz; 453 questions scored), the best was Planner + Expander at 4.2% (19 of 453) in the closed setup. Plain Archon got 3.3%, and zero-shot GPT-4o got 3.1%. Here search hurt: results were often off-topic or full of page clutter. Also, these HLE gaps are a few questions out of 453, so they are within noise.

## Worked example

Imagine the firm asks: "How many years passed between the client's first master agreement and its latest amendment?" The answer needs two documents and some arithmetic. A single search on the raw question may find only one of them.

Our notebook builds this in miniature. The questions are about made-up towns, so no model can know the answers without searching. The "web" is 16 local pages: eight real town pages, three look-alike decoys ("New Velmora" with different numbers), and five clutter pages. Search uses BM25, a classic keyword ranking. Here is the full pipeline in the notebook:

```
question → Planner (plan + search queries) → Web search (BM25) → 3 Generators (8B, 8B, 3B)
         → Critic + Ranker → Fuser → Verifier → answer          (7 model calls)
```

- The Planner writes up to 3 queries, one per fact needed. That is how it finds both pages for a two-step question.
- BM25 returns the top 3 pages per query. Rare words count more, and long pages count less.
- Three generators write answers: two from llama3.1:8b and one from llama3.2:3b, a tiny mixed team.
- One call critiques and ranks all three.
- The fuser writes one final answer from the top two.
- The verifier asks whether the pages support it. If not, we fall back to the top-ranked candidate.

We skip the Expander to save calls. The notebook compares four setups: a single call, a single call plus search (RAG, retrieval-augmented generation: fetch documents, then answer), planner plus search plus one generator, and the full pipeline. It reports accuracy and calls per question.

For the firm, the decoy pages are the key lesson. "Acme Holdings" and "Acme Holdings UK" may have different contracts. If search pulls the wrong one, the pipeline confidently gives a wrong answer.

## Common mistakes

- **Assuming search always helps.** On HLE it hurt. Check what the search actually returns before you trust it.
- **Adding blocks without testing each one.** The Expander helped in one setup and hurt in another. Remove one block at a time and measure.
- **Reading tiny differences as wins.** A gap of four questions out of 453 is noise.
- **Ignoring cost.** The full pipeline uses 7 calls per question. If "planner plus search plus one generator" ties it with 2 calls, use that.
- **Forgetting look-alike documents.** Similar names with different facts are where retrieval goes wrong. Test with decoys.

## Check your understanding

**Q: Why did web search hurt on Humanity's Last Exam?**
A: The results were often off-topic or full of page clutter. That confused the models instead of helping them.

**Q: What does the Planner add to the pipeline?**
A: It rephrases the question and writes a step-by-step plan and search queries. Every later block uses that plan.

**Q: Why does the notebook use questions about made-up towns?**
A: No model can know the answers, so search is truly needed. That makes the value of search easy to see.

## Try it in Colab

Run the notebook to compare the four setups and look at accuracy next to calls per question. Read the retrieved page lists to see whether a decoy page sneaks in. Then use the solutions to score the pipeline without the fuser and verifier, and to test a noisier web.
