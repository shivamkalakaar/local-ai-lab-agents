# How to build great AI agents

We built this from Stanford CS329Z, Engineering AI Agents (Fall 2026): 11 lecture decks, about 50 papers and posts, and the course code. Each rule lists its sources at the end.

---

## The one-paragraph version

A great agent is mostly not the model. It is:
- an eval you trust
- a simple, mostly fixed workflow
- a few well-designed tools
- a clean context
- hard rules enforced in code, not in the prompt
- a loop that turns every mistake into a test and every human correction into training data.

The weaker the model (local, open, small), the more these matter. Pick the model last, by measuring.

---

## The 12 rules

### 1. Build the eval before the agent
- Write 30–50 real tasks with known right answers, then grow to 100–150. About 100 well-chosen items estimate a score to within about 2 points.
- Hold out 20–30% of the tasks and never tune on them.
- Grade with code wherever you can: exact fields, numbers that match the source, required items present. Score both what the agent missed and what it wrongly flagged.
- Use an LLM judge only for open-ended parts. Give it the expert's answer, use a different model family, swap the order for pairwise comparisons, and check it against about 80–100 human labels.
- Test the eval itself:
  - The expert's own answer must score about 100%.
  - "Do nothing" and "flag everything" must score low.
  - Ask a strong model to cheat, and patch whatever works.
- Report pass^k (right on all k tries), not just the average. 75% per try is only about 42% right three times in a row. For repeated business work, consistency is the product.
- Metrics should return text ("expected X, got Y"), not just a score. Optimizers use that text.

*Sources: Terminal-Bench lecture, Anthropic "Demystifying evals", ABC checklist, tinyBenchmarks, LLM-as-judge, τ-bench.*

### 2. Start simple: one call, then a workflow, an agent only if needed
- If you know the steps, fix them in code. Let the LLM decide only where it must.
- Try the cheap baselines first: retry, and "small model first, escalate on failure". In one study they matched fancy agents, and one complex method (LATS) cost 50x more and scored lower.
- Five patterns cover most needs: chaining, routing, parallel checks or votes, orchestrator-workers, and generator plus checker.
- Plan first, then execute (ReWOO style). It used about 40–60% fewer tokens.
- Add complexity only when the eval shows it helps.

*Sources: Anthropic "Building effective agents", L1, L6, "AI Agents That Matter".*

### 3. Code decides. The model reads and writes
- Put rules, thresholds, comparisons, math and approvals in plain code.
- The model classifies, extracts and drafts.
- For fixed choices, use a classifier or an enum, not free text.
- Use typed schemas between every step, constrained decoding (forced JSON) where the server supports it, and validation before anything runs.

*Sources: L1–L2, DSPy, SGLang/vLLM structured output.*

### 4. Spend your design effort on tools, not the prompt
- Use a few distinct tools. Pi, with 4 tools and a prompt under 1,000 tokens, sat on the best cost/success frontier.
- Write tool docs as if for a junior developer. Make wrong use impossible: absolute paths, enums, required fields.
- Return short, paged outputs. In SWE-agent, showing whole files cut success from 18.0% to 12.7%.
- Return clear error messages so the agent can fix itself, and let it say "no tool fits".
- Prefer native function calling. It beat ReAct-style text on τ-bench for every model. Weak servers are the exception: there, a strict text format plus a parser is safer.

*Sources: SWE-agent, HarnessTax, MCP spec, BFCL, τ-bench.*

### 5. The harness matters most for weak models
- Same model, different harness: Haiku 4.5 scored 31% to 48%, while top models moved only 3–4 points. Cost varied up to 5x.
- The model maker's own harness was not the best in 9 of 12 cases.
- So choose the harness by measurement, per model, and track cost as well as accuracy.

*Sources: HarnessTax, L5.*

### 6. Keep the context small and clean
- Context quality drops early and steadily as length grows.
- Use a stable prompt prefix and only add to the end. Editing earlier text breaks the cache and costs time.
- Hide old tool outputs before reaching for summaries. Tool outputs are about 84% of the tokens, and hiding old ones halved cost with equal or better results.
- Compact near the limit, but keep critical rules outside anything that can be compacted. One agent deleted a user's inbox after "confirm before acting" was summarized away.
- Fetch data just in time. For very long inputs, keep the data out of the prompt and let the model search it. A small model doing this scored 51%, against 34% for a big model reading all 263K tokens.
- Write notes to files: a progress file and a checklist.

*Sources: L2, L6, Anthropic context engineering, MemGPT.*

### 7. Retrieval defaults that work
- Under about 200K tokens of documents? Put them all in the prompt.
- Otherwise:
  - Run two searches, keyword (BM25) and embeddings, and merge the results. Keyword search catches clause numbers and IDs. Embeddings catch rewording.
  - Split documents on their structure (headings, clauses), or into chunks of about 100–500 tokens.
  - Add a short "where this came from" note to each chunk. Failed retrievals dropped from 5.7% to 1.9% with reranking.
  - Fetch about 150, rerank, and give the model about 20.
- Measure retrieval on its own (did the right passage come back?).
- Check every citation in code: the quoted text must really be on that page.

*Sources: L3, Anthropic contextual retrieval, ColBERT/DPR.*

### 8. Memory: simple, cited and approved
- Plain files and plain search beat fancy memory products. Memory agents scored about 0.12–0.14 against 0.20 for basic search.
- Keep raw records forever. Every derived "lesson" must point to the records behind it.
- The agent may *propose* memory changes. A person approves them, and every change is versioned. Never let it silently rewrite what it remembers.
- Changing prompts, skills or rules counts as a code change and gets tested before release.
- Scope memory, for example per client or per matter, so a lesson from one client is never used for another.

*Sources: L7, CoALA, Generative Agents, MemoryArena, LoCoMo.*

### 9. One strong agent plus a separate checker, not a swarm
- Across 260 setups, multi-agent systems averaged −0.3% versus a single agent. Gains disappear once a single agent scores above about 45%.
- They fail badly on step-by-step work (−39% to −70%) and help on wide, separable work like research (+57% to +81%).
- Independent agents with merged outputs amplify errors 17x. With a central checker, 4x.
- When you do delegate:
  - Send a written brief: goal, output format, limits, when to stop.
  - Pass files, not retold messages.
  - Treat the sub-agent as a tool that returns a result.

*Sources: L8, Kim et al. 2025, MAST, Anthropic multi-agent research.*

### 10. Safety lives in the harness, not the prompt
- A privacy prompt barely helps: leaks went from 27% to 26% of actions. So remove the exits instead: no network for the agent process, no send or upload tools in the reviewer, read-only access to documents.
- All document text is data, never instructions. The reading step cannot call any action tools.
- Plant test injections in sample documents and re-run them after every change.
- Irreversible steps (send, file, release) are human-started only. Ask for sign-off at a few clear points, with evidence attached.
- Keep an append-only log of every step: model version, input hashes, outputs. It lets you replay any past decision.
- Guidance beats blocking: telling the agent *what to drop* kept it useful (94% private and 81% helpful), while bare blocks caused retry loops.

*Sources: PrivacyLens, Zhang & Yang, CDI, OpenAI on prompt injection, Claude Code docs.*

### 11. Make it durable and observable
- Each LLM call is a retryable step with a timeout. The main control code stays deterministic, so a crash resumes without redoing work or repeating side effects.
- Set caps on turns, tokens, time and money, and add a "stuck" detector (stop after about 4 identical repeats).
- Trace everything, and read the traces. Read the actual prompt your framework sends.
- For long jobs: a setup step, then one item per session, a JSON checklist the agent can only mark done with evidence, a progress file, and a commit per step.

*Sources: L5 (Temporal demo), OpenHands, Anthropic long-running harnesses.*

### 12. Improve in this order: prompts, then weights
1. Prompt tuning on the small model (DSPy GEPA or MIPROv2), with a strong model reflecting on its mistakes. In the course demo, Gemma 3 12B went from 55% to 100%. GEPA beat RL (GRPO) using far fewer runs.
2. Fine-tune (SFT) on outputs from a strong teacher, keeping only outputs your checker passes. A few hundred to a few thousand varied, high-quality examples can be enough.
3. On-policy distillation: the student writes and the teacher grades every token. It matched or beat RL at about a tenth of the compute. It needs the teacher's token probabilities, so the teacher must be an open-weight model.
4. RL (GRPO) only where answers can be checked automatically. Harden the checker, because models game weak ones, and keep checkpoints, because RL can collapse late.
5. Re-tune the prompts on the new model.
- After every training run, check general skills too. Narrow fine-tuning once cut instruction-following from 85% to 45%.
- Every human correction is stored with its reason. It becomes an example now, a test case at once, and training data later. The held-out eval is never trained on.

*Sources: L9, L11, GEPA, MIPROv2, DeepSeekMath, Thinking Machines on-policy distillation, LIMA, SWE-smith, Shankar.*

---

