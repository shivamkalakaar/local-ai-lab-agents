---
summary: How a Stanford team tested agents that design other agents, and why you should do the cost arithmetic before building one.
minutes: 11
---

## The big idea

A meta-agent is an agent that designs other agents. It writes a new agent, tests it, saves it in an archive (a list of every design so far with its score), and uses that archive to write the next one. It sounds like free progress.

Batu El, Mert Yuksekgonul and James Zou studied this loop in their CS329A project (Winter 2025). The course page calls it "Towards Effective Meta-Agents." The paper is titled "Inefficiencies of Meta Agents for Agent Design." They asked three hard questions about meta-agents like ADAS (the Automated Design of Agentic Systems paper, 2024):

1. Does the meta-agent really learn from its past designs?
2. Are the designs different enough to combine?
3. Does the design cost ever pay for itself?

Think of hiring a consultant to redesign your intake process. The new process may be a bit better. But you paid the consultant, and the new process takes longer per client. You only come out ahead after enough clients.

## Why it matters for us

A firm might want a system that keeps inventing better contract-review agents on its own. This project says: check the math first.

- Designed agents often cost more per question, because they make more model calls. Add the cost of designing them, and they may never pay back.
- Showing the designer everything it has tried can be worse than showing it only the best few.
- Running on the firm's own computers lowers the price per call, but not the number of calls. The arithmetic still matters.

## How it works, step by step

1. **Start with a library.** The archive began with 7 hand-written agents, such as Chain-of-Thought (reason step by step), Majority Voting, LLM Debate and Step-back.

2. **Pick what the designer sees.** This choice is called the context strategy. There were three:
   - Cumulative: show all past agents (what ADAS does).
   - Parallel: show only the 7 starting agents, so every new design starts fresh.
   - Evolutionary: show the 7 best agents so far, like picking parents.

3. **Design, test, save.** A gpt-4o meta-agent wrote agent code that ran on gpt-3.5. They repeated this for 30 rounds, on four test sets, 3 times each.

4. **Measure diversity.** Each agent gets a row of right and wrong answers. Cosine similarity (a number for how alike two rows are; near 1 means the same questions right) shows whether agents overlap. Coverage is the share of questions at least one agent gets right.

5. **Do the cost math.** Total cost is C0 + n × Cj. C0 is the one-time design cost. Cj is the cost per question once deployed. n is the number of questions.

What they found. Best training score, averaged: cumulative 55.0, parallel 60.0, evolutionary 60.3. So showing everything did not beat showing nothing new. Evolutionary gained up to 10 points over cumulative on MGSM (41.4 vs. 56.5). On test, the starting library scored 49.0, cumulative 52.2, parallel 55.9, evolutionary 55.2. Evolutionary agents were the least diverse.

The cost result is the sharpest. Designed agents cost more per question than the starting ones. Counting design cost, a designed agent was cheaper per correct answer only after about 15,000 questions. That happened only on MMLU and DROP with parallel context. Elsewhere it never broke even.

## Worked example

Say the firm's best plain agent reviews a clause in a single call. A meta-agent designs a "solve then check" agent that makes two calls and is a bit more accurate. Is it worth it?

Our notebook asks the same question at small scale. Designs are data, not code: a system prompt plus one of four fixed procedures. So nothing the model writes is ever run as code. llama3.1:8b designs, llama3.2:3b runs the agents, on six kinds of checkable puzzles. Here is the break-even step:

```python
C0 = sum(tok(a["chars"] + a.get("meta_chars", 0)) for a in designed)
c_init, c_new = tok(test_init["chars"]) / len(TEST), tok(test_new["chars"]) / len(TEST)
p_init, p_new = max(test_init["score"], 1e-9), max(test_new["score"], 1e-9)
gap = c_init / p_init - c_new / p_new
breakeven = C0 / (p_new * gap) if gap > 0 and best_new is not best_init else float("inf")
```

- Line 1 adds up the design cost, C0. It counts the designer's prompts plus testing every new design. `tok` turns characters into tokens (about 4 characters each).
- Line 2 finds the cost per question for the old agent and the new one.
- Line 3 takes each agent's accuracy on the test set. The tiny number avoids dividing by zero.
- Line 4 compares cost per correct answer. If the new agent is not cheaper per correct answer, the gap is zero or negative.
- Line 5 finds how many questions it takes to earn back C0. If there is no gap, the answer is "never."

For the firm, swap in real numbers: clauses per month, tokens per review, and accuracy. If break-even is 15,000 clauses and you review 500 a month, that is years away.

## Common mistakes

- **Skipping the cost math.** A better score can still lose money. Compute break-even first.
- **Feeding the designer everything.** The full history did worse than a curated set. Show the best few designs.
- **Expecting a "pick the best agent per question" boost.** Designed agents tend to solve the same questions, so combining them adds little unless you push for variety.
- **Trusting one run.** Results from small test sets are noisy. The authors used 3 runs; use several too.

## Check your understanding

**Q: Which context strategy did ADAS use, and how did it compare?**
A: Cumulative, which shows all past agents. It scored 55.0 on training, below parallel (60.0) and evolutionary (60.3).

**Q: What do C0 and Cj stand for in the cost formula?**
A: C0 is the one-time cost of designing the agent. Cj is the cost per question once it is in use.

**Q: When did a designed agent pay for itself?**
A: Only after about 15,000 questions, and only on MMLU and DROP with parallel context. On the other test sets it never did.

## Try it in Colab

Run the notebook to test the 3 starting agents, then let the meta-agent design 3 more with evolutionary context. Compare the best designed agent with the best starting one on train and test. Then read the similarity table and the break-even line, and try the 10x designer price in the solutions.
