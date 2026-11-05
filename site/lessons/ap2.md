---
summary: How a Stanford team let an agent pick the right thinking trick for each step of a plan, and why you must compare at the same call budget.
minutes: 11
---

## The big idea

There are many tricks to make a model think better. It can reflect on its own answer. It can refine a draft. Two copies can debate. Most teams pick one trick and use it everywhere.

Chelsea Zou, Samuel Liu and Jui Khankari (the poster also lists Miriam Cheng) asked a better question in their CS329A project, PRIME (Winter 2025). What if the agent picked the best trick for each step of the task?

Think of a head chef. They don't use the same method for every dish. Some get a quick sear. Some get slow tasting and adjusting. A good chef learns which method works for which dish, and keeps that knowledge for next time.

PRIME does this with a search method called Monte Carlo Tree Search (MCTS): try promising options, keep score of what worked, and spend more effort on the winners while still testing others now and then.

## Why it matters for us

A contract review has many steps. Some are easy, like "find the governing-law clause." Some are hard, like "does this indemnity clause break the playbook rule on liability caps?"

- Spending the same effort on every step wastes calls on the easy ones and starves the hard ones. Picking a method per step can help.
- On a firm's own computers, model calls are cheap but not free. PRIME's lesson is to compare methods at the same number of calls. Otherwise "smarter" may just mean "tried more times."
- A cheap test, called a reverse ablation, shows which part of a pipeline deserves the bigger model. You upgrade one part at a time and see which upgrade helps most.

## How it works, step by step

1. **Plan.** A planner splits the task into subgoals (smaller pieces of the job). Each subgoal gets a confidence score from the model.

2. **Choose an action at each step.** PRIME can Drill down (break the subgoal up further), Solve it with a chosen tool (self-reflection, self-refinement or debate), or Backtrack (give up on this branch and go back).

3. **Pick which option to try with UCT.** UCT stands for Upper Confidence bound for Trees. It is a simple score: how well this option has done so far, plus a bonus for options you haven't tried much. The bonus keeps the search curious.

4. **Score the result.** In the paper, values mix a model's own score with self-consistency (whether repeated answers agree).

5. **Back up the score.** The option's value becomes the running mean (average so far) of all its results. Good options rise. Bad ones sink.

PRIME used GPT-4o-mini inside the search and tested on PlanBench, WebShop and Game of 24, 60 questions each. In Table 1, PRIME scored 36.2% / 40% / 75%. LATS, an earlier tree-search agent, scored 2.2% / 38% / 44%. Plain GPT-4o-mini scored 0% on all three. A word of caution: the paper's text gives different figures for some of the same cells, so these numbers are not fully consistent. The authors estimate PRIME cost about 4x less than o1, even with more calls.

Two honest results matter most. On WebShop, PRIME and LATS nearly tied, and reflections were generic. And the search wandered when it was unclear whether a step had worked.

## Worked example

Picture a paralegal with a budget of five model calls for one tricky clause. The plan lists three angles: compare the cap to fees, check carve-outs, check the time window. For each angle, the agent can answer directly, answer then reflect, or break the angle down further. After each try, a checker says pass or fail. The agent learns which angle-and-method pairs pay off.

Contract checks are rarely exact, so our notebook uses Game of 24 instead. You must use four numbers once each, with + − × ÷, to make 24. A code checker can grade every answer perfectly. That makes the reward clear, which is exactly when tree search shines.

Here is the heart of the selection loop:

```python
uct = lambda a: a["V"] + C * math.sqrt(math.log(total) / a["N"])
arm = max(arms, key=uct)
ok, used = run_tool(arm["tool"], nums, steps, arm["left"], budget - calls)
calls += used
r = 1.0 if ok else 0.0
arm["V"] = (arm["V"] * arm["N"] + r) / (arm["N"] + 1); arm["N"] += 1; total += 1
```

- An "arm" is one pairing of a first step and a tool, like "8 − 2 = 6, then reflect."
- Line 1 is the UCT score. `V` is how well the arm has done. The square-root part is the curiosity bonus. It is big when the arm `N` has been tried rarely. `C` sets how curious to be.
- Line 2 picks the arm with the highest score.
- Line 3 runs the tool and reports success and how many calls it used.
- Line 4 counts calls against the budget of 5.
- Line 5 turns pass or fail into a reward of 1 or 0.
- Line 6 updates the running mean, the same rule the paper uses.

The notebook runs PRIME-lite next to two baselines. One is a single call. The other is "retry with feedback," which also gets up to 5 calls. If PRIME-lite only beats the single call, the extra calls did the work, not the cleverness.

## Common mistakes

- **Comparing at unequal budgets.** A method with 5 calls will usually beat 1 call. Always include a simple retry baseline with the same budget.
- **Using tree search without a clear reward.** If you can't tell whether a step worked, the search wanders. Use it where a checker exists.
- **Upgrading every part at once.** Swap in the bigger model one part at a time. That shows where the money should go.
- **Trusting one table.** The PRIME paper's own numbers disagree in places. Read results with care and rerun what matters.

## Check your understanding

**Q: What does the curiosity bonus in UCT do?**
A: It raises the score of options that haven't been tried much. That stops the search from locking onto its first favorite too early.

**Q: Why does the notebook include a "retry with feedback" baseline?**
A: It uses the same 5-call budget as PRIME-lite. If they tie, the gain came from extra tries, not from smart strategy choice.

**Q: Where did PRIME struggle?**
A: On WebShop it nearly tied with LATS, and its reflections were generic. It also wandered when a step's value was unclear.

## Try it in Colab

Run the notebook to compare single shot, retry with feedback, and PRIME-lite on six solvable puzzles. Look at the success rate next to the average number of calls. Then check the option statistics to see which tool the search learned to prefer, and try the exploration constant at 0.2 and 2.0.
