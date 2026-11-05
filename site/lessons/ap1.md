---
summary: How a Stanford team graded AI-built web apps by what a user can do, and how to check the grader first.
minutes: 9
source: CS329A past project · AppBench
slides: https://cs329a.stanford.edu/winter2025projects/appbench.pdf
---

## Learning objective
- Build a checklist grader for AI-built apps, and test the grader before you trust it.

## The problem
- AI tools can build a whole web app from one request.
- Code benchmarks check single functions with unit tests.
- Nothing checked if the app is usable:
  - Does navigation work?
  - Does saved state survive a refresh?
  - Are errors handled?

## "It compiles" vs "a user can use it"
| Unit test | User check |
|---|---|
| `tip(50, 20)` returns 10 | Type a bill, click Calculate, see the tip |
| The page loads | Add two tasks, refresh, both are still there |
| Passes on the happy path | A blank task is ignored |

## What the students built: AppBench
- Team: Ethan Hellman, Brendan McLaughlin, Abhinav Lalwani, Belinda Mo (CS329A, Winter 2025).
- 9 app requests, from L0 (static page) to L4 (live shared whiteboard).
- Six axes: UI, features, state, API integration, cross-page navigation, data processing.
- 7 builders, one shot each: Replit, Bolt, Cursor Agent, Lovable, Claude Sonnet 3.7, GPT-4.5, OpenHands CodeAct 2.1.

## How the checklist works
- Explicit check: what the user asked for.
- Implicit check: what a good app would also do (a back button, ignore blank input).
- Dependency graph: which check needs which other check to pass first.
  - Parent fails → children are skipped and score 0.
  - One root failure shows once, not ten times.
- Weighted scores, scaled to 0–100.

## How each check runs
- Screenshot + multimodal LLM: a model that reads images judges the look.
- LLM-driven browser (Stagehand on Playwright): clicks through the app.
- Network interception: fakes server errors to test error handling.

## What they found
| Grader vs. 3 human raters | Value |
|---|---|
| Pearson correlation | 0.8173 |
| Spearman correlation | 0.6827 |
| Mean MSE / MAE | 0.0302 / 0.1313 |

- All builders scored above 75% up to L2. Scores fell at L3 and dropped sharply at L4.
- 2 of 7 builders failed to produce a compiling app for one L3 and one L4 task.
- One-shot models were close to full agent products. Lovable and Bolt were top.

## Our Colab rebuild: smaller
| AppBench | Our rebuild |
|---|---|
| 9 apps, L0–L4 | 5 single-file HTML apps, L0–L3 |
| Real browser | No browser: Node.js runs the app's script in a fake page |
| 7 commercial builders | The local model, one shot, then one fix round |
| 3 human raters | Known-good and known-broken apps (hand score + LLM judge in solutions) |

- Each app exposes its logic as functions on `window.app`.
- Python's HTML parser checks structure.

## The checklist in code (L3 to-do app)
```python
("app loads", 1, True, [], APP_LOADS),
("add + toggle + remaining", 3, True, ["app loads"], js([["app.addTask('a')", "app.addTask('b')", "app.toggle(0)", "app.remaining()"]], 1)),
("saves JSON under 'todos'", 2, True, ["add + toggle + remaining"], js([["app.addTask('a')", "JSON.parse(localStorage.getItem('todos')).length"]], 1)),
("survives a refresh", 3, True, ["saves JSON under 'todos'"], js([["app.addTask('a')", "app.addTask('b')"], ["app.remaining()"]], 2)),
("ignores blank tasks", 1, False, ["app loads"], js([["app.addTask('   ')", "app.all().length"]], 0)),
```
- Each tuple: name, weight, explicit (True) or implicit (False), parents, test.
- Line 1: does `window.app` exist? Everything depends on it.
- Line 2: add two tasks, tick one, expect 1 left.
- Line 3: look in `localStorage` (the browser's small storage) for the saved list.
- Line 4: two inner lists = two page loads. Reload, expect 2 tasks.
- Line 5: implicit; nobody asked, but blank tasks should be ignored.

## Test the grader first
- Known-good tip calculator: must score 100.
- Known-broken one (no labels, wrong math, wrong button id): must score under 60.
- Crashing script: behavior checks must fail.
- Only then grade the model's apps.

## One shot vs one fix round
| One shot | One fix round |
|---|---|
| Same prompt, one turn, no feedback | Model gets the names of its failed checks, one chance to fix |
| 1 call per app | 1 more call per app (0 if nothing failed) |

- Our checks name the exact problem, so the fix round often helps.
- AppBench agents got no such feedback.

## How this applies to us
- A firm will ask a local model for small tools: a deadline tracker, a fee calculator, a playbook-breach list.
- Grade them on the firm's own machine with a checklist like this: "deadlines survive a refresh", "liability cap over 12 months is flagged".
- Check the grader on a known-good and known-broken tool before trusting it.

## Common mistakes
- Grading by "it compiles" → test what a user does, including a refresh.
- Only explicit checks → add implicit ones, weighted lower.
- One failure cascades into ten → skip children of failed parents.
- Trusting the grader blindly → known-good must score 100, known-broken low; then compare with a human.

## Check your understanding
**Q: Why does AppBench skip a check when its parent check fails?**
A: So one root problem shows up once. Otherwise one missing feature causes a pile of confusing failures.

**Q: What is the difference between an explicit and an implicit check?**
A: Explicit tests what the user asked for. Implicit tests what a good app would also do, like ignoring blank input.

**Q: How did the team know their grader was any good?**
A: They compared it with three human raters. Pearson correlation was 0.8173.

## Try it in Colab
- Grade the known-good, known-broken and crashing tip calculators; check the scores.
- Let the local model build all five apps; see which checks fail as the level rises.
- Run the fix round and compare the one-shot and after-fix scores.
