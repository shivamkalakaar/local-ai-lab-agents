---
summary: How a Stanford team graded AI-built web apps by what a user can actually do, and how to check that your grader deserves trust.
minutes: 11
---

## The big idea

AI tools can now build a whole web app from one request. But how do you know the app works? Most code tests only check small pieces, like "does this function return 10?" They don't ask whether a person can click, type and get the right result.

Four Stanford students, Ethan Hellman, Brendan McLaughlin, Abhinav Lalwani and Belinda Mo, built AppBench (CS329A, Winter 2025) to fill that gap. They wrote 9 app requests, from L0 (a plain static page) up to L4 (a live whiteboard several people share). For each request they wrote a checklist and turned it into an automatic grader.

Think of a building inspector. The inspector doesn't just confirm the house is standing. They flip switches, run taps and try the doors. And if the power is off, they don't fail every light separately. They note one root problem.

That is AppBench in one picture: test what a user would do, in a sensible order, and skip the checks that can't work yet.

## Why it matters for us

A law firm will soon ask a local model to build small internal tools. Think of a deadline tracker, a fee calculator, or a page that lists which clauses break the playbook (the firm's house rules, like "liability cap at most 12 months of fees").

- "It runs" is not enough. The tracker must still show your deadlines after you refresh the page.
- The grader itself can be wrong. AppBench checked theirs against three human raters before trusting it. We should do the same, even if only with a few known-good and known-broken examples.
- Fancy agent products did not clearly win. One-shot models (one request, one answer, no back-and-forth) were close to full agent products. Lovable and Bolt came out on top. So measure before you pay for the fancier tool.

## How it works, step by step

1. **Write the request.** Describe the app in plain words, as a user would. AppBench spread its 9 requests over six areas: looks, features, saved state, outside services, moving between pages, and data handling.

2. **Split the checklist in two.** Explicit checks are what the user asked for. Implicit checks are what a good app would also do, like ignoring a blank entry or having a back button.

3. **Order the checks as a dependency graph.** A dependency graph is a map of which check needs which other check to pass first. You can't test "add to cart" if the product page never opens. If a parent check fails, its children are skipped and score zero. One root failure then shows up as one failure, not ten.

4. **Turn each check into a test.** AppBench used screenshots judged by a model that reads images, a model-driven browser that clicks through the app, and network tricks to fake server errors.

5. **Weight and score.** Important checks count more. The total is scaled to 0–100.

6. **Grade the grader.** Compare the automatic scores with human scores. AppBench reported a Pearson correlation of 0.8173 and a Spearman correlation of 0.6827 with three raters. Correlation is a number from -1 to 1 that says how closely two sets of scores move together.

What they found: every kind of builder scored above 75% up to level L2. Scores fell at L3 and dropped sharply at L4. Two of the seven builders could not produce a working app for one L3 task and one L4 task.

## Worked example

Our rebuild is smaller. Colab has no browser, so each app must expose its logic as plain JavaScript functions on `window.app`. Python reads the page's structure. Node.js (a program that runs JavaScript outside a browser) runs the app's own code inside a fake page and calls those functions.

Picture the firm asking for a deadline tracker that saves tasks. The closest app in our notebook is the L3 to-do list. Here are its checks:

```python
("app loads", 1, True, [], APP_LOADS),
("add + toggle + remaining", 3, True, ["app loads"], js([["app.addTask('a')", "app.addTask('b')", "app.toggle(0)", "app.remaining()"]], 1)),
("saves JSON under 'todos'", 2, True, ["add + toggle + remaining"], js([["app.addTask('a')", "JSON.parse(localStorage.getItem('todos')).length"]], 1)),
("survives a refresh", 3, True, ["saves JSON under 'todos'"], js([["app.addTask('a')", "app.addTask('b')"], ["app.remaining()"]], 2)),
("ignores blank tasks", 1, False, ["app loads"], js([["app.addTask('   ')", "app.all().length"]], 0)),
```

Each line has five parts: a name, a weight, whether the user asked for it (True) or it is implicit (False), the checks it depends on, and the test itself.

- "app loads" just asks: does `window.app` exist? Everything else depends on it.
- The second check adds two tasks, ticks one off, and expects 1 task left. It is worth 3 points.
- The third check looks inside `localStorage` (the browser's small built-in storage) for the saved list.
- The fourth check is the clever one. The inner lists are two "page loads". It adds two tasks, then reloads the page and asks how many remain. The answer must be 2.
- The last check is implicit. Nobody asked, but a good app ignores a blank task.

If the app never saves, the refresh check is skipped instead of failing for a confusing reason. The report then points straight at the real problem.

## Common mistakes

- **Grading by "it compiles."** A page can load and still lose all your data on refresh. Test what a user does.
- **Only checking what was asked.** Implicit checks catch the gap between a literal answer and a good one. Keep them, but weight them lower.
- **Letting one failure cascade.** Without dependencies, a missing button looks like ten bugs. Skip children of failed parents.
- **Trusting the grader blindly.** First make sure a known-good app scores 100 and a known-broken app scores low. Then compare with a human.
- **Assuming the agent wins.** Measure a strong one-shot model next to the agent product before you pay for the agent.

## Check your understanding

**Q: Why does AppBench skip a check when the check it depends on fails?**
A: So one root problem shows up once. Otherwise a single missing feature would cause a pile of confusing failures.

**Q: What is the difference between an explicit and an implicit check?**
A: An explicit check tests what the user asked for. An implicit check tests what a good app would also do, like ignoring blank input.

**Q: How did the team know their automatic grader was any good?**
A: They compared its scores with three human raters. The Pearson correlation was 0.8173, so the grader mostly agreed with people.

## Try it in Colab

Run the notebook to grade a known-good and a known-broken app, then let the local model build all five apps in one shot. Look at which checks fail as the level goes up, especially the implicit ones and the refresh test. Then run the one fix round and see how much precise failure messages help.
