---
summary: How a Stanford team graded AI-built web apps by what a user can do, and how to check the grader first.
minutes: 19
source: CS329A past project · AppBench
slides: https://cs329a.stanford.edu/winter2025projects/appbench.pdf
---

## Learning objective
- Build a checklist grader for AI-built apps from a plain request, and test the grader against people before you trust it.

# The problem and why it matters

## The project at a glance
- AppBench: Benchmarking AI-Generated Web Applications.
- Team: Ethan Hellman, Brendan McLaughlin, Abhinav Lalwani, Belinda Mo (CS329A, Winter 2025).
- Sources: the 12-page report and the poster.
- Claim: "to our knowledge, the first comprehensive framework" to grade AI-built web interfaces in a scalable, automatic, user-centric way.

## The gap they saw
- LLMs now write multi-file programs and fix real GitHub issues.
- Tools can scaffold a whole web app from one sentence.
- Benchmarks did not keep up:
  - Most check function-level correctness or project-scale bug fixes.
  - None check if an interactive app is usable.
- "A web app is more than just working code — it's an experience."

## "It compiles" vs "a user can use it"
| Unit test | User check |
|---|---|
| `tip(50, 20)` returns 10 | Type a bill, click Calculate, see the tip |
| The page loads | Add two items to the cart, refresh, both are still there |
| Passes on the happy path | An empty search shows a clear error |

- Usability needs intuitive interaction, logical navigation and visual coherence.
- Measuring that needs reasoning about how real users use the app, not unit tests.

## The two questions (poster)
- 1. How do you build a dataset that tests whether frontier models can build high-quality web apps?
- 2. How do you build a system that fairly grades app quality and match to the user's request?
- Dataset rules: (a) must be feasible to grade automatically; (b) must span the diversity of real web apps.

## Their three contributions
- Name a gap: no benchmark for interactive, user-facing apps.
- A new simulation-based grader: scripted task plans plus heuristics that act like a user.
- Show automatic UX grading is feasible: strong agreement with human scores across models and agent systems.
- UX: user experience; how easy and pleasant an app is to use.

# Background they build on

## Code-generation benchmarks
| Benchmark | What it checks |
|---|---|
| HumanEval, APPS | Single functions, judged by test cases |
| CodeXGLUE, DS-1000, MultiPL-E | Many languages, data-science code, varied tasks |
| SWE-bench, ClassEval, InterCode | Project-scale bug fixing, class-level code, interactive coding |

- All still judge by test cases or unit assertions.
- Critics: a high pass rate may not mean reliable software, especially when users touch it.

## Web-agent benchmarks
- WebArena, Mind2Web, AgentBench, WebShop: test an LLM acting inside a browser.
  - Navigation, search, finishing a goal on a website.
- The gap: they test using the web, not building a usable app.
- They also assume fixed, pre-designed pages.
- AppBench asks: can the model create the interface from scratch?

## UX research (HCI)
- HCI: human-computer interaction, the field that studies usability.
- Decades of work: heuristic evaluation (Nielsen and Molich, 1990), automated usability testing, design scoring.
- WebQEM and layout-quality methods: structured ways to score effectiveness, efficiency, looks.
- UIcrit: a dataset of UI design critiques.
  - Limit: static screenshots or human labels, so it does not scale.

## Simulation-based evaluation
- Idea: grade a system by acting out tasks, not by fixed metrics.
- Used in NLP (Dynabench), agent benchmarks (AgentBench), test writing (LMUnit, HITS).
- SimulBench: simulation tasks to test creativity and procedure.
- AppBench applies this to web apps: an agent plays the user and grades the app as an experience.

# The method, step by step

## What one task contains
- A natural-language user request.
- An executable grading script written for that request.
- The script outputs:
  - A single 0–100 quality score.
  - A plain-language reason for the score.
- Each script is tuned to match human judgment.

## Six axes of web development
| # | Axis | What it tests |
|---|---|---|
| 1 | UI complexity | Visual and structural sophistication |
| 2 | Feature coverage | Breadth of capabilities |
| 3 | State management | User actions and saved data |
| 4 | API integration | Calling outside services |
| 5 | Cross-page functionality | Navigation, multi-page flows |
| 6 | Data processing | Handling structured data |

- They note the axes should later be weighted by real-world importance.

## Five difficulty levels
- L0: static app, almost no interaction.
- L1: basic interaction, simple state.
- L2: multi-page, navigation, moderate state.
- L3: complex state across views, plus an outside API.
- L4: advanced state, real-time updates, complex interaction.

## The nine tasks (Table 4)
| Task | Level | Axes | What it is |
|---|---|---|---|
| l0-about-us | L0 | 1, 2 | Static about page, responsive |
| l0-welcome-page | L0 | 1, 2 | Landing page with hero section |
| l1-color-switcher | L1 | 1, 2, 3 | Theme switcher |
| l1-weather-search | L1 | 1, 3, 4 | Weather app with an API |
| l2-bookstore | L2 | 1, 2, 5, 6 | Multi-page book catalog |
| l2-add-to-cart | L2 | 1, 3, 5 | Shop with a cart |
| l3-food-order | L3 | 1, 2, 3, 5, 6 | Restaurant menu with an order |
| l3-newsletter | L3 | 1, 3, 4 | Sign-up that sends an email |
| l4-whiteboard | L4 | 1, 2, 3, 4, 6 | Live shared drawing board |

## Caveat: the task labels don't agree
- Table 5 (full requests) uses other labels for five tasks:
  - l1-food-order, l2-newsletter, l2-weather-search, l3-add-to-cart, l3-bookstore.
- Figure 3 also says "l3-add-to-cart"; Tables 3 and 4 say l2.
- The poster calls the weather request an L2 example.
- Its L4 example (live stock prices) is not one of the nine report tasks.
- We use the Table 4 labels, as Table 3 does.

## Example requests (Table 5)
- L0 welcome page: "a header, a centered welcome message, and a footer... a warm, welcoming landing page."
- Theme switcher: three buttons (light, dark, blue); the whole page updates at once; "no backend... just simple state management."
- Newsletter: name and email, submit, "receive a welcome email" through the MailSlurp email service.
- Whiteboard: draw with the mouse, at least 3 colors, a connected-users list, an invite link; see other users' strokes live.
- For private APIs, the request includes the login keys.

## Step 1: request → checklist
- Example request (add-to-cart): product list from an API, detail page with "Add to Cart", a cart icon in the header, cart survives navigation and refresh.
- Human evaluators turned it into an ordered checklist.
- Two kinds of items:
  - Core (explicit): what the user asked for.
  - Quality-enhancing (implicit): what a good app would also do.
- The checklist sits between the request and the final grading graph.

## The add-to-cart checklist (Figure 2)
| Core (explicit) | Quality-enhancing (implicit) |
|---|---|
| Home page lists products | A sensible site title |
| Each shows name, price, thumbnail | A back button to the home page |
| Persistent header with cart | Add several items with a quantity control |
| Clicking a product opens a new page | View items in the cart |
| Detail page: description, big image, add button | Remove items from the cart |
| Cart icon still in header on detail page | |
| Add to cart → icon shows 1 | |
| Refresh or go back → cart persists | |

- Each item is tagged UI, navigation, or state.

## Step 2: checklist → dependency graph
- Some checks need others first.
  - "Add to cart" needs the detail page to open.
  - The detail page needs clickable products on the home page.
- Graph: each node = one testable requirement; each edge = "needs this first".
- The graph models a user's path through the app.

## The add-to-cart graph (Figure 3)
- Root: "Has home page with list of products".
- Its children: name/price/thumbnail, header with cart icon, site title (implicit), clicking opens detail page.
- Under "detail page": description + big image + add button, cart icon on detail page, back button (implicit).
- Under "add button": cart count updates, view cart (implicit).
- Leaves: cart persists on refresh; add several at once (implicit); remove from cart (implicit).
- Pink nodes = core, scored first. Purple = implicit, extra points.

## Step 3: graph → script (the harness)
- Harness: a shared toolbox of grading tools, chained per node.
- All tools run on Playwright (a library that drives a real browser).

| Tool | What it does |
|---|---|
| Multimodal LLM reasoning | A model reads a screenshot plus a question; no need to parse the page code |
| LLM-driven browser (Stagehand) | Plain commands like "click the add to cart button" |
| Network interception | Blocks or breaks API calls to test error screens |

## Screenshot checks with a fixed answer shape (A.3.1)
- The judge must fill a typed schema, so its answer is a number, not prose.
- Weather UI schema fields:
  - has_search, has_button, has_title: yes/no.
  - layout_score: a number from 0 to the layout maximum.
  - reasoning: text.
- Error states use a separate check, e.g. "empty search submission".

## Skip children of failed parents (A.3.3)
- Weather example: run the weather-display and error tests only if the UI score is at least 4.
- Otherwise log "Skipping weather tests due to insufficient UI score".
- Why: stops cascading failures when a basic requirement is missing.
- It "mirrors how human testers would approach" an app.

## Weighted scoring config (A.4)
| Category | Max | Subcategories (max) |
|---|---|---|
| Initial UI | 8 | Search input 2.5, search button 2.5, layout 2, title 1 |
| Weather display | 15 | Temperature 4, condition 3, extra data 3, display quality 5 |
| Error handling | 4 | Empty search 2, invalid city 2 |

- Weights are declared at the top of each script, mapped to the graph.
- Retuning weights to match humans is a one-line change.
- Scores are scaled to 0–100 so tasks of different size compare.

## A graded example (poster)
- Mobile landing page UI: 6.8 / 7.
- Search input found, search button found, page title found in mobile view.
- Note: "Clear, well-organized layout."
- Lost points: an "Edit with lovable" button overlaps the page.

# Setup

## Seven builders, one shot each
| Type | Builders |
|---|---|
| Closed-source agent products | Replit, Bolt, Cursor Agent, Lovable |
| Zero-shot models | Claude Sonnet 3.7, GPT-4.5 |
| Open-source agent | OpenHands CodeAct 2.1 |

- One-shot: same prompt, one turn to build a working app.
- The report's tables call the four products "Service 1–4" and don't map names to numbers.
- The poster's chart lists Replit, Bolt, Cursor Agent, Lovable in the same order and colors.

## Helping the plain models (A.5)
- Claude 3.7 and GPT-4.5 first got a description of a starter Next.js project.
  - Next.js 15.2.3, TypeScript, Tailwind CSS, ESLint, React 19.
  - The folder layout: src/pages, src/styles, public, config files.
- Then: "Build the following application as described starting from the default code."
- Why: avoid apps "overly simplistic" next to the agent products.
- Every app was then graded by the same task script.

## Human check of the grader
- 3 independent human raters scored every generated app.
- They did not see the automatic scores.
- Compare grader vs. human: MSE, MAE, Pearson, Spearman.
  - MSE: mean squared error (0–1 scale, lower is better).
  - MAE: mean absolute error.
  - Pearson: do the scores rise together in a straight line?
  - Spearman: do the two rank the apps in the same order?

# Results

## Grader vs. humans (Table 1)
| Metric | Value |
|---|---|
| Mean MSE | 0.0302 |
| Mean MAE | 0.1313 |
| Pearson correlation | 0.8173 |
| Spearman correlation | 0.6827 |

- An MAE of 0.1313 means the grader is off by about 13 points out of 100 on average.
- Pearson higher than Spearman: the overall trend matches better than the exact ordering.

## Error per builder (Table 2)
| Builder | MSE |
|---|---|
| Service 1 | 0.0466 |
| Service 2 | 0.0307 |
| Service 3 | 0.0303 |
| Service 4 | 0.0160 |
| Claude 3.7 | 0.0199 |
| GPT 4.5 | 0.0179 |
| CodeAct | 0.0501 |

- The grader agreed least with humans on CodeAct and Service 1.

## Error per task (Table 3)
| Task | MSE |
|---|---|
| l0-about-us | 0.0194 |
| l0-welcome-page | 0.0179 |
| l1-color-switcher | 0.0194 |
| l1-weather-search | 0.0163 |
| l2-bookstore | 0.0212 |
| l2-add-to-cart | 0.0042 |
| l3-food-order | 0.0751 |
| l3-newsletter | 0.0653 |
| l4-whiteboard | 0.0334 |

- Best agreement on add-to-cart, the task with the worked-out graph. Worst on the two L3 tasks.

## Human vs. grader scatter (Figure 6)
- x-axis: human score (0–1). y-axis: grader score (0–1). Dotted line: perfect agreement.
- Most points cluster in the top-right: both rate the app highly.
- Lower-scored apps scatter more widely around the line.
- A few points sit well below the line: the grader was harsher than the humans.

## Scores per builder per task (Figure 4)
- Bar chart: one group per task, one bar per builder, score 0–1.
- L0–L2 tasks: most bars high.
- l3-food-order and l3-newsletter: lower and more mixed.
- l4-whiteboard: most bars low; some builders have almost no bar.
- Poster: 2 of 7 builders produced no compiling app for one L3 and one L4 task.

## Scores by builder type and level (Figure 5)
- Three lines over L0–L4: closed-source agents, zero-shot models, open-source agent.
- All three fall as the level rises; all drop steeply at L4.
- Zero-shot models are on par with or above the agent products up to L1, then fall below.
- The open-source agent dips at L2, then is lowest at L4.
- Closed-source agents end highest at L4, but still low.

## Closed-source products compared (poster)
- A poster chart compares Replit, Bolt, Cursor Agent and Lovable by level.
- Lovable stays the steadiest through L3–L4.
- Cursor Agent falls to near zero at L4.
- Poster verdict: "Loveable and Bolt emerge as the top-performing agents."
- Qualitative table: whiteboard screenshots from Replit, Lovable, Claude Sonnet 3.7 and CodeAct differ a lot in polish.

## The headline findings (poster)
- All builder types score above 75% up to L2.
- Scores start to drop at L3 and fall sharply at L4.
- 2 of 7 builders could not produce a compiling app for 1 L3 and 1 L4 task.
- The gap between builder types narrows at higher levels: a sign of shared limits.
- Zero-shot models are "surprisingly competitive" despite a simpler design and lower compute.

# Analysis and discussion

## Do agent frameworks pay off?
- Expected: agent products far ahead of single prompts, especially on hard tasks.
- Found: the gap was "modest, and in some cases negligible".
- Their question: are agent pipelines worth the added complexity and cost?
- Or are returns shrinking without better design or supervision?

## Why L4 erases differences
- At L4 most systems can't build a usable app.
- "Partially functional" vs "completely broken" no longer matters to a user.
- Agents sometimes write cleaner code or better structure.
- That does not make the app more usable.

## Looks vs. usability
- Closed-source agent products tend to make more polished pages.
- They do not finish user tasks much better than zero-shot models.
- Lesson: visual appeal and functional usability are separate axes of quality.
- Agent infrastructure alone does not close the gap.

## What the grader result shows
- Strong correlation with humans: simulated UX grading can approximate people.
- Their point: automation and user-centered design can work together.
- Weakest spot: the middle band, apps neither clearly working nor clearly broken.

# Limitations and future work

## Limitations they state
- Only 9 tasks: "premature to draw strong generalizations".
- Grading scripts were written by hand per task.
- Graders need calibration with expert UX reviewers.
  - Sharper thresholds, better detection of failure types.
  - Better separation of builders in the middle band.
- Plus our note: one run per builder, 3 raters, and the label mix-up between Tables 4 and 5.

## Future work they list
- More request–grader pairs for wider coverage.
- Better grading scripts: more tools, stricter criteria.
- A more accurate grader agent.
- Automatic construction of graders.
- Finer experiments that score each of the six axes.

# Our smaller Colab rebuild

## What we change
| AppBench | Our rebuild |
|---|---|
| 9 apps, L0–L4 | 5 single-file HTML apps, L0–L3 |
| Real browser (Playwright, Stagehand) | No browser: Node.js runs the app's script in a fake page |
| Screenshot LLM judge | Python HTML parser for structure; LLM judge only in solutions |
| 7 commercial builders | The local model, one shot, then one fix round |
| 3 human raters | Known-good, known-broken, crashing apps; your own hand scores in solutions |

- Each app must expose its logic as functions on `window.app`, so code can "click" it.

## Our five apps
| App | Level | Key checks |
|---|---|---|
| l0-about-us | L0 | header names Acme Labs, 2+ paragraphs, footer address |
| l1-theme-switcher | L1 | default light, setTheme works, unknown theme ignored |
| l1-tip-calculator | L1 | tip(50, 20) = 10, total(80, 15) = 92, rounds to cents |
| l2-food-order | L2 | 2 Burgers + Soda = 18.75, remove never below zero |
| l3-todo-saved | L3 | saved under 'todos', survives a refresh |

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

## Skipping children, in code
```python
if any(not results.get(d, (False,))[0] for d in deps):
    results[cname] = (False, "skipped: depends on " + ", ".join(deps))
```
- `deps`: the parent checks listed in the tuple.
- If any parent failed, this check is not run.
- It scores 0 and says which parent blocked it, like AppBench's "insufficient UI score" skip.

## Test the grader first
- Known-good tip calculator: must score 100.
- Known-broken one (labels removed, tip not divided by 100, button id changed): must score under 60.
- Crashing script (calls an undefined function first): core behavior score under 50.
- Only then grade the model's apps.

## One shot vs one fix round
| One shot | One fix round |
|---|---|
| Same prompt, one turn, no feedback | Model gets the names of its failed checks, one chance to fix |
| 1 call per app | 1 more call per app (0 if nothing failed) |

- Our checks name the exact problem, so the fix round often helps.
- AppBench builders got no such feedback; one reason their agents' edge was small.

## Solutions: three extensions
- Add an implicit "clear the order" check nobody asked for; most apps fail it.
- Score apps by hand and compute Spearman with the grader (AppBench: 0.68 with 3 humans).
- A text-only LLM judge scores each file 0–10.
  - It often rates broken logic highly: reading code is not clicking.

# How this applies to us

## Small tools a firm will ask for
- A deadline tracker, a fee calculator, a playbook-breach list.
- A local model can build each one as a single page.
- The question is not "does it load" but "can a lawyer use it".

## A checklist for a contract tool
| Core (explicit) | Implicit |
|---|---|
| Upload a contract, see its liability clause | Blank upload shows an error |
| "Liability cap over 12 months of fees" is flagged | Flag links to the clause text |
| Flags survive a refresh | Export the flag list |

- Parents first: no clause found → skip the "flagged" check.

## Check the grader on your own machine
- Keep a known-good and a known-broken tool next to the grader.
- Known-good must score 100, known-broken low, before each run.
- Then have one or two lawyers score a few tools by hand; compute Spearman.
- Everything runs locally: Node.js, Python, a local model. No contract leaves the machine.

## Is an agent loop worth it for us?
- AppBench: big agent products barely beat one-shot models.
- Our cheap version of an agent: one fix round with named failures.
- Measure both on your tasks before paying for more calls.

## Common mistakes
- Grading by "it compiles" → test what a user does, including a refresh.
- Only explicit checks → add implicit ones, weighted lower.
- One failure cascades into ten → skip children of failed parents.
- Trusting the grader blindly → known-good must score 100, known-broken low; then compare with a human.
- Treating a good-looking page as a working one → polish and usability are separate axes.

## Check your understanding
**Q: Why does AppBench skip a check when its parent check fails?**
A: So one root problem shows up once. Otherwise one missing feature causes a pile of confusing failures.

**Q: What is the difference between an explicit and an implicit check?**
A: Explicit tests what the user asked for. Implicit tests what a good app would also do, like a back button.

**Q: How did the team know their grader was any good?**
A: Three humans scored every app blind. The grader's Pearson correlation with them was 0.8173 (Table 1).

**Q: Why is Spearman (0.6827) lower than Pearson (0.8173)?**
A: The overall trend matches, but the grader often orders close apps differently from the humans.

**Q: Which task did the grader match humans on best, and worst?**
A: Best on l2-add-to-cart (MSE 0.0042); worst on l3-food-order (MSE 0.0751), Table 3.

**Q: Did agent products beat one-shot models?**
A: Only modestly, sometimes not at all. At L4 nearly everything failed, so the gap disappeared.

**Q: Why does our rebuild need `window.app`?**
A: Colab has no browser. Exposed functions let Node.js call the app's logic like a user's clicks would.

## Try it in Colab
- Grade the known-good, known-broken and crashing tip calculators; check the scores.
- Let the local model build all five apps; see which checks fail as the level rises.
- Run the fix round and compare the one-shot and after-fix scores.
