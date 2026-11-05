# How to write a lesson (slide style)

Each module gets one file: `site/lessons/<id>.md`. The site shows it as a stack of **slide cards**, like a lecture deck. The user finds the CS329Z lecture slides easy to read and wants the same language. **Copy the slide style exactly.**

## What the slide style is
Read 2–3 of the lecture text files before writing (`scratchpad/slides/z/lectureNN.txt`). Notice:
- Every slide has a **short title** ("Why Retrieval?", "Chunking: Basic Unit in Retrieval").
- The body is **3–5 short bullets**, often fragments, not full paragraphs. Sub-bullets for detail ("Too small: the vector has no context").
- Ideas are taught by **concrete side-by-side examples**: "LLMs Alone" vs "LLMs + Retrieval"; "Chunk" vs "Contextualized Chunk"; "Same Text, Three Splits".
- **Definitions are one line**, written as `Term: what it does` ("Overlap: 10–20%; no sentence lost at chunk boundary").
- Small **tables** compare options (Type | Examples | What it covers).
- A **Learning Objective** in one sentence near the start.
- No stories, no analogies stretched over paragraphs, no "Think of it like...". No filler.

Reuse the lecture's own wording, definitions and examples where they fit (credit is given by the source link). Do not paste whole slides; rewrite to fit the module. Skip the math formulas; say in one line what the formula does.

## File format (exact)

```markdown
---
summary: One plain sentence for the home-page card (max 25 words).
minutes: 8
source: CS329Z Lecture 3 · RAG + Agents
slides: https://web.stanford.edu/class/cs329z/slides/lecture03.pdf
---

## Learning objective
- One sentence: what you will be able to do after this lesson.

## Slide title
- bullet
- bullet
  - sub-bullet

## Slide title with a comparison
| LLM alone | LLM + retrieval |
|---|---|
| "Try Wednesday afternoon, midweek is usually quiet." | Retrieved queue log → "Thursday 9am, about 3 students ahead of you." |

## Check your understanding
**Q: question?**
A: answer in 1–2 short sentences.

## Try it in Colab
- 2–3 bullets: what to run in this module's notebook and what to look for.
```

Rules:
- 8–14 content slides between "Learning objective" and "Check your understanding". Each slide under ~60 words.
- Each `##` heading becomes one slide card. Use `###` never.
- Explain each technical term the first time, in a few words on the same bullet ("BM25: keyword search; scores chunks by shared words").
- At least 2 slides with a side-by-side example. Use the slide's own example (campus, office hours, ACME) or our contract example (a law firm checking contracts against its playbook, e.g. "liability cap at most 12 months of fees") — whichever is clearer.
- One slide "How this applies to us": 3 bullets on running this on the firm's own machine with a local model.
- One slide "Common mistakes": 3–4 bullets, `Mistake → fix`.
- Code: at most one slide with ≤8 lines from the notebook, each line explained in a bullet below it.
- Only numbers from the lecture or the module's material. Never invent results.
- Exactly 3 Q/A in "Check your understanding".
- `source`/`slides`: the matching CS329Z lecture. Part 2 (CS329A) has no posted slides: set `source: CS329A · <lecture title>` and `slides:` to the main paper's URL.

## Module → lecture (CS329Z slides: https://web.stanford.edu/class/cs329z/slides/lectureNN.pdf)
m0 → 01 (and 02) · m1 → 01 · m2 → 02 · m3 → 02 · m4 → 03 · m5 → 04 · m6 → 06 · m7 → 05 · m8 → 07 · m9 → 08 · m10 → 10 and 13 · m11 → 09 · m12 → 11 and 12 · m13 → 01 (key challenges) · m14 → 10 · m15, p1, p2 → the lectures they draw on (03, 04, 09, 10).

## Where the module material is
- Part 1 m0–m15: `scripts/src/notebooks/NN_*.py`; terms `site/terms.py`; reference prose in `site/template.html` (`<section class="mod" id="mN"`).
- p1, p2: `scripts/src/projects/`.
- Part 2 a01–a11: `scripts/src/cs329a/aNN_*.py`, `site/cs329a/aNN_*.json`.
- ap1–ap5: `scripts/src/cs329a_projects/`, `site/cs329a/projects.json`.
- The old prose lesson is in the same file; reuse its correct facts, drop its style.
