# How to write a lesson page

Each module gets one file: `site/lessons/<id>.md`, where `<id>` is `m0`…`m15`, `p1`, `p2`, `a01`…`a11` or `ap1`…`ap5`. The site shows it as the main teaching on that module's page. The dense reference material (key terms, papers, notebook code) appears below it on the same page, so the lesson should **teach**, not repeat the reference.

## Who reads it
The readers are smart startup people (MBAs, a CS PhD) who are new to AI engineering. They find the current pages too dense. Write the way a great teacher talks to a bright friend over coffee.

## Language rules (strict)
- Short sentences (under 20 words). Everyday words. One idea per paragraph.
- **Every technical term is explained the first time it appears, in plain words, right there.** For example: "the context window (the amount of text the model can read at once)".
- Use analogies from daily life: a new intern, a library, a kitchen, a checklist.
- Use the running example: **a small law firm reviewing contracts against its playbook** (its rules, e.g. "liability cap at most 12 months of fees").
- No hype, no filler ("In today's fast-paced world…"), no emoji. Don't overuse bold.
- Only use numbers that appear in the module's existing material. Never invent results.
- Credit papers by short name, e.g. "(the ReAct paper, 2023)". The page lists the full papers below.

## Exact structure (use these headings, in this order)

```markdown
---
summary: One or two plain sentences for the home-page card (max 30 words). What you'll understand after this lesson.
minutes: 10
---

## The big idea
2–4 short paragraphs. The concept in plain words, with one everyday analogy.

## Why it matters for us
2–3 short paragraphs or bullets. Why this matters for running AI on a firm's own computers, and for the contract-review example.

## How it works, step by step
A numbered list of 3–7 steps, or short subsections. Build the idea up gradually. Explain each new term inline.
You may include ONE small ASCII diagram or a short table if it makes the structure clearer.

## Worked example
A concrete walk-through using the law-firm contract example. Show what goes in, what happens, what comes out.
You may quote 3–10 lines of code from the module's notebook, but explain every line in plain words.

## Common mistakes
3–5 bullets: what people get wrong, and what to do instead.

## Check your understanding
3 questions, each followed by its answer in this exact format:

**Q: question?**
A: answer in 1–3 plain sentences.

## Try it in Colab
2–3 sentences: what to run in this module's notebook and what to look for.
```

Length: about 700–1,100 words. Reaching for long words is a failure; clarity is the goal.

## Where the source material is
- Part 1 modules m0–m15: notebook source `scripts/src/notebooks/NN_*.py`; key terms in `site/terms.py` (TERMS["mN"]); reference prose in `site/template.html` (find `<section class="mod" id="mN"`).
- Projects p1, p2: `scripts/src/projects/P1_*.py`, `P2_*.py`; terms in `site/terms.py`.
- Part 2 a01–a11: `scripts/src/cs329a/aNN_*.py` and `site/cs329a/aNN_*.json` (papers, terms, traps).
- Past projects ap1–ap5: `scripts/src/cs329a_projects/apN_*.py` and `site/cs329a/projects.json`.
