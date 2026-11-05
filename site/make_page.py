"""Build the course page from the notebook sources, so page code == Colab code.

    uv run --no-project --with markdown --with nbformat python site/make_page.py

Writes docs/index.html (GitHub Pages) and site/artifact.html (claude.ai copy).
"""
import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "site"))
from build import SETUP, REPO, parse  # noqa: E402
from terms import TERMS  # noqa: E402

SOURCES = {f"m{i}": p for i, p in enumerate(sorted((ROOT / "scripts/src/notebooks").glob("*.py")))}
SOURCES["p1"] = ROOT / "scripts/src/projects/P1_build_your_own_harness.py"
SOURCES["p2"] = ROOT / "scripts/src/projects/P2_build_an_eval_task.py"
for _p in sorted((ROOT / "scripts/src/cs329a").glob("a*.py")):
    SOURCES[_p.stem.split("_")[0]] = _p                 # a01, a02, ...
for _p in sorted((ROOT / "scripts/src/cs329a_projects").glob("ap*.py")):
    SOURCES[_p.stem.split("_")[0]] = _p                 # ap1, ap2, ...
FOLDER = {"m": "notebooks", "p": "projects", "a": "cs329a"}
SOLUTION_HEAD = re.compile(r"^#+\s*(Solutions?|Stretch goal solutions|Solution:)", re.I)

EXTRA_CSS = """
.terms{margin:6px 0 22px;padding:14px 18px;background:var(--paper);border:1px solid var(--line);border-radius:10px;max-width:76ch}
.terms h4,.nbhead h4{font:600 12px var(--f-mono);text-transform:uppercase;letter-spacing:.08em;color:var(--accent);margin:2px 0 8px}
.terms dl{margin:0;display:grid;gap:8px}
.terms dt{font-weight:600}
.terms dd{margin:0;color:var(--ink)}
.terms .from{display:block;font-size:12.5px;color:var(--muted);margin-top:1px}
.nbhead{margin:30px 0 6px;padding:14px 18px;border-left:4px solid var(--ink);background:var(--paper);border-radius:0 10px 10px 0;max-width:76ch}
.nbhead p{margin:4px 0}
.nb-md{max-width:72ch}
.nb-md h3{margin:26px 0 6px}
.nb-md h4{font:700 16px/1.3 var(--f-display);margin:18px 0 6px}
.nb-md pre{background:var(--code-bg);border-radius:10px}
.nb-md table{min-width:0}
details.solutions{margin:18px 0;border:1.5px solid var(--accent);border-radius:10px;padding:8px 16px;background:var(--paper)}
details.solutions>summary{font:600 15px var(--f-body);color:var(--accent);padding:4px 0}
.code .cap .cell{color:#c7d3de}
"""


def md_to_html(text: str) -> str:
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("**Next:**"))
    out = markdown.markdown(text, extensions=["tables", "fenced_code"])
    out = out.replace("<h2>", "<h3>").replace("</h2>", "</h3>").replace("<h1>", "<h3>").replace("</h1>", "</h3>")
    out = re.sub(r"<table>", '<div class="tbl"><table>', out).replace("</table>", "</table></div>")
    out = re.sub(r'<pre><code class="language-(\w+)">', r'<pre><code class="language-\1">', out)
    return f'<div class="nb-md">{out}</div>' if out.strip() else ""


def code_block(code: str, label: str, lang="python") -> str:
    return (f'<div class="code"><div class="cap"><span class="cell">{label}</span><button type="button" class="copy">Copy</button></div>'
            f'<pre><code class="language-{lang}">{html.escape(code)}</code></pre></div>')


def colab_url(mod: str) -> str:
    folder = "cs329a_projects" if mod.startswith("ap") else FOLDER[mod[0]]
    return f"https://colab.research.google.com/github/{REPO}/blob/main/{folder}/{SOURCES[mod].stem}.ipynb"


def render_notebook(mod: str) -> str:
    src = SOURCES[mod].read_text()
    cells = parse(src)[1:]                      # skip the title cell; the page has its own heading
    name = SOURCES[mod].stem + ".ipynb"
    parts = [f'<div class="nbhead"><h4>The code, cell by cell</h4><p>Everything below is exactly what runs in '
             f'<a href="{colab_url(mod)}">{name}</a>, in the same order. The notebook starts with the shared setup cell '
             f'(shown in Module 00), then these cells.</p></div>']
    lesson, solutions, n = [], [], 1
    target = lesson
    for kind, text in cells:
        if kind == "md" and SOLUTION_HEAD.match(text.strip()):
            target = solutions
            rest = text.strip().split("\n", 1)
            if len(rest) > 1:
                target.append(md_to_html(rest[1]))
            continue
        if kind == "md":
            target.append(md_to_html(text))
        else:
            n += 1
            target.append(code_block(text, f"{name} · cell {n}"))
    parts += lesson
    if solutions:
        parts.append('<details class="solutions"><summary>Show worked solutions (try the exercises first)</summary>'
                     + "".join(solutions) + "</details>")
    return "\n".join(parts)


def render_terms(mod: str) -> str:
    items = TERMS.get(mod)
    if not items:
        return ""
    rows = "".join(f"<div><dt>{html.escape(t)}</dt><dd>{html.escape(d)}<span class=\"from\">Source: {html.escape(s)}</span></dd></div>"
                   for t, d, s in items)
    return f'<div class="terms"><h4>Key terms</h4><dl>{rows}</dl></div>'


def render_setup() -> str:
    setup = SETUP.format(repo=REPO, extra="[]", models="")
    llm = (ROOT / "lal/llm.py").read_text()
    return ("<h3>The setup cell (first cell of every notebook)</h3>"
            "<p>On Colab it clones the course repo, installs the packages, installs Ollama, starts it, and downloads the model. "
            "On a laptop with Ollama it just connects. Notebooks that need extra packages add them to <code>EXTRA</code>.</p>"
            + code_block(setup, "every notebook · cell 1")
            + "<h3>The helper every notebook uses: <code>lal/llm.py</code></h3>"
            "<p>A thin wrapper around the OpenAI-compatible API that Ollama and vLLM expose. Point <code>LAL_BASE_URL</code> "
            "somewhere else and the same notebooks run on another server.</p>"
            + code_block(llm, "lal/llm.py")
            + render_notebook("m0"))


PART2_CSS = """
.part-banner{margin:56px 0 0;padding:26px 24px;border-radius:12px;background:var(--ink);color:var(--paper)}
.part-banner h2{color:var(--paper);margin:4px 0 8px}
.part-banner p{color:var(--paper);opacity:.9;margin:6px 0;max-width:70ch}
.part-banner a{color:var(--paper)}
.papers{display:grid;gap:12px;margin:12px 0 18px}
.paper{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:14px 18px;max-width:80ch}
.paper h4{font:700 16px/1.35 var(--f-display);margin:0 0 2px}
.paper .by{font-size:13px;color:var(--muted);margin:0 0 8px}
.paper dl{margin:0;display:grid;grid-template-columns:max-content minmax(0,1fr);gap:4px 12px}
.paper dt{font:600 11.5px var(--f-mono);text-transform:uppercase;letter-spacing:.06em;color:var(--accent);padding-top:3px}
.paper dd{margin:0}
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def section_head(mod, num_label, title, src_line):
    return (f'<section class="mod" id="{mod}" data-mod="{mod}">\n  <div class="mod-head"><div><div class="mod-num">{esc(num_label)}</div>'
            f'<h2>{esc(title)}</h2></div><span class="actions"><a class="colab-btn" href="{colab_url(mod)}">Open in Colab</a>'
            f'<button class="done-btn" data-done="{mod}" aria-pressed="false">Mark done</button></span></div>\n'
            f'  <p class="src">{esc(src_line)}</p>\n')


def render_papers(papers, heading="The papers"):
    cards = []
    for pp in papers:
        title = esc(pp.get("title"))
        link = f'<a href="{esc(pp["url"])}">{title}</a>' if pp.get("url") else title
        rows = "".join(f"<dt>{lab}</dt><dd>{esc(pp.get(key))}</dd>" for lab, key in
                       (("What", "what"), ("Result", "result"), ("Takeaway", "takeaway")) if pp.get(key))
        cards.append(f'<div class="paper"><h4>{link}</h4><p class="by">{esc(pp.get("authors"))} · {esc(pp.get("year"))}</p><dl>{rows}</dl></div>')
    return f"<h3>{heading}</h3><div class=\"papers\">" + "".join(cards) + "</div>"


LESSONS = """<section class="mod" id="part2-lessons">
  <div class="mod-head"><div><div class="mod-num">Part 2 · Start here</div><h2>What this means for building agents</h2></div></div>
  <p class="big"><b>The big idea of CS329A:</b> an agent gets better only as fast as you can check its work automatically. Almost every paper below is a way to use a checker: to pick the best answer, to give feedback, to search, or to train.</p>
  <ol>
    <li><b>More tries plus a good checker can beat a bigger model.</b> The chance that at least one of many samples is right keeps rising as you sample more, roughly following a power law. The hard part is picking the right answer: with a checker you capture the gain, while majority vote flattens early (Brown et al., <i>Large Language Monkeys</i>; module A01).</li>
    <li><b>Weak checkers, combined, make a strong one.</b> Several imperfect checks, such as a model judge, answer agreement and simple rules, can be weighted into one reliable verifier (<i>Weaver</i>; module A02). Contract review needs exactly this: quote checks + rule checks + a judge.</li>
    <li><b>Feedback from running code is the best teacher.</b> Train or loop on real test results, and pick among many attempts by running tests (<i>RLEF</i>, <i>CodeMonkeys</i>; modules A03 and A08).</li>
    <li><b>Searching for better designs beats hand-designing them, but only with an automatic score.</b> AlphaEvolve, ADAS and the AI Scientist all propose, score automatically, keep the best and repeat. If the score has a loophole, the search will find it (module A06 shows a cheat winning once a validity check is removed).</li>
    <li><b>Reinforcement learning on checkable rewards is now cheap enough to use.</b> GRPO and DAPO's fixes make it practical on small models; module A05 trains a 360M model on a free Colab GPU.</li>
    <li><b>Measure how long a task an agent can finish, not just whether it's right.</b> METR's "time horizon" is the length of task an AI can complete half the time, and it has been roughly doubling every seven months. GDPval grades AI work on real paid tasks against expert work (module A10).</li>
  </ol>
  <div class="thesis"><b>For Local AI Lab:</b> the moat isn't the model. It's the <b>checkers and tests we build for each customer's repeated task.</b> Once a task has an automatic score, every technique in this part (sample and select, combined verifiers, search, RL on small models) can keep improving a local model on that task, cheaply. The pitch: <i>we turn your repeated work into a scored task, and the model on your own computers keeps getting better at it.</i></div>
</section>"""


def render_part2():
    import json
    mods = sorted((ROOT / "site/cs329a").glob("a*.json"))
    projects_file = ROOT / "site/cs329a/projects.json"
    if not mods and not projects_file.exists():
        return "", ""
    out = ['<div class="part-banner" id="part2"><p class="eyebrow" style="color:var(--paper)">Part 2</p>'
           '<h2>Self-improving AI agents</h2>'
           '<p>Everything from Stanford CS329A "Self-Improving AI Agents" (Autumn 2025, Azalia Mirhoseini and Aakanksha Chowdhery): '
           'every assigned paper summarized, a runnable Colab notebook per lecture, and small rebuilds of the past student projects. '
           'Course site: <a href="https://cs329a.stanford.edu/">cs329a.stanford.edu</a>.</p></div>',
           LESSONS]
    nav = ['    <li class="part">Part 2 · Self-improving agents</li>',
           '    <li><a href="#part2-lessons"><span class="n">★</span>What this means</a></li>']
    for f in mods:
        d = json.loads(f.read_text())
        mod = d["id"]
        if mod not in SOURCES:
            continue
        nav.append(f'    <li><a href="#{mod}"><span class="n">{mod.upper()}</span>{esc(d["title"])}</a></li>')
        sec = [section_head(mod, f"Part 2 · {mod.upper()}", d["title"], d.get("lecture", "")),
               f'  <p class="big">{esc(d.get("big_idea"))}</p>']
        if d.get("terms"):
            rows = "".join(f'<div><dt>{esc(t[0])}</dt><dd>{esc(t[1])}<span class="from">Source: {esc(t[2] if len(t) > 2 else "")}</span></dd></div>' for t in d["terms"])
            sec.append(f'<div class="terms"><h4>Key terms</h4><dl>{rows}</dl></div>')
        if d.get("papers"):
            sec.append(render_papers(d["papers"]))
        sec.append(render_notebook(mod))
        if d.get("traps"):
            sec.append('<div class="pit"><strong>Traps</strong><ul>' + "".join(f"<li>{esc(t)}</li>" for t in d["traps"]) + "</ul></div>")
        if d.get("check"):
            sec.append('<div class="check"><h4>Check yourself</h4>' + "".join(
                f"<details><summary>{esc(q)}</summary><p>{esc(a)}</p></details>" for q, a in d["check"]) + "</div>")
        sec.append("</section>")
        out.append("\n".join(sec))
    if projects_file.exists():
        nav.append('    <li class="part">Part 2 · Past projects</li>')
        for pr in json.loads(projects_file.read_text()):
            mod = pr["id"]
            if mod not in SOURCES:
                continue
            nav.append(f'    <li><a href="#{mod}"><span class="n">{mod.upper()}</span>{esc({"ap3": "Math reasoning (distillation, GRPO)"}.get(mod, pr["title"].split(":")[0]))}</a></li>')
            links = " · ".join(f'<a href="{esc(u)}">{lab}</a>' for lab, u in (("Paper", pr.get("paper_url")), ("Poster", pr.get("poster_url"))) if u)
            sec = [section_head(mod, f"Past project · {mod.upper()}", pr["title"], f"CS329A Winter 2025 · {pr.get('authors', '')}"),
                   f'  <p class="src">{links}</p>',
                   '<div class="paper"><dl>' + "".join(f"<dt>{lab}</dt><dd>{esc(pr.get(k))}</dd>" for lab, k in
                       (("Problem", "problem"), ("Method", "method"), ("Results", "results"), ("Lessons", "lessons"), ("Our rebuild", "rebuild")) if pr.get(k)) + "</dl></div>",
                   render_notebook(mod), "</section>"]
            out.append("\n".join(sec))
    return "\n\n".join(out), "\n".join(nav)


def build():
    s = (ROOT / "site/template.html").read_text()
    s = s.replace("</style>", EXTRA_CSS + PART2_CSS + "</style>", 1)
    part2, part2_nav = render_part2()
    s = s.replace("<!--PART2-->", part2).replace("<!--PART2NAV-->", part2_nav)
    for mod in SOURCES:
        s = s.replace(f"<!--TERMS:{mod}-->", render_terms(mod))
        s = s.replace(f"<!--NB:{mod}-->", render_setup() if mod == "m0" else render_notebook(mod))
    left = re.findall(r"<!--(?:NB|TERMS):\w+-->", s)
    assert not left, left
    (ROOT / "site/artifact.html").write_text(s)

    # GitHub Pages version: a full HTML document, normal new-tab links, mermaid loaded from a CDN
    p = re.sub(r'\s*// Open in Colab: try a new window first.*?\n  \}\);\n\n', "\n", s, flags=re.S)
    p = re.sub(r'\s*var cr = document.getElementById\("copyRepo"\);.*?\}\);\n', "\n", p, flags=re.S)
    p = re.sub(r"<p><b>For one-click Colab, use the team site:</b>.*?</p>", "", p, flags=re.S)
    p = p.replace('.ipynb">Open in Colab</a>', '.ipynb" target="_blank" rel="noopener">Open in Colab</a>')
    p = re.sub(r'(<a href="https://colab[^"]+")>', r'\1 target="_blank" rel="noopener">', p)
    p = p.replace('target="_blank" rel="noopener" target="_blank" rel="noopener"', 'target="_blank" rel="noopener"')
    p = p.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    head, body = p.split("</style>", 1)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="description" content="Local AI Lab: a hands-on course on building AI agents that run on a local model, with Colab notebooks.">\n'
            + head + "</style>\n</head>\n<body>\n" + body.rstrip()
            + '\n<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"; '
              'mermaid.initialize({startOnLoad: true, theme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "default"});</script>\n'
              "</body>\n</html>\n")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/index.html").write_text(page)
    n_code, n_terms = s.count('class="code"'), s.count("<dt>")
    print(f"artifact.html {len(s)//1024} KB, docs/index.html {len(page)//1024} KB, {n_code} code blocks, {n_terms} key terms")


if __name__ == "__main__":
    build()
