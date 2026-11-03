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
    folder = "projects" if mod.startswith("p") else "notebooks"
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


def build():
    s = (ROOT / "site/template.html").read_text()
    s = s.replace("</style>", EXTRA_CSS + "</style>", 1)
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
