"""Split the course into a short home page plus one page per module.

    uv run --no-project --with markdown --with nbformat python site/multipage.py

docs/index.html          home: what the course is, then one short card per module
docs/m/<id>.html         lesson in plain English (site/lessons/<id>.md), then the deeper material
"""
import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "site"))
import make_page  # noqa: E402

GROUPS = [
    ("Part 1 · Engineering AI agents", [
        ("Basics", ["m0", "m1", "m2", "m3"]),
        ("Building blocks", ["m4", "m5", "m6", "m7", "m8", "m9"]),
        ("Make it good", ["m10", "m11", "m12", "m13", "m14"]),
        ("Put it together", ["m15", "p1", "p2"]),
    ]),
    ("Part 2 · Self-improving agents", [
        ("Start here", ["part2-lessons"]),
        ("Lectures", ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10", "a11"]),
        ("Past student projects, rebuilt small", ["ap1", "ap2", "ap3", "ap4", "ap5"]),
    ]),
    ("Reference", [("", ["glossary"])]),
]
ORDER = [i for _, gs in GROUPS for _, ids in gs for i in ids]

PAGE_CSS = """
.wrap.single{grid-template-columns:250px minmax(0,1fr)}
.topbar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 18px;font-size:14px}
.topbar a{text-decoration:none}
.crumbs{color:var(--muted)}
.lesson{max-width:860px;font-size:17px;line-height:1.72}
.lesson h2{font:700 24px/1.25 var(--f-display);margin:34px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
.lesson h2:first-of-type{border-top:0;padding-top:0;margin-top:8px}
.lesson h3{font:700 19px/1.3 var(--f-display);margin:22px 0 6px}
.lesson p,.lesson li{max-width:70ch}
.lesson pre{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto;font:13.5px/1.5 var(--f-mono);color:var(--ink)}
.lesson pre code{background:none;padding:0}
.lesson table{min-width:0;font-size:15px}
.lesson .tbl{max-width:72ch}
.deck{display:flex;flex-direction:column;gap:18px;margin:10px 0 30px}
.deck-src{color:var(--muted);font-size:14px;margin:0 0 12px}
.slide{border:1px solid var(--line);border-radius:12px;background:var(--paper);padding:20px 26px 18px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.slide-top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;border-bottom:2px solid var(--accent);padding-bottom:8px;margin-bottom:12px}
.slide-top h3{font:700 21px/1.25 var(--f-display);margin:0}
.slide-no{font:600 12px var(--f-mono);color:var(--muted);white-space:nowrap}
.slide ul{margin:6px 0;padding-left:22px}.slide li{margin:5px 0}.slide li ul{margin:2px 0}
.slide p{margin:8px 0}
.slide.objective{background:var(--accent-soft);border-color:var(--accent)}
.slide table{font-size:15px}.slide td,.slide th{vertical-align:top}
.slide pre{margin:8px 0}
.lesson-meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:14px;margin:4px 0 22px}
.lesson-summary{font-size:19px;line-height:1.55;padding:16px 20px;border-left:4px solid var(--accent);background:var(--accent-soft);border-radius:0 10px 10px 0;max-width:68ch;margin:6px 0 24px}
.deeper{margin-top:48px;padding-top:8px;border-top:3px solid var(--ink)}
.deeper>h2.deeper-title{font:800 26px/1.2 var(--f-display);margin:18px 0 4px}
.deeper>p.deeper-sub{color:var(--muted);margin:0 0 10px}
.pager{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:44px 0 10px}
.pager a{display:block;padding:14px 16px;border:1px solid var(--line);border-radius:10px;background:var(--paper);text-decoration:none;color:var(--ink)}
.pager a:hover{border-color:var(--accent)}
.pager small{display:block;color:var(--muted);font-size:12px;font:600 11px var(--f-mono);text-transform:uppercase;letter-spacing:.08em}
.pager .next{text-align:right;grid-column:2}
.cards-group{margin:30px 0 8px}
.cards-group h3{font:600 12px var(--f-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 10px}
.part-title{font:800 clamp(24px,3vw,30px)/1.15 var(--f-display);margin:46px 0 4px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{display:flex;flex-direction:column;gap:6px;padding:16px 18px;background:var(--paper);border:1px solid var(--line);border-radius:12px;text-decoration:none;color:var(--ink);min-width:0;transition:border-color .15s}
.card:hover{border-color:var(--accent)}
.card .num{font:500 12px var(--f-mono);color:var(--accent)}
.card .t{font:700 18px/1.25 var(--f-display)}
.card .s{font-size:14.5px;color:var(--muted);line-height:1.5}
.card .go{margin-top:auto;padding-top:6px;font:600 13px var(--f-body);color:var(--accent)}
.card.done .num::after{content:" · done ✓"}
.howto{max-width:72ch;margin:22px 0 0;padding:16px 20px;background:var(--paper);border:1px solid var(--line);border-radius:10px}
.howto ol{margin:6px 0 0}
"""

SCRIPT = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>
(function(){
  if (window.hljs) document.querySelectorAll(".code pre code").forEach(function(el){ hljs.highlightElement(el); });
  document.querySelectorAll(".code").forEach(function(box){
    var btn = box.querySelector(".copy"), code = box.querySelector("code");
    if (!btn || !code) return;
    btn.addEventListener("click", function(){
      function ok(){ btn.textContent = "Copied"; setTimeout(function(){ btn.textContent = "Copy"; }, 1400); }
      function fb(){ var r = document.createRange(); r.selectNodeContents(code); var s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = "Selected"; }
      try { navigator.clipboard.writeText(code.innerText).then(ok, fb); } catch (e) { fb(); }
    });
  });
  var KEY = "agent-school-done", done = {}, TOTAL = +document.body.dataset.total || 1;
  try { done = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { done = {}; }
  function save(){ try { localStorage.setItem(KEY, JSON.stringify(done)); } catch (e) {} }
  function render(){
    var n = Object.keys(done).filter(function(k){ return done[k]; }).length;
    document.querySelectorAll(".done-btn").forEach(function(b){
      var on = !!done[b.dataset.done];
      b.setAttribute("aria-pressed", on ? "true" : "false"); b.textContent = on ? "Done ✓" : "Mark done";
    });
    document.querySelectorAll("[data-mod-link]").forEach(function(a){ a.classList.toggle("done", !!done[a.dataset.modLink]); });
    var l = document.getElementById("progLabel"), bar = document.getElementById("progBar");
    if (l) l.textContent = Math.min(n, TOTAL) + " of " + TOTAL + " modules done";
    if (bar) bar.style.width = (100 * Math.min(n, TOTAL) / TOTAL) + "%";
  }
  document.querySelectorAll(".done-btn").forEach(function(b){
    b.addEventListener("click", function(){ done[b.dataset.done] = !done[b.dataset.done]; save(); render(); });
  });
  render();
})();
</script>
"""


def read_lesson(mid):
    f = ROOT / "site/lessons" / f"{mid}.md"
    if not f.exists():
        return None
    text = f.read_text()
    meta = {}
    m = re.match(r"---\s*\n(.*?)\n---\s*\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]
    # Q/A pairs -> click-to-reveal
    def qa(mm):
        return (f'<details><summary>{html.escape(mm.group(1).strip())}</summary>'
                f'<p>{html.escape(mm.group(2).strip())}</p></details>\n')
    md = lambda s: re.sub(r"<table>", '<div class="tbl"><table>',
                          markdown.markdown(re.sub(r"(?m)^  (?=[-*] )", "    ", s), extensions=["tables", "fenced_code", "md_in_html"])).replace("</table>", "</table></div>")
    parts = re.split(r"(?m)^## ", text)
    slides, tail = [], []
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        low = head.lower().strip()
        if low.startswith("check your understanding"):
            body = re.sub(r"\*\*Q:\s*(.+?)\*\*\s*\n\s*A:\s*(.+?)(?=\n\s*\n|\n\*\*Q:|\Z)", qa, body, flags=re.S)
            tail.append(f'<h2>{html.escape(head.strip())}</h2><div class="check">{md(body)}</div>')
        elif low.startswith("try it in colab"):
            tail.append(f'<h2>{html.escape(head.strip())}</h2>{md(body)}')
        else:
            slides.append((head.strip(), md(body)))
    n = len(slides)
    deck = "".join(
        f'<div class="slide{" objective" if h.lower().startswith("learning objective") else ""}">'
        f'<div class="slide-top"><h3>{html.escape(h)}</h3><span class="slide-no">{k}/{n}</span></div>{b}</div>'
        for k, (h, b) in enumerate(slides, 1))
    src = ""
    if meta.get("source"):
        link = (f' · <a href="{html.escape(meta["slides"])}" target="_blank" rel="noopener">'
                f'{"original slides (PDF)" if "cs329z/slides" in meta.get("slides", "") else "main paper"}</a>') if meta.get("slides") else ""
        src = f'<p class="deck-src">Based on {html.escape(meta["source"])}{link}</p>'
    body_html = md(parts[0]) + src + f'<div class="deck">{deck}</div>' + "".join(tail)
    return meta, body_html


def strip_tags(x):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip()


def build():
    make_page.build()                                   # refreshes site/artifact.html with every section
    s = (ROOT / "site/artifact.html").read_text()
    s = s.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    css = s[s.index("<style>", s.index("<title>")):s.index("</style>", s.index("--bg:")) + 8]
    pre_head = s[:s.index("<title>")]
    links = re.findall(r'<link[^>]+>', pre_head + s[:s.index("<style>", s.index("<title>"))])
    hero = re.search(r'<header class="hero">.*?</header>', s, re.S).group(0)
    rail = re.search(r'<nav class="rail".*?</nav>', s, re.S).group(0)
    sections = {m.group(1): m.group(0) for m in re.finditer(r'<section class="mod" id="([\w-]+)".*?</section>', s, re.S)}
    footer = re.search(r"<footer>.*?</footer>", s, re.S)
    footer = footer.group(0) if footer else ""
    missing = [i for i in ORDER if i not in sections]
    assert not missing, missing

    def info(mid):
        sec = sections[mid]
        title = strip_tags(re.search(r"<h2>(.*?)</h2>", sec, re.S).group(1))
        num = re.search(r'<div class="mod-num">(.*?)</div>', sec, re.S)
        num = strip_tags(num.group(1)) if num else ("Glossary" if mid == "glossary" else "")
        les = read_lesson(mid)
        big = re.search(r'<p class="big">(.*?)</p>', sec, re.S)
        summary = (les[0].get("summary") if les else None) or (strip_tags(big.group(1)) if big else "")
        if len(summary) > 190:
            summary = summary[:185].rsplit(" ", 1)[0] + "…"
        minutes = les[0].get("minutes") if les else None
        return {"title": title, "num": num or mid, "summary": summary, "minutes": minutes, "lesson": les}

    INFO = {m: info(m) for m in ORDER}
    total = len([m for m in ORDER if m not in ("glossary", "part2-lessons")])

    def fix_links(x, here_is_sub):
        pre = "" if here_is_sub else "m/"
        x = re.sub(r'href="#([\w-]+)"', lambda mm: f'href="{pre}{mm.group(1)}.html"' if mm.group(1) in INFO else mm.group(0), x)
        x = re.sub(r'(<a href="https://colab[^"]+")(?![^>]*target=)', r'\1 target="_blank" rel="noopener"', x)
        return x

    def rail_for(here, sub):
        r = fix_links(rail, sub)
        r = r.replace('<p class="brand">Agent Engineering School</p>',
                      f'<p class="brand"><a href="{"../" if sub else ""}index.html" style="color:inherit;text-decoration:none">Agent Engineering School</a></p>')
        r = re.sub(r'<a href="([\w-]+)\.html">', lambda mm: f'<a href="{mm.group(1)}.html" data-mod-link="{mm.group(1)}"'
                   + (' aria-current="page" style="background:var(--accent-soft);font-weight:600"' if mm.group(1) == here else "") + ">", r)
        r = re.sub(r'<a href="m/([\w-]+)\.html">', lambda mm: f'<a href="m/{mm.group(1)}.html" data-mod-link="{mm.group(1)}">', r)
        r = r.replace('<div class="progress" id="progLabel">0 modules done</div>', '<div class="progress" id="progLabel"></div>')
        return r

    def page(title, body, sub, needs_mermaid=False, desc=""):
        mer = ('<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"; '
               'mermaid.initialize({startOnLoad: true, theme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "default"});</script>\n') if needs_mermaid else ""
        return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                f"<title>{html.escape(title)}</title>\n<meta name=\"description\" content=\"{html.escape(desc)}\">\n"
                + "\n".join(links) + "\n" + css.replace("</style>", PAGE_CSS + "</style>") + "\n</head>\n"
                f"<body data-total=\"{total}\">\n" + body + SCRIPT + mer + "</body>\n</html>\n")

    # ---------- home page
    cards = []
    for part, groups in GROUPS:
        cards.append(f'<h2 class="part-title">{html.escape(part)}</h2>')
        for gname, ids in groups:
            cs = []
            for mid in ids:
                i = INFO[mid]
                mins = f' · {i["minutes"]} min read' if i["minutes"] else ""
                cs.append(f'<a class="card" href="m/{mid}.html" data-mod-link="{mid}"><span class="num">{html.escape(i["num"])}{mins}</span>'
                          f'<span class="t">{html.escape(i["title"])}</span><span class="s">{html.escape(i["summary"])}</span>'
                          f'<span class="go">Start lesson →</span></a>')
            head = f"<h3>{html.escape(gname)}</h3>" if gname else ""
            cards.append(f'<div class="cards-group">{head}<div class="cards">{"".join(cs)}</div></div>')
    howto = ('<div class="howto"><b>How to use this course</b><ol>'
             '<li>Pick a card below. Each lesson is a short set of slides: one idea per slide, short bullets, side-by-side examples.</li>'
             '<li>Answer the three questions at the end of the lesson to check you got it.</li>'
             '<li>Open the module in Colab and run the code. It runs a free open model on Colab\'s GPU, with no API keys.</li>'
             '<li>Want more detail? Each page has a "Go deeper" section with key terms, papers and the full code.</li></ol></div>')
    hero_home = re.sub(r"<p><b>For one-click Colab, use the team site:</b>.*?</p>", "", hero, flags=re.S)
    hero_home = re.sub(r'<p class="eyebrow">.*?</p>', '<p class="eyebrow">Self-paced course · 34 short lessons · every line of code runs in Colab</p>', hero_home, flags=re.S)
    hero_home = re.sub(r'<p class="lede">.*?</p>', '<p class="lede">Slide-style lessons on building AI agents that run on your own computers, drawn from Stanford CS329Z (Engineering AI Agents) and CS329A (Self-Improving AI Agents). Each lesson links to the original lecture slides and a runnable Colab notebook.</p>', hero_home, flags=re.S)
    hero_home = re.sub(r'<div class="meta">.*?</div>', '', hero_home, flags=re.S)
    hero_home = fix_links(hero_home, False).replace("</header>", howto + "</header>")
    home_body = (f'<div class="wrap">{rail_for(None, False)}<main>{hero_home}{"".join(cards)}'
                 f'<footer>{strip_tags(footer)}</footer></main></div>')
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/index.html").write_text(page("Agent Engineering School", home_body, False,
                                                desc="Local AI Lab: a hands-on course on building AI agents that run on a local model, with Colab notebooks."))

    # ---------- one page per module
    (ROOT / "docs/m").mkdir(exist_ok=True)
    for k, mid in enumerate(ORDER):
        i = INFO[mid]
        sec = sections[mid]
        mh = re.search(r'<div class="mod-head">.*?</h2></div>(?:<span class="actions">.*?</span>)?</div>', sec, re.S)
        head_html = mh.group(0) if mh else ""
        inner = sec.replace(head_html, "", 1)
        inner = re.sub(r'^<section[^>]*>', "", inner).rsplit("</section>", 1)[0]
        inner = fix_links(inner, True)
        head_html = fix_links(head_html, True).replace('<div class="mod-head">', '<div class="mod-head" style="margin-bottom:6px">')
        if not head_html:
            head_html = f'<div class="mod-head"><div><div class="mod-num">{html.escape(i["num"])}</div><h2>{html.escape(i["title"])}</h2></div></div>'
        les = i["lesson"]
        if les:
            meta = f'<div class="lesson-meta">{("<span>" + i["minutes"] + " min read</span>") if i["minutes"] else ""}<span>Slides first, then the deeper material</span></div>'
            body = (f'{head_html}{meta}<p class="lesson-summary">{html.escape(i["summary"])}</p>'
                    f'<article class="lesson">{les[1]}</article>'
                    f'<div class="deeper"><h2 class="deeper-title">Go deeper</h2><p class="deeper-sub">Key terms, papers, evidence, the full notebook code and worked solutions.</p>{inner}</div>')
        else:
            body = head_html + inner
        prev_l = f'<a class="prev" href="{ORDER[k-1]}.html"><small>← Previous</small>{html.escape(INFO[ORDER[k-1]]["title"])}</a>' if k > 0 else ""
        next_l = f'<a class="next" href="{ORDER[k+1]}.html"><small>Next →</small>{html.escape(INFO[ORDER[k+1]]["title"])}</a>' if k + 1 < len(ORDER) else ""
        top = (f'<div class="topbar"><a href="../index.html">← All modules</a>'
               f'<span class="crumbs">{html.escape(i["num"])}</span></div>')
        main = f'<main>{top}<section class="mod" id="{mid}" style="border-bottom:0;padding-top:0">{body}</section><nav class="pager">{prev_l}{next_l}</nav></main>'
        full = f'<div class="wrap">{rail_for(mid, True)}{main}</div>'
        full = full.replace('href="m/', 'href="')
        (ROOT / f"docs/m/{mid}.html").write_text(page(f'{i["title"]} · Agent Engineering School', full, True,
                                                      needs_mermaid="mermaid" in body, desc=i["summary"]))
    have = sum(1 for m in ORDER if INFO[m]["lesson"])
    print(f"home + {len(ORDER)} module pages; {have} with slide lessons")


if __name__ == "__main__":
    build()
