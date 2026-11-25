"""Build the Local AI Lab home page (docs/index.html).

    python3 site/lab_home.py
Sections: what we build, products, team, school, try it, principles, supporters.
Same palette and fonts as local-ai.stanford.edu.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/shivamkalakaar/local-ai-lab-agents"

TEAM = [
    ("Anvita Gupta", "PhD Student, Computer Science", "img/team/anvita-gupta.jpg", "https://www.linkedin.com/in/avgupta/"),
    ("Shivam Kalkar", "MBA Student, Graduate School of Business", "img/team/shivam-kalkar.jpg", "https://www.linkedin.com/in/shivamkalkar/"),
    ("Swetha Srinivasan", "MBA Student, Graduate School of Business", "img/team/swetha_headshot.jpeg", "https://www.linkedin.com/in/swetha-srinivasan-b92353154/"),
]

PRINCIPLES = [
    ("Open Source AI Models", "Everything we build uses openly licensed models and is released as open source, open to inspect and modify."),
    ("Local Hardware", "Our tools run entirely on hardware you own, from laptops to community servers, with no cloud subscription and no data leaving the device."),
    ("Open Models That Keep Improving", "We build on open models and post-train them on your approved work, so agents get better over time."),
]

STACK = [
    ("Hardware", "We pick and set up the machines you own, from a laptop to a small server."),
    ("Open models", "Open-weight models, chosen for your task by testing several on the same work."),
    ("Agents", "Ready agents that run on our own harness, the code that drives the model. They look things up and check their answers."),
    ("Training", "Your approved corrections become training data, so the models improve over time."),
]

CSS = """
:root{--cardinal:#8c1515;--cardinal-dark:#6f0f0f;--ink:#16161a;--ink-2:#3a3a42;--muted:#6b6b76;--line:#e6e4df;--paper:#fff;--paper-2:#faf9f6;--sand:#f0ede4;
--serif:"Alegreya",Georgia,serif;--sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,SFMono-Regular,Menlo,monospace;
--shadow:0 1px 2px rgba(22,22,26,.04),0 8px 30px rgba(22,22,26,.06);--ease:cubic-bezier(.22,.61,.36,1);--nav-bg:rgba(250,249,246,.85)}
@media (prefers-color-scheme:dark){:root{--cardinal:#e8837b;--cardinal-dark:#f3a59f;--ink:#ecebe6;--ink-2:#c3c1ba;--muted:#94928c;--line:#2c2b30;--paper:#1c1c20;--paper-2:#141417;--sand:#28272c;--nav-bg:rgba(20,20,23,.85)}}
*{box-sizing:border-box}html{scroll-behavior:smooth;background:var(--paper-2)}body{margin:0;background:var(--paper-2);color:var(--ink);font:16px/1.65 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--cardinal)}img{max-width:100%;display:block}
.w{max-width:1140px;margin:0 auto;padding:0 16px}
nav.top{position:sticky;top:0;z-index:10;background:var(--nav-bg);backdrop-filter:saturate(1.4) blur(10px);border-bottom:1px solid var(--line)}
nav.top .w{display:flex;align-items:center;gap:18px;height:58px}
.brand{font:700 20px var(--serif);color:var(--ink);text-decoration:none;margin-right:auto}
.links{display:flex;gap:16px;font-size:14px;overflow-x:auto;scrollbar-width:none}.links a{color:var(--ink-2);text-decoration:none;white-space:nowrap}.links a:hover{color:var(--cardinal)}
.hero{padding:76px 0 56px;background-image:radial-gradient(var(--line) 1px,transparent 1px);background-size:22px 22px}
.eyebrow{font:600 12px var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--cardinal);margin:0}
h1{font:700 clamp(44px,8vw,84px)/1 var(--serif);letter-spacing:-.01em;margin:12px 0 6px}
.tag{font:italic 500 clamp(22px,3vw,30px) var(--serif);color:var(--ink-2);margin:0 0 18px}
.lede{font-size:19px;color:var(--ink-2);max-width:60ch;margin:0}
.btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:26px}
.btn{display:inline-block;padding:10px 16px;border-radius:10px;border:1px solid var(--line);background:var(--paper);color:var(--ink);text-decoration:none;font-weight:500;font-size:15px;transition:transform .2s var(--ease),box-shadow .2s var(--ease)}
.btn:hover{transform:translateY(-1px);box-shadow:var(--shadow)}.btn.p{background:var(--cardinal);border-color:var(--cardinal);color:#fff}
section{padding:64px 0;border-top:1px solid var(--line)}
h2{font:700 clamp(30px,4vw,40px)/1.1 var(--serif);margin:6px 0 10px}
.sub{color:var(--muted);max-width:62ch;margin:0 0 26px}
.grid{display:grid;gap:16px}.g2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.g3{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.g4{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:var(--shadow);transition:transform .25s var(--ease),box-shadow .25s var(--ease)}
a.card{display:block;color:inherit;text-decoration:none}a.card:hover{transform:translateY(-3px);box-shadow:0 4px 12px rgba(22,22,26,.06),0 24px 60px rgba(22,22,26,.1)}
.card h3{font:700 22px/1.2 var(--serif);margin:0 0 6px}.card p{margin:0;color:var(--ink-2);font-size:15px}
.num{font:600 12px var(--mono);color:var(--cardinal);letter-spacing:.06em}
.prod{display:grid;grid-template-columns:1.1fr 1fr;gap:0;padding:0;overflow:hidden}
.prod .txt{padding:26px}.prod .shot{background:#0a0d12;display:flex;align-items:center}.prod .shot img{width:100%}
.stats{display:flex;gap:22px;flex-wrap:wrap;margin:16px 0 18px}.stats div b{display:block;font:700 26px var(--serif);color:var(--cardinal)}.stats div span{font-size:13px;color:var(--muted)}
.soon{border-style:dashed;box-shadow:none;background:transparent;display:flex;flex-direction:column;justify-content:center}
.pill{display:inline-block;font:600 11px var(--sans);letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:999px;background:var(--sand);color:var(--ink-2);margin-bottom:10px}
.pill.live{background:color-mix(in srgb,var(--cardinal) 14%,transparent);color:var(--cardinal)}
.person{text-align:center}.person img{width:132px;height:132px;border-radius:50%;object-fit:cover;margin:0 auto 14px;border:3px solid var(--paper);box-shadow:var(--shadow)}
.person h3{font-size:21px}.person p{font-size:14px}.person .ln{font-size:13px;color:var(--cardinal);margin-top:8px}
.school{display:grid;grid-template-columns:1fr 1.1fr;gap:28px;align-items:center}
.collage{display:grid;grid-template-columns:1fr 1fr;gap:10px}.collage img{border-radius:10px;border:1px solid var(--line);aspect-ratio:16/9;object-fit:contain;background:#e9e9e9}
.collage img:first-child{grid-column:span 2}
.pr{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px}.pr .n{font:700 40px var(--serif);color:var(--cardinal);line-height:1}.pr h3{font:700 21px var(--serif);margin:8px 0 6px}.pr p{margin:0;color:var(--ink-2);font-size:15px}
.sup{display:flex;gap:16px;flex-wrap:wrap;align-items:stretch}.sup .card{display:flex;align-items:center;gap:14px;min-width:280px;flex:1}
.sup img{height:46px;width:auto}.gsb{font:700 18px var(--serif);line-height:1.2}.gsb span{display:block;font:500 13px var(--sans);color:var(--muted)}
footer{background:#16161a;color:#c3c1ba;padding:40px 0;font-size:14px}footer a{color:#ecebe6}footer .w{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
.reveal{opacity:0;transform:translateY(14px);transition:opacity .6s var(--ease),transform .6s var(--ease)}.reveal.in{opacity:1;transform:none}
@media (max-width:820px){.prod,.school{grid-template-columns:1fr}.links{display:none}}
@media (prefers-reduced-motion:reduce){.reveal{opacity:1;transform:none;transition:none}html{scroll-behavior:auto}}
"""

REVEAL_JS = """<script>
(()=>{const els=document.querySelectorAll('.reveal');if(!('IntersectionObserver' in window)){els.forEach(e=>e.classList.add('in'));return}
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target)}}),{threshold:.12});
els.forEach(e=>io.observe(e))})();
</script>"""


def esc(x) -> str:
    return html.escape(str(x))


def build() -> None:
    bake = ROOT / "products" / "agent-studio" / "studio" / "bakeoff_results.json"
    rows = sorted(json.loads(bake.read_text()), key=lambda r: (-r["correct"], r["cost_usd"])) if bake.exists() else []
    best = rows[0] if rows else None

    stack = "".join(f'<div class="card reveal"><div class="num">0{i + 1}</div><h3>{esc(t)}</h3><p>{esc(d)}</p></div>' for i, (t, d) in enumerate(STACK))
    team = "".join(
        f'<a class="card person reveal" href="{esc(link)}" target="_blank" rel="noopener"><img src="{esc(img)}" alt="{esc(name)}" loading="lazy">'
        f'<h3>{esc(name)}</h3><p>{esc(role)}</p><div class="ln">LinkedIn ↗</div></a>'
        for name, role, img, link in TEAM
    )
    principles = "".join(f'<div class="reveal"><div class="n">0{i + 1}</div><h3>{esc(t)}</h3><p>{esc(d)}</p></div>' for i, (t, d) in enumerate(PRINCIPLES))
    best_line = f"{best['correct']}/{best['total']}" if best else "–"
    best_name = esc(best["label"]) if best else ""

    body = f"""
<nav class="top"><div class="w">
  <a class="brand" href="#top">The Local AI Lab</a>
  <div class="links"><a href="#stack">What we build</a><a href="#products">Products</a><a href="#team">Team</a><a href="#school">School</a><a href="#try">Try it</a><a href="#principles">Principles</a><a href="#supporters">Supporters</a></div>
  <a class="btn" href="{REPO}" style="padding:6px 12px;font-size:14px">GitHub ↗</a>
</div></nav>

<header class="hero" id="top"><div class="w">
  <p class="eyebrow">Stanford University</p>
  <h1>The Local AI Lab</h1>
  <p class="tag">Own Your AI Stack</p>
  <p class="lede">A Stanford student lab building the full stack to run AI on your own computers: hardware, tuned open models, ready agents, and training that keeps them improving. Strong AI for the private work that cloud AI can't touch.</p>
  <div class="btns"><a class="btn p" href="products/agent-studio.html">See Product 1: Agent Studio</a><a class="btn" href="school/">Start learning</a><a class="btn" href="#try">Try it</a></div>
</div></header>

<section id="stack"><div class="w">
  <p class="eyebrow">What we build</p><h2>Every layer, on your hardware</h2>
  <p class="sub">We set up and run each layer of the AI stack on machines you own.</p>
  <div class="grid g4">{stack}</div>
</div></section>

<section id="products"><div class="w">
  <p class="eyebrow">Products</p><h2>What we've shipped</h2>
  <p class="sub">We number products as we launch them. Each one is open source and comes with a page that shows how well it works.</p>
  <div class="card prod reveal">
    <div class="txt">
      <span class="pill live">Product 1 · live</span>
      <h3 style="font-size:30px">Agent Studio</h3>
      <p>An AI agent on open-weight models with a harness we wrote ourselves. You ask about company policies or a 26,000-email archive and watch each step as it runs. Every answer points to its source.</p>
      <div class="stats"><div><b>{best_line}</b><span>tasks solved by {best_name}</span></div><div><b>4</b><span>open models tested</span></div><div><b>287</b><span>tests passing</span></div></div>
      <div class="btns" style="margin-top:0"><a class="btn p" href="products/agent-studio.html">Product page</a><a class="btn" href="{REPO}/tree/main/products/agent-studio">Code</a></div>
    </div>
    <div class="shot"><img src="products/img/agent-studio.png" alt="Agent Studio: the chat on the left and every step the agent takes on the right" loading="lazy"></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card soon reveal"><span class="pill">Product 2</span><h3>Coming next</h3><p>We'll add it here when it launches.</p></div>
    <div class="card soon reveal"><span class="pill">Product 3</span><h3>Later</h3><p>Follow the repo on GitHub to see new products first.</p></div>
  </div>
</div></section>

<section id="team"><div class="w">
  <p class="eyebrow">People</p><h2>The team</h2>
  <div class="grid g3">{team}</div>
</div></section>

<section id="school"><div class="w school">
  <div class="reveal">
    <p class="eyebrow">Agent Engineering School</p><h2>Learn to build agents that run on your own computer</h2>
    <p class="sub">A free course in Colab notebooks that follows Stanford's CS329Z and CS329A. Each lesson is a short set of slides, then code you can run on a free GPU.</p>
    <div class="stats"><div><b>16</b><span>modules, from a first model call to a contract-review agent</span></div><div><b>2</b><span>build-it-yourself projects</span></div><div><b>11</b><span>lessons on self-improving agents</span></div><div><b>5</b><span>past student projects, rebuilt small</span></div></div>
    <div class="btns" style="margin-top:0"><a class="btn p" href="school/">Start learning</a><a class="btn" href="m/m1.html">Lesson 1: what an agent is</a></div>
  </div>
  <div class="collage reveal">
    <img src="img/lectures/agent-key-components.jpg" alt="Lecture slide: the key parts of an agent" loading="lazy">
    <img src="img/lectures/rag.jpg" alt="Lecture slide: retrieval-augmented generation" loading="lazy">
    <img src="img/lectures/multi-head-attention.jpg" alt="Lecture slide: multi-head attention" loading="lazy">
  </div>
</div></section>

<section id="try"><div class="w">
  <p class="eyebrow">Try it</p><h2>Things you can try today</h2>
  <div class="grid g3" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">
    <a class="card reveal" href="products/agent-studio.html"><span class="pill">Coming soon</span><h3>Live demo</h3><p>Ask Agent Studio a question in your browser, with nothing to install. Until then, run it on your laptop with one command.</p></a>
    <a class="card reveal" href="school/"><span class="pill">Free</span><h3>Colab notebooks</h3><p>16 notebooks and 2 projects that run an open model on Colab's free GPU. No API keys.</p></a>
    <a class="card reveal" href="products/agent-studio.html#bake-off"><span class="pill">Results</span><h3>Model bake-off</h3><p>Four open models on the same harness and the same 30 tasks, with the cost of each.</p></a>
    <a class="card reveal" href="research/harness.html"><span class="pill">Read</span><h3>How good harnesses work</h3><p>What we learned from reading the code of Pi, Codex CLI, Gemini CLI, OpenHands and more.</p></a>
    <a class="card reveal" href="research/playbook.html"><span class="pill">Read</span><h3>12 rules for building agents</h3><p>The lessons from Stanford CS329Z, in plain words, with the evidence behind each one.</p></a>
  </div>
</div></section>

<section id="principles"><div class="w">
  <p class="eyebrow">Principles</p><h2>Our Principles</h2>
  <div class="pr">{principles}</div>
</div></section>

<section id="supporters"><div class="w">
  <p class="eyebrow">Partners</p><h2>Supported by</h2>
  <div class="sup">
    <a class="card reveal" href="https://knight-hennessy.stanford.edu/" target="_blank" rel="noopener"><img src="img/partners/knight-hennessy.png" alt="Knight-Hennessy Scholars, Stanford University"></a>
    <a class="card reveal" href="https://www.gsb.stanford.edu/experience/learning/entrepreneurship/courses/startup-garage" target="_blank" rel="noopener"><div class="gsb">Startup Garage<span>Stanford Graduate School of Business</span></div></a>
  </div>
</div></section>

<footer><div class="w">
  <div><b style="font:700 18px var(--serif);color:#ecebe6">The Local AI Lab</b><br>Open models on hardware you own.</div>
  <div><a href="products/agent-studio.html">Agent Studio</a> · <a href="school/">School</a> · <a href="{REPO}">GitHub</a></div>
  <div>© 2026 The Local AI Lab · Stanford University</div>
</div></footer>
{REVEAL_JS}"""

    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>The Local AI Lab · Own Your AI Stack</title>\n"
            "<meta name=\"description\" content=\"A Stanford student lab building the full stack to run AI on your own computers: hardware, open models, agents and training.\">\n"
            "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"><link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n"
            "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Alegreya:ital,wght@0,400;0,500;0,700;1,400;1,500&family=Inter:wght@400;500;600&display=swap\">\n"
            f"<style>{CSS}</style>\n</head>\n<body>{body}\n</body>\n</html>\n")
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
