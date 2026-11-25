"""Build the Local AI Lab product page for Agent Studio (Product 1).

    python3 site/product_agent_studio.py
writes docs/products/agent-studio.html from products/agent-studio/studio/bakeoff_results.json.
Self-contained HTML (inline SVG graphics), same fonts and colours as the course site.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRODUCT = ROOT / "products" / "agent-studio"
OUT = ROOT / "docs" / "products" / "agent-studio.html"
REPO = "https://github.com/shivamkalakaar/local-ai-lab-agents"
SRC = f"{REPO}/tree/main/products/agent-studio"

MAKERS = {"openai": "OpenAI", "nvidia": "NVIDIA", "google": "Google", "mistralai": "Mistral"}


def esc(x) -> str:
    return html.escape(str(x))


# ----------------------------------------------------------------- graphics --

def architecture_svg() -> str:
    """The harness at a glance: request -> loop -> guardrails -> tools, with the model on the side."""
    def box(x, y, w, h, title, lines, kind="node"):
        rows = "".join(
            f'<text x="{x + 14}" y="{y + 44 + i * 17}" class="ar-small">{esc(t)}</text>' for i, t in enumerate(lines)
        )
        return (f'<g class="ar-{kind}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12"/>'
                f'<text x="{x + 14}" y="{y + 24}" class="ar-title">{esc(title)}</text>{rows}</g>')

    def arrow(x1, y1, x2, y2, label="", dy=-6):
        if x1 == x2:  # vertical: label to the right of the line
            mid = f'<text x="{x1 + 8}" y="{(y1 + y2) / 2 + 4}" class="ar-label">{esc(label)}</text>' if label else ""
        else:
            mid = f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 + dy}" class="ar-label" text-anchor="middle">{esc(label)}</text>' if label else ""
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ar-edge" marker-end="url(#ah)"/>{mid}'

    parts = [
        '<svg viewBox="0 0 980 470" class="arch" role="img" aria-labelledby="arch-t arch-d">',
        '<title id="arch-t">How the Agent Studio harness works</title>',
        '<desc id="arch-d">A request goes from the web UI to the agent loop, which keeps the context small, parses tool calls, '
        'repairs and nudges the model, checks answers, passes every action through guardrails, and calls tools. '
        'An open-weight model is called through one plain chat API.</desc>',
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" class="ar-head"/></marker></defs>',
        box(10, 40, 160, 120, "Studio UI", ["chat + live steps", "approve / deny", "sources, memory", "cost per call"], "ui"),
        box(250, 20, 440, 230, "Agent loop (our harness)", [], "loop"),
        box(270, 62, 196, 82, "Context", ["budget per call", "mask old results", "summary as last resort"]),
        box(476, 62, 196, 82, "Call protocol", ["one call per reply", "tolerant parser", "per-model call tag"]),
        box(270, 154, 196, 82, "Self-correction", ["nudge 'I'll check…'", "repair repeated calls", "errors fed back"]),
        box(476, 154, 196, 82, "Answer checker", ["facts vs. evidence", "code check first", "side model if unsure"]),
        box(760, 40, 200, 120, "Open model", ["gpt-oss · Nemotron", "Gemma · Mistral", "plain chat API", "no tool-calling API"], "model"),
        box(250, 290, 440, 70, "Guardrails", ["deny rules · copied-command check · approval · OS sandbox"], "guard"),
        box(10, 400, 950, 56, "Tools", ["search_docs · read_doc · search_emails · read_email · email_qa · daily_digest · run_terminal · remember · recall · ask_user"], "tools"),
        arrow(170, 100, 248, 100, "request"),
        arrow(690, 100, 758, 100, "messages"),
        arrow(758, 128, 692, 128, "reply"),
        arrow(470, 250, 470, 288, "every action"),
        arrow(470, 360, 470, 398, "allowed"),
        arrow(248, 140, 172, 140, "live steps", 16),
        "</svg>",
    ]
    return "".join(parts)


def bakeoff_svg(rows: list[dict]) -> str:
    """Horizontal bars: tasks solved per model, with cost and the harness fixes that ran."""
    rows = sorted(rows, key=lambda r: (-r["correct"], r["cost_usd"]))
    total = rows[0]["total"] if rows else 30
    w, left, right, bar_h, gap, top = 980, 230, 210, 34, 22, 46
    usable = w - left - right
    h = top + len(rows) * (bar_h + gap) + 40
    out = [f'<svg viewBox="0 0 {w} {h}" class="bake" role="img" aria-labelledby="bk-t bk-d">',
           '<title id="bk-t">Bake-off: tasks solved per open model</title>',
           '<desc id="bk-d">' + esc("; ".join(f"{r['label']}: {r['correct']} of {r['total']}, ${r['cost_usd']:.3f}" for r in rows)) + "</desc>"]
    for t in range(0, total + 1, 5):
        x = left + usable * t / total
        out.append(f'<line x1="{x}" y1="{top - 14}" x2="{x}" y2="{h - 30}" class="bk-grid"/>'
                   f'<text x="{x}" y="{top - 20}" class="bk-axis" text-anchor="middle">{t}</text>')
    for i, r in enumerate(rows):
        y = top + i * (bar_h + gap)
        bw = usable * r["correct"] / max(1, r["total"])
        best = " bk-best" if i == 0 else ""
        maker = MAKERS.get(r["model"].split("/")[0], "")
        out.append(f'<text x="{left - 14}" y="{y + 15}" class="bk-name" text-anchor="end">{esc(r["label"])}</text>'
                   f'<text x="{left - 14}" y="{y + 31}" class="bk-maker" text-anchor="end">{esc(maker)}</text>'
                   f'<rect x="{left}" y="{y}" width="{usable}" height="{bar_h}" rx="6" class="bk-track"/>'
                   f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="6" class="bk-bar{best}" style="--d:{i * 120}ms"/>'
                   f'<text x="{left + bw - 10:.1f}" y="{y + 22}" class="bk-val" text-anchor="end">{r["correct"]}/{r["total"]}</text>'
                   f'<text x="{w - right + 14}" y="{y + 15}" class="bk-cost">${r["cost_usd"]:.3f} total</text>'
                   f'<text x="{w - right + 14}" y="{y + 31}" class="bk-maker">docs {r["doc"]} · terminal {r["terminal"]}</text>')
    out.append(f'<text x="{left}" y="{h - 8}" class="bk-axis">tasks solved (of {total}) · same harness, same tasks, same answer checker</text></svg>')
    return "".join(out)


# --------------------------------------------------------------------- page --

CSS = """
:root{--bg:#f3f5f7;--paper:#fff;--ink:#17202a;--muted:#56626f;--line:#d9dee4;--accent:#0d6e6a;--accent-soft:#e1f0ef;--mark:#f6dc6b;--warn:#a4421d;
--code-bg:#0f1720;--code-fg:#dde5ec;--f-display:"Bricolage Grotesque","Avenir Next",system-ui,sans-serif;--f-body:"IBM Plex Sans",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0e1419;--paper:#151d24;--ink:#e4eaef;--muted:#97a4b0;--line:#28343e;--accent:#5cc2bb;--accent-soft:#16302f;--mark:#c9a92a;--warn:#f0a07a}}
*{box-sizing:border-box}html{background:var(--bg)}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--f-body)}
a{color:var(--accent)}.wrap{max-width:1040px;margin:0 auto;padding:28px 16px 80px}
.top{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:14px;margin-bottom:26px}.top a{text-decoration:none}
.eyebrow{font:600 12px var(--f-mono);letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin:0}
h1{font:800 clamp(34px,6vw,56px)/1.04 var(--f-display);letter-spacing:-.02em;margin:10px 0 12px}
h2{font:700 26px/1.2 var(--f-display);margin:54px 0 10px;padding-top:18px;border-top:1px solid var(--line)}
.lede{font-size:19px;color:var(--muted);max-width:62ch;margin:0}
.btns{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 0}.btn{font:600 14px var(--f-mono);padding:9px 14px;border-radius:9px;border:1px solid var(--line);background:var(--paper);color:var(--ink);text-decoration:none}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:30px 0 0}
.fact{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:14px 16px}.fact b{display:block;font:800 26px var(--f-display);color:var(--accent)}.fact span{color:var(--muted);font-size:14px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:18px;overflow-x:auto}
.shot{border-radius:14px;border:1px solid var(--line);width:100%;display:block;margin-top:28px;box-shadow:0 10px 40px rgba(0,0,0,.12)}
svg{width:100%;height:auto;display:block;font-family:var(--f-mono)}
.ar-node rect{fill:var(--paper);stroke:var(--line)}.ar-loop rect{fill:var(--accent-soft);stroke:var(--accent);stroke-dasharray:5 4}
.ar-ui rect,.ar-model rect{fill:var(--paper);stroke:var(--accent);stroke-width:1.5}.ar-guard rect{fill:color-mix(in srgb,var(--warn) 9%,var(--paper));stroke:var(--warn)}
.ar-tools rect{fill:var(--paper);stroke:var(--line)}.ar-title{font:700 14px var(--f-display);fill:var(--ink)}.ar-small{font-size:11.5px;fill:var(--muted)}
.ar-edge{stroke:var(--accent);stroke-width:1.6}.ar-head{fill:var(--accent)}.ar-label{font-size:11px;fill:var(--accent)}
.bk-grid{stroke:var(--line)}.bk-axis{font-size:11px;fill:var(--muted)}.bk-name{font:700 14px var(--f-display);fill:var(--ink)}.bk-maker{font-size:11px;fill:var(--muted)}
.bk-track{fill:color-mix(in srgb,var(--line) 45%,transparent)}.bk-bar{fill:color-mix(in srgb,var(--accent) 55%,var(--paper));transform-box:fill-box;transform-origin:left;animation:grow .9s cubic-bezier(.22,1,.36,1) both;animation-delay:var(--d)}
.bk-best{fill:var(--accent)}.bk-val{font:700 13px var(--f-mono);fill:#fff}.bk-cost{font:600 13px var(--f-mono);fill:var(--ink)}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}@media (prefers-reduced-motion:reduce){.bk-bar{animation:none}}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font:600 12px var(--f-mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
td.num{font-family:var(--f-mono);white-space:nowrap}.ok{color:var(--accent);font-weight:600}
.fix{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.fix .card h3{font:700 17px var(--f-display);margin:0 0 6px}.fix .card p{margin:0;color:var(--muted);font-size:15px}
pre{background:var(--code-bg);color:var(--code-fg);border-radius:12px;padding:16px 18px;overflow-x:auto;font:13.5px/1.6 var(--f-mono)}
.note{color:var(--muted);font-size:14px}code{font-family:var(--f-mono);font-size:.92em}
"""


def build() -> None:
    results_path = PRODUCT / "studio" / "bakeoff_results.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []
    rows = sorted(rows, key=lambda r: (-r["correct"], r["cost_usd"]))
    best = rows[0] if rows else None
    total_cost = sum(r["cost_usd"] for r in rows)
    shot = (OUT.parent / "img" / "agent-studio.png").exists()

    table = "".join(
        f'<tr><td><b>{esc(r["label"])}</b><br><span class="note">{esc(MAKERS.get(r["model"].split("/")[0], ""))} · <code>{esc(r["model"])}</code></span></td>'
        f'<td class="num{" ok" if r is best else ""}">{r["correct"]}/{r["total"]}</td><td class="num">{esc(r["doc"])}</td><td class="num">{esc(r["terminal"])}</td>'
        f'<td class="num">${r["cost_usd"]:.3f}</td><td class="num">{r["nudges"]}</td><td class="num">{r["repairs"]}</td><td class="num">{r["format_errors"]}</td></tr>'
        for r in rows
    )

    body = f"""
<div class="wrap">
  <div class="top"><a href="../index.html">← The Local AI Lab</a><a href="{SRC}">Source on GitHub ↗</a></div>
  <p class="eyebrow">Local AI Lab · Product 1</p>
  <h1>Agent Studio</h1>
  <p class="lede">An AI agent on <b>open-weight models</b>. We wrote the code around the model ourselves (people call it the harness): no agent framework and no tool-calling API, only a plain chat call. You can watch each step as it happens, on your own machine.</p>
  <div class="btns"><a class="btn primary" href="#quick-start">Run it locally →</a><a class="btn" href="{SRC}">View the code</a><a class="btn" href="#bake-off">See the bake-off</a></div>
  <div class="facts">
    <div class="fact"><b>{best["correct"] if best else "–"}/{best["total"] if best else "–"}</b><span>known-answer tasks solved by the best open model ({esc(best["label"]) if best else "–"})</span></div>
    <div class="fact"><b>4</b><span>open models from OpenAI, NVIDIA, Google and Mistral, same harness</span></div>
    <div class="fact"><b>${total_cost:.2f}</b><span>total cost of the whole four-model bake-off (120 tasks)</span></div>
    <div class="fact"><b>287/287</b><span>offline tests passing (Stanford CS329Z HW1 suite)</span></div>
  </div>
  {'<img class="shot" src="img/agent-studio.png" alt="Agent Studio: chat on the left, live trace of every model call and tool call on the right">' if shot else ''}
  <p class="note" style="margin-top:8px">Left: the conversation. Right: the live trace, with every model call (and how full its context is), every tool call and its result. Shown here on Gemma 4 31B.</p>

  <h2>How the harness works</h2>
  <p class="note">The model only ever sees text and writes text. Everything that makes it reliable lives in the harness: what goes into each call, how replies are parsed, how mistakes are caught, and what is allowed to run.</p>
  <div class="card">{architecture_svg()}</div>

  <h2>Every answer points to its source</h2>
  <p class="note">Click a source chip under an answer and the document opens in an editor-style viewer: the passages the agent found are marked in the margin, and the facts from the answer are highlighted in the text. Here the agent found the newer announcement that replaced the old 5-day rule.</p>
  {'<img class="shot" src="img/agent-studio-sources.png" alt="Source viewer: the PTO announcement with the cited line and the facts 10 days, 80 hours and June 30 highlighted">' if (OUT.parent / "img" / "agent-studio-sources.png").exists() else ''}

  <h2 id="bake-off">Bake-off: same harness, four open models</h2>
  <p class="note">Each model ran the same {best["total"] if best else 30} tasks through the full agent: 20 questions about the company documents (some policies were replaced by later announcements) and 10 computations over a 26,000-email archive using a sandboxed terminal. Answers were scored by the course's own answer matcher. No model-specific prompts, only one per-model setting (the call tag).</p>
  <div class="card">{bakeoff_svg(rows) if rows else "<p>Run studio/bakeoff.py to fill this in.</p>"}</div>
  <div class="card" style="margin-top:12px">
    <table><thead><tr><th>Model</th><th>Solved</th><th>Docs</th><th>Terminal</th><th>Cost</th><th>Nudges</th><th>Repairs</th><th>Format errors</th></tr></thead><tbody>{table}</tbody></table>
    <p class="note" style="margin:10px 0 0">Nudges, repairs and format errors count how often the harness had to step in. Prices from OpenRouter on 2026-11-25. Results vary a little from run to run.</p>
  </div>

  <h2>What the harness fixed along the way</h2>
  <div class="fix">
    <div class="card"><h3>A model that drops its own call tag</h3><p>Nemotron's provider treats <code>&lt;tool_call&gt;</code> as a special token and cuts the reply to <code>&lt;to</code>. The parser now also accepts <code>&lt;call&gt;</code>, set per model. Nemotron went from 0 to working.</p></div>
    <div class="card"><h3>Thinking out loud instead of acting</h3><p>gpt-oss sometimes answers "We need to search the docs." and stops. A reply that names a tool but makes no call now gets one short reminder with an example call.</p></div>
    <div class="card"><h3>The same call written three times</h3><p>Models often repeat one call with junk in between. A reply with different calls is still an error. The same call written twice now runs once, instead of failing the turn.</p></div>
    <div class="card"><h3>Answers that sound right</h3><p>Before an answer goes out, every number and id in it is checked against what the tools returned. Only unclear cases go to a second model. Unsupported details go back to the agent once.</p></div>
    <div class="card"><h3>Planted instructions</h3><p>Some documents tell AI agents to run commands. Results are marked as data, the user's request is repeated after each result, and a command copied from a document is blocked in auto mode: 0 of 7 planted commands ran.</p></div>
    <div class="card"><h3>Long conversations</h3><p>Every call fits a fixed budget. Old tool results become one-line stubs first; a summary is written only if that is not enough. The user's own messages are never cut.</p></div>
  </div>

  <h2 id="quick-start">Run it locally</h2>
  <p class="note">Needs macOS or Linux, <a href="https://docs.astral.sh/uv/">uv</a>, Node 18+, and an <a href="https://openrouter.ai/keys">OpenRouter key</a> (a typical question costs well under a cent; the launcher sets a $2 cap). The terminal tool runs commands in an OS sandbox: <code>sandbox-exec</code> on macOS, <code>bubblewrap</code> on Linux.</p>
<pre>git clone {REPO}.git
cd local-ai-lab-agents/products/agent-studio
export OPENROUTER_API_KEY=sk-or-...
./start.sh                      # opens http://localhost:3000
STUDIO_MODEL=google/gemma-4-31b-it ./start.sh   # pick a model</pre>

  <h2>Built on</h2>
  <p class="note">The harness began as the author's solution to Stanford CS329Z (Engineering AI Agents) Assignment 1, built on its MIT-licensed starter code by Michael Ryan and the CS329Z staff, and on patterns studied in the source of Pi, Codex CLI, Gemini CLI, OpenHands, mini-swe-agent and AgentDojo. The email archive is a slice of the public Enron corpus (Klimt and Yang, 2004); the company documents are fictional.</p>
</div>"""

    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>Agent Studio · Local AI Lab</title>\n"
            "<meta name=\"description\" content=\"Local AI Lab Product 1: an open-weight AI agent with a custom harness, with a four-model bake-off.\">\n"
            "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"><link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n"
            "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap\">\n"
            f"<style>{CSS}</style>\n</head>\n<body>{body}\n</body>\n</html>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} models)")


if __name__ == "__main__":
    build()
