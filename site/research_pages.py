"""Render the readable research pages from site/content/*.md into docs/research/.

    uv run --no-project --with markdown python site/research_pages.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    ("harness-research.md", "harness.html", "How good agent harnesses work",
     "What we learned from reading the source code of Pi, Codex CLI, Gemini CLI, OpenHands, mini-swe-agent and more."),
    ("playbook.md", "playbook.html", "12 rules for building agents",
     "The lessons of Stanford CS329Z (Engineering AI Agents), in plain words."),
]

CSS = """
:root{--cardinal:#8c1515;--ink:#16161a;--ink-2:#3a3a42;--muted:#6b6b76;--line:#e6e4df;--paper:#fff;--paper-2:#faf9f6;--sand:#f0ede4;
--serif:"Alegreya",Georgia,serif;--sans:"Inter",-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--cardinal:#e8837b;--ink:#ecebe6;--ink-2:#c3c1ba;--muted:#94928c;--line:#2c2b30;--paper:#1c1c20;--paper-2:#141417;--sand:#28272c}}
*{box-sizing:border-box}html{background:var(--paper-2)}body{margin:0;background:var(--paper-2);color:var(--ink);font:17px/1.7 var(--sans)}
a{color:var(--cardinal)}.w{max-width:820px;margin:0 auto;padding:28px 16px 80px}
.top{display:flex;justify-content:space-between;font-size:14px;margin-bottom:28px}.top a{text-decoration:none}
.eyebrow{font:600 12px var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--cardinal);margin:0}
h1{font:700 clamp(34px,6vw,52px)/1.05 var(--serif);margin:10px 0 8px}.lede{color:var(--ink-2);font-size:19px;margin:0 0 30px}
article h1{font-size:30px;margin-top:44px}article h2{font:700 26px/1.2 var(--serif);margin:40px 0 8px;padding-top:16px;border-top:1px solid var(--line)}
article h3{font:700 20px/1.3 var(--serif);margin:26px 0 6px}article h4{font:600 16px var(--sans);margin:18px 0 4px}
article p,article li{max-width:72ch}article li{margin:4px 0}
article code{font:14px var(--mono);background:var(--sand);padding:1px 5px;border-radius:5px}
article pre{background:#16161a;color:#ecebe6;border-radius:12px;padding:14px 16px;overflow-x:auto}article pre code{background:none;padding:0;color:inherit}
.tbl{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 12px var(--sans);text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}hr{border:0;border-top:1px solid var(--line);margin:30px 0}
"""


def build() -> None:
    out_dir = ROOT / "docs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst, title, lede in PAGES:
        text = (ROOT / "site" / "content" / src).read_text()
        text = re.sub(r"\A#\s+.*\n", "", text.lstrip())  # the page has its own title
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
        body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
        page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                f"<title>{html.escape(title)} · The Local AI Lab</title>\n<meta name=\"description\" content=\"{html.escape(lede)}\">\n"
                "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Alegreya:wght@700&family=Inter:wght@400;600&display=swap\">\n"
                f"<style>{CSS}</style>\n</head>\n<body><div class=\"w\">"
                "<div class=\"top\"><a href=\"../index.html\">← The Local AI Lab</a><a href=\"../school/\">Agent Engineering School →</a></div>"
                f"<p class=\"eyebrow\">Read</p><h1>{html.escape(title)}</h1><p class=\"lede\">{html.escape(lede)}</p>"
                f"<article>{body}</article></div></body>\n</html>\n")
        (out_dir / dst).write_text(page)
        print(f"wrote docs/research/{dst}")


if __name__ == "__main__":
    build()
