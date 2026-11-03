"""Turn scripts/src/*.py (percent-format cell files) into notebooks/*.ipynb.

Cell markers:  '# %% [markdown]' starts a markdown cell (lines start with '# '),
               '# %%' starts a code cell.
The shared setup cell is inserted after the first markdown cell. A line
'# EXTRA: pkg1 pkg2' anywhere in the file adds packages to that notebook's setup.
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
REPO = "shivamkalakaar/local-ai-lab-agents"

SETUP = '''# Local AI Lab setup: run this cell first (about 2-3 minutes on Colab, a few seconds locally)
import os, sys, subprocess
EXTRA = {extra}
if "google.colab" in sys.modules:
    if not os.path.exists("/content/local-ai-lab-agents"):
        subprocess.run(["git", "clone", "-q", "https://github.com/{repo}", "/content/local-ai-lab-agents"], check=True)
    else:
        subprocess.run(["git", "-C", "/content/local-ai-lab-agents", "pull", "-q"])   # get the latest fixes
    os.chdir("/content/local-ai-lab-agents")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", *EXTRA], check=True)
elif os.path.basename(os.getcwd()) in ("notebooks", "projects", "cs329a", "cs329a_projects"):
    os.chdir("..")
sys.path.insert(0, os.getcwd())
from lal.runtime import setup
MODEL = setup({models})'''


def colab_badge(path: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/{path}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def parse(src: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
        if text.strip():
            cells.append((kind, text))

    for line in src.splitlines():
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "md", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        elif line.startswith("# EXTRA:") or line.startswith("# MODELS:"):
            continue
        else:
            buf.append(line)
    flush()
    return cells


def build(src_path: Path, out_dir: Path):
    src = src_path.read_text()
    extra = re.search(r"^# EXTRA: (.+)$", src, re.M)
    models = re.search(r"^# MODELS: (.+)$", src, re.M)
    cells = parse(src)
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}}
    rel = f"{out_dir.name}/{src_path.stem}.ipynb"
    out = []
    for i, (kind, text) in enumerate(cells):
        if kind == "md":
            if i == 0:
                text = colab_badge(rel) + "\n\n" + text
            out.append(nbf.v4.new_markdown_cell(text))
        else:
            out.append(nbf.v4.new_code_cell(text))
        if i == 0:
            out.append(nbf.v4.new_code_cell(SETUP.format(
                repo=REPO, extra=repr(extra.group(1).split() if extra else []),
                models=repr(models.group(1).split()) if models else "")))
    nb.cells = out
    out_dir.mkdir(exist_ok=True)
    nbf.write(nb, out_dir / f"{src_path.stem}.ipynb")
    return rel


if __name__ == "__main__":
    for sub in ("notebooks", "projects", "cs329a", "cs329a_projects"):
        for p in sorted((ROOT / "scripts" / "src" / sub).glob("*.py")):
            print("built", build(p, ROOT / sub))
