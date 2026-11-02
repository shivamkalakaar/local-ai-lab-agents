"""Load the synthetic contracts, their labels and the sample playbook."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"


def contract(cid: str) -> str:
    return (ROOT / "contracts" / f"{cid}.txt").read_text()


def labels() -> dict:
    return {l["id"]: l for l in map(json.loads, (ROOT / "labels.jsonl").read_text().splitlines())}


def playbook() -> dict:
    return json.loads((ROOT / "playbook.json").read_text())


def ids() -> list[str]:
    return sorted(labels())


def clauses(text: str) -> dict[str, str]:
    """Split a contract on its numbered sub-clauses: {'7.1': 'EACH PARTY...'}"""
    out = {}
    for m in re.finditer(r"^(\d+\.\d+) (.+)$", text, flags=re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


def sections(text: str) -> list[dict]:
    """Split on section headings, keeping the heading as context: [{'id','title','text'}]"""
    parts = re.split(r"\n(?=\d+\. [A-Z])", text)
    out = []
    for p in parts[1:]:
        head, _, body = p.partition("\n")
        num, _, title = head.partition(". ")
        out.append({"id": num, "title": title.title(), "text": body.strip()})
    return out
