"""Bake-off: the same known-answer tasks through our harness on each open model.

    cd ../hw1 && uv run python ../agent-studio/bakeoff.py [--models a,b] [--slice N]

Tasks: HW1's 20 company-document questions and 9 terminal questions (+1
follow-up), each in a fresh session of the full Cardinal agent, auto mode,
16k context. Scored by the course's own answer matcher. Results are written
to agent-studio/bakeoff_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW1 = next((p for p in (HERE.parent / "harness", HERE.parent / "hw1") if (p / "cs329z_hw1").is_dir()), HERE.parent / "hw1")
sys.path.insert(0, str(HW1))
sys.path.insert(0, str(HERE))

from cs329z_hw1 import adapters  # noqa: E402
from cs329z_hw1.llm import LM  # noqa: E402
from cs329z_hw1.types import AgentConfig  # noqa: E402
from models import OPEN_MODELS, configure, register  # noqa: E402
from tests.helpers import contains_any  # noqa: E402

register()


def tasks(n: int | None):
    fx = HW1 / "tests" / "fixtures"
    docs = json.loads((fx / "doc_questions.json").read_text())
    term = json.loads((fx / "terminal_questions.json").read_text())
    out = [("doc", [(d["id"], d["question"], d["answers"])]) for d in docs]
    for t in term:
        turns = [(t["id"], t["question"], t["answers"])]
        if t.get("followup"):
            turns.append((t["id"] + "-followup", t["followup"]["question"], t["followup"]["answers"]))
        out.append(("terminal", turns))
    return out[:n] if n else out


def run_task(model: str, kind: str, turns):
    lm = LM(model, tag=f"bakeoff/{model}")
    aux = LM(model, tag=f"bakeoff/{model}/aux")
    session = adapters.run_agent_session(lm, None, AgentConfig(mode="auto", aux_lm=aux, context_budget=16_000))
    configure(session, model)
    rows = []
    for tid, question, answers in turns:
        t0 = time.time()
        try:
            result = session.send(question)
            text, status = result.text, result.status
        except Exception as exc:
            text, status = f"{type(exc).__name__}: {exc}", "error"
        rows.append({"id": tid, "kind": kind, "ok": contains_any(text, answers), "status": status,
                     "seconds": round(time.time() - t0, 1), "answer": text[:300]})
    events = [e.get("type") for e in session.transcript]
    for r in rows:
        r.update(nudges=events.count("nudge"), repairs=events.count("repaired"),
                 errors=events.count("error"), checks=events.count("check"))
    return rows, lm.cost_usd + aux.cost_usd


def bake(model: str, n: int | None):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda kt: run_task(model, *kt), tasks(n)))
    rows = [r for rs, _ in results for r in rs]
    cost = sum(c for _, c in results)
    summary = {
        "model": model,
        "label": OPEN_MODELS[model]["label"],
        "correct": sum(r["ok"] for r in rows),
        "total": len(rows),
        "doc": f"{sum(r['ok'] for r in rows if r['kind'] == 'doc')}/{sum(r['kind'] == 'doc' for r in rows)}",
        "terminal": f"{sum(r['ok'] for r in rows if r['kind'] == 'terminal')}/{sum(r['kind'] == 'terminal' for r in rows)}",
        "cost_usd": round(cost, 4),
        "minutes": round((time.time() - t0) / 60, 1),
        "format_errors": sum(r["errors"] for r in rows if r["id"][-9:] != "-followup"),
        "nudges": sum(r["nudges"] for r in rows if r["id"][-9:] != "-followup"),
        "repairs": sum(r["repairs"] for r in rows if r["id"][-9:] != "-followup"),
        "rows": rows,
    }
    print(f"{summary['label']:24s} {summary['correct']:>2}/{summary['total']}  docs {summary['doc']:>5}  "
          f"terminal {summary['terminal']:>5}  ${summary['cost_usd']:.4f}  {summary['minutes']} min  "
          f"fmt-errors {summary['format_errors']}  nudges {summary['nudges']}  repairs {summary['repairs']}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(OPEN_MODELS))
    ap.add_argument("--slice", type=int, default=None)
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        summaries = list(pool.map(lambda m: bake(m, args.slice), models))
    out = HERE / "bakeoff_results.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
