"""Hidden verifier. Runs AFTER the agent finishes, in a step the agent never sees.

Usage: python test.py <workspace_dir> <reward_file>
Reward = fraction of contracts whose rule list is exactly right (0.0 to 1.0).
Each check maps to one requirement in instruction.md.
"""
import json
import sys
from pathlib import Path

ws, reward_file = Path(sys.argv[1]), Path(sys.argv[2])
expected = json.loads((Path(__file__).parent / "expected.json").read_text())


def finish(score, why):
    reward_file.write_text(f"{score:.3f}")
    print(f"reward={score:.3f} ({why})")
    sys.exit(0)


report_path = ws / "report.json"
if not report_path.exists():                                   # requirement: write report.json
    finish(0.0, "report.json missing")
try:
    report = json.loads(report_path.read_text())
except json.JSONDecodeError:
    finish(0.0, "report.json is not valid JSON")
if not isinstance(report, dict):
    finish(0.0, "report.json must be an object")

rules = set(json.loads((ws / "playbook.json").read_text())["rules"]) if (ws / "playbook.json").exists() else set()
right = 0
for cid, want in expected.items():                             # requirement: every contract, exact rule names
    got = report.get(cid)
    if isinstance(got, list) and all(isinstance(x, str) for x in got) and sorted(set(got)) == want:
        right += 1
extra = set(report) - set(expected)                            # requirement: only the five contracts
if extra:
    finish(0.0, f"unexpected keys {sorted(extra)}")
finish(right / len(expected), f"{right}/{len(expected)} contracts exactly right")
