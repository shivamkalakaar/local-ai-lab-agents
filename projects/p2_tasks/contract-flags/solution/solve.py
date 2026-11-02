"""Reference (oracle) solution. Must score 1.0 through the same verifier.

Usage: python solve.py <workspace_dir>
"""
import json
import re
import sys
from pathlib import Path

ws = Path(sys.argv[1])
pb = json.loads((ws / "playbook.json").read_text())
report = {}
for f in sorted(ws.glob("msa_*.txt")):
    t = f.read_text()
    broken = []
    cap = re.search(r"IN THE (?:[A-Z-]+ \()?(\d+)\)? MONTHS PRECEDING", t)
    if "BE LIMITED, AND EACH PARTY SHALL BE LIABLE" in t or not cap or int(cap.group(1)) > pb["max_cap_months"]:
        broken.append("liability_cap")
    law = re.search(r"laws of the State of ([A-Za-z ]+?),", t).group(1)
    if law not in pb["allowed_law"]:
        broken.append("governing_law")
    years = int(re.search(r"successive \w+ \((\d+)\) year", t).group(1))
    if years > pb["max_renewal_years"]:
        broken.append("auto_renewal")
    days = int(re.search(r"at least (?:[a-z-]+ \()?(\d+)\)? days", t).group(1))
    if days < pb["min_notice_days"]:
        broken.append("termination_notice")
    report[f.stem] = sorted(broken)
(ws / "report.json").write_text(json.dumps(report, indent=1))
