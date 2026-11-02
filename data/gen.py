"""Generate synthetic service agreements with known problems.

Every contract starts from a clean template. Some get "faults" injected on purpose
(SWE-smith idea): because we inject them, we know the right answer for grading.
All parties and numbers are made up.

Run:  python data/gen.py        (writes data/contracts/*.txt, data/labels.jsonl, data/playbook.json)
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 2026

PLAYBOOK = {
    "name": "Sample firm playbook for vendor service agreements",
    "rules": {
        "liability_cap": "The cap must be stated and must not exceed fees paid in the prior 12 months.",
        "governing_law": "Governing law must be California or Delaware.",
        "auto_renewal": "Automatic renewal terms must be one year or shorter.",
        "termination_notice": "Notice to terminate or not renew must be at least 30 days.",
    },
    "allowed_law": ["California", "Delaware"],
    "max_cap_months": 12,
    "max_renewal_years": 1,
    "min_notice_days": 30,
}

CUSTOMERS = ["Northwind Legal LLP", "Bayview Health Partners", "Granite Ridge Capital", "Harbor Point Logistics",
             "Summit Peak Manufacturing", "Redwood Analytics Inc.", "Lakeshore Insurance Co.", "Cedar & Pine Accounting"]
VENDORS = ["Acme Cloud Services, Inc.", "Brightline Data Corp.", "Orbit Software LLC", "Keystone IT Solutions",
           "Pinnacle Records Management", "Vector Systems Ltd.", "Atlas Document Tech", "Meridian Support Group"]
WORDS = {1: "one", 2: "two", 3: "three", 5: "five", 24: "twenty-four", 36: "thirty-six"}


def num(n):
    return f"{WORDS[n]} ({n})" if n in WORDS else str(n)


def render(p):
    cap = ("IN NO EVENT SHALL EITHER PARTY'S TOTAL LIABILITY UNDER THIS AGREEMENT BE LIMITED, AND EACH PARTY "
           "SHALL BE LIABLE FOR ALL DAMAGES ARISING HEREUNDER." if p["cap_months"] is None else
           f"EACH PARTY'S TOTAL LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE FEES PAID BY CUSTOMER "
           f"IN THE {num(p['cap_months']).upper()} MONTHS PRECEDING THE CLAIM.")
    clauses = [
        ("1", "Services", f"Vendor shall provide the document management and support services described in "
                          f"Exhibit A (the \"Services\") to Customer."),
        ("2", "Fees", f"Customer shall pay Vendor ${p['fee']:,} per month, invoiced monthly and due within thirty (30) days."),
        ("3", "Term", f"This Agreement begins on {p['start']} and continues for an initial term of "
                      f"{num(p['term_months'])} months."),
        ("4", "Renewal", f"After the initial term, this Agreement shall automatically renew for successive "
                         f"{num(p['renew_years'])} year periods unless either party gives at least "
                         f"{num(p['notice_days'])} days' prior written notice of non-renewal."),
        ("5", "Confidentiality", "Each party shall keep the other party's Confidential Information secret and use it "
                                 "only to perform this Agreement."),
        ("6", "Data Protection", "Vendor shall process Customer data only on Customer's documented instructions and "
                                 "shall notify Customer of any security incident within seventy-two (72) hours."),
        ("7", "Limitation of Liability", cap),
        ("8", "Indemnification", "Vendor shall defend and indemnify Customer against third-party claims that the "
                                 "Services infringe any intellectual property right."),
        ("9", "Termination", f"Either party may terminate this Agreement for material breach that remains uncured "
                             f"{num(p['cure_days'])} days after written notice."),
        ("10", "Governing Law", f"This Agreement is governed by the laws of the State of {p['law']}, without regard "
                                f"to its conflict of laws rules."),
        ("11", "Entire Agreement", "This Agreement, including its exhibits, is the entire agreement between the parties."),
    ]
    head = (f"MASTER SERVICES AGREEMENT\n\nThis Master Services Agreement (the \"Agreement\") is entered into by "
            f"{p['customer']} (\"Customer\") and {p['vendor']} (\"Vendor\").\n")
    body = "\n".join(f"\n{n}. {title.upper()}\n{n}.1 {text}" for n, title, text in clauses)
    return head + body + "\n"


def make(i, rng, faults):
    p = {"customer": rng.choice(CUSTOMERS), "vendor": rng.choice(VENDORS), "fee": rng.choice([4500, 8000, 12500, 20000]),
         "start": f"{rng.choice(['January', 'March', 'June', 'September'])} 1, 2027", "term_months": rng.choice([12, 24, 36]),
         "renew_years": 1, "notice_days": rng.choice([30, 60, 90]), "cap_months": 12, "law": rng.choice(["California", "Delaware"]),
         "cure_days": 30}
    issues = []
    if "liability_cap" in faults:
        p["cap_months"] = rng.choice([None, 24, 36])
        issues.append({"rule": "liability_cap", "clause": "7.1"})
    if "governing_law" in faults:
        p["law"] = rng.choice(["New York", "Texas", "Florida"])
        issues.append({"rule": "governing_law", "clause": "10.1"})
    if "auto_renewal" in faults:
        p["renew_years"] = rng.choice([2, 3, 5])
        issues.append({"rule": "auto_renewal", "clause": "4.1"})
    if "termination_notice" in faults:
        p["notice_days"] = rng.choice([5, 10, 15])
        issues.append({"rule": "termination_notice", "clause": "4.1"})
    text = render(p)
    cid = f"msa_{i:02d}"
    terms = {"governing_law": p["law"], "liability_cap_months": p["cap_months"], "renewal_years": p["renew_years"],
             "notice_days": p["notice_days"], "monthly_fee": p["fee"]}
    return cid, text, {"id": cid, "terms": terms, "issues": issues}


def main():
    rng = random.Random(SEED)
    rules = list(PLAYBOOK["rules"])
    plan = [[] for _ in range(6)] + [[r] for r in rules for _ in range(2)] + [rng.sample(rules, 2) for _ in range(6)]
    (HERE / "contracts").mkdir(exist_ok=True)
    labels = []
    for i, faults in enumerate(plan, 1):
        cid, text, label = make(i, rng, faults)
        (HERE / "contracts" / f"{cid}.txt").write_text(text)
        labels.append(label)
    (HERE / "labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
    (HERE / "playbook.json").write_text(json.dumps(PLAYBOOK, indent=2))
    print(f"wrote {len(labels)} contracts ({sum(1 for l in labels if not l['issues'])} clean)")


if __name__ == "__main__":
    main()
