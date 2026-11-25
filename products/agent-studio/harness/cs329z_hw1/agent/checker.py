"""Answer checker (class rule 9: one agent plus a separate checker).

Before a final answer goes to the user:
1. Code check (free): every specific detail in the answer (numbers, dates,
   email ids, addresses) must appear in this conversation's evidence (tool
   results, user messages, saved memory). Most answers pass here.
2. Only for details the code check cannot find, the side model judges
   whether they are supported or follow from the evidence (e.g. a sum).
If something is unsupported, the agent gets the answer back once with the
list, to verify with a tool or remove it.
"""

from __future__ import annotations

import re

from cs329z_hw1.llm import OutputTruncated

_DETAIL = re.compile(
    r"em-\d+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|\$?\d[\d,]*(?:\.\d+)?%?"
)
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
EVIDENCE_CHARS = 24000

CHECK_SYSTEM = """You check an assistant's answer against the evidence it had (tool results and the user's messages).
For each listed detail, decide if the evidence states it or it follows directly from the evidence (for example a simple sum or count of listed items).
Reply with exactly one line:
VERDICT: SUPPORTED
or
VERDICT: UNSUPPORTED: <the unsupported details, comma-separated>"""


def _norm_num(tok: str) -> str:
    return tok.replace(",", "").replace("$", "").rstrip("%").rstrip(".")


def details(answer: str) -> list[str]:
    found = []
    for m in _DETAIL.finditer(answer or ""):
        tok = m.group(0).rstrip(".,")
        digits = re.sub(r"\D", "", tok)
        if tok[0].isdigit() or tok[0] == "$":
            if len(digits) < 2:  # single digits ("step 1", "3 items") are too common to check
                continue
            if re.fullmatch(r"(19|20)\d\d", digits) and len(tok) == 4:  # bare years
                continue
        found.append(tok)
    return list(dict.fromkeys(found))


def missing_details(answer: str, evidence: str) -> list[str]:
    ev = evidence.lower()
    ev_nums = set(_norm_num(t) for t in re.findall(r"\$?\d[\d,]*(?:\.\d+)?", evidence))
    missing = []
    for tok in details(answer):
        low = tok.lower()
        if low in ev:
            continue
        if any(c.isdigit() for c in tok) and _norm_num(tok) in ev_nums:
            continue
        missing.append(tok)
    return missing


def side_check(aux_lm, answer: str, missing: list[str], evidence: str) -> list[str]:
    """Ask the side model about the details the code check could not find.
    Returns the details it calls unsupported ([] if it says supported or its
    reply cannot be read: the checker never blocks on its own failure)."""
    ev = evidence if len(evidence) <= EVIDENCE_CHARS else evidence[-EVIDENCE_CHARS:]
    messages = [
        {"role": "system", "content": CHECK_SYSTEM},
        {"role": "user", "content": f"Evidence:\n{ev}\n\nAnswer:\n{answer}\n\nDetails to check: {', '.join(missing)}"},
    ]
    try:
        reply = aux_lm(messages)
    except OutputTruncated as exc:
        reply = exc.text or ""
    m = re.search(r"VERDICT:\s*UNSUPPORTED\s*:?\s*(.*)", reply or "", re.IGNORECASE)
    if not m:
        return []
    listed = [x.strip() for x in m.group(1).split(",") if x.strip()]
    return listed or missing


def feedback(unsupported: list[str]) -> str:
    return ("[Checker] Before this answer goes to the user: these details do not appear in any tool result "
            f"or user message in this conversation: {', '.join(unsupported)}. Verify them with a tool, or "
            "remove them and say what you could not confirm. Then give the final answer.")
