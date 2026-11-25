"""Problem 1 (priority): classify one email as urgent / normal / ignore.

One model call. The prompt holds the rubric; the reply ends with two lines,
``REASON: ...`` and ``LABEL: <label>``. The parser reads the LABEL line first
and falls back to the last label word in the reply. No label word at all
gives "unknown". Nothing here raises.
"""

from __future__ import annotations

import re
from datetime import datetime

from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.types import Email, LMCallable

LABELS = ("urgent", "normal", "ignore")
BODY_CHARS = 3000  # long bodies are mostly quoted replies; the top carries the ask

SYSTEM = """You sort email for a busy employee. Label one email for the person it is addressed to, at the time it was sent.

Labels:
- urgent: BOTH (1) the email asks the recipient to do or decide something, or reports a live problem the recipient is responsible for (outage, failed payment, an escalation), AND (2) the action is needed the same day, the next working day, or by a stated deadline at most two working days away. Working days are Monday to Friday, counted from the email's date.
- normal: legitimate mail written for this recipient or a small working group, with no deadline in the next two working days: status updates, FYIs, questions, documents for review, scheduling for later dates, replies in an ongoing discussion, and personal notes from people the recipient knows.
- ignore: mail the recipient can leave unread: advertising and spam, newsletters and news digests, announcements to large employee lists (events, training, facilities), and automated notifications that ask nothing of this recipient.

Rules for close calls:
- Wording does not decide. "URGENT" in a subject that asks nothing of the recipient is not urgent. A calm "I need your signature by 3:00 PM today" is urgent.
- A request with no deadline, or a deadline more than two working days away, is normal.
- Mass mailings and ads are ignore even if they say "act now" or "important".
- An automated notice asking THIS recipient to act (e.g. an expense report awaiting their approval) is normal, or urgent if its deadline is within two working days.
- A short personal or social note from someone the recipient knows is normal, not ignore.

Reply with exactly two lines and nothing else:
REASON: <one sentence: who it is from, what it asks of the recipient, and any deadline>
LABEL: <urgent|normal|ignore>"""

_LABEL_LINE = re.compile(r"label\s*[:=]\s*\**\s*(urgent|normal|ignore)\b", re.IGNORECASE)
_LABEL_WORD = re.compile(r"\b(urgent|normal|ignore)\b", re.IGNORECASE)
_REASON_LINE = re.compile(r"reason\s*[:=]\s*(.+)", re.IGNORECASE)


def _when(date: str) -> str:
    """'2001-05-14T15:02:00Z' -> 'Monday 2001-05-14 15:02 UTC' (or the raw string)."""
    try:
        dt = datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%A %Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(date)


def render_email(email: Email, body_chars: int = BODY_CHARS) -> str:
    """The email as the model sees it."""
    body = (email.get("body") or "").strip()
    if len(body) > body_chars:
        body = body[:body_chars] + "\n[... rest of email cut ...]"
    to = email.get("to") or []
    to_line = ", ".join(to[:8]) + (f" and {len(to) - 8} more" if len(to) > 8 else "")
    return (
        f"From: {email.get('from') or '(unknown)'}\n"
        f"To: {to_line or '(none listed)'}\n"
        f"Sent: {_when(email.get('date', ''))}\n"
        f"Subject: {email.get('subject') or '(no subject)'}\n\n"
        f"{body or '(empty body)'}"
    )


def parse_priority(reply: str) -> dict:
    """Turn any model reply into {"category", "reason"}. Never raises."""
    text = reply if isinstance(reply, str) else ""
    m = _LABEL_LINE.search(text)
    if m:
        category = m.group(1).lower()
    else:
        words = _LABEL_WORD.findall(text)
        category = words[-1].lower() if words else "unknown"

    reason_match = _REASON_LINE.search(text)
    reason = reason_match.group(1).strip() if reason_match else ""
    if not reason:
        # Fall back to the reply itself (first non-empty line), then a stock line.
        reason = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")[:300]
    if not reason:
        reason = "The model gave no usable reply." if category == "unknown" else f"Labelled {category}."
    return {"category": category, "reason": reason}


def classify_priority(email: Email, lm: LMCallable) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render_email(email)},
    ]
    try:
        reply = lm(messages)
    except OutputTruncated as exc:  # a cut-off reply may still hold the label
        reply = exc.text or ""
    return parse_priority(reply)
