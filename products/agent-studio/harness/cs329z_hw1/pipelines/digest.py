"""Problem 2 (daily_digest): a morning summary of one day's email.

Code does the aggregation, the model only labels and writes:
1. classify every email with the priority pipeline, in parallel
2. drop the ignore pile (it never reaches the writer, so it cannot leak)
3. keep one item per thread (the most pressing, then the latest)
4. one writer call: urgent items first, each named by its company/project/site
5. if the writer went over 300 words, ask once for a shorter version
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.pipelines.priority import classify_priority
from cs329z_hw1.types import Email, LMCallable

MAX_WORDS = 300
TARGET_WORDS = 220  # aim well under the limit; models overshoot
MAX_WORKERS = 8
MAX_NORMAL_ITEMS = 40  # a busy day has ~280 emails; keep the writer prompt small
RANK = {"urgent": 0, "normal": 1, "unknown": 1, "ignore": 2}

WRITER = f"""You write a morning email digest for a busy employee. Use only the items given.

Format (plain text, no greeting, no sign-off):
URGENT
- one line per urgent item: what is needed, by when, from whom. ALWAYS name the specific company, project, site, deal or counterparty the email is about, using the exact name from the item (for example "Orlin Valley", not "a site").
OTHER
- short lines grouping the remaining items by theme; skip trivial ones.

Rules: at most {TARGET_WORDS} words in total. Every urgent item must appear. Do not invent anything. If there are no urgent items, write "URGENT\\n- Nothing urgent today." """


def _excerpt(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "..."


def _item_line(email: Email, label: dict, body_chars: int) -> str:
    sender = email.get("from") or "unknown sender"
    subject = email.get("subject") or "(no subject)"
    return (
        f"- From {sender} | Subject: {subject} | Why: {label['reason']} | "
        f"Text: {_excerpt(email.get('body', ''), body_chars)}"
    )


def _classify_all(emails: list[Email], lm: LMCallable) -> list[dict]:
    if len(emails) <= 1:
        return [classify_priority(e, lm) for e in emails]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return list(pool.map(lambda e: classify_priority(e, lm), emails))


def _one_per_thread(pairs: list[tuple[Email, dict]]) -> list[tuple[Email, dict]]:
    """Keep the most pressing email of each thread (latest on ties)."""
    best: dict[str, tuple[Email, dict]] = {}
    for email, label in pairs:
        key = email.get("thread_id") or email.get("id")
        cur = best.get(key)
        if cur is None:
            best[key] = (email, label)
            continue
        new_rank, cur_rank = RANK[label["category"]], RANK[cur[1]["category"]]
        if new_rank < cur_rank or (new_rank == cur_rank and email.get("date", "") >= cur[0].get("date", "")):
            best[key] = (email, label)
    return list(best.values())


def _write(lm: LMCallable, messages: list[dict]) -> str:
    try:
        reply = lm(messages)
    except OutputTruncated as exc:
        reply = exc.text or ""
    return reply if isinstance(reply, str) else ""


def build_digest(emails: list[Email], lm: LMCallable) -> str:
    if not emails:
        return "No emails today."

    labels = _classify_all(emails, lm)
    kept = [(e, l) for e, l in zip(emails, labels) if l["category"] != "ignore"]
    if not kept:
        return "URGENT\n- Nothing urgent today.\nOTHER\n- Only newsletters, ads and mass announcements today."

    kept = _one_per_thread(kept)
    urgent = [(e, l) for e, l in kept if l["category"] == "urgent"]
    normal = [(e, l) for e, l in kept if l["category"] != "urgent"][:MAX_NORMAL_ITEMS]

    parts = [f"URGENT ITEMS ({len(urgent)}):"]
    parts += [_item_line(e, l, 400) for e, l in urgent] or ["(none)"]
    parts += [f"\nOTHER ITEMS ({len(normal)}):"]
    parts += [_item_line(e, l, 160) for e, l in normal] or ["(none)"]
    messages = [
        {"role": "system", "content": WRITER},
        {"role": "user", "content": "\n".join(parts)},
    ]

    digest = _write(lm, messages)
    if len(digest.split()) > MAX_WORDS:
        retry = messages + [
            {"role": "assistant", "content": digest},
            {
                "role": "user",
                "content": f"That is {len(digest.split())} words. Rewrite it in at most {TARGET_WORDS - 40} words. "
                "Keep every urgent item and its company/project/site name; cut the OTHER section first.",
            },
        ]
        shorter = _write(lm, retry)
        if shorter.strip():
            digest = shorter
    return digest if digest.strip() else "Digest unavailable: the model returned no text."
