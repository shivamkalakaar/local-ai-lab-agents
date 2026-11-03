"""Small task sets with exact answers, so notebooks can grade themselves.

Pure Python, no downloads, fully seeded: the same (n, seed, difficulty) always gives
the same problems.

    from lal.tasks import math_problems, extract_int
    probs = math_problems(8, seed=0, difficulty="easy")
    probs[0]  # {"id": "easy-0-000", "question": "...", "answer": 37, "difficulty": "easy"}
    extract_int("... so the total is 37.\nANSWER: 37")  # 37

- "easy" problems take 2 steps (one multiply or add, then one more operation).
- "hard" problems take 3-5 steps (percentages, rounding up, rates, a little algebra).
All answers are whole numbers (int), never negative.
"""
from __future__ import annotations

import random
import re

__all__ = ["math_problems", "extract_int"]

_NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jo", "Kofi", "Lena"]


def _easy(rng: random.Random) -> tuple[str, int]:
    name = rng.choice(_NAMES)
    kind = rng.randrange(5)
    if kind == 0:
        a, b = rng.randint(3, 12), rng.randint(4, 15)
        c = rng.randint(1, a * b - 1)
        return (f"{name} has {a} boxes with {b} pencils in each box. {name} gives away {c} pencils. "
                f"How many pencils does {name} have left?"), a * b - c
    if kind == 1:
        a = rng.randint(20, 60)
        b, c = rng.randint(1, a - 1), rng.randint(1, 30)
        return (f"A bus starts with {a} people. At the first stop {b} people get off and {c} people get on. "
                f"How many people are on the bus now?"), a - b + c
    if kind == 2:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        c = next(x for x in (20, 50, 100) if x > a * b)
        return (f"{name} buys {a} notebooks at ${b} each and pays with a ${c} bill. "
                f"How many dollars of change does {name} get?"), c - a * b
    if kind == 3:
        a, b, c = rng.randint(5, 30), rng.randint(3, 9), rng.randint(5, 40)
        return (f"{name} reads {a} pages a day for {b} days, then reads {c} more pages. "
                f"How many pages did {name} read in total?"), a * b + c
    a, b = rng.randint(4, 12), rng.randint(5, 12)
    c = rng.randint(1, a * b - 1)
    return (f"A farm has {a} rows of trees with {b} trees in each row. {c} trees are cut down. "
            f"How many trees are left?"), a * b - c


def _hard(rng: random.Random) -> tuple[str, int]:
    name = rng.choice(_NAMES)
    kind = rng.randrange(5)
    if kind == 0:
        while True:
            a, b = rng.randint(6, 24), rng.randint(12, 48)
            p, q = rng.choice([10, 20, 25, 40, 50]), rng.randint(6, 15)
            if (a * b * p) % 100 == 0:
                break
        good = a * b - a * b * p // 100
        return (f"A shop orders {a} crates with {b} apples in each crate. {p}% of the apples are bruised and "
                f"thrown away. The rest are packed into bags of {q} apples. How many full bags can the shop make?"), good // q
    if kind == 1:
        w, r, h, d = rng.randint(3, 9), rng.randint(4, 12), rng.randint(5, 8), rng.randint(2, 6)
        total = w * r * h * d
        s, q = rng.randint(5, total // 4), rng.randint(20, 60)
        good = total - s
        return (f"A factory has {w} workers. Each worker makes {r} parts per hour and works {h} hours a day. "
                f"After {d} days, {s} parts fail inspection. The good parts ship in boxes of {q}. "
                f"How many boxes are needed to ship all the good parts, counting a partly full box?"), -(-good // q)
    if kind == 2:
        while True:
            a, b, c = rng.randint(400, 900), rng.randint(150, 350), rng.randint(60, 150)
            wk, d = rng.randint(4, 12), rng.randint(200, 1500)
            left = wk * (a - b - c) - d
            if a - b - c > 0 and left >= 0:
                break
        return (f"{name} earns ${a} per week. Each week {name} spends ${b} on rent and ${c} on food, and saves the rest. "
                f"After {wk} weeks, {name} buys a bike for ${d} with the savings. How many dollars of savings are left?"), left
    if kind == 3:
        v, t1, u, t2 = rng.randint(40, 120), rng.randint(2, 6), rng.randint(40, 120), rng.randint(1, 5)
        dist = v * t1 + u * t2
        other = rng.randint(dist // 3, dist - 1)
        return (f"Train A travels at {v} km per hour for {t1} hours, stops, then travels at {u} km per hour for {t2} hours. "
                f"Train B travels {other} km. How many more km did train A travel than train B?"), dist - other
    y = rng.randint(2, 10)
    x = rng.randint(y + 4, y + 30)
    other = rng.choice([n for n in _NAMES if n != name])
    return (f"{name} is {x} years older than {other}. In {y} years, {name} will be exactly twice as old as {other}. "
            f"How old is {other} now?"), x - y          # (B + x + y) = 2 (B + y)  ->  B = x - y


def math_problems(n: int, seed: int = 0, difficulty: str = "easy") -> list[dict]:
    """n word problems with exact integer answers.

    Returns a list of dicts: {"id", "question", "answer" (int), "difficulty"}.
    The same arguments always return the same problems.
    """
    if difficulty not in ("easy", "hard"):
        raise ValueError("difficulty must be 'easy' or 'hard'")
    rng = random.Random(f"lal-math-{seed}-{difficulty}")
    make = _easy if difficulty == "easy" else _hard
    out = []
    for i in range(n):
        q, a = make(rng)
        out.append({"id": f"{difficulty}-{seed}-{i:03d}", "question": q, "answer": int(a), "difficulty": difficulty})
    return out


_NUM = r"-?\d+(?:\.\d+)?"


def _to_int(s: str):
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def extract_int(text) -> int | None:
    """Pull the final integer answer out of a model reply.

    Looks, in order, for "ANSWER: n", "#### n" (GSM8K style) and "\\boxed{n}"; the last match wins.
    Falls back to the last number in the text. Thousands separators ("1,250") and "$" are handled.
    Returns None if there is no number, or if the answer is not a whole number (e.g. 12.5).
    """
    if not text:
        return None
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text))
    for pat in (rf"ANSWER\s*\**\s*[:=]\s*\**\s*\$?\s*({_NUM})", rf"####\s*\$?\s*({_NUM})", rf"\\boxed\{{\s*\$?\s*({_NUM})"):
        hits = re.findall(pat, t, flags=re.IGNORECASE)
        if hits:
            return _to_int(hits[-1])
    nums = re.findall(_NUM, t)
    return _to_int(nums[-1]) if nums else None
