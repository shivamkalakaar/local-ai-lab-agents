"""Checks that code, not the model, performs: citations and scoring."""
import random
import re
from math import comb


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def quote_is_in_doc(quote: str, doc: str) -> bool:
    """A quote counts only if it is really in the document (whitespace and case ignored)."""
    return bool(quote and quote.strip()) and norm(quote) in norm(doc)


def prf(predicted: set, gold: set) -> dict:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 1.0
    r = tp / len(gold) if gold else 1.0
    return {"precision": p, "recall": r, "tp": tp, "fp": len(predicted - gold), "fn": len(gold - predicted)}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k tries succeeds, given c successes in n tries."""
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance that ALL k tries succeed. The right metric for repeated business work."""
    return comb(c, k) / comb(n, k)


def bootstrap_ci(scores, iters=5000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(iters))
    return means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2)) - 1]


def check(name: str, ok: bool):
    """Print PASS/FAIL so learners know a step worked."""
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return ok
