"""Problem 12 (memory): facts that persist across sessions.

Store: <memory_dir>/memory.json, a list of {"id": "m3", "fact": str, "saved_at": "YYYY-MM-DD"}.
* remember(fact): append (exact duplicates are skipped). If an older fact
  looks similar, the result says so; both stay and "newer wins".
* recall(query): keyword overlap ranking with a small recency tie-break.
* At session start every fact is loaded into the system prompt, newest
  first, up to a size cap; recall still reaches facts past the cap.
With memory_dir=None (a Cardinal session without one) the store lives in
memory for this session only.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path

from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens

PROMPT_CAP_TOKENS = 600
_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or", "my", "i", "me", "user", "s"}


def _terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP}


class MemoryStore:
    def __init__(self, memory_dir: str | Path | None):
        self.dir = Path(memory_dir) if memory_dir is not None else None
        self._facts: list[dict] = []
        self._lock = threading.Lock()
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / "memory.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self._facts = [f for f in data if isinstance(f, dict) and f.get("fact")]
                except (ValueError, OSError):
                    self._facts = []

    @property
    def path(self) -> Path | None:
        return self.dir / "memory.json" if self.dir is not None else None

    def facts(self) -> list[dict]:
        return list(self._facts)

    def _save(self) -> None:
        if self.path is not None:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._facts, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    def remember(self, fact: str) -> str:
        fact = " ".join((fact or "").split())
        if not fact:
            raise ValueError("fact is empty; pass one short sentence to save.")
        with self._lock:
            for f in self._facts:
                if f["fact"].lower() == fact.lower():
                    return f"Already saved as [{f['id']}]. Nothing changed."
            new_terms = _terms(fact)
            similar = []
            for f in self._facts:
                old = _terms(f["fact"])
                if new_terms and old and len(new_terms & old) / len(new_terms | old) >= 0.4:
                    similar.append(f)
            next_id = 1 + max((int(f["id"][1:]) for f in self._facts if str(f.get("id", "")).startswith("m") and f["id"][1:].isdigit()), default=0)
            entry = {"id": f"m{next_id}", "fact": fact, "saved_at": date.today().isoformat()}
            self._facts.append(entry)
            self._save()
        msg = f"Saved as [{entry['id']}]."
        if similar:
            listed = "; ".join(f"[{f['id']}] {f['fact']}" for f in similar[-3:])
            msg += f" Similar older facts: {listed}. If the new fact corrects one, the newer fact wins."
        return msg

    def recall(self, query: str) -> str:
        facts = self.facts()
        if not facts:
            return "No memories saved yet."
        q = _terms(query or "")
        n = len(facts)
        scored = []
        for rank_from_new, f in enumerate(reversed(facts)):
            overlap = len(q & _terms(f["fact"])) if q else 0
            scored.append((overlap + 0.1 * (0.9 ** rank_from_new), overlap, f))
        hits = [s for s in sorted(scored, key=lambda s: -s[0]) if s[1] > 0][:5]
        if not hits:
            newest = list(reversed(facts))[:5]
            listed = "\n".join(f"[{f['id']} {f['saved_at']}] {f['fact']}" for f in newest)
            return f"No saved fact matches {query!r}. The {len(newest)} newest of {n} saved facts:\n{listed}"
        return "\n".join(f"[{f['id']} {f['saved_at']}] {f['fact']}" for _, _, f in hits)

    def prompt_block(self) -> str:
        facts = list(reversed(self.facts()))  # newest first
        if not facts:
            return "No facts saved yet."
        lines, used = [], 0
        for f in facts:
            line = f"[{f['id']} {f['saved_at']}] {f['fact']}"
            if used + count_tokens(line) > PROMPT_CAP_TOKENS:
                lines.append(f"({len(facts) - len(lines)} older facts not shown; use recall to search them.)")
                break
            lines.append(line)
            used += count_tokens(line)
        return "\n".join(lines)
