"""Problem 9 (search_docs): retrieval over the Cardinal documents for an agent.

Design (see HW1_PLAN.md Step 9):
* Index unit = one paragraph (the docs have one paragraph per line). Each
  chunk is indexed with its document title and nearest "##" heading in
  front, so a query for a section name still finds its paragraphs.
* Search returns at most 2 hits per document, so one long document (the
  handbook) cannot crowd out a newer announcement.
* A snippet is the hit paragraph trimmed around the query words (<=150 tokens).
* read_doc returns whole lines from start_line until ~700 tokens.
* Every document carries its date (Effective/Posted line), because later
  announcements supersede older policies.
"""

from __future__ import annotations

import re

from cs329z_hw1.pipelines.bm25 import BM25Index
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens
from cs329z_hw1.types import DocWindow, SearchResult

SNIPPET_MAX_TOKENS = 140  # adapter limit is 150
WINDOW_TOKENS = 700  # adapter limit is 1500
MAX_HITS_PER_DOC = 2
MIN_SNIPPET_TOKENS = 12  # short lines borrow the next lines so a snippet says something

_DATE = re.compile(r"^(?:effective|posted|date|updated|last updated)\s*:\s*(.+)$", re.IGNORECASE)


def _doc_date(lines: list[str]) -> str:
    for line in lines[:15]:
        m = _DATE.match(line.strip())
        if m:
            return m.group(1).strip()
    return ""


class DocIndex:
    def __init__(self, docs: list[dict]):
        self.docs: dict[str, dict] = {}
        self.order: list[str] = []
        chunk_texts: list[str] = []
        self.chunks: list[tuple[str, int, str]] = []  # (doc_id, 1-based line, heading)
        for doc in docs:
            lines = doc["text"].splitlines()
            self.docs[doc["doc_id"]] = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "lines": lines,
                "date": _doc_date(lines),
            }
            self.order.append(doc["doc_id"])
            heading = ""
            for n, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    heading = stripped.lstrip("#").strip()
                    continue
                self.chunks.append((doc["doc_id"], n, heading))
                chunk_texts.append(f"{doc['title']}\n{heading}\n{stripped}")
        self.bm25 = BM25Index(chunk_texts)

    # ------------------------------------------------------------ search --
    def _excerpt(self, doc_id: str, line_no: int, terms: list[str]) -> str:
        lines = self.docs[doc_id]["lines"]
        text = lines[line_no - 1].strip()
        j = line_no
        while count_tokens(text) < MIN_SNIPPET_TOKENS and j < len(lines):  # borrow following lines
            nxt = lines[j].strip()
            j += 1
            if nxt and not nxt.startswith("#"):
                text = f"{text} {nxt}"
        max_chars = SNIPPET_MAX_TOKENS * 4 - 8
        if len(text) <= max_chars:
            return text
        lower = text.lower()
        hits = [m.start() for t in terms for m in [re.search(rf"\b{re.escape(t)}\b", lower)] if m]
        center = min(hits) if hits else 0
        start = max(0, center - max_chars // 3)
        start = text.rfind(" ", 0, start) + 1 if start > 0 else 0
        end = min(len(text), start + max_chars)
        if end < len(text):
            end = text.rfind(" ", start, end) if text.rfind(" ", start, end) > start else end
        return ("..." if start > 0 else "") + text[start:end].strip() + ("..." if end < len(text) else "")

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if k <= 0:
            return []
        terms = list(dict.fromkeys(tokenize(query or "")))
        if not terms:
            return []
        results: list[SearchResult] = []
        per_doc: dict[str, int] = {}
        for chunk_id, score in self.bm25.search(query, max(50, k * 10)):
            doc_id, line_no, heading = self.chunks[chunk_id]
            if per_doc.get(doc_id, 0) >= MAX_HITS_PER_DOC:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            snippet = self._excerpt(doc_id, line_no, terms)
            if heading and not any(re.search(rf"\b{re.escape(t)}\b", snippet.lower()) for t in terms):
                snippet = f"{heading}: {snippet}"  # the match was in the heading
                if count_tokens(snippet) > SNIPPET_MAX_TOKENS:
                    snippet = snippet[: SNIPPET_MAX_TOKENS * 4 - 3] + "..."
            doc = self.docs[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=doc["title"],
                    snippet=snippet,
                    line=line_no,
                    heading=heading,
                    date=doc["date"],
                    score=round(score, 3),
                )
            )
            if len(results) >= k:
                break
        return results

    # -------------------------------------------------------------- read --
    def read(self, doc_id: str, start_line: int = 1, max_tokens: int = WINDOW_TOKENS) -> DocWindow:
        if doc_id not in self.docs:
            known = ", ".join(self.order)
            raise LookupError(f"No document with doc_id {doc_id!r}. Known doc_ids: {known}")
        lines = self.docs[doc_id]["lines"]
        total = len(lines)
        if not isinstance(start_line, int) or start_line < 1 or start_line > total:
            raise ValueError(f"start_line must be between 1 and {total} for {doc_id!r}, got {start_line}")
        end = start_line
        size = count_tokens(lines[start_line - 1])
        while end < total:
            nxt = size + 1 + count_tokens(lines[end])  # +1 for the newline join, roughly
            if count_tokens("\n".join(lines[start_line - 1: end + 1])) > max_tokens:
                break
            size = nxt
            end += 1
        text = "\n".join(lines[start_line - 1: end])
        while count_tokens(text) > 1500 and end > start_line:  # hard adapter limit
            end -= 1
            text = "\n".join(lines[start_line - 1: end])
        return DocWindow(doc_id=doc_id, start_line=start_line, end_line=end, total_lines=total, text=text)


def build_doc_index(docs: list[dict]) -> DocIndex:
    return DocIndex(docs)
