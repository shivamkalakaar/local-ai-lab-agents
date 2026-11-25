"""Problem 8 (cardinal): the ten Cardinal tools.

Indexes are built lazily, the first time a tool needs them, behind a lock,
and shared by every session in the process (the live tests run sessions in
threads). Creating a session reads nothing from the archive.
"""

from __future__ import annotations

import threading

from cs329z_hw1 import data
from cs329z_hw1.agent.docs import DocIndex
from cs329z_hw1.agent.prompts import DESCRIPTIONS
from cs329z_hw1.agent.terminal import make_run_terminal
from cs329z_hw1.cardinal import cardinal_spec
from cs329z_hw1.pipelines.bm25 import BM25Index
from cs329z_hw1.pipelines.digest import build_digest
from cs329z_hw1.pipelines.email_qa import answer_question
from cs329z_hw1.tokens import count_tokens

DOC_TOOL_BUDGET = 5600  # search_docs + read_doc results per conversation (limit 6000)
_lock = threading.Lock()
_shared: dict = {}


def email_index():
    with _lock:
        if "emails" not in _shared:
            emails = data.load_emails()
            _shared["emails"] = (emails, BM25Index([data.email_text(e) for e in emails]),
                                 {e["id"]: e for e in emails})
        return _shared["emails"]


def doc_index() -> DocIndex:
    with _lock:
        if "docs" not in _shared:
            _shared["docs"] = DocIndex(data.load_docs())
        return _shared["docs"]


def _one_line(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "..."


def build_cardinal_tools(session) -> list:
    """The ten tools, bound to one session (for aux_lm, user, memory, budgets)."""
    config = session.config
    used = {"docs": 0}

    # ------------------------------------------------------------ emails --
    def search_emails(query: str, k: int = 5) -> str:
        emails, index, _ = email_index()
        hits = index.search(query, min(int(k), 20))
        if not hits:
            return f"No emails match {query!r}. Try other keywords (names, code names, places)."
        lines = [f"{len(hits)} emails for {query!r}:"]
        for doc_id, score in hits:
            e = emails[doc_id]
            lines.append(f"- {e['id']} | {e['date'][:10]} | from {e.get('from') or '?'} | "
                         f"subject: {e.get('subject') or '(none)'} | {_one_line(e.get('body', ''), 160)}")
        return "\n".join(lines)

    def read_email(email_id: str) -> str:
        _, _, by_id = email_index()
        e = by_id.get(email_id.strip())
        if e is None:
            raise LookupError(f"No email with id {email_id!r}. Ids look like em-01234; get them from search_emails.")
        body = e.get("body") or ""
        if len(body) > 6000:
            body = body[:6000] + f"\n[... {len(body) - 6000} more characters cut ...]"
        return (f"id: {e['id']} (thread {e.get('thread_id')}, reply to {e.get('reply_to')})\n"
                f"from: {e.get('from')}\nto: {', '.join(e.get('to') or [])}\ndate: {e.get('date')}\n"
                f"subject: {e.get('subject')}\n\n{body}")

    def email_qa(question: str) -> str:
        emails, index, _ = email_index()

        def search(q: str, k: int = 5):
            return [emails[i] for i, _ in index.search(q, k)]

        result = answer_question(question, search, session.aux_lm)
        if not result["answer"]:
            return "No answer found in the archive. Try search_emails with other keywords."
        return f"Answer: {result['answer']}\nSupporting emails: {', '.join(result['support']) or '(none)'}"

    def daily_digest(date: str) -> str:
        day = data.emails_on(date)
        if not day:
            return f"No emails on {date}. The archive covers 2001-06-01 to 2001-08-31."
        return f"Digest for {date} ({len(day)} emails):\n" + build_digest(day, session.aux_lm)

    # -------------------------------------------------------------- docs --
    def _docs_left() -> int:
        return DOC_TOOL_BUDGET - used["docs"]

    def _charge(text: str) -> str:
        used["docs"] += count_tokens(text)
        return text

    def search_docs(query: str, k: int = 5) -> str:
        left = _docs_left()
        if left < 120:
            return _charge("Document tool budget used up for this conversation; answer with what you have.")
        results = doc_index().search(query, min(int(k), 8))
        if not results:
            return _charge(f"No document passages match {query!r}. Try other words (e.g. the policy or product name).")
        lines = [f"{len(results)} passages for {query!r}:"]
        for i, r in enumerate(results, 1):
            dated = f" ({r['date']})" if r.get("date") else ""
            section = f" > {r['heading']}" if r.get("heading") else ""
            entry = (f"[{i}] {r['doc_id']} \"{r['title']}\"{dated}{section}, line {r['line']}: {r['snippet']} "
                     f"-> read_doc(\"{r['doc_id']}\", start_line={max(1, r['line'] - 2)})")
            if count_tokens("\n".join(lines + [entry])) + 40 > left:
                break
            lines.append(entry)
        out = "\n".join(lines)
        out += f"\n[document tool budget left: ~{max(0, left - count_tokens(out) - 15)} tokens]"
        return _charge(out)

    def read_doc(doc_id: str, start_line: int = 1) -> str:
        left = _docs_left()
        if left < 150:
            return _charge("Document tool budget used up for this conversation; answer with what you have.")
        index = doc_index()
        window = index.read(doc_id.strip(), int(start_line), max_tokens=min(700, left - 90))
        doc = index.docs[window["doc_id"]]
        dated = f" | dated {doc['date']}" if doc["date"] else ""
        footer = (f"[next: read_doc(\"{window['doc_id']}\", start_line={window['end_line'] + 1})]"
                  if window["end_line"] < window["total_lines"] else "[end of document]")
        out = (f"[{window['doc_id']} \"{doc['title']}\"{dated} | lines {window['start_line']}-{window['end_line']} "
               f"of {window['total_lines']}]\n{window['text']}\n{footer}")
        out += f"\n[document tool budget left: ~{max(0, left - count_tokens(out) - 15)} tokens]"
        return _charge(out)

    run_terminal = make_run_terminal(config)

    tools = [
        cardinal_spec("search_emails", DESCRIPTIONS["search_emails"], search_emails),
        cardinal_spec("read_email", DESCRIPTIONS["read_email"], read_email),
        cardinal_spec("email_qa", DESCRIPTIONS["email_qa"], email_qa),
        cardinal_spec("daily_digest", DESCRIPTIONS["daily_digest"], daily_digest),
        cardinal_spec("search_docs", DESCRIPTIONS["search_docs"], search_docs),
        cardinal_spec("read_doc", DESCRIPTIONS["read_doc"], read_doc),
        cardinal_spec("run_terminal", DESCRIPTIONS["run_terminal"], run_terminal),
    ]
    return tools


def memory_tools(session) -> list:
    store = session.memory

    def remember(fact: str) -> str:
        return store.remember(fact)

    def recall(query: str) -> str:
        return store.recall(query)

    return [
        cardinal_spec("remember", DESCRIPTIONS["remember"], remember),
        cardinal_spec("recall", DESCRIPTIONS["recall"], recall),
    ]


def ask_user_tool(session):
    def ask_user(question: str) -> str:
        if session.config.user is None:
            return "No user is available to answer right now; make a reasonable assumption and say what you assumed."
        answer = session.config.user.ask(question)
        return answer if (answer or "").strip() else "(The user gave no answer.)"

    return cardinal_spec("ask_user", DESCRIPTIONS["ask_user"], ask_user)
