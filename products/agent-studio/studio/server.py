"""Agent Studio server: runs the HW1 Cardinal agent and streams every step.

Run from the hw1 folder (so its environment and .env are used):
    cd ../hw1 && uv run python ../agent-studio/server.py        # http://127.0.0.1:8765

Standard library only. Endpoints (JSON in, JSON out, CORS open for localhost):
    POST /api/session  {mode}                -> {session_id}
    POST /api/send     {session_id, message} -> {ok}   (runs in a background thread)
    GET  /api/events?session_id=..&since=N   -> text/event-stream of step events
    POST /api/approve  {session_id, approved, reason}
    POST /api/answer   {session_id, text}
    POST /api/steer    {session_id, text}
    POST /api/mode     {session_id, mode}
    GET  /api/memory?session_id=..  /api/doc?doc_id=..  /api/email?id=..  /api/stats
Every conversation is saved to hw1/runs/chat-studio-<time>.json after each
message, in the same shape as cs329z_hw1.chat (usable for the field notes).
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
# The harness lives next to the studio: ../harness in the published repo, ../hw1 while developing.
HW1 = next((p for p in (HERE.parent / "harness", HERE.parent / "hw1") if (p / "cs329z_hw1").is_dir()), HERE.parent / "hw1")
sys.path.insert(0, str(HW1))
sys.path.insert(0, str(HERE))

from cs329z_hw1 import adapters, data  # noqa: E402
from cs329z_hw1.agent import cardinal_tools  # noqa: E402
from cs329z_hw1.llm import LM, OutputTruncated, total_spend  # noqa: E402
from cs329z_hw1.tokens import count_message_tokens, count_tokens  # noqa: E402
from cs329z_hw1.types import AgentConfig, Approval  # noqa: E402
from cs329z_hw1.user import UserIO  # noqa: E402
from models import OPEN_MODELS, configure, register  # noqa: E402

register()
DEFAULT_MODEL = os.environ.get("STUDIO_MODEL", "")  # empty = best model in bakeoff_results.json


def model_table() -> list[dict]:
    """Open models with their bake-off results, best first."""
    path = HERE / "bakeoff_results.json"
    results = {r["model"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    rows = []
    for mid, info in OPEN_MODELS.items():
        r = results.get(mid, {})
        rows.append({"id": mid, "label": info["label"], "maker": info["maker"],
                     "price_in": info["price"][0], "price_out": info["price"][2],
                     "correct": r.get("correct"), "total": r.get("total"), "cost_usd": r.get("cost_usd")})
    rows.sort(key=lambda r: (-(r["correct"] or 0), r["cost_usd"] or 9))
    return rows


def default_model() -> str:
    return DEFAULT_MODEL or model_table()[0]["id"]

PORT = 8765
MEMORY_DIR = HW1 / ".lm_cache" / "studio_memory"
RUNS_DIR = HW1 / "runs"
CONTEXT_BUDGET = 16_000
SESSIONS: dict[str, "StudioSession"] = {}


class WebUser(UserIO):
    """Approvals and questions go to the browser; the agent thread waits."""

    def __init__(self, studio: "StudioSession"):
        self.studio = studio
        self._reply: "queue.Queue[dict]" = queue.Queue()
        self._steer: "queue.Queue[str]" = queue.Queue()

    def approve(self, call) -> Approval:
        self.studio.emit({"type": "approval_request", "name": call.name, "args": call.args})
        reply = self._reply.get()
        self.studio.emit({"type": "approval_resolved", "approved": reply["approved"], "reason": reply.get("reason", "")})
        return Approval(bool(reply["approved"]), reply.get("reason", ""))

    def ask(self, question: str) -> str:
        self.studio.emit({"type": "ask_request", "question": question})
        reply = self._reply.get()
        self.studio.emit({"type": "ask_resolved", "text": reply.get("text", "")})
        return reply.get("text", "")

    def pending_message(self):
        try:
            return self._steer.get_nowait()
        except queue.Empty:
            return None


class Watched:
    """Wraps an LM so each call shows up in the stream (start, end, cost)."""

    def __init__(self, lm: LM, role: str, studio: "StudioSession"):
        self.lm, self.role, self.studio = lm, role, studio

    def __getattr__(self, name):
        return getattr(self.lm, name)

    def __call__(self, messages):
        tokens_in = count_message_tokens(messages)
        cost0, cached0, t0 = self.lm.cost_usd, self.lm.cached_calls, time.time()
        self.studio.emit({"type": "model_start", "role": self.role, "tokens_in": tokens_in,
                          "budget": CONTEXT_BUDGET if self.role == "agent" else None})
        try:
            reply = self.lm(messages)
            out = reply
            return reply
        except OutputTruncated as exc:
            out = exc.text or ""
            raise
        finally:
            self.studio.emit({"type": "model_end", "role": self.role, "ms": int((time.time() - t0) * 1000),
                              "tokens_in": tokens_in, "tokens_out": count_tokens(locals().get("out", "") or ""),
                              "cost": round(self.lm.cost_usd - cost0, 6),
                              "cached": self.lm.cached_calls > cached0})


class StudioSession:
    def __init__(self, mode: str, model: str):
        self.id = uuid.uuid4().hex[:10]
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.busy = False
        self.turns: list[dict] = []
        self.started = time.strftime("%Y%m%d-%H%M%S")
        self.user = WebUser(self)
        self.model = model
        self.agent_lm = LM(model, tag="studio")
        self.aux_lm = LM(model, tag="studio/aux")
        config = AgentConfig(mode=mode, user=self.user, memory_dir=MEMORY_DIR,
                             aux_lm=self.aux_lm, context_budget=CONTEXT_BUDGET)
        self.session = adapters.run_agent_session(self.agent_lm, None, config)
        configure(self.session, model)  # per-model harness settings (call tag)
        # Stream every transcript event and every model call.
        original_log = self.session._log
        self.session._log = lambda event: (original_log(event), self.emit(dict(event)))
        self.session.lm = Watched(self.agent_lm, "agent", self)
        self.session.aux_lm = Watched(self.aux_lm, "side", self)
        self.mode = mode

    def emit(self, event: dict) -> None:
        with self.cond:
            event = {**event, "seq": len(self.events), "ts": time.time()}
            self.events.append(event)
            self.cond.notify_all()

    def run(self, message: str) -> None:
        self.busy = True
        spend0 = self.agent_lm.cost_usd + self.aux_lm.cost_usd
        self.emit({"type": "run_start", "message": message})
        try:
            result = self.session.send(message)
            text, status = result.text, result.status
        except Exception as exc:  # show model/network failures instead of hanging
            text, status = f"The run failed: {type(exc).__name__}: {exc}", "error"
        spent = self.agent_lm.cost_usd + self.aux_lm.cost_usd - spend0
        self.turns.append({"user": message, "agent": text, "status": status})
        self.emit({"type": "run_end", "status": status, "text": text, "cost": round(spent, 6),
                   "session_cost": round(self.agent_lm.cost_usd + self.aux_lm.cost_usd, 6)})
        self.busy = False
        self.save()

    def save(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        record = {"started": self.started, "mode": self.session.mode, "model": self.agent_lm.model,
                  "turns": self.turns, "transcript": self.session.transcript,
                  "usage": {"agent": self.agent_lm.usage(), "aux": self.aux_lm.usage()}, "source": "agent-studio"}
        path = RUNS_DIR / f"chat-studio-{self.started}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _session(params) -> StudioSession:
    sid = params.get("session_id")
    if isinstance(sid, list):
        sid = sid[0]
    if sid not in SESSIONS:
        raise KeyError("unknown session_id; create one with POST /api/session")
    return SESSIONS[sid]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the console quiet
        pass

    def _headers(self, status=200, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self._headers(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._headers(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/events":
                return self._stream(_session(q), int(q.get("since", 0)))
            if url.path == "/api/memory":
                store = _session(q).session.memory
                return self._json({"facts": store.facts() if store else []})
            if url.path == "/api/doc":
                index = cardinal_tools.doc_index()
                doc = index.docs.get(q.get("doc_id", ""))
                if not doc:
                    return self._json({"error": "unknown doc_id"}, 404)
                return self._json({"doc_id": doc["doc_id"], "title": doc["title"], "date": doc["date"], "lines": doc["lines"]})
            if url.path == "/api/email":
                _, _, by_id = cardinal_tools.email_index()
                email = by_id.get(q.get("id", ""))
                return self._json(email if email else {"error": "unknown id"}, 200 if email else 404)
            if url.path == "/api/docs":
                index = cardinal_tools.doc_index()
                return self._json({"docs": [{"doc_id": d, "title": index.docs[d]["title"], "date": index.docs[d]["date"]} for d in index.order]})
            if url.path == "/api/stats":
                return self._json({"total_spend": round(total_spend(), 6)})
            if url.path == "/api/health":
                return self._json({"ok": True, "model": default_model()})
            if url.path == "/api/session":
                st = _session(q)
                return self._json({"session_id": st.id, "model": st.model, "mode": st.session.mode})
            if url.path == "/api/models":
                return self._json({"models": model_table(), "default": default_model()})
            self._json({"error": "not found"}, 404)
        except KeyError as exc:
            self._json({"error": str(exc)}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "bad JSON"}, 400)
        path = urlparse(self.path).path
        try:
            if path == "/api/session":
                model = body.get("model") or default_model()
                if model not in OPEN_MODELS:
                    return self._json({"error": f"unknown model {model!r}"}, 400)
                studio = StudioSession(body.get("mode", "confirm"), model)
                SESSIONS[studio.id] = studio
                return self._json({"session_id": studio.id, "model": studio.agent_lm.model, "mode": studio.mode})
            studio = _session(body)
            if path == "/api/send":
                if studio.busy:
                    return self._json({"error": "the agent is still working; send a steering note instead"}, 409)
                message = (body.get("message") or "").strip()
                if not message:
                    return self._json({"error": "empty message"}, 400)
                threading.Thread(target=studio.run, args=(message,), daemon=True).start()
                return self._json({"ok": True})
            if path == "/api/approve":
                studio.user._reply.put({"approved": bool(body.get("approved")), "reason": body.get("reason", "")})
                return self._json({"ok": True})
            if path == "/api/answer":
                studio.user._reply.put({"text": body.get("text", "")})
                return self._json({"ok": True})
            if path == "/api/steer":
                text = (body.get("text") or "").strip()
                if text:
                    studio.user._steer.put(text)
                    studio.emit({"type": "steer_queued", "content": text})
                return self._json({"ok": True})
            if path == "/api/mode":
                studio.session.set_mode(body.get("mode", "confirm"))
                studio.emit({"type": "mode", "mode": studio.session.mode})
                return self._json({"ok": True, "mode": studio.session.mode})
            self._json({"error": "not found"}, 404)
        except (KeyError, ValueError) as exc:
            self._json({"error": str(exc)}, 400)

    def _stream(self, studio: StudioSession, since: int):
        self._headers(200, "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        sent = since
        try:
            while True:
                with studio.cond:
                    if len(studio.events) <= sent:
                        studio.cond.wait(timeout=15)
                    batch = studio.events[sent:]
                if not batch:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                for event in batch:
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n".encode())
                sent += len(batch)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    if not data.EMAILS_PATH.exists():
        print("The email archive is missing: run `uv run python data/download.py` in hw1 first.")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"Agent Studio server on http://127.0.0.1:{PORT}  (default model {default_model()}; Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
