"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { deriveSources, deriveTurns, pendingPrompt, thinking } from "./derive";
import type { Mode, ModelRow, StudioEvent } from "./types";

export type Conn = "connecting" | "online" | "offline";

export function useStudio() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [models, setModels] = useState<ModelRow[]>([]);
  const [mode, setModeState] = useState<Mode>("confirm");
  const [events, setEvents] = useState<StudioEvent[]>([]);
  const [conn, setConn] = useState<Conn>("connecting");
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const seen = useRef(0);

  const start = useCallback(async (m: Mode = mode, chosen?: string) => {
    setError(null);
    esRef.current?.close();
    setEvents([]);
    seen.current = 0;
    try {
      const s = await api.newSession(m, chosen ?? (model || undefined));
      setSessionId(s.session_id);
      setModel(s.model);
      window.history.replaceState(null, "", `?session=${s.session_id}`);
      setModeState(s.mode);
      setConn("online");
    } catch {
      setConn("offline");
      setSessionId(null);
    }
  }, [mode, model]);

  // First session on load (with the model list for the dropdown).
  useEffect(() => {
    api.models().then((r) => setModels(r.models)).catch(() => setModels([]));
    // ?session=<id> reopens a running session (survives reloads, shareable on this machine).
    const existing = new URLSearchParams(window.location.search).get("session");
    if (existing) {
      api.session(existing)
        .then((s) => { setSessionId(s.session_id); setModel(s.model); setModeState(s.mode); setConn("online"); })
        .catch(() => start("confirm"));
    } else start("confirm");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Live event stream for the current session (reconnects from the last seen event).
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const open = () => {
      const es = new EventSource(api.eventsUrl(sessionId, seen.current));
      esRef.current = es;
      es.onopen = () => setConn("online");
      es.onmessage = (msg) => {
        const e = JSON.parse(msg.data) as StudioEvent;
        if (e.seq < seen.current) return;
        seen.current = e.seq + 1;
        if (e.type === "mode" && e.mode) setModeState(e.mode);
        setEvents((prev) => [...prev, e]);
      };
      es.onerror = () => {
        es.close();
        if (cancelled) return;
        setConn("offline");
        retry = setTimeout(open, 1500);
      };
    };
    open();
    return () => {
      cancelled = true;
      clearTimeout(retry);
      esRef.current?.close();
    };
  }, [sessionId]);

  const turns = useMemo(() => deriveTurns(events), [events]);
  const sources = useMemo(() => deriveSources(events), [events]);
  const prompt = useMemo(() => pendingPrompt(events), [events]);
  const inFlight = useMemo(() => thinking(events), [events]);
  const running = turns.length > 0 && turns[turns.length - 1].endTs === undefined;

  const totals = useMemo(() => {
    let cost = 0, calls = 0, sideCalls = 0, tin = 0, tout = 0, cached = 0;
    let lastContext = 0;
    for (const e of events) {
      if (e.type !== "model_end") continue;
      cost += e.cost ?? 0;
      tin += e.tokens_in ?? 0;
      tout += e.tokens_out ?? 0;
      if (e.cached) cached++;
      if (e.role === "agent") {
        calls++;
        lastContext = e.tokens_in ?? 0;
      } else sideCalls++;
    }
    const count = (t: string) => events.filter((e) => e.type === t).length;
    return {
      cost, calls, sideCalls, tin, tout, cached, lastContext,
      nudges: count("nudge"), repairs: count("repaired"), checks: count("check"),
      compactions: count("compaction"), errors: count("error"),
      denials: events.filter((e) => e.type === "tool_result" && e.status === "denied").length,
    };
  }, [events]);

  const run = async (fn: () => Promise<unknown>) => {
    try {
      setError(null);
      await fn();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return {
    sessionId, model, models, mode, events, turns, sources, prompt, inFlight, running, totals, conn, error,
    newSession: () => start(mode),
    switchModel: (id: string) => start(mode, id),
    send: (text: string) => sessionId && run(() => api.send(sessionId, text)),
    steer: (text: string) => sessionId && run(() => api.steer(sessionId, text)),
    approve: (ok: boolean, reason = "") => sessionId && run(() => api.approve(sessionId, ok, reason)),
    answer: (text: string) => sessionId && run(() => api.answer(sessionId, text)),
    setMode: (m: Mode) => sessionId && run(async () => { const r = await api.mode(sessionId, m); setModeState(r.mode); }),
  };
}

export type Studio = ReturnType<typeof useStudio>;
