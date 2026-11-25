import type { DocBody, EmailBody, Fact, Mode, ModelRow } from "./types";

export const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://127.0.0.1:8765";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${AGENT_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data as T;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${AGENT_URL}${path}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
  return data as T;
}

export const api = {
  health: () => get<{ ok: boolean; model: string }>("/api/health"),
  newSession: (mode: Mode, model?: string) =>
    post<{ session_id: string; model: string; mode: Mode }>("/api/session", { mode, model }),
  session: (id: string) => get<{ session_id: string; model: string; mode: Mode }>(`/api/session?session_id=${id}`),
  models: () => get<{ models: ModelRow[]; default: string }>("/api/models"),
  send: (session_id: string, message: string) => post("/api/send", { session_id, message }),
  approve: (session_id: string, approved: boolean, reason = "") =>
    post("/api/approve", { session_id, approved, reason }),
  answer: (session_id: string, text: string) => post("/api/answer", { session_id, text }),
  steer: (session_id: string, text: string) => post("/api/steer", { session_id, text }),
  mode: (session_id: string, mode: Mode) => post<{ mode: Mode }>("/api/mode", { session_id, mode }),
  memory: (session_id: string) => get<{ facts: Fact[] }>(`/api/memory?session_id=${session_id}`),
  doc: (doc_id: string) => get<DocBody>(`/api/doc?doc_id=${encodeURIComponent(doc_id)}`),
  email: (id: string) => get<EmailBody>(`/api/email?id=${encodeURIComponent(id)}`),
  stats: () => get<{ total_spend: number }>("/api/stats"),
  eventsUrl: (session_id: string, since = 0) => `${AGENT_URL}/api/events?session_id=${session_id}&since=${since}`,
};
