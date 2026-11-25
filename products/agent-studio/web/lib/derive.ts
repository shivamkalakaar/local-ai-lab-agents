import type { SourceRef, Step, StudioEvent, Turn } from "./types";

/** Turn the raw event stream into chat turns with their steps. */
export function deriveTurns(events: StudioEvent[]): Turn[] {
  const turns: Turn[] = [];
  let turn: Turn | null = null;
  let pendingCall: Step | null = null;
  let pendingModel: Step | null = null;

  for (const e of events) {
    if (e.type === "run_start") {
      turn = { id: e.seq, user: e.message ?? "", startTs: e.ts, steps: [] };
      turns.push(turn);
      continue;
    }
    if (!turn) continue;
    switch (e.type) {
      case "model_start": {
        pendingModel = { id: e.seq, ts: e.ts, kind: "model", label: e.role === "side" ? "side model" : "model", model: { start: e } };
        turn.steps.push(pendingModel);
        break;
      }
      case "model_end": {
        if (pendingModel?.model && pendingModel.model.start.role === e.role) pendingModel.model.end = e;
        pendingModel = null;
        break;
      }
      case "tool_call": {
        pendingCall = { id: e.seq, ts: e.ts, kind: "tool", label: e.name ?? "tool", call: e };
        turn.steps.push(pendingCall);
        break;
      }
      case "tool_result": {
        if (pendingCall && pendingCall.call?.name === e.name && !pendingCall.result) pendingCall.result = e;
        else turn.steps.push({ id: e.seq, ts: e.ts, kind: "tool", label: e.name ?? "tool", result: e });
        pendingCall = null;
        break;
      }
      case "nudge":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "nudge", label: "nudge", event: e });
        break;
      case "check":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "check", label: "checker", event: e });
        break;
      case "repaired":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "repaired", label: "repaired", event: e });
        break;
      case "compaction":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "compaction", label: "compaction", event: e });
        break;
      case "error":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "error", label: "format error", event: e });
        break;
      case "steer_queued":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "steer", label: "steer", event: e });
        break;
      case "approval_request":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "approval", label: "approval", event: e });
        break;
      case "approval_resolved": {
        const s = [...turn.steps].reverse().find((x) => x.kind === "approval");
        if (s) s.result = e;
        break;
      }
      case "ask_request":
        turn.steps.push({ id: e.seq, ts: e.ts, kind: "ask", label: "ask_user", event: e });
        break;
      case "ask_resolved": {
        const s = [...turn.steps].reverse().find((x) => x.kind === "ask");
        if (s) s.result = e;
        break;
      }
      case "run_end":
        turn.answer = e.text;
        turn.status = e.status;
        turn.cost = e.cost;
        turn.endTs = e.ts;
        break;
    }
  }
  return turns;
}

/** The open approval or question, if the agent is waiting on the user. */
export function pendingPrompt(events: StudioEvent[]): StudioEvent | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const t = events[i].type;
    if (t === "approval_resolved" || t === "ask_resolved" || t === "run_end") return null;
    if (t === "approval_request" || t === "ask_request") return events[i];
  }
  return null;
}

/** Is a model call in flight right now? */
export function thinking(events: StudioEvent[]): StudioEvent | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const t = events[i].type;
    if (t === "model_end" || t === "run_end") return null;
    if (t === "model_start") return events[i];
  }
  return null;
}

const SEARCH_DOC = /^\[\d+\]\s+([\w-]+)\s+"([^"]*)"(?:\s+\(([^)]*)\))?.*?,\s*line\s+(\d+):/;
const READ_DOC = /^\[([\w-]+)\s+"([^"]*)"(?:\s*\|\s*dated\s+([^|]+))?\s*\|\s*lines\s+(\d+)-(\d+)\s+of/;
const EMAIL_ID = /\bem-\d{4,}\b/g;

/** Which documents and emails the agent touched, and where. */
export function deriveSources(events: StudioEvent[]): SourceRef[] {
  const map = new Map<string, SourceRef>();
  const ref = (kind: "doc" | "email", id: string, seq: number): SourceRef => {
    const key = `${kind}:${id}`;
    let r = map.get(key);
    if (!r) {
      r = { kind, id, hitLines: [], readRanges: [], firstSeq: seq };
      map.set(key, r);
    }
    return r;
  };
  for (const e of events) {
    if (e.type !== "tool_result" || e.status !== "ok" || !e.content) continue;
    if (e.name === "search_docs") {
      for (const line of e.content.split("\n")) {
        const m = line.match(SEARCH_DOC);
        if (m) {
          const r = ref("doc", m[1], e.seq);
          r.title = m[2];
          r.date = m[3];
          r.hitLines.push(Number(m[4]));
        }
      }
    } else if (e.name === "read_doc") {
      const m = e.content.split("\n")[0].match(READ_DOC);
      if (m) {
        const r = ref("doc", m[1], e.seq);
        r.title = m[2];
        r.date = m[3]?.trim();
        r.readRanges.push([Number(m[4]), Number(m[5])]);
      }
    } else if (["read_email", "search_emails", "email_qa"].includes(e.name ?? "")) {
      const ids = new Set(e.content.match(EMAIL_ID) ?? []);
      for (const id of ids) ref("email", id, e.seq);
    }
  }
  return [...map.values()].sort((a, b) => b.firstSeq - a.firstSeq);
}

/** Bold phrases in an answer are its key facts; used to mark them in sources. */
export function keyPhrases(answer?: string): string[] {
  if (!answer) return [];
  const out = new Set<string>();
  for (const m of answer.matchAll(/\*\*([^*]{2,80})\*\*/g)) out.add(m[1].trim());
  // Models that don't bold: numbers with units, money, percentages and dates are the key facts.
  const MONTH = "(?:January|February|March|April|May|June|July|August|September|October|November|December)";
  const facts = new RegExp(
    `\\$\\d[\\d,.]*(?:\\s?(?:million|billion|MM|M|k))?|\\d[\\d,.]*\\s?(?:%|percent|days?|hours?|minutes?|weeks?|months?|years?|business days|working days|characters)` +
      `|${MONTH}\\s\\d{1,2}(?:,\\s\\d{4})?|\\d{1,2}\\s${MONTH}`,
    "g",
  );
  for (const m of answer.matchAll(facts)) if (m[0].length >= 3) out.add(m[0].trim());
  return [...out];
}

export const fmtCost = (c: number) => (c < 0.01 ? `$${c.toFixed(4)}` : `$${c.toFixed(3)}`);
export const fmtNum = (n: number) => n.toLocaleString("en-US");
export const fmtMs = (ms: number) => (ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`);
