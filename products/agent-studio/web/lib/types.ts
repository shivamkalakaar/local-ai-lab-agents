export type Mode = "confirm" | "auto";

export type StudioEvent = {
  seq: number;
  ts: number;
  type: string;
  // transcript events
  content?: string;
  name?: string;
  args?: Record<string, unknown>;
  status?: string;
  // model calls
  role?: "agent" | "side";
  tokens_in?: number;
  tokens_out?: number;
  budget?: number | null;
  ms?: number;
  cost?: number;
  cached?: boolean;
  // runs, prompts
  message?: string;
  text?: string;
  question?: string;
  approved?: boolean;
  reason?: string;
  session_cost?: number;
  mode?: Mode;
  // compaction
  dropped_items?: number;
  tokens_before?: number;
  summary?: string;
};

export type Step = {
  id: number;
  ts: number;
  kind:
    | "tool"
    | "model"
    | "nudge"
    | "check"
    | "repaired"
    | "compaction"
    | "error"
    | "steer"
    | "approval"
    | "ask";
  label: string;
  call?: StudioEvent;
  result?: StudioEvent;
  model?: { start: StudioEvent; end?: StudioEvent };
  event?: StudioEvent;
};

export type Turn = {
  id: number;
  user: string;
  startTs: number;
  endTs?: number;
  answer?: string;
  status?: string;
  cost?: number;
  steps: Step[];
};

export type SourceRef = {
  kind: "doc" | "email";
  id: string;
  title?: string;
  date?: string;
  hitLines: number[];
  readRanges: [number, number][];
  firstSeq: number;
};

export type DocBody = { doc_id: string; title: string; date: string; lines: string[] };
export type EmailBody = {
  id: string;
  thread_id: string;
  reply_to: string | null;
  from: string;
  to: string[];
  date: string;
  subject: string;
  body: string;
};
export type Fact = { id: string; fact: string; saved_at: string };

export type ModelRow = {
  id: string;
  label: string;
  maker: string;
  price_in: number;
  price_out: number;
  correct: number | null;
  total: number | null;
  cost_usd: number | null;
};
