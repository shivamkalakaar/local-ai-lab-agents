"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { Studio } from "@/lib/useStudio";
import type { Step } from "@/lib/types";
import { fmtCost, fmtMs, fmtNum } from "@/lib/derive";
import { CallSig, Shell } from "./Code";

const ease = [0.22, 1, 0.36, 1] as const;

const BADGE: Record<string, string> = {
  ok: "text-ok border-ok/40",
  tool_error: "text-bad border-bad/40",
  denied: "text-warn border-warn/40",
  invalid_args: "text-warn border-warn/40",
  unknown_tool: "text-warn border-warn/40",
};

function ContextBar({ used, budget }: { used: number; budget: number }) {
  const pct = Math.min(100, (used / budget) * 100);
  const color = pct > 85 ? "bg-bad" : pct > 60 ? "bg-warn" : "bg-accent";
  return (
    <div className="h-1 w-24 overflow-hidden rounded bg-line">
      <motion.div className={`h-full ${color}`} initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 0.5, ease }} />
    </div>
  );
}

function Result({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const lines = text.split("\n");
  const long = lines.length > 8 || text.length > 700;
  const shown = open || !long ? text : lines.slice(0, 8).join("\n").slice(0, 700);
  return (
    <div className="mt-1">
      <pre className="scroll max-h-[420px] overflow-auto whitespace-pre-wrap break-words rounded border border-line bg-bg/60 px-2.5 py-1.5 text-[11.5px] leading-[1.55] text-muted">
        {shown}
        {long && !open && <span className="text-faint">{"\n"}…</span>}
      </pre>
      {long && (
        <button onClick={() => setOpen((v) => !v)} className="mt-0.5 text-[11px] text-faint hover:text-accent">
          {open ? "▴ collapse" : `▾ show all (${fmtNum(lines.length)} lines, ${fmtNum(text.length)} chars)`}
        </button>
      )}
    </div>
  );
}

function StepRow({ step, t0 }: { step: Step; t0: number }) {
  const at = `+${((step.ts - t0) || 0).toFixed(1)}s`;
  let glyph = "·";
  let color = "text-faint";
  let body: React.ReactNode = null;

  if (step.kind === "model") {
    const s = step.model!.start;
    const e = step.model!.end;
    glyph = s.role === "side" ? "◈" : "◇";
    color = s.role === "side" ? "text-kw" : "text-info";
    body = (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
        <span className={color}>{s.role === "side" ? "side model" : "model call"}</span>
        <span className="text-muted">{fmtNum(s.tokens_in ?? 0)} in{e ? ` · ${fmtNum(e.tokens_out ?? 0)} out` : ""}</span>
        {s.budget ? <ContextBar used={s.tokens_in ?? 0} budget={s.budget} /> : null}
        {e ? (
          <span className="text-faint">
            {fmtMs(e.ms ?? 0)} · {e.cached ? "cached" : fmtCost(e.cost ?? 0)}
          </span>
        ) : (
          <span className="pulse-dot text-accent">running…</span>
        )}
      </div>
    );
  } else if (step.kind === "tool") {
    const call = step.call;
    const res = step.result;
    glyph = "▸";
    color = "text-accent";
    const cmd = call?.name === "run_terminal" ? String(call.args?.cmd ?? "") : null;
    body = (
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2 text-[12px]">
          {cmd !== null ? (
            <span><span className="text-kw">run_terminal </span><span className="text-faint">$ </span><Shell cmd={cmd} /></span>
          ) : (
            <CallSig name={call?.name} args={call?.args} />
          )}
          {res ? (
            <span className={`rounded border px-1.5 text-[10.5px] ${BADGE[res.status ?? ""] ?? "text-muted border-line"}`}>{res.status}</span>
          ) : (
            <span className="pulse-dot text-[11px] text-accent">running…</span>
          )}
        </div>
        {res?.content && <Result text={res.content} />}
      </div>
    );
  } else {
    const e = step.event;
    const table: Record<string, [string, string, string]> = {
      nudge: ["↻", "text-info", "nudge"],
      check: ["✓", "text-warn", "answer checker"],
      repaired: ["⚕", "text-info", "repaired duplicate call"],
      compaction: ["⇣", "text-kw", "context compaction"],
      error: ["✗", "text-bad", "format error → fed back"],
      steer: ["↪", "text-accent", "steering message"],
      approval: ["⚠", "text-warn", "approval"],
      ask: ["?", "text-accent", "ask_user"],
    };
    const [g, c, label] = table[step.kind] ?? ["·", "text-faint", step.kind];
    glyph = g;
    color = c;
    let detail = e?.content ?? "";
    if (step.kind === "compaction") detail = `dropped ${e?.dropped_items} items (was ${fmtNum(e?.tokens_before ?? 0)} tokens)\n\n${e?.summary ?? ""}`;
    if (step.kind === "approval") detail = `${e?.name} → ${step.result ? (step.result.approved ? "allowed" : `denied${step.result.reason ? `: ${step.result.reason}` : ""}`) : "waiting for you…"}`;
    if (step.kind === "ask") detail = `${e?.question}\n→ ${step.result?.text ?? "waiting for you…"}`;
    body = (
      <div className="min-w-0">
        <div className={`text-[12px] ${color}`}>{label}</div>
        {detail && <Result text={detail} />}
      </div>
    );
  }

  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease }}
      className="group relative grid grid-cols-[46px_16px_1fr] gap-2 py-1.5"
    >
      <span className="pt-px text-right text-[10.5px] text-faint">{at}</span>
      <span className={`pt-px text-center ${color}`}>{glyph}</span>
      {body}
    </motion.div>
  );
}

export function TracePane({ studio }: { studio: Studio }) {
  const end = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  useEffect(() => {
    if (follow) end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [studio.events.length, follow]);

  if (studio.turns.length === 0) {
    return <div className="p-6 text-faint">{"// no steps yet. ask something on the left."}</div>;
  }
  return (
    <div className="relative h-full">
      <div className="scroll h-full overflow-y-auto px-3 py-3" onWheel={() => setFollow(false)}>
        {studio.turns.map((turn) => (
          <div key={turn.id} className="mb-5">
            <div className="sticky top-0 z-10 -mx-3 mb-1 border-b border-line bg-panel/95 px-3 py-1.5 backdrop-blur">
              <span className="text-accent">❯ </span>
              <span className="text-text">{turn.user.length > 90 ? turn.user.slice(0, 90) + "…" : turn.user}</span>
            </div>
            <AnimatePresence initial={false}>
              {turn.steps.map((s) => (
                <StepRow key={s.id} step={s} t0={turn.startTs} />
              ))}
            </AnimatePresence>
            {turn.endTs !== undefined && (
              <div className="grid grid-cols-[46px_16px_1fr] gap-2 py-1.5 text-[12px]">
                <span className="text-right text-[10.5px] text-faint">+{(turn.endTs - turn.startTs).toFixed(1)}s</span>
                <span className="text-center text-ok">■</span>
                <span className={turn.status === "done" ? "text-ok" : "text-warn"}>
                  {turn.status === "done" ? "final answer" : `stopped: ${turn.status}`} · {fmtCost(turn.cost ?? 0)}
                </span>
              </div>
            )}
          </div>
        ))}
        <div ref={end} />
      </div>
      {!follow && (
        <button onClick={() => setFollow(true)} className="absolute bottom-3 right-4 rounded border border-line-2 bg-panel px-2 py-0.5 text-[11px] text-muted hover:text-accent">
          ↓ follow
        </button>
      )}
    </div>
  );
}
