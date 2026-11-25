"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { Studio } from "@/lib/useStudio";
import type { SourceRef, Step, Turn } from "@/lib/types";
import { fmtCost, fmtMs } from "@/lib/derive";
import { CallSig, Shell } from "./Code";
import { Typewriter } from "./Typewriter";
import { Markdown } from "./Markdown";

const SUGGESTIONS = [
  { tag: "docs", text: "How many PTO days can I carry over into next year?" },
  { tag: "email", text: "For the 2001 Foxglove Classic, how much did the caterer want up front?" },
  { tag: "terminal", text: "How many emails in the archive were sent in July 2001?" },
  { tag: "clarify", text: "What's the daily meal limit for my business trip?" },
  { tag: "memory", text: "I'm on the firmware team. When I say 'the spec' I mean the firmware spec." },
  { tag: "risky", text: "Delete every file in the workspace to free up space." },
];

const ease = [0.22, 1, 0.36, 1] as const;

function statusColor(status?: string) {
  if (status === "ok") return "bg-ok";
  if (status === "denied") return "bg-warn";
  if (status === "tool_error") return "bg-bad";
  if (status === "invalid_args" || status === "unknown_tool") return "bg-warn";
  return "bg-faint";
}

function StepLine({ step }: { step: Step }) {
  if (step.kind === "model") {
    const end = step.model?.end;
    if (end) return null; // finished model calls live in the trace tab
    return (
      <div className="relative overflow-hidden rounded px-2 py-0.5 text-muted">
        <span className="pulse-dot mr-2 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />
        {step.model?.start.role === "side" ? "side model checking" : "model thinking"}
        <span className="ml-2 text-faint">{(step.model?.start.tokens_in ?? 0).toLocaleString()} tok in</span>
        <div className="scan absolute bottom-0 left-0 h-px w-full" />
      </div>
    );
  }
  if (step.kind === "tool") {
    const cmd = step.call?.name === "run_terminal" ? String(step.call?.args?.cmd ?? "") : null;
    return (
      <div className="flex items-start gap-2 px-2 py-0.5">
        <span className={`mt-[7px] inline-block h-1.5 w-1.5 shrink-0 rounded-full ${step.result ? statusColor(step.result.status) : "pulse-dot bg-accent"}`} />
        <span className="min-w-0 text-[12px]">
          {cmd !== null ? (
            <>
              <span className="text-kw">$ </span>
              <Shell cmd={cmd.length > 160 ? cmd.slice(0, 160) + "…" : cmd} />
            </>
          ) : (
            <CallSig name={step.call?.name} args={step.call?.args} />
          )}
          {step.result && step.result.status !== "ok" && <span className="ml-2 text-warn">[{step.result.status}]</span>}
        </span>
      </div>
    );
  }
  const map: Record<string, [string, string]> = {
    nudge: ["text-info", "↻ nudge: the model announced an action without calling a tool"],
    check: ["text-warn", "✓? checker: sent the answer back to verify a detail"],
    repaired: ["text-info", "⚕ repaired: a duplicated tool call was run once"],
    compaction: ["text-kw", "⇣ compaction: older context folded into a summary"],
    error: ["text-bad", "✗ format error fed back to the model"],
    steer: ["text-accent", `↪ steer: ${step.event?.content ?? ""}`],
  };
  const m = map[step.kind];
  if (!m) return null;
  return <div className={`px-2 py-0.5 text-[12px] ${m[0]}`}>{m[1]}</div>;
}

function SourceChips({ refs, onOpen }: { refs: SourceRef[]; onOpen: (r: SourceRef) => void }) {
  if (!refs.length) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {refs.slice(0, 6).map((r) => (
        <button
          key={`${r.kind}:${r.id}`}
          onClick={() => onOpen(r)}
          className="group rounded border border-line-2 bg-panel-2 px-2 py-0.5 text-[11px] text-muted transition hover:border-accent hover:text-text"
        >
          <span className="text-faint group-hover:text-accent">{r.kind === "doc" ? "doc" : "mail"}:</span>
          {r.id}
          {r.hitLines.length > 0 && <span className="text-faint">:L{Math.min(...r.hitLines)}</span>}
        </button>
      ))}
    </div>
  );
}

function TurnView({ turn, latest, sources, until, onOpen }: { turn: Turn; latest: boolean; sources: SourceRef[]; until: number; onOpen: (r: SourceRef) => void }) {
  const done = turn.endTs !== undefined;
  const elapsed = done ? (turn.endTs! - turn.startTs) * 1000 : 0;
  const toolSteps = turn.steps.filter((s) => s.kind === "tool").length;
  const fresh = latest && done && Date.now() / 1000 - turn.endTs! < 6;
  const mine = sources.filter((r) => r.firstSeq >= turn.id && r.firstSeq < until);
  const [showSteps, setShowSteps] = useState(!done);
  useEffect(() => {
    if (done) setShowSteps(false);
  }, [done]);

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease }} className="space-y-2">
      <div className="flex gap-3">
        <span className="select-none pt-px text-accent">❯</span>
        <div className="min-w-0 whitespace-pre-wrap text-text">{turn.user}</div>
      </div>

      <div className="ml-[18px] border-l border-line pl-3">
        <button onClick={() => setShowSteps((v) => !v)} className="mb-1 flex items-center gap-2 text-[11px] text-faint hover:text-muted">
          <span>{showSteps ? "▾" : "▸"}</span>
          <span>
            {toolSteps} tool call{toolSteps === 1 ? "" : "s"}
            {done && <> · {fmtMs(Math.round(elapsed))} · {fmtCost(turn.cost ?? 0)}</>}
          </span>
          {!done && <span className="text-accent">running</span>}
        </button>
        <AnimatePresence initial={false}>
          {showSteps && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease }}
              className="overflow-hidden"
            >
              <AnimatePresence initial={false}>
                {turn.steps.map((s) => (
                  <motion.div key={s.id} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.18, ease }}>
                    <StepLine step={s} />
                  </motion.div>
                ))}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>

        {done && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2 }} className="mt-1">
            <div className="mb-1 flex items-center gap-2 text-[11px]">
              <span className="text-kw">agent</span>
              {turn.status !== "done" && <span className="rounded bg-warn/15 px-1.5 text-warn">stopped: {turn.status}</span>}
            </div>
            <div className="text-text">
              {fresh ? <Typewriter text={turn.answer ?? ""} animate /> : <Markdown text={turn.answer ?? ""} />}
            </div>
            <SourceChips refs={mine} onOpen={onOpen} />
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

function PromptCard({ studio }: { studio: Studio }) {
  const p = studio.prompt!;
  const [text, setText] = useState("");
  const isApproval = p.type === "approval_request";
  const cmd = isApproval && p.name === "run_terminal" ? String(p.args?.cmd ?? "") : null;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 6 }}
      transition={{ duration: 0.22, ease }}
      className="ml-[18px] overflow-hidden rounded-md border border-warn/50 bg-panel shadow-[0_0_0_4px_color-mix(in_srgb,var(--warn)_10%,transparent)]"
    >
      <div className="flex items-center justify-between border-b border-line bg-warn/10 px-3 py-1.5 text-[11px]">
        <span className="text-warn">{isApproval ? "⚠ approval needed" : "? the agent asks"}</span>
        <span className="text-faint">{isApproval ? `confirm mode · ${p.name}` : "ask_user"}</span>
      </div>
      <div className="px-3 py-2.5">
        {isApproval ? (
          <pre className="whitespace-pre-wrap break-words rounded bg-panel-2 px-3 py-2 text-[12px]">
            {cmd !== null ? (<><span className="text-kw">$ </span><Shell cmd={cmd} /></>) : <CallSig name={p.name} args={p.args} />}
          </pre>
        ) : (
          <div className="text-text">{p.question}</div>
        )}
        <form
          className="mt-2.5 flex flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (isApproval) studio.approve(false, text);
            else if (text.trim()) studio.answer(text.trim());
            setText("");
          }}
        >
          <input
            autoFocus
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={isApproval ? "reason to deny (optional)" : "type your answer…"}
            className="min-w-[180px] flex-1 rounded border border-line-2 bg-bg px-2 py-1 text-[12px] outline-none focus:border-accent"
          />
          {isApproval ? (
            <>
              <button type="button" onClick={() => studio.approve(true)} className="rounded bg-ok/20 px-3 py-1 text-[12px] text-ok transition hover:bg-ok/30">
                allow
              </button>
              <button type="submit" className="rounded bg-bad/15 px-3 py-1 text-[12px] text-bad transition hover:bg-bad/25">
                deny
              </button>
            </>
          ) : (
            <button type="submit" className="rounded bg-accent/20 px-3 py-1 text-[12px] text-accent transition hover:bg-accent/30">
              reply ⏎
            </button>
          )}
        </form>
      </div>
    </motion.div>
  );
}

export function ChatPane({ studio, onOpenSource }: { studio: Studio; onOpenSource: (r: SourceRef) => void }) {
  const bottom = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState("");
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [studio.events.length, studio.prompt]);

  const submit = () => {
    const t = draft.trim();
    if (!t) return;
    if (studio.running) studio.steer(t);
    else studio.send(t);
    setDraft("");
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="scroll min-h-0 flex-1 space-y-6 overflow-y-auto px-5 py-5">
        {studio.turns.length === 0 && (
          <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, ease }} className="mx-auto max-w-xl pt-6">
            <div className="text-faint">{"// cardinal energy · internal assistant"}</div>
            <div className="mt-1 text-[22px] font-semibold tracking-tight text-text">
              ask<span className="text-accent">(</span>anything<span className="text-accent">)</span>
            </div>
            <p className="mt-2 text-muted">
              Company documents and a 2001 email archive. Every search, read, command and check streams on the right.
            </p>
            <div className="mt-5 grid gap-2">
              {SUGGESTIONS.map((s, i) => (
                <motion.button
                  key={s.text}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 * i + 0.1, duration: 0.2, ease }}
                  onClick={() => studio.send(s.text)}
                  disabled={!studio.sessionId}
                  className="group flex items-center gap-3 rounded border border-line bg-panel/70 px-3 py-2 text-left transition hover:border-accent hover:bg-panel"
                >
                  <span className="w-16 shrink-0 text-[11px] text-faint group-hover:text-accent">{s.tag}</span>
                  <span className="text-muted group-hover:text-text">{s.text}</span>
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}
        {studio.turns.map((t, i) => (
          <TurnView key={t.id} turn={t} latest={i === studio.turns.length - 1} sources={studio.sources} until={studio.turns[i + 1]?.id ?? Infinity} onOpen={onOpenSource} />
        ))}
        <AnimatePresence>{studio.prompt && <PromptCard key={studio.prompt.seq} studio={studio} />}</AnimatePresence>
        <div ref={bottom} />
      </div>

      <div className="border-t border-line bg-panel/80 px-4 py-3 backdrop-blur">
        {studio.error && <div className="mb-2 text-[12px] text-bad">✗ {studio.error}</div>}
        <div className={`flex items-end gap-2 rounded-md border bg-bg px-3 py-2 transition ${studio.running ? "border-accent/50" : "border-line-2 focus-within:border-accent"}`}>
          <span className="select-none pb-px text-accent">❯</span>
          <textarea
            rows={1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={studio.running ? "steer the agent… (it reads this before its next step)" : "ask about a policy, an email, or a computation…"}
            className="scroll max-h-40 min-h-[22px] flex-1 resize-none bg-transparent text-text outline-none placeholder:text-faint"
          />
          <button
            onClick={submit}
            disabled={!draft.trim() || !studio.sessionId}
            className="rounded bg-accent/15 px-2.5 py-0.5 text-[12px] text-accent transition enabled:hover:bg-accent/25 disabled:opacity-40"
          >
            {studio.running ? "steer" : "run"} ⏎
          </button>
        </div>
      </div>
    </div>
  );
}
