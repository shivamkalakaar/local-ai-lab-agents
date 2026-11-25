"use client";

import { animate, motion, useMotionValue, useTransform } from "motion/react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Studio } from "@/lib/useStudio";
import type { Fact } from "@/lib/types";
import { fmtCost, fmtNum } from "@/lib/derive";

const ease = [0.22, 1, 0.36, 1] as const;

/** A number that counts up to its new value. */
export function Counter({ value, format }: { value: number; format: (n: number) => string }) {
  const mv = useMotionValue(value);
  const text = useTransform(mv, (v) => format(v));
  useEffect(() => {
    const c = animate(mv, value, { duration: 0.6, ease });
    return c.stop;
  }, [value, mv]);
  return <motion.span>{text}</motion.span>;
}

export function MemoryPane({ studio }: { studio: Studio }) {
  const [facts, setFacts] = useState<Fact[] | null>(null);
  const runs = studio.turns.filter((t) => t.endTs !== undefined).length;
  useEffect(() => {
    if (!studio.sessionId) return;
    api.memory(studio.sessionId).then((r) => setFacts(r.facts)).catch(() => setFacts([]));
  }, [studio.sessionId, runs]);

  return (
    <div className="scroll h-full overflow-y-auto p-4 text-[12px]">
      <div className="mb-2 text-faint">{"// .lm_cache/studio_memory/memory.json · shared by every session · newest first in the prompt"}</div>
      {facts === null ? (
        <div className="text-faint">loading…</div>
      ) : facts.length === 0 ? (
        <div className="text-muted">
          <span className="text-kw">[]</span> no facts yet. Tell the agent something lasting about yourself, like your team.
        </div>
      ) : (
        <pre className="whitespace-pre-wrap break-words leading-[1.7]">
          <span className="text-muted">[</span>
          {"\n"}
          {[...facts].reverse().map((f, i) => (
            <motion.div key={f.id} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04, duration: 0.2, ease }} className="pl-4">
              <span className="text-muted">{"{ "}</span>
              <span className="text-fn">&quot;id&quot;</span><span className="text-muted">: </span><span className="text-str">&quot;{f.id}&quot;</span>
              <span className="text-muted">, </span>
              <span className="text-fn">&quot;saved_at&quot;</span><span className="text-muted">: </span><span className="text-str">&quot;{f.saved_at}&quot;</span>
              <span className="text-muted">,</span>
              {"\n  "}
              <span className="text-fn">&quot;fact&quot;</span><span className="text-muted">: </span><span className="text-str">&quot;{f.fact}&quot;</span>
              <span className="text-muted">{" }"}{i < facts.length - 1 ? "," : ""}</span>
            </motion.div>
          ))}
          <span className="text-muted">]</span>
        </pre>
      )}
    </div>
  );
}

function Tile({ label, children, tone = "text-text" }: { label: string; children: React.ReactNode; tone?: string }) {
  return (
    <div className="rounded border border-line bg-panel-2/60 px-3 py-2">
      <div className="text-[10.5px] uppercase tracking-wider text-faint">{label}</div>
      <div className={`mt-0.5 text-[18px] font-semibold ${tone}`}>{children}</div>
    </div>
  );
}

export function StatsPane({ studio }: { studio: Studio }) {
  const t = studio.totals;
  const calls = studio.events.filter((e) => e.type === "model_end" && e.role === "agent");
  const budget = 16000;
  const harness: [string, number, string][] = [
    ["nudges", t.nudges, "model announced an action without a call"],
    ["repairs", t.repairs, "duplicated call run once instead of failing"],
    ["checker", t.checks, "answer sent back to verify a detail"],
    ["compactions", t.compactions, "context folded to stay under budget"],
    ["format errors", t.errors, "bad call format fed back to the model"],
    ["denials", t.denials, "blocked by a rule, the copy check, or you"],
  ];
  return (
    <div className="scroll h-full space-y-5 overflow-y-auto p-4 text-[12px]">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Tile label="cost" tone="text-accent"><Counter value={t.cost} format={(n) => fmtCost(n)} /></Tile>
        <Tile label="model calls"><Counter value={t.calls} format={(n) => fmtNum(Math.round(n))} /></Tile>
        <Tile label="side calls"><Counter value={t.sideCalls} format={(n) => fmtNum(Math.round(n))} /></Tile>
        <Tile label="tokens in/out">
          <span className="text-[14px]">{fmtNum(t.tin)}<span className="text-faint"> / </span>{fmtNum(t.tout)}</span>
        </Tile>
      </div>

      <div>
        <div className="mb-2 text-faint">{"// context per model call (budget 16,000 tokens; compaction keeps it under)"}</div>
        {calls.length === 0 ? (
          <div className="text-faint">no calls yet</div>
        ) : (
          <div className="space-y-1">
            {calls.map((c, i) => {
              const pct = Math.min(100, ((c.tokens_in ?? 0) / budget) * 100);
              return (
                <div key={c.seq} className="grid grid-cols-[28px_1fr_70px] items-center gap-2">
                  <span className="text-right text-faint">#{i + 1}</span>
                  <div className="h-2.5 overflow-hidden rounded-sm bg-line">
                    <motion.div
                      className={`h-full ${pct > 85 ? "bg-bad" : pct > 60 ? "bg-warn" : "bg-accent"}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.6, ease, delay: Math.min(i, 10) * 0.03 }}
                    />
                  </div>
                  <span className="text-right text-muted">{fmtNum(c.tokens_in ?? 0)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div>
        <div className="mb-2 text-faint">{"// harness interventions this session"}</div>
        <div className="space-y-1">
          {harness.map(([name, n, what]) => (
            <div key={name} className="grid grid-cols-[110px_36px_1fr] gap-2">
              <span className="text-fn">{name}</span>
              <span className={n > 0 ? "text-accent" : "text-faint"}>{n}</span>
              <span className="text-faint">{what}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
