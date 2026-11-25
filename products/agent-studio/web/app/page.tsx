"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { ChatPane } from "@/components/ChatPane";
import { TracePane } from "@/components/TracePane";
import { SourcesPane } from "@/components/SourcesPane";
import { Counter, MemoryPane, StatsPane } from "@/components/SidePanes";
import { useStudio } from "@/lib/useStudio";
import { fmtCost, fmtNum, keyPhrases } from "@/lib/derive";
import type { SourceRef } from "@/lib/types";

type Tab = "trace" | "sources" | "memory" | "stats";
const TABS: { id: Tab; file: string; icon: string }[] = [
  { id: "trace", file: "trace.log", icon: "≡" },
  { id: "sources", file: "sources/", icon: "▤" },
  { id: "memory", file: "memory.json", icon: "{}" },
  { id: "stats", file: "stats.ts", icon: "◔" },
];
const ease = [0.22, 1, 0.36, 1] as const;

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("studio-theme");
      if (saved === "light" || saved === "dark") setTheme(saved);
    } catch {}
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("studio-theme", theme);
    } catch {}
  }, [theme]);
  return [theme, setTheme] as const;
}

function Offline({ retry }: { retry: () => void }) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease }} className="max-w-lg rounded-md border border-line bg-panel p-5">
        <div className="text-bad">✗ agent server not reachable on 127.0.0.1:8765</div>
        <p className="mt-2 text-muted">Start it in a terminal, then retry:</p>
        <pre className="mt-2 rounded bg-bg px-3 py-2 text-[12px]">
          <span className="text-faint">$ </span><span className="text-fn">cd</span> ~/Desktop/Tinkering/cs329z-homework/hw1{"\n"}
          <span className="text-faint">$ </span><span className="text-fn">uv</span> run python ../agent-studio/server.py
        </pre>
        <button onClick={retry} className="mt-3 rounded bg-accent/15 px-3 py-1 text-accent hover:bg-accent/25">retry ⏎</button>
      </motion.div>
    </div>
  );
}

export default function StudioPage() {
  const studio = useStudio();
  const [theme, setTheme] = useTheme();
  const [tab, setTab] = useState<Tab>("trace");
  const [selected, setSelected] = useState<SourceRef | null>(null);
  const [split, setSplit] = useState(48);

  const allAnswers = studio.turns.map((t) => t.answer ?? "").join("\n");
  const phrases = useMemo(() => keyPhrases(allAnswers), [allAnswers]);
  const openSource = (r: SourceRef) => {
    setSelected(r);
    setTab("sources");
  };
  const t = studio.totals;
  const ctxPct = Math.min(100, (t.lastContext / 16000) * 100);

  const startDrag = (e: React.PointerEvent) => {
    const total = window.innerWidth;
    const move = (ev: PointerEvent) => setSplit(Math.min(70, Math.max(30, (ev.clientX / total) * 100)));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    e.preventDefault();
  };

  return (
    <div className="canvas flex h-dvh flex-col">
      {/* top bar */}
      <header className="flex h-11 shrink-0 items-center gap-3 border-b border-line bg-panel/90 px-4 backdrop-blur">
        <div className="flex items-center gap-2">
          <span className="text-accent">◆</span>
          <span className="font-semibold tracking-tight">cardinal-agent</span>
          <span className="hidden text-faint sm:inline">~/studio/{studio.sessionId ?? "…"}</span>
        </div>
        <div className="mx-auto flex items-center rounded border border-line-2 bg-bg p-0.5 text-[11.5px]">
          {(["confirm", "auto"] as const).map((m) => (
            <button key={m} onClick={() => studio.setMode(m)} className="relative px-3 py-0.5">
              {studio.mode === m && (
                <motion.span layoutId="mode-pill" className="absolute inset-0 rounded-sm bg-accent/15" transition={{ type: "spring", stiffness: 500, damping: 38 }} />
              )}
              <span className={`relative ${studio.mode === m ? "text-accent" : "text-muted"}`}>{m}</span>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 text-[11.5px]">
          <label className="hidden items-center gap-1.5 lg:flex">
            <span className="text-faint">model</span>
            <select
              value={studio.model}
              onChange={(e) => studio.switchModel(e.target.value)}
              disabled={studio.running}
              className="rounded border border-line-2 bg-bg px-1.5 py-0.5 text-text outline-none focus:border-accent"
              title="Switching starts a new session"
            >
              {studio.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}{m.correct != null ? ` · ${m.correct}/${m.total}` : ""} · ${m.price_in}/${m.price_out}
                </option>
              ))}
            </select>
          </label>
          <span className="hidden text-muted md:inline">
            spent <span className="text-accent"><Counter value={t.cost} format={fmtCost} /></span>
          </span>
          <button onClick={() => studio.newSession()} className="rounded border border-line-2 px-2 py-0.5 text-muted transition hover:border-accent hover:text-accent">
            + new session
          </button>
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} className="rounded border border-line-2 px-2 py-0.5 text-muted transition hover:text-text" aria-label="toggle theme">
            {theme === "dark" ? "☾" : "☀"}
          </button>
        </div>
      </header>

      {/* body */}
      <main className="flex min-h-0 flex-1 flex-col md:flex-row">
        {studio.conn === "offline" && !studio.sessionId ? (
          <Offline retry={() => studio.newSession()} />
        ) : (
          <>
            <section className="min-h-0 flex-1 bg-panel/40 md:flex-none" style={{ flexBasis: `${split}%` }}>
              <ChatPane studio={studio} onOpenSource={openSource} />
            </section>
            <div onPointerDown={startDrag} className="hidden w-1 cursor-col-resize bg-line transition hover:bg-accent/60 md:block" />
            <section className="flex min-h-0 flex-1 flex-col border-t border-line bg-panel/70 md:border-t-0">
              <nav className="flex shrink-0 items-end gap-px border-b border-line bg-bg/60 px-2 pt-1.5">
                {TABS.map((x) => {
                  const badge = x.id === "sources" ? studio.sources.length : x.id === "trace" ? studio.turns.reduce((n, tt) => n + tt.steps.filter((s) => s.kind === "tool").length, 0) : 0;
                  return (
                    <button
                      key={x.id}
                      onClick={() => setTab(x.id)}
                      className={`relative rounded-t px-3 py-1.5 text-[12px] transition ${tab === x.id ? "bg-panel text-text" : "text-muted hover:text-text"}`}
                    >
                      {tab === x.id && <motion.span layoutId="tab-line" className="absolute inset-x-0 top-0 h-px bg-accent" />}
                      <span className="mr-1.5 text-faint">{x.icon}</span>
                      {x.file}
                      {badge > 0 && <span className="ml-1.5 rounded bg-accent/15 px-1 text-[10px] text-accent">{badge}</span>}
                    </button>
                  );
                })}
              </nav>
              <div className="min-h-0 flex-1">
                <AnimatePresence mode="wait">
                  <motion.div key={tab} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }} className="h-full">
                    {tab === "trace" && <TracePane studio={studio} />}
                    {tab === "sources" && <SourcesPane sources={studio.sources} selected={selected} onSelect={setSelected} phrases={phrases} />}
                    {tab === "memory" && <MemoryPane studio={studio} />}
                    {tab === "stats" && <StatsPane studio={studio} />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </section>
          </>
        )}
      </main>

      {/* status bar */}
      <footer className="flex h-6 shrink-0 items-center gap-4 overflow-hidden border-t border-line bg-accent/10 px-3 text-[11px] text-muted">
        <span className="flex items-center gap-1.5">
          <span className={`inline-block h-1.5 w-1.5 rounded-full ${studio.conn === "online" ? "bg-ok" : studio.conn === "offline" ? "bg-bad" : "pulse-dot bg-warn"}`} />
          {studio.conn}
        </span>
        <span className="text-text">{studio.model || "model"}</span>
        <span>mode: <span className="text-accent">{studio.mode}</span></span>
        <span className="hidden sm:inline">calls {t.calls}+{t.sideCalls}</span>
        <span className="hidden items-center gap-1.5 sm:flex">
          ctx
          <span className="inline-block h-1.5 w-16 overflow-hidden rounded bg-line">
            <motion.span className="block h-full bg-accent" animate={{ width: `${ctxPct}%` }} transition={{ duration: 0.4, ease }} />
          </span>
          {fmtNum(t.lastContext)}/16k
        </span>
        {studio.running && <span className="pulse-dot text-accent">● running</span>}
        <span className="ml-auto">session {fmtCost(t.cost)}</span>
      </footer>
    </div>
  );
}
