"use client";

import { AnimatePresence, motion } from "motion/react";
import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import type { DocBody, EmailBody, SourceRef } from "@/lib/types";

const ease = [0.22, 1, 0.36, 1] as const;

/** Wrap each occurrence of the answer's key phrases in a highlighter mark. */
function marked(text: string, phrases: string[], key: string): ReactNode {
  const usable = phrases.filter((p) => p.length >= 2);
  if (!usable.length) return text;
  const esc = usable.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).sort((a, b) => b.length - a.length);
  const re = new RegExp(`(${esc.join("|")})`, "gi");
  const parts = text.split(re);
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark key={`${key}-${i}`} className="sweep px-0.5" style={{ animationDelay: `${150 + i * 60}ms` }}>
        {part}
      </mark>
    ) : (
      <Fragment key={`${key}-${i}`}>{part}</Fragment>
    ),
  );
}

/** Pieces of the key phrases that are likely to appear verbatim in a source. */
function searchTerms(phrases: string[]): string[] {
  const out = new Set<string>();
  for (const p of phrases) {
    out.add(p);
    // numbers only with a unit, $ or % ("10 days", "$750"), never a bare "31"
    for (const m of p.matchAll(/\$\d[\d,.]*|\d[\d,.]*\s*(?:days?|hours?|minutes?|weeks?|months?|percent|%|business days|working days)/gi)) {
      if (m[0].trim().length >= 3) out.add(m[0].trim());
    }
  }
  return [...out];
}

function DocView({ ref_, phrases }: { ref_: SourceRef; phrases: string[] }) {
  const [doc, setDoc] = useState<DocBody | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const first = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setDoc(null);
    api.doc(ref_.id).then(setDoc).catch((e) => setErr(String(e)));
  }, [ref_.id]);
  const hits = useMemo(() => new Set(ref_.hitLines), [ref_.hitLines]);
  const inRead = (n: number) => ref_.readRanges.some(([a, b]) => n >= a && n <= b);
  const terms = useMemo(() => searchTerms(phrases), [phrases]);
  const focusLine = useMemo(() => {
    if (!doc) return 1;
    const withFact = doc.lines.findIndex((l) => terms.some((t) => l.toLowerCase().includes(t.toLowerCase())));
    if (withFact >= 0) return withFact + 1;
    return ref_.hitLines.length ? Math.min(...ref_.hitLines) : ref_.readRanges[0]?.[0] ?? 1;
  }, [doc, terms, ref_]);
  useEffect(() => {
    if (doc) setTimeout(() => first.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 80);
  }, [doc, focusLine]);

  if (err) return <div className="p-4 text-bad">{err}</div>;
  if (!doc) return <div className="p-4 text-faint">loading {ref_.id}…</div>;
  return (
    <div className="text-[12px] leading-[1.7]">
      <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-panel/95 px-3 py-1.5 backdrop-blur">
        <span>
          <span className="text-faint">docs/</span>
          <span className="text-text">{doc.doc_id}.md</span>
        </span>
        <span className="text-faint">{doc.date && `dated ${doc.date} · `}{doc.lines.length} lines</span>
      </div>
      <div className="py-2">
        {doc.lines.map((line, i) => {
          const n = i + 1;
          const hit = hits.has(n);
          const read = inRead(n);
          return (
            <div
              key={n}
              ref={n === focusLine ? first : undefined}
              className={`grid grid-cols-[44px_1fr] border-l-2 ${hit ? "border-mark bg-mark/5" : read ? "border-accent/50" : "border-transparent"}`}
            >
              <span className={`select-none pr-3 text-right ${hit ? "text-mark" : "text-faint"}`}>{n}</span>
              <span className={`whitespace-pre-wrap break-words pr-3 ${line.startsWith("#") ? "font-semibold text-kw" : read || hit ? "text-text" : "text-muted"}`}>
                {line ? marked(line, terms, `l${n}`) : " "}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EmailView({ ref_, phrases }: { ref_: SourceRef; phrases: string[] }) {
  const [email, setEmail] = useState<EmailBody | null>(null);
  useEffect(() => {
    setEmail(null);
    api.email(ref_.id).then(setEmail).catch(() => setEmail(null));
  }, [ref_.id]);
  const terms = useMemo(() => searchTerms(phrases), [phrases]);
  if (!email) return <div className="p-4 text-faint">loading {ref_.id}…</div>;
  return (
    <div className="text-[12px] leading-[1.7]">
      <div className="sticky top-0 z-10 border-b border-line bg-panel/95 px-3 py-1.5 backdrop-blur">
        <span className="text-faint">emails.jsonl › </span>
        <span className="text-text">{email.id}</span>
      </div>
      <div className="space-y-0.5 px-4 py-3">
        {(["from", "to", "date", "subject"] as const).map((k) => (
          <div key={k} className="grid grid-cols-[64px_1fr]">
            <span className="text-fn">{k}:</span>
            <span className="break-words text-text">{k === "to" ? (email.to || []).join(", ") : String(email[k] || "")}</span>
          </div>
        ))}
      </div>
      <pre className="whitespace-pre-wrap break-words border-t border-line px-4 py-3 text-muted">{marked(email.body, terms, "b")}</pre>
    </div>
  );
}

export function SourcesPane({ sources, selected, onSelect, phrases }: {
  sources: SourceRef[];
  selected: SourceRef | null;
  onSelect: (r: SourceRef) => void;
  phrases: string[];
}) {
  const current = selected ? sources.find((s) => s.kind === selected.kind && s.id === selected.id) ?? selected : sources[0];
  if (!sources.length) return <div className="p-6 text-faint">{"// documents and emails the agent opens appear here."}</div>;
  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(160px,34%)_1fr]">
      <div className="scroll min-h-0 overflow-y-auto border-r border-line py-2">
        <div className="px-3 pb-1 text-[10.5px] uppercase tracking-wider text-faint">explorer</div>
        <AnimatePresence initial={false}>
          {sources.map((s) => {
            const active = current && current.kind === s.kind && current.id === s.id;
            return (
              <motion.button
                key={`${s.kind}:${s.id}`}
                layout="position"
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, ease }}
                onClick={() => onSelect(s)}
                className={`block w-full truncate px-3 py-1 text-left text-[12px] transition ${active ? "bg-accent/10 text-text" : "text-muted hover:bg-panel-2 hover:text-text"}`}
                title={s.title ?? s.id}
              >
                <span className={s.kind === "doc" ? "text-fn" : "text-kw"}>{s.kind === "doc" ? "▤ " : "✉ "}</span>
                {s.id}
                {s.hitLines.length > 0 && <span className="text-faint"> ·{s.hitLines.length}</span>}
              </motion.button>
            );
          })}
        </AnimatePresence>
      </div>
      <div className="scroll min-h-0 overflow-y-auto">
        {current && (current.kind === "doc" ? <DocView key={current.id} ref_={current} phrases={phrases} /> : <EmailView key={current.id} ref_={current} phrases={phrases} />)}
      </div>
    </div>
  );
}
