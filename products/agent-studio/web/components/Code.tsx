"use client";

import { Fragment, type ReactNode } from "react";

/** Minimal JSON syntax colouring (keys, strings, numbers, literals). */
export function JsonInline({ value }: { value: unknown }) {
  const text = JSON.stringify(value ?? {}, null, 0);
  const parts: ReactNode[] = [];
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(<span key={i++} className="text-muted">{text.slice(last, m.index)}</span>);
    if (m[1]) {
      parts.push(
        <span key={i++} className={m[2] ? "text-fn" : "text-str"}>
          {m[1]}
        </span>,
      );
      if (m[2]) parts.push(<span key={i++} className="text-muted">{m[2]}</span>);
    } else if (m[3]) parts.push(<span key={i++} className="text-kw">{m[3]}</span>);
    else if (m[4]) parts.push(<span key={i++} className="text-num">{m[4]}</span>);
    last = re.lastIndex;
  }
  if (last < text.length) parts.push(<span key={i++} className="text-muted">{text.slice(last)}</span>);
  return <>{parts}</>;
}

/** A tool call written like a function call: name({...}). */
export function CallSig({ name, args }: { name?: string; args?: Record<string, unknown> }) {
  return (
    <span className="break-words">
      <span className="text-kw">{name}</span>
      <span className="text-muted">(</span>
      <JsonInline value={args} />
      <span className="text-muted">)</span>
    </span>
  );
}

const SHELL_RE = /(\|\||&&|\||;|>>?|<)|('(?:[^'])*'|"(?:\\.|[^"\\])*")|(^|\s)(-{1,2}[\w-]+)|(^|[|;&]\s*)([a-z][\w.-]*)/g;

/** Shell command colouring for terminal calls. */
export function Shell({ cmd }: { cmd: string }) {
  const out: ReactNode[] = [];
  let last = 0;
  let i = 0;
  let m: RegExpExecArray | null;
  const re = new RegExp(SHELL_RE.source, "g");
  while ((m = re.exec(cmd))) {
    if (m.index > last) out.push(<Fragment key={i++}>{cmd.slice(last, m.index)}</Fragment>);
    if (m[1]) out.push(<span key={i++} className="text-kw">{m[1]}</span>);
    else if (m[2]) out.push(<span key={i++} className="text-str">{m[2]}</span>);
    else if (m[4]) out.push(<Fragment key={i++}>{m[3]}<span className="text-num">{m[4]}</span></Fragment>);
    else if (m[6]) out.push(<Fragment key={i++}>{m[5]}<span className="text-fn">{m[6]}</span></Fragment>);
    last = re.lastIndex;
  }
  if (last < cmd.length) out.push(<Fragment key={i++}>{cmd.slice(last)}</Fragment>);
  return <span className="whitespace-pre-wrap break-words">{out}</span>;
}
