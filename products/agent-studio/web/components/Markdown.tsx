"use client";

import { Fragment, type ReactNode } from "react";

function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\*([^*\s][^*]*)\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(<Fragment key={`${key}-${i++}`}>{text.slice(last, m.index)}</Fragment>);
    if (m[2]) out.push(<strong key={`${key}-${i++}`} className="font-semibold text-accent">{m[2]}</strong>);
    else if (m[4]) out.push(<code key={`${key}-${i++}`} className="rounded bg-panel-2 px-1 py-px text-str">{m[4]}</code>);
    else if (m[6]) out.push(<em key={`${key}-${i++}`} className="text-text/90">{m[6]}</em>);
    last = re.lastIndex;
  }
  if (last < text.length) out.push(<Fragment key={`${key}-${i++}`}>{text.slice(last)}</Fragment>);
  return out;
}

/** Just enough Markdown for agent answers: bold, code, italics, bullets, numbered lists, paragraphs. */
export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  const lines = text.split("\n");
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = (k: number) => {
    if (!list) return;
    const Tag = list.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={`l${k}`} className={`my-1 space-y-0.5 pl-5 ${list.ordered ? "list-decimal" : "list-[square]"} marker:text-faint`}>
        {list.items.map((it, j) => (
          <li key={j}>{inline(it, `li${k}-${j}`)}</li>
        ))}
      </Tag>,
    );
    list = null;
  };
  lines.forEach((line, k) => {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const num = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || num) {
      const ordered = Boolean(num);
      if (!list || list.ordered !== ordered) {
        flush(k);
        list = { ordered, items: [] };
      }
      list.items.push((bullet ?? num)![1]);
      return;
    }
    flush(k);
    if (!line.trim()) return;
    const heading = line.match(/^#{1,4}\s+(.*)$/);
    if (heading) blocks.push(<div key={k} className="mt-2 font-semibold text-text">{inline(heading[1], `h${k}`)}</div>);
    else blocks.push(<p key={k} className="my-1">{inline(line, `p${k}`)}</p>);
  });
  flush(lines.length);
  return <div className="break-words">{blocks}</div>;
}
