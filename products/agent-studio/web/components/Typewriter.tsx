"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "motion/react";
import { Markdown } from "./Markdown";

/** Reveals an answer at reading speed with a caret, then renders it as Markdown. */
export function Typewriter({ text, animate }: { text: string; animate: boolean }) {
  const reduce = useReducedMotion();
  const [n, setN] = useState(animate && !reduce ? 0 : text.length);

  useEffect(() => {
    if (!animate || reduce) {
      setN(text.length);
      return;
    }
    let raf = 0;
    const startAt = performance.now();
    const cps = Math.max(220, text.length / 2.2); // long answers finish in ~2s
    const tick = (now: number) => {
      const next = Math.min(text.length, Math.floor(((now - startAt) / 1000) * cps));
      setN(next);
      if (next < text.length) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [text, animate, reduce]);

  const done = n >= text.length;
  return (
    <div className={done ? "" : "caret"}>
      <Markdown text={done ? text : text.slice(0, n)} />
    </div>
  );
}
