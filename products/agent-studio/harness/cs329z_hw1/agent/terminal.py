"""Problem 10 (terminal): the run_terminal tool and its observation policy.

Every command goes through sandbox.run_terminal. What the model sees:
    exit code / timed out / line counts first (never cut), then stdout and
    stderr. Large output keeps ~30% head and ~70% tail on line boundaries,
    sized to ~20% of the context budget, and the full text is saved to
    outputs/cmd-<n>.txt in the workspace (a relative path, so prompts stay
    cache-friendly) for grep/sed. Status is "ok" only for exit code 0; a
    rejection, a timeout or a non-zero exit raises, which the registry turns
    into a tool_error whose content is exactly what the model is shown.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from cs329z_hw1 import sandbox

HEAD_SHARE = 0.3
MIN_SHOW_TOKENS = 150
MAX_SHOW_TOKENS = 3000
_SANDBOX_HINTS = ("operation not permitted", "read-only file system", "permission denied")


class TerminalFailure(RuntimeError):
    """Raised with the full observation so the registry records it as tool_error."""


def _cut(text: str, max_chars: int) -> tuple[str, int, int]:
    """Head + tail on line boundaries. Returns (shown, omitted_lines, first_omitted_line)."""
    if len(text) <= max_chars:
        return text, 0, 0
    lines = text.splitlines()
    head_budget, tail_budget = int(max_chars * HEAD_SHARE), int(max_chars * (1 - HEAD_SHARE))
    head, used = [], 0
    for line in lines:
        if used + len(line) + 1 > head_budget:
            break
        head.append(line)
        used += len(line) + 1
    tail, used = [], 0
    for line in reversed(lines[len(head):]):
        if used + len(line) + 1 > tail_budget:
            break
        tail.append(line)
        used += len(line) + 1
    tail.reverse()
    if not head and not tail:  # one enormous line
        return text[:head_budget] + "\n" + text[-tail_budget:], max(0, len(lines) - 2), 1
    omitted = len(lines) - len(head) - len(tail)
    marker = f"[... {omitted} lines cut here ...]"
    return "\n".join(head + [marker] + tail), omitted, len(head) + 1


def make_run_terminal(config, counter=None):
    """Build the run_terminal function for one session."""
    counter = counter or itertools.count(1)
    show_tokens = max(MIN_SHOW_TOKENS, min(int(0.2 * config.context_budget), MAX_SHOW_TOKENS))
    max_chars = show_tokens * 4

    def run_terminal(cmd: str) -> str:
        result = sandbox.run_terminal(cmd, workspace=config.workspace, timeout=config.terminal_timeout)
        if result.rejected:
            raise TerminalFailure(
                f"Command rejected by the sandbox, not run: {result.reason}\n"
                "Use only allowed programs (python3, grep, sort, awk, wc, head, tail, sed, ...); "
                "no here-documents, no $(...); put multi-line Python in python3 -c '...'."
            )

        out, err = result.stdout or "", result.stderr or ""
        n = next(counter)
        total_chars = len(out) + len(err)
        saved_note = ""
        if total_chars > max_chars:
            rel = Path("outputs") / f"cmd-{n}.txt"  # stdout only, so its line numbers match the output
            try:
                ws = Path(config.workspace) if config.workspace else sandbox.default_workspace()
                (ws / "outputs").mkdir(parents=True, exist_ok=True)
                (ws / rel).write_text(out, encoding="utf-8")
                if err:
                    (ws / rel.with_suffix(".err.txt")).write_text(err, encoding="utf-8")
                saved_note = (f"Full stdout saved to {rel.as_posix()} (line N of the file = line N of the output; "
                              f"use sed -n 'Np' or grep -n on it).")
            except OSError as exc:
                saved_note = f"(Could not save the full output: {exc}.) Re-run a narrower command."
        out_budget = max_chars if not err else int(max_chars * 0.75)
        err_budget = max_chars - min(len(out), out_budget) if err else 0
        shown_out, out_cut, out_first = _cut(out, out_budget)
        shown_err, err_cut, _ = _cut(err, max(err_budget, 400)) if err else ("", 0, 0)

        if result.timed_out:
            status_line = f"Command timed out after {config.terminal_timeout:g} seconds and was killed."
        else:
            status_line = f"Exit code: {result.exit_code}"
        header = [status_line, f"stdout: {len(out.splitlines())} lines, {len(out)} chars"
                  + (f"; {out_cut} lines cut starting at line {out_first}" if out_cut else "")]
        if err:
            header.append(f"stderr: {len(err.splitlines())} lines" + (f"; {err_cut} lines cut" if err_cut else ""))
        if saved_note:
            header.append(saved_note)
        body = f"--- stdout ---\n{shown_out if out else '(no output)'}"
        if err:
            body += f"\n--- stderr ---\n{shown_err}"
        observation = "\n".join(header) + "\n" + body
        if result.exit_code not in (0, None) and any(h in err.lower() for h in _SANDBOX_HINTS):
            observation += ("\nNote: this was likely blocked by the sandbox (no network, no writes outside "
                            "the workspace). Do not retry it; find another way.")
        if result.timed_out or result.exit_code != 0:
            raise TerminalFailure(observation)
        return observation

    return run_terminal
