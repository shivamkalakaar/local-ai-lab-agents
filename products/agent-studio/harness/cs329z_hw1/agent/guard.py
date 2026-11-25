"""Problem 14 (guardrails): deny rules and the copied-command check.

* Deny rules: re.search over the command (and over a normalized copy, so
  zero-width characters or quote tricks do not dodge a rule).
* Copied-command check (Gemini-CLI style, auto mode only): a terminal command
  that reuses command-like text from a document or email, which the user
  never typed, is refused. Two triggers, both after normalizing:
    1. a risky token copied from untrusted text: a URL, an absolute or ../
       path, or a script file name;
    2. a copied run of >= 24 characters that itself looks like shell code
       (pipes, redirects, $, curl, rm, chmod, base64, ...).
  Plain words copied from an email subject (grep "Basin Analytics") pass.
Mode, approval policy and rules live only in the session's own fields; no
code path reads them from tool output.
"""

from __future__ import annotations

import re
import unicodedata

_INVISIBLE = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏‪-‮⁠-⁤﻿]")
_RISKY = [
    re.compile(r"https?://[^\s'\"]+"),
    re.compile(r"(?<![\w.])/(?:[\w.-]+/)*[\w.-]+"),  # absolute path
    re.compile(r"\.\./[\w./-]*"),
    re.compile(r"\b[\w-]+\.(?:sh|bash|py|pl|rb|exe|bat|ps1|command)\b"),
    re.compile(r"~/[\w./-]+"),
]
_SHELLY = ("|", ";", "&&", ">", "<", "$", "`", "http", "curl", "wget", "rm ", "chmod", "base64",
           "bash", " sh ", "python3 -c", "nc ", "scp", "ssh", "eval", "export", "sudo")
WINDOW = 24


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE.sub("", text)
    text = text.replace("'", "").replace('"', "").replace("\\", "")
    return " ".join(text.lower().split())


def deny_rule_match(cmd: str, rules) -> str | None:
    norm = normalize(cmd)
    for rule in rules or ():
        try:
            if re.search(rule, cmd) or re.search(rule, norm):
                return rule
        except re.error:
            if rule in cmd:
                return rule
    return None


def copied_from_untrusted(cmd: str, untrusted: list[str], trusted: list[str]) -> str | None:
    """Return a short description of the copied piece, or None."""
    if not untrusted:
        return None
    c = normalize(cmd)
    bad = "\n".join(normalize(t) for t in untrusted)
    good = "\n".join(normalize(t) for t in trusted)
    for pattern in _RISKY:
        for m in pattern.finditer(c):
            tok = m.group(0)
            if len(tok) >= 4 and tok in bad and tok not in good:
                return f"'{tok}'"
    for i in range(0, max(0, len(c) - WINDOW + 1), 2):
        piece = c[i:i + WINDOW]
        if piece in bad and piece not in good and any(s in f" {piece} " for s in _SHELLY):
            return f"'{piece}'"
    return None
