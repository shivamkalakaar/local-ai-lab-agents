"""Problem 6 (tool_protocol): the text format for tool calls.

A call is one JSON object between tags, after any text (Qwen/Hermes style):

    I'll search the policies.
    <tool_call>
    {"name": "search_docs", "arguments": {"query": "parental leave"}}
    </tool_call>

Parsing: find the first <tool_call>, then let json.JSONDecoder.raw_decode
read exactly one JSON value. raw_decode understands strings, so tags or
braces inside an argument never confuse it. Then check there is no second
call. Rules: never raise; a cut inside the JSON is an error; a cut only in
the closing tag still gives the exact original call; two calls is an error.
"""

from __future__ import annotations

import json
import re

from cs329z_hw1.types import ParsedResponse, ToolCall

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
# Some open models (e.g. Nemotron via its providers) treat <tool_call> as a
# special token and drop it, so <call>...</call> is accepted too.
TAGS = {"<tool_call>": "</tool_call>", "<call>": "</call>"}


def _find_open(reply: str, pos: int = 0):
    """Earliest opening tag at or after pos: (index, open, close) or None."""
    best = None
    for o, c in TAGS.items():
        i = reply.find(o, pos)
        if i != -1 and (best is None or i < best[0]):
            best = (i, o, c)
    return best
_DECODER = json.JSONDecoder(strict=False)  # tolerate raw newlines/tabs inside strings
_FENCE_BEFORE = re.compile(r"```[a-zA-Z]*\s*$")
_FENCE_AFTER = re.compile(r"^\s*```")
_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("arguments", "args", "parameters", "input")

FORMAT_HELP = (
    "Write exactly one call at the end of your reply, like this:\n"
    f'{OPEN}\n{{"name": "<tool name>", "arguments": {{<arguments as JSON>}}}}\n{CLOSE}'
)


# ------------------------------------------------------------------ write --

def format_tool_call(text: str, call: ToolCall) -> str:
    payload = json.dumps({"name": call.name, "arguments": call.args}, ensure_ascii=False)
    # Escape '<' so an argument that quotes a whole call holds no literal tag.
    payload = payload.replace("<", "\\u003c")
    text = (text or "").strip()
    return f"{text}\n{OPEN}\n{payload}\n{CLOSE}" if text else f"{OPEN}\n{payload}\n{CLOSE}"


def escape_result(content: str) -> str:
    """Stop tool output from closing our result tag (prompt-injection hardening)."""
    return content.replace("</tool_result", "<\\/tool_result").replace("<tool_result", "<\\tool_result")


def format_tool_result(name: str, status: str, content: str) -> str:
    return f'<tool_result name="{name}" status="{status}">\n{escape_result(content)}\n</tool_result>'


# ------------------------------------------------------------------ read ---

def _snippet(s: str, pos: int, width: int = 12) -> str:
    return s[max(0, pos - width): pos + width].replace("\n", "\\n")


def _call_from_obj(obj) -> tuple[ToolCall | None, str | None]:
    if not isinstance(obj, dict):
        return None, f"The call must be a JSON object with \"name\" and \"arguments\", got {type(obj).__name__}."
    name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str)), None)
    if not name:
        return None, 'The call has no "name" string.'
    args = next((obj[k] for k in _ARG_KEYS if k in obj), {})
    if isinstance(args, str):  # some models send the arguments as a JSON string
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None, f'"arguments" for {name!r} is a string that is not valid JSON.'
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None, f'"arguments" for {name!r} must be a JSON object.'
    return ToolCall(name=name, args=args), None


def _trailing_call(before: str) -> tuple[ToolCall | None, int]:
    """A complete {"name", "arguments"} object that ends the text, if any."""
    end_text = before.rstrip()
    for m in re.finditer(r"\{", end_text):
        try:
            obj, end = _DECODER.raw_decode(end_text, m.start())
        except ValueError:
            continue
        if end == len(end_text) and isinstance(obj, dict) and any(k in obj for k in _ARG_KEYS):
            call, problem = _call_from_obj(obj)
            if call is not None and not problem:
                return call, m.start()
    return None, -1


def repeated_call(reply: str) -> ToolCall | None:
    """If a reply holds the SAME call written two or more times (a common model
    slip), return that call; otherwise None. parse_response still reports such a
    reply as an error; the session decides to run the call once."""
    try:
        calls, pos = [], 0
        while True:
            found = _find_open(reply, pos)
            if found is None:
                break
            start, open_tag, _ = found
            i = start + len(open_tag)
            while i < len(reply) and reply[i].isspace():
                i += 1
            obj, end = _DECODER.raw_decode(reply, i)
            call, problem = _call_from_obj(obj)
            if problem:
                return None
            calls.append(call)
            pos = end
        if len(calls) >= 2 and all(c.name == calls[0].name and json.dumps(c.args, sort_keys=True) ==
                                   json.dumps(calls[0].args, sort_keys=True) for c in calls):
            return calls[0]
    except (ValueError, RecursionError):
        return None
    return None


def parse_response(text: str) -> ParsedResponse:
    try:
        return _parse(text if isinstance(text, str) else "")
    except Exception as exc:  # belt and braces: the parser never raises
        return ParsedResponse(text="", tool_call=None, error=f"Could not read the reply: {type(exc).__name__}: {exc}")


def _parse(reply: str) -> ParsedResponse:
    found = _find_open(reply)
    if found is None:
        return ParsedResponse(text=reply.strip())
    start, open_tag, close_tag = found

    before = reply[:start]
    body_start = start + len(open_tag)
    i = body_start
    while i < len(reply) and reply[i].isspace():
        i += 1
    if i >= len(reply) or reply[i:].strip() == close_tag:
        # Some models write the JSON first and the tag after it. Accept that only
        # when nothing but an (empty) tag follows a complete call object.
        call, call_start = _trailing_call(before)
        if call is not None and _find_open(before) is None:
            return ParsedResponse(text=before[:call_start].strip(), tool_call=call)
        return ParsedResponse(text=before.strip(), error="Your reply was cut off right after <tool_call>. "
                              "Re-send the whole call, and keep the reply shorter.\n" + FORMAT_HELP)
    try:
        obj, end = _DECODER.raw_decode(reply, i)
    except json.JSONDecodeError as exc:
        if close_tag not in reply[i:]:
            msg = ("Your reply was cut off inside the tool call, so it was not run. "
                   "Re-send the complete call, and keep the reply shorter.")
        else:
            msg = (f"The JSON in your tool call is invalid ({exc.msg}) near: '{_snippet(reply, exc.pos)}'. "
                   "Check quotes (escape \" inside strings as \\\") and braces.")
        return ParsedResponse(text=before.strip(), error=msg + "\n" + FORMAT_HELP)

    call, problem = _call_from_obj(obj)
    if problem:
        return ParsedResponse(text=before.strip(), error=problem + "\n" + FORMAT_HELP)

    rest = reply[end:]
    stripped = rest.lstrip()
    if stripped.startswith(close_tag):
        rest = stripped[len(close_tag):]
    elif close_tag.startswith(stripped):  # only the closing tag was cut: the call is complete
        rest = ""
    if _find_open(rest) is not None:
        return ParsedResponse(text=before.strip(), error="Your reply has more than one tool call. "
                              "Send exactly ONE call per reply; you will see its result before the next.")

    if _FENCE_BEFORE.search(before) and _FENCE_AFTER.match(rest):  # tolerate ``` around the call
        before = _FENCE_BEFORE.sub("", before)
        rest = _FENCE_AFTER.sub("", rest, count=1)
    text = (before.strip() + ("\n" + rest.strip() if rest.strip() else "")).strip()
    return ParsedResponse(text=text, tool_call=call)


# ---------------------------------------------------------------- render ---

def _type_of(schema: dict) -> str:
    t = schema.get("type", "any")
    if isinstance(t, list):
        t = " or ".join(t)
    if t == "array" and isinstance(schema.get("items"), dict):
        t = f"array of {schema['items'].get('type', 'any')}"
    return t


def _render_param(name: str, schema: dict, required: bool) -> str:
    bits = [_type_of(schema), "required" if required else "optional"]
    if "default" in schema:
        bits.append(f"default {json.dumps(schema['default'])}")
    if "enum" in schema:
        bits.append("one of " + ", ".join(json.dumps(v) for v in schema["enum"]))
    if "pattern" in schema:
        bits.append(f"format {schema['pattern']}")
    if "minimum" in schema:
        bits.append(f"min {schema['minimum']}")
    line = f"  - {name} ({', '.join(bits)})"
    if schema.get("description"):
        line += f": {schema['description']}"
    return line


def render_tools(schemas: list[dict], tag: str = "tool_call") -> str:
    OPEN, CLOSE = f"<{tag}>", f"</{tag}>"  # noqa: N806 - shadows the defaults for this render
    blocks = []
    for s in schemas:
        params = s.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        lines = [f"### {s['name']}", (s.get("description") or "").strip()]
        if props:
            lines.append("Arguments:")
            lines += [_render_param(n, p if isinstance(p, dict) else {}, n in required) for n, p in props.items()]
        else:
            lines.append("Arguments: none (use {})")
        blocks.append("\n".join(l for l in lines if l))

    example_args = '{"text": "She said \\"ship it\\" twice."}'
    return (
        "## Tools\n"
        "You can call the tools below. To call one, end your reply with exactly one call:\n"
        f'{OPEN}\n{{"name": "<tool name>", "arguments": {{...}}}}\n{CLOSE}\n'
        "Rules:\n"
        f"- At most ONE call per reply. Short reasoning may come BEFORE the call. STOP right after {CLOSE}: no text, no second copy, no check of the format after it.\n"
        "- The arguments are a JSON object: use the exact argument names below, double quotes, and escape "
        "quotes inside strings as \\\". Numbers and booleans are not quoted.\n"
        "- When a tool can do what the user asks, ALWAYS call it, even for easy things like arithmetic or "
        "repeating text back: the user needs the tool's result, not your own answer.\n"
        "- Do not say you will use a tool without calling it: write the call in the same reply.\n"
        f"- The {OPEN} tag comes BEFORE the JSON and {CLOSE} after it.\n"
        "- If no tool fits the request, answer in plain text with no call.\n"
        f'Example (the tool name is a placeholder; use a real one from the list below):\nI\'ll pass the sentence on.\n{OPEN}\n{{"name": "some_tool", "arguments": {example_args}}}\n{CLOSE}\n\n'
        + "\n\n".join(blocks)
    )
