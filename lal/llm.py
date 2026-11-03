"""One small wrapper around the OpenAI-compatible API that Ollama (and vLLM) expose.

Change LAL_BASE_URL to point the same code at vLLM or any other server.
"""
import json
import os
import time

from openai import OpenAI

client = OpenAI(base_url=os.getenv("LAL_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("LAL_API_KEY", "local"))


def model() -> str:
    return os.getenv("LAL_MODEL", "llama3.1:8b")


def chat(messages, model_name=None, **kw):
    """One model call. Returns the assistant message object."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    kw.setdefault("temperature", 0)
    resp = client.chat.completions.create(model=model_name or model(), messages=messages, **kw)
    return resp.choices[0].message


def ask(prompt, system=None, **kw) -> str:
    """Shortcut: prompt in, text out."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return chat(msgs, **kw).content or ""


def structured(messages, schema_model, model_name=None, retries=2, **kw):
    """Force the reply to match a Pydantic model (the server enforces the JSON schema), then validate.

    Small models occasionally return a cut-off or empty reply, so we retry a couple of times
    before giving up, and cap the length so a runaway reply can't hang the loop.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    kw.setdefault("temperature", 0)
    kw.setdefault("max_tokens", 1024)
    fmt = {"type": "json_schema", "json_schema": {"name": schema_model.__name__, "schema": schema_model.model_json_schema()}}
    last = None
    for _ in range(retries + 1):
        resp = client.chat.completions.create(model=model_name or model(), messages=messages, response_format=fmt, **kw)
        try:
            return schema_model.model_validate_json(resp.choices[0].message.content or "")
        except Exception as e:      # invalid or empty JSON: try again
            last = e
    raise last


def timed(fn, *a, **kw):
    """Run fn and print how long it took."""
    t0 = time.time()
    out = fn(*a, **kw)
    print(f"[{time.time() - t0:.1f}s]")
    return out
