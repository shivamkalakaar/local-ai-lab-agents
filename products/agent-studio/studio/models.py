"""Open-weight models the Studio can run, with OpenRouter prices (USD per 1M
tokens: input, cached input, output; checked 2026-11-25 from
openrouter.ai/api/v1/models).

The course's LM wrapper only knows OpenAI prices, so register() adds these to
its price table at startup. HW1 itself is not changed.
"""

from __future__ import annotations

OPEN_MODELS = {
    "openai/gpt-oss-120b": {"label": "gpt-oss-120b", "maker": "OpenAI (open weights)", "price": (0.05, 0.05, 0.25)},
    "nvidia/nemotron-3-super-120b-a12b": {"label": "Nemotron 3 Super 120B", "maker": "NVIDIA", "price": (0.075, 0.075, 0.40),
                                          "tag": "call"},
    "google/gemma-4-31b-it": {"label": "Gemma 4 31B", "maker": "Google", "price": (0.20, 0.20, 0.40)},
    "mistralai/mistral-small-2603": {"label": "Mistral Small 2603", "maker": "Mistral", "price": (0.15, 0.15, 0.60)},
}


def register() -> None:
    from cs329z_hw1 import llm

    for model_id, info in OPEN_MODELS.items():
        llm.PRICES[model_id.rsplit("/", 1)[-1]] = info["price"]


def configure(session, model_id: str) -> None:
    """Per-model harness settings (today: the call tag)."""
    tag = OPEN_MODELS.get(model_id, {}).get("tag")
    if tag:
        session.use_call_tag(tag)
