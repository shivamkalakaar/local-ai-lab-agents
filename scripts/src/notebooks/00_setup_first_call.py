# %% [markdown]
# # 00 · Setup and your first local model call
# **Local AI Lab · Build and run agents on your own computer**
#
# This course runs every agent on a model that lives on this machine. No API keys, no cloud AI, no data leaving the box.
#
# **Before you start (Colab):** `Runtime → Change runtime type → T4 GPU`. Then run the setup cell below. It installs [Ollama](https://ollama.com), starts it, and downloads Meta's Llama 3.1 8B (about 5 GB). It takes 2–3 minutes the first time.
#
# Companion reading: [Agent Engineering School, Module 00](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m0)

# %% [markdown]
# ## 1. Talk to the model on this machine
# Ollama runs the model on this computer and serves it at `http://localhost:11434`. It speaks the same API as OpenAI, so we use the normal `openai` Python library and only change two things:
# - `base_url` points at **localhost**, not at OpenAI's servers;
# - the API key is a dummy value, because nothing leaves this machine.
#
# The setup cell already loaded the model into GPU memory (that first load takes about a minute), so this call should take a second or two.

# %%
import time
from openai import OpenAI

local = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")   # the model server on THIS machine
t0 = time.time()
resp = local.chat.completions.create(
    model=MODEL,                                           # llama3.1:8b, downloaded by the setup cell
    messages=[{"role": "user", "content": "In one sentence: why would a law firm want AI that runs on its own computers?"}],
    temperature=0,
)
reply = resp.choices[0].message.content
print(reply)
print(f"\n{time.time() - t0:.1f} seconds, model = {MODEL}, server = {local.base_url}")

# %% [markdown]
# Every later notebook uses two short helpers from `lal/llm.py`, so the code stays readable. They wrap exactly the call above, pointed at the same local server:
# - `ask(prompt)`: prompt in, text out;
# - `structured(prompt, Schema)`: forced JSON that matches a Pydantic schema.

# %%
from lal.llm import ask, chat, client
print("lal.llm talks to:", client.base_url)            # the same localhost address
print(ask("Reply with exactly: ready"))

# %% [markdown]
# ## 2. Measure speed and token use
# On your own hardware there is no per-token bill, but speed still matters.

# %%
t0 = time.time()
resp = local.chat.completions.create(model=MODEL, temperature=0,
    messages=[{"role": "user", "content": "List five common clauses in a services contract, one line each."}])
secs = time.time() - t0
u = resp.usage
print(resp.choices[0].message.content)
print(f"\nprompt tokens: {u.prompt_tokens}, output tokens: {u.completion_tokens}, "
      f"speed: {u.completion_tokens / secs:.0f} tokens/second")

# %% [markdown]
# ## 3. Temperature: same question, different settings
# Low temperature gives the same answer every time (good for extraction). Higher temperature gives variety (good for brainstorming, or for sampling several candidates to check).

# %%
for temp in (0, 1.0):
    answers = {ask("Name one risky clause in a vendor contract. Answer in 3 words or fewer.", temperature=temp).strip()
               for _ in range(3)}
    print(f"temperature {temp}: {answers}")

# %% [markdown]
# ## 4. Our practice data
# We use 20 made-up service agreements. Some have problems planted on purpose: an uncapped or oversized liability cap, the wrong governing law, a long auto-renewal, or a short notice period. Because we planted them, we know the right answers, so we can grade our agents. (See `data/gen.py`.)

# %%
from lal.data import contract, labels, playbook, ids
print(contract("msa_07")[:900], "...")
print("\nTrue answer for msa_07:", labels()["msa_07"])
print("\nPlaybook rules:")
for k, v in playbook()["rules"].items():
    print(f"  - {k}: {v}")

# %%
from lal.checks import check
check("model answers", len(reply) > 10)
check("20 practice contracts loaded", len(ids()) == 20)

# %% [markdown]
# ## Your turn
# 1. Ask the model to summarize `msa_07` in three bullet points. Time it.
# 2. Change `temperature` and see how the summary changes.
#
# **Next:** 01 · Build an agent loop from scratch

# %% [markdown]
# ## Solutions
# Try the exercises yourself first, then compare.

# %%
# Solution 1: summarize msa_07 in three bullets and time it
t0 = time.time()
summary = ask("Summarize this contract in exactly three short bullet points:\n\n" + contract("msa_07"))
print(summary)
print(f"\n{time.time() - t0:.1f} seconds")

# %%
# Solution 2: the same summary at three temperatures
for temp in (0, 0.7, 1.3):
    print(f"--- temperature {temp} ---")
    print(ask("Summarize this contract in exactly three short bullet points:\n\n" + contract("msa_07"), temperature=temp))
# What to notice: at 0 the summary is the same every run. At higher temperatures the wording changes,
# and at very high values facts can drift (for example, the cap period). Use 0 for anything you will check.
