# How to write a notebook for this repo

Notebooks are written as Python files in "percent" format, and `scripts/build.py` turns them into `.ipynb` files. Never edit `.ipynb` files by hand.

- Course part 1 (CS329Z): `scripts/src/notebooks/*.py` becomes `notebooks/`. Its projects: `scripts/src/projects/*.py` becomes `projects/`.
- Course part 2 (CS329A, Self-Improving AI Agents): `scripts/src/cs329a/aNN_name.py` becomes `cs329a/`. Past-project rebuilds: `scripts/src/cs329a_projects/apN_name.py` becomes `cs329a_projects/`.

## Cell format

```python
# %% [markdown]
# # a01 · Title of the notebook
# **Local AI Lab** · CS329A Lecture 2 · one-line summary
#
# Two or three plain sentences: what you will build and why it matters.

# %%
# EXTRA: rank-bm25            <- optional: extra pip packages for this notebook (setup installs them)
# MODELS: llama3.1:8b llama3.2:3b   <- optional: extra Ollama models to download
import ...

# %% [markdown]
# ## 1. Section title
# Explanation in plain language.

# %%
code...
```

- The first markdown cell is the title cell. The build inserts the shared setup cell right after it (see `SETUP` in `scripts/build.py`). After setup, the variable `MODEL` holds the main model name, Ollama is running at `http://localhost:11434/v1`, and the repo root is the working directory.
- End every notebook with the same three parts, in this order:
  - a check cell that uses `from lal.checks import check` and prints PASS or FAIL;
  - a `## Your turn` markdown cell with 2–3 exercises;
  - a `## Solutions` markdown cell followed by code cells that solve each exercise, with a short comment on what to notice.
- **Writing style:** very plain English, short sentences, no jargon unless you define it the first time you use it. Cite the paper behind each idea, for example "(Brown et al. 2024, *Large Language Monkeys*)".

## Helpers you can use (`lal/`)

| Function | What it does |
|---|---|
| `lal.llm.ask(prompt, system=None, temperature=0, **kw) -> str` | One model call, returns the text. |
| `lal.llm.chat(messages, model_name=None, **kw)` | Returns the OpenAI message object. Tool calls are supported (pass `tools=`). |
| `lal.llm.structured(messages_or_str, PydanticModel, model_name=None, temperature=0, max_tokens=1024)` | JSON output forced to match a schema, with retries. |
| `lal.llm.client` | The raw OpenAI client pointed at Ollama. |
| `lal.checks` | `check(name, ok)`, `quote_is_in_doc`, `prf(pred_set, gold_set)`, `pass_at_k(n, c, k)`, `pass_hat_k(n, c, k)`, `bootstrap_ci(scores)`. |
| `lal.data` | `contract(cid)`, `labels()`, `playbook()`, `ids()`, `clauses(text)`, `sections(text)`. These are the 20 synthetic contracts from part 1. |
| `lal.runtime.pull(model)` | Downloads another Ollama model. |

You may add a small new helper module under `lal/` if several notebooks need it, for example `lal/tasks.py` with small verifiable task sets. Make it pure Python, generated with a fixed seed.

## Hard rules

- **Hardware:** a free Colab T4 GPU (16 GB). Ollama serves `llama3.1:8b` (main) and `llama3.2:3b`, with an 8k context and 2 parallel requests. Other allowed Ollama models: `gemma3:4b`, `gemma3:12b`, `granite3.3:8b`, `phi4-mini`, `mistral:7b`.
- **No Chinese-origin models.** That rules out Qwen, DeepSeek, GLM, Yi and their fine-tunes. For small trainable models (GRPO or SFT with TRL), use `HuggingFaceTB/SmolLM2-135M-Instruct` / `SmolLM2-360M-Instruct` or `meta-llama/Llama-3.2-1B-Instruct`. The Llama models are gated, so prefer SmolLM2.
- **Keep it fast:** at most about 80 model calls and about 15 minutes per notebook. Use small samples and `ThreadPoolExecutor(2)` for parallel calls. Put anything longer behind a flag such as `RUN_LONG = False`.
- **Tasks must have checkable answers**, so the notebook can grade itself. Good choices:
  - arithmetic or word problems generated with a seed, with exact answers;
  - small coding tasks with unit tests run in a subprocess with a timeout;
  - the contract data.
- **Never run untrusted model code outside a subprocess with a timeout.**
- **Do not download datasets bigger than a few MB.** Generate data instead.
- Every model-dependent check must be able to fail, and should be lenient enough that an 8B model usually passes.
- Pin nothing exotic. Colab already has torch, transformers, numpy and pandas. Add `trl`, `peft` or `datasets` through `# EXTRA:` only where needed.
- The author cannot run models locally, so write code that is obviously correct. Guard parsing of model output (`try/except`, `.get`, regex fallbacks). Compile the file with `python3 -m py_compile` before finishing.

## Page content file (one per module)

Alongside each notebook, write `site/cs329a/<same-stem>.json`:

```json
{
  "id": "a01",
  "title": "Test-time compute scaling",
  "lecture": "CS329A Lecture 2 (Sep 26)",
  "big_idea": "One or two plain sentences.",
  "papers": [
    {"title": "...", "authors": "Brown et al.", "year": 2024, "url": "https://arxiv.org/abs/...",
     "what": "2-3 sentences: what they did.",
     "result": "the key numbers, concretely",
     "takeaway": "what a builder should do because of this paper"}
  ],
  "terms": [["Term", "plain definition", "source"]],
  "traps": ["...", "..."],
  "check": [["question?", "answer."]]
}
```

Papers must be summarized from the actual paper (abstract plus the key results sections). Mark anything you infer as such, and never invent numbers.
