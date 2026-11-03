# %% [markdown]
# # a09 · Augmenting agents with memory: paging, KV-cache reuse and cartridges
# **Local AI Lab** · CS329A Lecture 14 (Nov 7, guest Junchen Jiang, LMCache) · three ways to give an agent more memory than its context window
#
# A model only "sees" what is in its context window, and a long context is slow and costly. This lecture covers three fixes at three layers. **MemGPT** lets the agent page facts in and out of a small window, like an operating system. **KV-cache reuse** (CacheBlend, LMCache) keeps the model's work on text it has already read, so it does not have to read it again. **Cartridges** compress a big document into a small, trained memory ahead of time. We build a small version of each and measure it.

# %%
import json, re, time, math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from lal.llm import client, ask, chat
from lal.data import contract, labels, ids, sections
from lal.checks import check

LAB = labels()

# %% [markdown]
# ## 1. MemGPT: an agent that manages its own memory
# MemGPT (Packer et al. 2023, *MemGPT: Towards LLMs as Operating Systems*) splits memory into tiers, like RAM and disk:
# - **Main context** (what the model sees): fixed system instructions, a small **working context** the agent writes notes into, and a **FIFO queue** (first in, first out) of recent messages.
# - **External context** (outside the window): **recall storage**, every message ever, and **archival storage**, a searchable store of documents.
#
# The agent moves data between tiers by calling functions. When the prompt reaches about 70% of the window, the system adds a "memory pressure" warning so the agent saves what matters. When it is full, old messages are evicted (they stay in recall storage). Search results come back in **pages**, so one search can never flood the window.
#
# First the pure-Python parts: an archival store with a small TF-IDF search (a word-overlap score that weights rare words more), and a main context with a character budget.

# %%
def tokenize(s):
    return re.findall(r"[a-z0-9_]+", s.lower())

class ArchivalStore:
    """Archival storage: text passages plus a tiny TF-IDF search. Results come back in pages."""
    def __init__(self, passages):
        self.passages = passages
        self.docs = [Counter(tokenize(p)) for p in passages]
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + len(passages) / c) for w, c in df.items()}

    def search(self, query, page=0, page_size=3):
        q = set(tokenize(query))
        named = {w for w in q if re.fullmatch(r"msa_\d+", w)}      # a contract id in the query acts as a filter
        scores = [(sum(d[w] * self.idf.get(w, 0) for w in q) / (1 + sum(d.values())) ** 0.5, i)
                  for i, d in enumerate(self.docs) if not named or named & set(d)]
        ranked = [i for s, i in sorted(scores, reverse=True) if s > 0]
        hits = ranked[page * page_size:(page + 1) * page_size]
        more = len(ranked) > (page + 1) * page_size
        out = "\n".join(f"- {self.passages[i]}" for i in hits) or "(no results)"
        return out + (f"\n[page {page}; more results: call again with page={page + 1}]" if more else f"\n[page {page}; no more results]")

# One passage per contract section, tagged with the contract id so the search can find it.
PASSAGES = [f"[{cid} section {s['id']} {s['title']}] {s['text']}" for cid in ids() for s in sections(contract(cid))]
store = ArchivalStore(PASSAGES)
print(len(PASSAGES), "passages in archival storage")
print(store.search("msa_07 fees per month"))

# %%
class MainContext:
    """Main context = system prompt + working context + FIFO queue, kept under a character budget.
    The queue holds 'units' (one assistant turn plus its tool results), so we never split a tool call from its result."""
    def __init__(self, system, budget_chars=2500, warn_at=0.7):
        self.system, self.budget, self.warn_at = system, budget_chars, warn_at
        self.working = []          # notes the agent chose to keep (read/write via a tool)
        self.queue = []            # list of units; each unit is a list of messages
        self.recall = []           # recall storage: every message ever, never deleted
        self.evicted = 0

    def size(self):
        return len(self.render_system()) + sum(len(json.dumps(m)) for u in self.queue for m in u)

    def render_system(self):
        notes = "\n".join(f"- {n}" for n in self.working) or "(empty)"
        return f"{self.system}\n\nWORKING CONTEXT (your saved notes):\n{notes}"

    def add(self, unit):
        self.queue.append(unit)
        self.recall.extend(unit)
        while self.size() > self.budget and len(self.queue) > 2:   # keep the task and the newest unit
            self.queue.pop(1)                                      # queue[0] is the user's task: never evicted
            self.evicted += 1

    def messages(self):
        msgs = [{"role": "system", "content": self.render_system()}]
        msgs += [m for u in self.queue for m in u]
        if self.evicted:
            msgs.insert(2, {"role": "user", "content": f"[system note: {self.evicted} older steps were evicted to recall storage. Your notes are in WORKING CONTEXT.]"})
        if self.size() > self.warn_at * self.budget:
            msgs.append({"role": "user", "content": "[system warning: memory pressure. Save any facts you still need with working_context_append now.]"})
        return msgs

# Pure-Python test: fill the context past its budget and confirm it stays under budget.
mc = MainContext("test", budget_chars=600)
mc.add([{"role": "user", "content": "the task"}])
for i in range(10):
    mc.add([{"role": "user", "content": "x" * 100 + str(i)}])
print("size", mc.size(), "<= budget", mc.budget, "| evicted", mc.evicted, "| recall keeps", len(mc.recall), "messages")

# %% [markdown]
# Now the agent. It has three tools: `archival_search`, `working_context_append` and `finish`. The task needs facts from four different contracts, but the window budget (2,500 characters, about 600 tokens) is far smaller than the four contracts together (about 7,000 characters). So the agent must search, write a note, and let old results be evicted.

# %%
TOOLS = [
    {"type": "function", "function": {"name": "archival_search",
        "description": "Search archival storage (all contract sections). Include the contract id and topic in the query, e.g. 'msa_07 fees'. Results are paged.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "page": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "working_context_append",
        "description": "Save one short fact to your working context so it survives eviction, e.g. 'msa_07 fee = 4500'.",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}}},
    {"type": "function", "function": {"name": "finish",
        "description": "Give the final answer.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}},
]
SYSTEM = ("You are an agent with a small memory. Facts live in archival storage; search it one contract at a time. "
          "After each search, save the key number with working_context_append, because old search results will be evicted. "
          "When you have every fact, call finish.")

def memgpt_agent(task, budget_chars=2500, max_steps=14, verbose=True):
    ctx = MainContext(SYSTEM, budget_chars)
    ctx.add([{"role": "user", "content": task}])
    peak = 0
    for step in range(max_steps):
        msgs = ctx.messages()
        peak = max(peak, sum(len(json.dumps(m)) for m in msgs))
        msg = chat(msgs, tools=TOOLS)
        calls = msg.tool_calls or []
        if not calls:                        # the model answered in plain text: treat it as final
            return {"answer": msg.content or "", "steps": step + 1, "peak_chars": peak, "notes": ctx.working, "evicted": ctx.evicted}
        unit = [{"role": "assistant", "content": msg.content or "",
                 "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]}]
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except Exception:
                args = {}
            name = c.function.name
            if name == "finish":
                return {"answer": str(args.get("answer", "")), "steps": step + 1, "peak_chars": peak, "notes": ctx.working, "evicted": ctx.evicted}
            if name == "archival_search":
                try:
                    result = store.search(str(args.get("query", "")), int(args.get("page", 0) or 0))
                except Exception as e:
                    result = f"ERROR: {e}"
            elif name == "working_context_append":
                ctx.working.append(str(args.get("fact", ""))[:200])
                result = "saved"
            else:
                result = f"ERROR: unknown tool {name}"
            if verbose:
                print(f"  step {step+1}: {name}({c.function.arguments[:60]}) -> {result[:70]!r}")
            unit.append({"role": "tool", "tool_call_id": c.id, "content": result})
        ctx.add(unit)
    return {"answer": "STOPPED: step limit", "steps": max_steps, "peak_chars": peak, "notes": ctx.working, "evicted": ctx.evicted}

def first_number(s):
    m = re.findall(r"\d[\d,]*", s or "")
    return int(m[-1].replace(",", "")) if m else None

TARGETS = ["msa_03", "msa_07", "msa_12", "msa_18"]
TRUTH = sum(LAB[c]["terms"]["monthly_fee"] for c in TARGETS)
task = f"What is the total monthly fee, in dollars, across contracts {', '.join(TARGETS)}? Answer with one number."
t0 = time.time()
out = memgpt_agent(task)
print(f"\n{out['steps']} steps in {time.time()-t0:.0f}s | peak prompt {out['peak_chars']} chars | evicted {out['evicted']} units")
print("notes:", out["notes"])
print("answer:", out["answer"], "| truth:", TRUTH)
print("the 4 contracts in full would be", sum(len(contract(c)) for c in TARGETS), "chars")

# %% [markdown]
# **What to notice:** the prompt never grew past the budget, yet the agent used facts from four contracts. The working-context notes are what made that work: when a search result was evicted, the number survived as a note. The weak spot is also clear: if the agent forgets to save a note, the fact is gone from its view (it is still in recall storage, but only a search would bring it back).

# %% [markdown]
# ## 2. KV-cache reuse: why CacheBlend and LMCache matter
# Before a model writes its first word it must read the whole prompt (the **prefill**). While reading, it builds a **KV cache**: key and value vectors for every token at every layer. Prefill time grows with prompt length, so a long prompt means a long wait for the first token (**time to first token, TTFT**).
#
# If the next request starts with the *exact same* text, the server can reuse the cached keys and values and skip that work. This is **prefix caching**. Ollama does it automatically within one model slot. We measure it:
# 1. a long shared prefix (8 contracts) + question A (cold: nothing cached);
# 2. the same prefix + question B (warm: the prefix is reused);
# 3. one changed word at the **start** of the prefix + question A (everything after the change must be recomputed);
# 4. one changed word at the **end** of the prefix + question A (most of the prefix is still reusable).
#
# Caveat: Ollama runs 2 slots here. If a request lands in the other slot, it will look "cold". Run the cell twice if the numbers look odd.

# %%
def ttft(prompt, max_tokens=8):
    """Time to the first streamed token, in seconds."""
    t0 = time.time()
    stream = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                            temperature=0, max_tokens=max_tokens, stream=True)
    first = None
    for chunk in stream:
        if first is None and chunk.choices and (chunk.choices[0].delta.content or ""):
            first = time.time() - t0
    return first if first is not None else time.time() - t0

DOCS = ids()[:8]
PREFIX = "\n\n".join(f"=== {c} ===\n{contract(c)}" for c in DOCS)
QA = "\n\nQuestion: What governing law does msa_03 use? Answer in one word."
QB = "\n\nQuestion: What is the monthly fee in msa_05? Answer with a number."
ask("Say OK.", max_tokens=2)                      # load the model so loading time is not counted
results = {}
results["1 cold: new prefix"] = ttft(PREFIX + QA)
results["2 warm: same prefix, new question"] = ttft(PREFIX + QB)
results["3 changed first word of prefix"] = ttft("NOTE. " + PREFIX + QA)
results["4 changed last part of prefix"] = ttft(PREFIX + "\n(end of documents)" + QA)
for k, v in results.items():
    print(f"{k:38} TTFT {v:5.2f}s")
print(f"prefix is about {len(PREFIX)//4} tokens")

# %% [markdown]
# **What to notice:** the warm request should be much faster than the cold one, and changing the very first word should make it slow again. Prefix caching is all or nothing from the point of change: one new word at the start throws away the whole cache.
#
# That is the problem in RAG (retrieval-augmented generation), where each request glues together *different* retrieved chunks in a different order. Each chunk may have been seen before, but never with the same prefix, so prefix caching misses. **CacheBlend** (Yao et al. 2024, *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion*) reuses each chunk's cached KV even when it is not at the start, and recomputes only a small share of tokens (15% per layer by default), the ones whose KV values would change most because of the text before them. It cut TTFT 2.2-3.3x versus full recompute with about the same quality. **LMCache** (Cheng et al. 2025, *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*) is the open-source layer that stores these caches outside GPU memory (CPU, disk, network) and shares them across vLLM/SGLang engines and queries.
#
# Builder rule you can use today: **put stable text first** (system prompt, rules, documents) and changing text last (the question). Never put a timestamp at the top of a prompt.

# %% [markdown]
# ## 3. Cartridges: study the document once, answer many questions cheaply
# Cartridges (Eyuboglu et al. 2025, *Cartridges: Lightweight and general-purpose long context representations via self-study*) train a **small KV cache** for one document collection, offline. At question time the model loads this small cache instead of the whole document. The training recipe is **self-study**:
# 1. split the corpus into chunks;
# 2. the model talks to itself about each chunk: one copy asks (from generic seed prompts like "summarize", "structure", "ask a question"), another answers with the chunk in view;
# 3. train the small cache with **context distillation**: make the model *with the cartridge* produce the same next-token probabilities as the model *with the full chunk* in context.
#
# The paper reports matching in-context quality with 38.6x less memory and 26.4x higher throughput. Training a cartridge needs gradient access to the KV vectors, which Ollama does not give us. So we build a **text analogy**: self-study generates Q&A, the Q&A becomes compact "study notes", and we compare answering from notes vs from the full document. The real method trains vectors, not text.

# %%
CORPUS = ids()[6:12]                           # 6 contracts, mostly ones with issues
FULL_DOC = "\n\n".join(f"=== {c} ===\n{contract(c)}" for c in CORPUS)
SEEDS = {"structuring": "List the key commercial terms of this contract as 'term: value' lines (fees, term, renewal, notice, liability cap, governing law).",
         "question": "Write 3 questions a lawyer would ask about this contract, each followed by its answer from the text."}

def self_study(cid):
    """Self-study on one chunk: generic seed prompts, the chunk in view. Returns synthetic Q&A text."""
    chunk = f"=== {cid} ===\n{contract(cid)}"
    return cid, "\n".join(ask(f"{chunk}\n\n{p}", max_tokens=300) for p in SEEDS.values())

with ThreadPoolExecutor(2) as ex:                # 6 contracts x 2 seeds = 12 calls
    studied = dict(ex.map(self_study, CORPUS))

notes = []
for cid in CORPUS:                               # compress each chunk's study output into a short card (6 calls)
    card = ask(f"Study notes for contract {cid}:\n{studied[cid]}\n\nCompress these into at most 6 short lines of the form "
               f"'{cid} <term>: <value>'. Keep numbers and state names exact.", max_tokens=200)
    notes.append(card.strip())
CARTRIDGE = "\n".join(notes)
print(CARTRIDGE[:800])
print(f"\nfull document: {len(FULL_DOC)} chars | cartridge: {len(CARTRIDGE)} chars ({len(FULL_DOC)/max(len(CARTRIDGE),1):.1f}x smaller)")

# %% [markdown]
# Now the test. The questions come from the labels, not from the study step, so the notes cannot "know" the questions in advance.

# %%
QS = ([(f"What state's law governs {c}? Answer with the state name only.", LAB[c]["terms"]["governing_law"].lower()) for c in CORPUS[:4]]
      + [(f"How many days' notice of non-renewal does {c} require? Answer with a number only.", str(LAB[c]["terms"]["notice_days"])) for c in CORPUS[2:]])
print(len(QS), "questions, e.g.", QS[0])

def answer_with(context, q):
    return ask(f"{context}\n\n{q}", max_tokens=20).strip().lower()

def grade(pred, gold):
    words = re.sub(r"[^a-z0-9 ]", " ", pred).split()
    return gold in words if gold.isdigit() else gold in pred

def run(context):
    with ThreadPoolExecutor(2) as ex:
        preds = list(ex.map(lambda qg: answer_with(context, qg[0]), QS))
    return sum(grade(p, g) for p, (_, g) in zip(preds, QS)) / len(QS), preds

acc_full, _ = run(FULL_DOC)                       # 8 calls
acc_cart, preds_cart = run(CARTRIDGE)             # 8 calls
print(f"full document in context: {acc_full:.0%} | cartridge notes: {acc_cart:.0%} | size ratio {len(FULL_DOC)/max(len(CARTRIDGE),1):.1f}x")
for (q, g), p in zip(QS, preds_cart):
    print(f"  {'ok ' if grade(p, g) else 'MISS'} {q[:55]:55} -> {p[:25]!r} (truth {g})")

# %% [markdown]
# **What to notice:** the notes are several times smaller, and on facts that the generic seeds happened to cover, they do about as well. Facts that self-study never touched are simply missing. The paper saw the same thing in a sharper form: training on the raw text with plain next-token prediction memorized it but generalized poorly, and the varied synthetic conversations of self-study were what made cartridges work across question types. Composition also works there: two cartridges can be loaded together without retraining.
#
# When is each tool right?
# - **MemGPT-style paging:** many documents, the agent decides what to look at, and you want an audit trail of what it read.
# - **Prefix / KV caching:** the same long text is sent again and again (system prompt, playbook, a contract under review). Free speed-up; just keep stable text first.
# - **Cartridges:** one large corpus queried thousands of times (a codebase, a company's contract archive). Pay once to train, then serve cheaply.

# %%
check("archival search finds the right contract section", "msa_07" in store.search("msa_07 fees per month").split("\n")[0])
check("main context stays under budget after eviction", mc.size() <= mc.budget and mc.evicted > 0 and len(mc.recall) == 11)
check("MemGPT agent gets the total fee right", first_number(out["answer"]) == TRUTH)
check("warm prefix is faster than cold prefix", results["2 warm: same prefix, new question"] < results["1 cold: new prefix"])
check("cartridge is at least 2x smaller than the document", len(FULL_DOC) >= 2 * len(CARTRIDGE))
check("cartridge keeps at least half the accuracy of the full document", acc_cart >= 0.5 * acc_full)

# %% [markdown]
# ## Your turn
# 1. Shrink the MemGPT budget to 1,500 characters. Does the agent still get the right total? Look at how many units were evicted and whether its notes were complete.
# 2. Move the question to the *start* of the prompt (question, then the 8 contracts) and re-time two different questions. What happens to the cache, and why?
# 3. Add a third self-study seed ("summarize the risks for the customer") and re-run the cartridge test. Does broader self-study raise accuracy?
#
# **Next:** a10 · Agentic evaluations and long-horizon tasks

# %% [markdown]
# ## Solutions

# %%
# Solution 1: a tighter budget
small = memgpt_agent(task, budget_chars=1500, verbose=False)
print("answer:", small["answer"], "| truth:", TRUTH, "| evicted:", small["evicted"], "| notes:", small["notes"])
# Notice: with less room more units are evicted, so success depends entirely on whether the agent wrote each fee as a note.
# This is the MemGPT trade-off: the model, not the system, decides what to remember.

# %%
# Solution 2: question first breaks the prefix cache
a = ttft(QA.strip() + "\n\n" + PREFIX)
b = ttft(QB.strip() + "\n\n" + PREFIX)
print(f"question-first A: {a:.2f}s, question-first B: {b:.2f}s")
# Notice: B is slow too. The prompts differ from the first token, so nothing can be reused.
# CacheBlend's idea is to reuse the documents' KV anyway, recomputing only the tokens that need it.

# %%
# Solution 3: one more seed prompt (6 more calls for study, 6 for cards, 8 for the test)
SEEDS["risks"] = "Summarize the risks of this contract for the customer, quoting the clause numbers and values."
with ThreadPoolExecutor(2) as ex:
    studied2 = dict(ex.map(self_study, CORPUS))
CART2 = "\n".join(ask(f"Study notes for contract {c}:\n{studied2[c]}\n\nCompress into at most 6 short lines '{c} <term>: <value>'. Keep numbers and state names exact.", max_tokens=200).strip() for c in CORPUS)
acc2, _ = run(CART2)
print(f"2 seeds: {acc_cart:.0%} | 3 seeds: {acc2:.0%} | cartridge sizes {len(CARTRIDGE)} vs {len(CART2)} chars")
# Notice: more varied self-study usually covers more facts. The paper found the same: the diversity of the synthetic
# conversations matters more than training on the raw text.
