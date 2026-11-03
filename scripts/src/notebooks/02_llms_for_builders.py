# %% [markdown]
# # 02 · LLMs for builders: forced JSON, validation, and sampling
# **Local AI Lab** · CS329Z Lecture 2 · [Module 02](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m2)
#
# Treat the model as an unreliable part with a typed interface. Force its output into a schema, check the values against the source, and when you can check answers, sample several and keep one that passes.

# %% [markdown]
# ## 1. A typed contract with Pydantic
# Note the "null if absent" option. Without it, models invent values to fill required fields.

# %%
from typing import Literal
from pydantic import BaseModel, Field
# lal.llm = the OpenAI client pointed at the local Ollama server, http://localhost:11434/v1 (see notebook 00)
from lal.llm import structured, ask
from lal.data import contract, labels, ids
from lal.checks import quote_is_in_doc, check

class Terms(BaseModel):
    governing_law: str | None = Field(description="US state name, e.g. 'Delaware', or null")
    liability_cap_months: int | None = Field(description="Cap as months of fees, e.g. 12. null if liability is unlimited")
    renewal_years: int | None = Field(description="Length of each automatic renewal period in years")
    notice_days: int | None = Field(description="Days of notice needed to stop renewal")
    cap_quote: str | None = Field(description="Exact sentence from the contract stating the liability cap")

SYSTEM = "Extract the requested terms from the contract. Copy quotes exactly. If a term is absent, use null. Never guess."

def extract(cid):
    return structured([{"role": "system", "content": SYSTEM}, {"role": "user", "content": contract(cid)}], Terms)

t = extract("msa_07")
print(t.model_dump_json(indent=2))
print("Truth:", labels()["msa_07"]["terms"])

# %% [markdown]
# ## 2. Valid JSON is not correct content
# The server guarantees the *format*. We still check the *content*: is the quote really in the document?

# %%
print("cap quote really in the contract?", quote_is_in_doc(t.cap_quote, contract("msa_07")))
print("made-up quote in the contract?   ", quote_is_in_doc("Liability is capped at $1 million.", contract("msa_07")))

# %% [markdown]
# ## 3. Field accuracy on 8 contracts

# %%
FIELDS = ["governing_law", "liability_cap_months", "renewal_years", "notice_days"]
sample = ids()[:4] + ids()[-4:]
right = {f: 0 for f in FIELDS}
quotes_ok = 0
for cid in sample:
    t = extract(cid)
    truth = labels()[cid]["terms"]
    for f in FIELDS:
        right[f] += getattr(t, f) == truth[f]
    quotes_ok += quote_is_in_doc(t.cap_quote, contract(cid))
for f in FIELDS:
    print(f"{f:22} {right[f]}/{len(sample)}")
print(f"{'cap quote verified':22} {quotes_ok}/{len(sample)}")

# %% [markdown]
# ## 4. Majority vote: when you have no checker
# Ask several times at a higher temperature and take the most common answer. The agreement rate is a useful confidence signal; low agreement means "send to a human".

# %%
from collections import Counter
def vote(question, n=5):
    answers = [ask(question, temperature=0.8).strip().lower().rstrip(".") for _ in range(n)]
    label, count = Counter(answers).most_common(1)[0]
    return label, count / n, answers

q = ("Is this clause favorable to the customer? Answer only 'yes' or 'no'.\n\n"
     + contract("msa_08").split("7.1 ")[1].split("\n")[0])
label, agree, answers = vote(q)
print(f"vote: {label} (agreement {agree:.0%}) from {answers}")

# %% [markdown]
# ## 5. Best-of-n: when you do have a checker
# Sample until an answer passes a check. Here the check is code: the quote must be in the document.

# %%
def best_of_n(cid, n=4):
    for i in range(n):
        t = structured([{"role": "system", "content": SYSTEM}, {"role": "user", "content": contract(cid)}], Terms, temperature=0.7)
        if quote_is_in_doc(t.cap_quote, contract(cid)):
            return t, i + 1
    return None, n   # nothing passed: escalate to a person instead of guessing

t, tries = best_of_n("msa_12")
print(f"passed after {tries} tries:" if t else "no sample passed", t.cap_quote if t else "")

# %%
check("extraction returns a valid Terms object", isinstance(extract("msa_01"), Terms))
check("governing law right on at least 6 of 8", right["governing_law"] >= 6)

# %% [markdown]
# ## Your turn
# 1. Remove "use null" from the system prompt and run `extract` on a contract with unlimited liability (`msa_16`). Does the model invent a cap?
# 2. Change `governing_law` to a `Literal[...]` of US states. Does accuracy change?
#
# **Next:** 03 · Context engineering

# %% [markdown]
# ## Solutions

# %%
# Solution 1: remove the "use null" escape hatch and force an integer cap on an UNLIMITED contract
class CapForced(BaseModel):
    liability_cap_months: int          # no None allowed
    cap_quote: str

forced = structured([{"role": "system", "content": "Extract the requested terms from the contract. Copy quotes exactly."},
                     {"role": "user", "content": contract("msa_16")}], CapForced)
print(forced)
print("truth:", labels()["msa_16"]["terms"]["liability_cap_months"], "(None = unlimited)")
# What to notice: when the schema demands a number, the model has to invent one. Always give an explicit "absent" option.

# %%
# Solution 2: governing law as a fixed list of choices
STATES = Literal["California", "Delaware", "New York", "Texas", "Florida", "Nevada", "Other"]
class LawOnly(BaseModel):
    governing_law: STATES

hits = sum(structured([{"role": "system", "content": SYSTEM}, {"role": "user", "content": contract(c)}], LawOnly).governing_law
           == labels()[c]["terms"]["governing_law"] for c in sample)
print(f"governing law with a fixed list: {hits}/{len(sample)}  (free text was {right['governing_law']}/{len(sample)})")
# A fixed list makes off-menu answers impossible ("State of Delaware", "DE", "Delaware law") and makes grading exact.
