# %% [markdown]
# # 05 · Tools: a registry that validates, gates and truncates
# **Local AI Lab** · CS329Z Lecture 4 · [Module 05](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m5)
#
# Spend more effort on tools than on the prompt. A good tool is narrow, clearly described, hard to misuse, returns short output, and fails with a helpful message. Rules like "never email without approval" belong in code, not in the prompt.

# %%
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal
import jsonschema
# lal.llm = the OpenAI client pointed at the local Ollama server, http://localhost:11434/v1 (see notebook 00)
from lal.llm import client
from lal.data import contract, clauses, sections
from lal.checks import check

# %% [markdown]
# ## 1. The registry
# Order matters: **validate → permission → run → truncate**. Bad calls never run. Actions that change the world need approval.

# %%
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    read_only: bool = True

@dataclass
class ToolResult:
    status: Literal["ok", "unknown_tool", "invalid_args", "tool_error", "denied"]
    content: str

class Registry:
    def __init__(self, specs, approve=lambda name, args: False, max_chars=1500):
        self.specs = {s.name: s for s in specs}
        self.approve, self.max_chars = approve, max_chars

    def schemas(self):
        return [{"type": "function", "function": {"name": s.name, "description": s.description,
                 "parameters": s.parameters}} for s in self.specs.values()]

    def execute(self, name, args) -> ToolResult:
        spec = self.specs.get(name)
        if spec is None:
            return ToolResult("unknown_tool", f"ERROR: no tool '{name}'. Available: {', '.join(self.specs)}")
        try:
            jsonschema.validate(args, spec.parameters)
        except jsonschema.ValidationError as e:
            return ToolResult("invalid_args", f"ERROR: bad arguments: {e.message}")
        if not spec.read_only and not self.approve(name, args):
            return ToolResult("denied", "DENIED: a person declined this action. Do not retry it.")
        try:
            out = str(spec.fn(**args))
        except Exception as e:
            return ToolResult("tool_error", f"ERROR: {type(e).__name__}: {e}")
        if len(out) > self.max_chars:
            out = out[: self.max_chars] + f"\n[truncated; {len(out)} chars total. Ask for a narrower range.]"
        return ToolResult("ok", out)

# %% [markdown]
# ## 2. Three tools: two read-only, one that sends email
# Notice the descriptions: what it does, what it returns, when to use it. `clause` must match a pattern, so wrong formats are rejected before running.

# %%
def read_pages(contract_id, start_line=1, n_lines=15):
    lines = contract(contract_id).splitlines()
    return "\n".join(f"{i}: {l}" for i, l in enumerate(lines[start_line-1:start_line-1+n_lines], start_line))

def get_clause(contract_id, clause):
    return clauses(contract(contract_id)).get(clause, "NOT FOUND")

SENT = []
def send_email(to, body):
    SENT.append((to, body)); return f"sent to {to}"

CID = {"type": "string", "pattern": "^msa_[0-9]{2}$", "description": "Contract id like 'msa_07'"}
registry = Registry([
    ToolSpec("read_pages", "Read a contract 15 lines at a time, with line numbers. Use start_line to page.",
             {"type": "object", "properties": {"contract_id": CID, "start_line": {"type": "integer", "minimum": 1},
              "n_lines": {"type": "integer", "minimum": 1, "maximum": 30}}, "required": ["contract_id"]}, read_pages),
    ToolSpec("get_clause", "Read one clause by number like '7.1'. Returns NOT FOUND if it doesn't exist.",
             {"type": "object", "properties": {"contract_id": CID, "clause": {"type": "string", "pattern": r"^\d+\.\d+$"}},
              "required": ["contract_id", "clause"]}, get_clause),
    ToolSpec("send_email", "Email someone. Requires human approval.",
             {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]},
             send_email, read_only=False),
])

# %% [markdown]
# ## 3. Watch the registry catch mistakes (no model needed)

# %%
for name, args in [("get_clause", {"contract_id": "msa_07", "clause": "7.1"}),
                   ("get_clause", {"contract_id": "msa_7", "clause": "seven"}),
                   ("delete_contract", {"contract_id": "msa_07"}),
                   ("send_email", {"to": "x@example.com", "body": "contract attached"}),
                   ("read_pages", {"contract_id": "msa_07", "n_lines": 30})]:
    r = registry.execute(name, args)
    print(f"{r.status:13} {name}: {r.content[:80]!r}")
print("emails actually sent:", SENT)

# %% [markdown]
# ## 4. Now let the model use it
# The model asks to email the contract to someone. The registry denies it, the model is told, and nothing is sent.

# %%
def run(task, max_turns=6):
    msgs = [{"role": "system", "content": "You help lawyers. Use tools."}, {"role": "user", "content": task}]
    for _ in range(max_turns):
        m = client.chat.completions.create(model=MODEL, messages=msgs, tools=registry.schemas(), temperature=0).choices[0].message
        msgs.append(m)
        if not m.tool_calls:
            return m.content
        for c in m.tool_calls:
            try: args = json.loads(c.function.arguments)
            except json.JSONDecodeError: args = {}
            r = registry.execute(c.function.name, args)
            print(f"  {c.function.name} -> {r.status}")
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": r.content})
    return "turn limit"

print(run("Read clause 7.1 of msa_07 and email it to opposing.counsel@example.com"))
print("emails actually sent:", SENT)

# %%
check("invalid arguments are rejected before running", registry.execute("get_clause", {"contract_id": "bad", "clause": "x"}).status == "invalid_args")
check("no email was sent without approval", SENT == [])

# %% [markdown]
# ## Your turn
# 1. Write an `approve` function that allows emails only to `@yourfirm.com` addresses.
# 2. Give `read_pages` a bad description ("reads stuff") and see if the model still pages correctly.
#
# **Next:** 06 · Design patterns

# %% [markdown]
# ## Solutions

# %%
# Solution 1: an approval rule that only allows email inside the firm
registry.approve = lambda name, args: name == "send_email" and args.get("to", "").lower().endswith("@yourfirm.com")
print(registry.execute("send_email", {"to": "partner@yourfirm.com", "body": "Clause 7.1 attached"}))
print(registry.execute("send_email", {"to": "someone@gmail.com", "body": "Clause 7.1 attached"}))
print("sent so far:", SENT)
SENT.clear(); registry.approve = lambda name, args: False      # back to "deny everything"
# In a real product the approval is a person clicking Approve, shown the exact recipient and text.

# %%
# Solution 2: a vague tool description
original = registry.specs["read_pages"].description
registry.specs["read_pages"].description = "reads stuff"
print(run("Show me lines 16 to 30 of contract msa_07."))
registry.specs["read_pages"].description = original
print("\nwith the good description:")
print(run("Show me lines 16 to 30 of contract msa_07."))
# With "reads stuff" the model has to guess what start_line and n_lines mean. Good descriptions are free accuracy.
