# %% [markdown]
# # Project 2 · Build an eval task, then test agents on it
# **Local AI Lab** · Homework-style project based on the skills of CS329Z Homework 2 and the Terminal-Bench guest lecture · [Module 10](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w#m10)
#
# A good eval task has four parts, the same structure Terminal-Bench and Harbor use:
#
# ```
# contract-flags/
#   instruction.md        what the agent is asked to do (short, natural, fully specified)
#   environment/          the files the agent starts with
#   tests/test.py         HIDDEN verifier: runs after the agent, writes a reward from 0 to 1
#   solution/solve.py     reference (oracle) solution: must score 1.0
# ```
#
# Harbor runs each task in a Docker container. Colab can't run Docker, so we use a **fresh temporary folder per attempt** and run the verifier as a separate step the agent never sees. The ideas are the same.
#
# You will:
# 1. inspect the task;
# 2. **test the task itself**: the oracle scores 1, doing nothing scores 0, a different correct solution passes, a shortcut fails;
# 3. run our local agent 5 times and report pass@k and pass^k;
# 4. build your own task.

# %%
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
from lal.checks import pass_at_k, pass_hat_k, check

TASK = Path("projects/p2_tasks/contract-flags")
print(TASK.joinpath("instruction.md").read_text())
print("environment:", sorted(p.name for p in (TASK / "environment").iterdir()))

# %% [markdown]
# ## 1. The runner: fresh workspace → attempt → hidden verifier → reward

# %%
def run_attempt(task: Path, attempt_fn) -> float:
    ws = Path(tempfile.mkdtemp(prefix="attempt_"))
    shutil.copytree(task / "environment", ws, dirs_exist_ok=True)      # fresh copy every time; tests are NOT copied
    try:
        attempt_fn(ws, (task / "instruction.md").read_text())
    except Exception as e:
        print("attempt crashed:", type(e).__name__, e)
    reward_file = ws / ".reward"
    out = subprocess.run([sys.executable, str(task / "tests" / "test.py"), str(ws), str(reward_file)],
                         capture_output=True, text=True, timeout=120)
    print("  verifier:", out.stdout.strip())
    reward = float(reward_file.read_text()) if reward_file.exists() else 0.0
    shutil.rmtree(ws, ignore_errors=True)
    return reward

# %% [markdown]
# ## 2. Test the task before testing any agent
# A broken task gives meaningless scores. These four checks come straight from the course's benchmark checklist.

# %%
def oracle(ws, instruction):
    subprocess.run([sys.executable, str(TASK / "solution" / "solve.py"), str(ws)], check=True)

def do_nothing(ws, instruction):
    pass

def alternative(ws, instruction):
    """A different correct approach: parse numbers with a different method. Must also pass."""
    import re
    pb = json.loads((ws / "playbook.json").read_text())
    rep = {}
    for f in ws.glob("msa_*.txt"):
        t = f.read_text().lower(); out = []
        nums = lambda pat: [int(x) for x in re.findall(pat, t)]
        cap = nums(r"\((\d+)\) months preceding") or nums(r"in the (\d+) months preceding")
        if not cap or cap[0] > 12: out.append("liability_cap")
        if not any(f"state of {s.lower()}" in t for s in pb["allowed_law"]): out.append("governing_law")
        if nums(r"successive \w+ \((\d+)\) year")[0] > 1: out.append("auto_renewal")
        d = nums(r"at least \w+ \((\d+)\) days") or nums(r"at least (\d+) days")
        if d[0] < 30: out.append("termination_notice")
        rep[f.stem] = sorted(out)
    (ws / "report.json").write_text(json.dumps(rep))

def shortcut(ws, instruction):
    """Gaming attempt: flag every rule for every contract."""
    pb = json.loads((ws / "playbook.json").read_text())
    (ws / "report.json").write_text(json.dumps({f.stem: sorted(pb["rules"]) for f in ws.glob("msa_*.txt")}))

scores = {name: run_attempt(TASK, fn) for name, fn in
          [("oracle", oracle), ("do nothing", do_nothing), ("alternative correct", alternative), ("flag everything", shortcut)]}
print(scores)

# %%
check("oracle scores 1.0", scores["oracle"] == 1.0)
check("doing nothing scores 0", scores["do nothing"] == 0.0)
check("a different correct solution also scores 1.0", scores["alternative correct"] == 1.0)
check("the flag-everything shortcut scores low", scores["flag everything"] < 0.5)

# %% [markdown]
# ## 3. Run our local agent on the task, 5 times
# The agent uses the harness from Project 1 with three tools limited to its workspace: list files, read a file, write a file. It never sees the tests.

# %%
from lal import harness_solution as S
from lal.harness_solution import ToolSpec, Config
from lal.llm import chat

def agent_attempt(ws: Path, instruction: str, temperature=0.3):
    def safe(name):                                   # keep the agent inside its workspace
        p = (ws / name).resolve()
        if ws.resolve() not in p.parents and p != ws.resolve():
            raise PermissionError("outside workspace")
        return p
    tools = S.Registry([
        ToolSpec("list_files", "List files in the workspace.", {"type": "object", "properties": {}}, lambda: "\n".join(sorted(x.name for x in ws.iterdir()))),
        ToolSpec("read_file", "Read a text file from the workspace.", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                 lambda name: safe(name).read_text()),
        ToolSpec("write_file", "Write a text file in the workspace.", {"type": "object", "properties": {"name": {"type": "string"}, "content": {"type": "string"}},
                 "required": ["name", "content"]}, lambda name, content: (safe(name).write_text(content), "written")[1], read_only=False),
    ], approve=lambda n, a: True, max_chars=4000)
    lm = lambda messages: chat(messages, temperature=temperature).content or ""
    agent = S.Agent(lm, tools, Config(max_turns=16, token_budget=40_000, compact_at_chars=30_000))
    r = agent.send(instruction)
    print(f"  agent stopped: {r.stop_reason} after {r.turns} turns")

K = 5
rewards = []
for i in range(K):
    print(f"attempt {i+1}")
    rewards.append(run_attempt(TASK, agent_attempt))

# %% [markdown]
# ## 4. Report it like a benchmark
# - **Mean reward:** partial credit across contracts.
# - **pass@k:** at least one of k attempts fully solved it.
# - **pass^k:** every attempt fully solved it. For repeated business work, this is the number that matters.

# %%
solved = sum(r == 1.0 for r in rewards)
print(f"rewards: {rewards}")
print(f"mean reward {sum(rewards)/K:.2f} | pass@1 {pass_at_k(K, solved, 1):.0%} | pass@{K} {pass_at_k(K, solved, K):.0%} | "
      f"pass^2 {pass_hat_k(K, solved, 2):.0%} | pass^{K} {pass_hat_k(K, solved, K):.0%}")

# %%
check("agent ran 5 attempts through the hidden verifier", len(rewards) == K)

# %% [markdown]
# ## 5. Your turn: build your own task
# Make a folder `projects/p2_tasks/my-task/` with the same four parts. Ideas:
# - **NDA check:** is the confidentiality period at least 3 years? Is there a non-solicit clause?
# - **Invoice vs. contract:** do the monthly fees on 3 invoices match clause 2.1?
# - **Redline:** rewrite clause 7.1 so the cap is 12 months; the verifier checks the new text with code.
#
# Then prove your task is sound with the same four checks (oracle 1.0, do-nothing 0, alternative correct solution 1.0, shortcut low), and run the agent 5 times.
#
# **Writing tips from the Terminal-Bench lecture:** keep the instruction short and complete; each test should match one requirement in the instruction; test the final result, not the steps; keep the tests where the agent can't read or edit them.
