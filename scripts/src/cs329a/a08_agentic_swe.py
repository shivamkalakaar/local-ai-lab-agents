# %% [markdown]
# # a08 · Agents for software engineering: many fixes, test-based selection, and faster code
# **Local AI Lab** · CS329A Lecture 13 (Nov 3) · CodeMonkeys-style sample-and-select for bug fixes, and KernelBench-style "make it faster" with a profiler interface
#
# Software is a great place for self-improving agents because the computer can check the work: tests pass or fail, and code runs faster or slower. We build two small loops. **Part A:** sample several bug fixes in parallel, let the model write its own tests, and pick a fix by test votes plus a model judge (CodeMonkeys). **Part B:** ask the model to speed up slow NumPy code, check it on random inputs, and measure `fast_p` (KernelBench). Then we give the optimizer a profiler and see if better feedback helps (agent-system interfaces).

# %%
import cProfile, difflib, json, os, re, shutil, subprocess, sys, tempfile, textwrap
from concurrent.futures import ThreadPoolExecutor
from lal.llm import ask, chat
from lal.checks import pass_at_k, check

WORK = tempfile.mkdtemp(prefix="a08_")

def extract_code(text):
    """Take the longest ```python block from a reply, or the whole reply if there is none."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.S)
    return max(blocks, key=len) if blocks else (text or "")

# %% [markdown]
# ## Part A · CodeMonkeys at small scale
# CodeMonkeys (Ehrlich, Brown, Juravsky et al. 2025, *CodeMonkeys: Scaling Test-Time Compute for Software Engineering*) solves real GitHub issues in three steps:
# 1. **Context:** find the files that matter.
# 2. **Generation:** run many attempts in parallel. Each attempt writes an edit **and** a test script.
# 3. **Selection:** run every model-written test on every edit, keep the top 3 edits by votes, and let a model pick among them.
#
# Two numbers matter. **Coverage:** did *any* attempt fix the bug? **Score:** did the one we *picked* fix it? The gap between them is the selection problem.
#
# ### A1. Four tiny buggy repos
# Each has one file, an issue written like a bug report, and a **hidden** test (like SWE-bench's tests). The model never sees the hidden test. We also keep a reference fix, only to prove each task is sound.

# %%
REPOS = [
    dict(name="textstats", file="textstats.py",
         code='''def word_count(text):
    """Number of words. Words are separated by any whitespace."""
    return len(text.split(" "))


def char_count(text, ignore_spaces=False):
    if ignore_spaces:
        return len(text.replace(" ", ""))
    return len(text)
''',
         issue="word_count gives wrong numbers. word_count('hello  world') returns 3 and word_count('a\\nb') returns 1. "
               "Both should be 2. Also word_count('') should be 0.",
         fix=("return len(text.split(\" \"))", "return len(text.split())"),
         test='''from textstats import word_count, char_count
assert word_count("hello  world") == 2
assert word_count("a\\nb") == 2
assert word_count("") == 0
assert word_count("  lead and trail  ") == 3
assert word_count("one") == 1
assert char_count("a b") == 3 and char_count("a b", ignore_spaces=True) == 2
'''),
    dict(name="late_fee", file="late_fee.py",
         code='''GRACE_DAYS = 5
DAILY_RATE = 0.01
MAX_FRACTION = 0.10


def late_fee(amount, days_late):
    """Fee for paying an invoice late.
    No fee within the grace period (up to and including GRACE_DAYS days late).
    After that, DAILY_RATE of the amount for each day beyond the grace period,
    capped at MAX_FRACTION of the amount. Rounded to cents."""
    if days_late < GRACE_DAYS:
        return 0.0
    fee = amount * DAILY_RATE * days_late
    return round(fee, 2)
''',
         issue="late_fee(1000, 5) returns 50.0, but 5 days late is still inside the grace period, so it should be 0.0. "
               "late_fee(1000, 8) returns 80.0 but should be 30.0 (only the 3 days after the grace period count). "
               "And late_fee(1000, 40) returns 400.0 even though the docstring says the fee is capped at 10% of the amount.",
         fix=("    if days_late < GRACE_DAYS:\n        return 0.0\n    fee = amount * DAILY_RATE * days_late\n",
              "    if days_late <= GRACE_DAYS:\n        return 0.0\n    fee = min(amount * DAILY_RATE * (days_late - GRACE_DAYS), amount * MAX_FRACTION)\n"),
         test='''from late_fee import late_fee
close = lambda a, b: abs(a - b) < 1e-6
assert close(late_fee(1000, 5), 0.0)
assert close(late_fee(1000, 0), 0.0)
assert close(late_fee(1000, 6), 10.0)
assert close(late_fee(1000, 8), 30.0)
assert close(late_fee(1000, 40), 100.0)
assert close(late_fee(200, 7), 4.0)
'''),
    dict(name="dates_util", file="dates_util.py",
         code='''from datetime import date


def add_months(d, n):
    """Return the date n months after d (n may be negative).
    If that day does not exist in the target month, use the last day of that month."""
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return date(year, month, d.day)
''',
         issue="add_months(date(2027, 1, 31), 1) crashes with 'ValueError: day is out of range for month'. "
               "It should return date(2027, 2, 28). In a leap year, add_months(date(2028, 1, 31), 1) should return date(2028, 2, 29).",
         fix=("    return date(year, month, d.day)\n",
              "    import calendar\n    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))\n"),
         test='''from datetime import date
from dates_util import add_months
assert add_months(date(2027, 1, 31), 1) == date(2027, 2, 28)
assert add_months(date(2028, 1, 31), 1) == date(2028, 2, 29)
assert add_months(date(2027, 5, 31), 1) == date(2027, 6, 30)
assert add_months(date(2027, 12, 31), 2) == date(2028, 2, 29)
assert add_months(date(2027, 3, 15), -2) == date(2027, 1, 15)
assert add_months(date(2027, 1, 10), 12) == date(2028, 1, 10)
'''),
    dict(name="lru", file="lru.py",
         code='''from collections import OrderedDict


class LRUCache:
    """Keeps at most `capacity` items. When full, it evicts the least recently USED item.
    Reading a key with get() counts as using it."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
''',
         issue="Keys I just read still get evicted. c = LRUCache(2); c.put('a', 1); c.put('b', 2); c.get('a'); c.put('c', 3). "
               "Now 'a' is gone, but 'b' is the least recently used key and should be the one evicted.",
         fix=("        return self.data.get(key, default)\n",
              "        if key not in self.data:\n            return default\n        self.data.move_to_end(key)\n        return self.data[key]\n"),
         test='''from lru import LRUCache
c = LRUCache(2)
c.put("a", 1); c.put("b", 2)
assert c.get("a") == 1
c.put("c", 3)
assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
assert c.get("zzz", "missing") == "missing"
d = LRUCache(1); d.put("x", 1); d.put("y", 2)
assert d.get("x") is None and d.get("y") == 2
e = LRUCache(2); e.put("a", 1); e.put("b", 2); e.put("a", 10); e.put("c", 3)
assert e.get("a") == 10 and e.get("b") is None
'''),
]
for r in REPOS:
    assert r["fix"][0] in r["code"], r["name"]
    r["fixed"] = r["code"].replace(*r["fix"])
print(REPOS[0]["issue"])

# %% [markdown]
# ### A2. Run a test against a version of the file, safely
# Every run gets a **fresh folder** with the file and one test script, in a separate process with a timeout. Exit code 0 means pass.

# %%
def run_test(repo, file_code, test_code, timeout=10):
    d = tempfile.mkdtemp(dir=WORK)
    open(os.path.join(d, repo["file"]), "w").write(file_code)
    open(os.path.join(d, "test_run.py"), "w").write(test_code)
    try:
        out = subprocess.run([sys.executable, "test_run.py"], cwd=d, capture_output=True, text=True, timeout=timeout)
        ok, msg = out.returncode == 0, (out.stderr or out.stdout).strip()[-300:]
    except subprocess.TimeoutExpired:
        ok, msg = False, "TIMEOUT"
    shutil.rmtree(d, ignore_errors=True)
    return ok, msg

# Prove each task is sound: the hidden test FAILS on the buggy code and PASSES on the reference fix.
sound = {r["name"]: (not run_test(r, r["code"], r["test"])[0], run_test(r, r["fixed"], r["test"])[0]) for r in REPOS}
print("(fails on bug, passes on fix):", sound)

# %% [markdown]
# ### A3. Generation: 4 edits per repo, in parallel
# Each attempt sees the issue and the file and returns the whole fixed file. 16 calls, 2 at a time.

# %%
N_EDITS = 4

def edit_prompt(r):
    return (f"You are fixing a bug in a small Python repository.\n\nIssue:\n{r['issue']}\n\nFile `{r['file']}`:\n```python\n{r['code']}```\n\n"
            "Rewrite the file so the issue is fixed. Keep the same function and class names. "
            "Return the complete fixed file in one ```python block.")

def gen_edit(r):
    return extract_code(ask(edit_prompt(r), temperature=0.8, max_tokens=900))

jobs = [ri for ri in range(len(REPOS)) for _ in range(N_EDITS)]
with ThreadPoolExecutor(2) as ex:
    edits = list(ex.map(lambda ri: gen_edit(REPOS[ri]), jobs))
EDITS = {ri: [e for j, e in zip(jobs, edits) if j == ri] for ri in range(len(REPOS))}

# Ground truth, for grading only: does each edit pass the hidden test?
HIDDEN = {ri: [run_test(REPOS[ri], e, REPOS[ri]["test"])[0] for e in EDITS[ri]] for ri in EDITS}
for ri, oks in HIDDEN.items():
    print(f"{REPOS[ri]['name']:11} edits passing the hidden test: {oks}")

# %% [markdown]
# ### A4. Coverage grows with the number of samples
# Coverage@k is the chance that at least one of k attempts is correct. We estimate it with the unbiased pass@k formula. Large Language Monkeys (Brown et al. 2024) found coverage often grows roughly log-linearly with samples; CodeMonkeys reached 69.8% coverage on SWE-bench Verified.

# %%
COVERAGE = {}
for k in range(1, N_EDITS + 1):
    COVERAGE[k] = sum(pass_at_k(N_EDITS, sum(oks), k) for oks in HIDDEN.values()) / len(HIDDEN)
    print(f"coverage@{k}: {COVERAGE[k]:.0%}  " + "#" * round(40 * COVERAGE[k]))

# %% [markdown]
# ### A5. The model writes its own tests
# We ask for 3 short test scripts per repo (12 calls). A useful test must **fail on the buggy code**; otherwise it doesn't reproduce the issue. We drop tests that pass on the original file (our own filter, in the spirit of CodeMonkeys running tests before and after an edit).

# %%
N_TESTS = 3

def test_prompt(r):
    module = r["file"][:-3]
    return (f"Issue:\n{r['issue']}\n\nFile `{r['file']}`:\n```python\n{r['code']}```\n\n"
            f"Write a short Python test script that does `from {module} import ...` and uses plain assert statements to check "
            "the behavior the issue asks for. It must FAIL on the current buggy code and PASS once the bug is fixed. "
            "No test framework, no explanation. Return only code in one ```python block.")

tjobs = [ri for ri in range(len(REPOS)) for _ in range(N_TESTS)]
with ThreadPoolExecutor(2) as ex:
    tests = list(ex.map(lambda ri: extract_code(ask(test_prompt(REPOS[ri]), temperature=0.8, max_tokens=500)), tjobs))
GEN_TESTS = {}
for ri in range(len(REPOS)):
    mine = [t for j, t in zip(tjobs, tests) if j == ri]
    GEN_TESTS[ri] = [t for t in mine if not run_test(REPOS[ri], REPOS[ri]["code"], t)[0]]   # keep tests that fail on the bug
    print(f"{REPOS[ri]['name']:11} model tests kept: {len(GEN_TESTS[ri])}/{len(mine)}")

# %% [markdown]
# ### A6. Selection: test votes, then a model judge
# - **Votes:** each edit scores one point per model test it passes.
# - **Top-3 + judge:** keep the 3 best by votes and show their diffs to the model, which picks one. CodeMonkeys found that the model judge did worse than plain voting *without* the top-3 filter, and better *with* it.

# %%
def votes(ri):
    return [sum(run_test(REPOS[ri], e, t)[0] for t in GEN_TESTS[ri]) for e in EDITS[ri]]

def diff(a, b):
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "before", "after", n=1))[:1500]

def judge(ri, idxs):
    r = REPOS[ri]
    cands = "\n\n".join(f"Candidate {n + 1}:\n```diff\n{diff(r['code'], EDITS[ri][i])}```" for n, i in enumerate(idxs))
    reply = ask(f"Issue:\n{r['issue']}\n\nThese are candidate fixes as diffs:\n\n{cands}\n\n"
                f"Which candidate fixes the issue correctly without breaking anything else? Reply with only the number (1-{len(idxs)}).",
                max_tokens=20)
    m = re.search(r"\d+", reply or "")
    n = int(m.group(0)) - 1 if m else 0
    return idxs[n] if 0 <= n < len(idxs) else idxs[0]

SEL = {}
for ri in range(len(REPOS)):
    v = votes(ri)
    order = sorted(range(len(v)), key=lambda i: -v[i])          # stable: ties keep the sampling order
    top3 = order[:3]
    distinct = len({EDITS[ri][i] for i in top3}) > 1
    SEL[ri] = {"votes": order[0], "votes + judge": judge(ri, top3) if distinct else order[0]}
    print(f"{REPOS[ri]['name']:11} votes {v} | vote pick {order[0]} | judge pick {SEL[ri]['votes + judge']}")

scores = {"random": sum(sum(o) / len(o) for o in HIDDEN.values()) / len(HIDDEN)}
for m in ("votes", "votes + judge"):
    scores[m] = sum(HIDDEN[ri][SEL[ri][m]] for ri in SEL) / len(SEL)
scores["oracle (coverage)"] = COVERAGE[N_EDITS]
for m, s in scores.items():
    print(f"{m:18} {s:.0%}")

# %% [markdown]
# ## Part B · KernelBench at CPU scale
# KernelBench (Ouyang, Guo et al. 2025, *KernelBench: Can LLMs Write Efficient GPU Kernels?*) asks a model to rewrite a PyTorch program as faster GPU code. It checks the output against the reference on 5 random inputs and times both. Its metric is
#
# $$\text{fast}_p = \frac{1}{N}\sum_i \mathbf{1}[\text{correct}_i \wedge \text{speedup}_i > p]$$
#
# so `fast_0` is just the correctness rate, and `fast_1` means "correct **and** faster than the reference".
#
# We have no GPU budget, so we do the same thing with slow Python/NumPy functions on the CPU.
#
# ### B1. Four slow functions

# %%
KTASKS = [
    dict(fn="pairwise_dist", ref='''import numpy as np

def pairwise_dist(X, Y):
    """Euclidean distance between every row of X (n, d) and every row of Y (m, d). Returns (n, m)."""
    n, m = X.shape[0], Y.shape[0]
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    return out
''', gen='''def make_inputs(rng, big, case):
    n, m, d = (150, 150, 8) if big else (int(rng.integers(1, 30)), int(rng.integers(1, 30)), int(rng.integers(1, 6)))
    return (rng.normal(size=(n, d)), rng.normal(size=(m, d)))
'''),
    dict(fn="moving_average", ref='''import numpy as np

def moving_average(x, w):
    """Mean of every window of length w in the 1-D array x. Returns len(x) - w + 1 values."""
    out = np.empty(len(x) - w + 1)
    for i in range(len(out)):
        out[i] = x[i:i + w].mean()
    return out
''', gen='''def make_inputs(rng, big, case):
    n = 40000 if big else int(rng.integers(1, 200))
    w = 25 if big else [1, n, int(rng.integers(1, n + 1)), int(rng.integers(1, n + 1)), 1][case]
    return (rng.normal(size=n) * 10, w)
'''),
    dict(fn="softmax_rows", ref='''import math
import numpy as np

def softmax_rows(M):
    """Softmax of each row of the 2-D array M."""
    out = np.empty(M.shape, dtype=float)
    for i in range(M.shape[0]):
        row = M[i]
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        for j in range(len(row)):
            out[i, j] = e[j] / s
    return out
''', gen='''def make_inputs(rng, big, case):
    r, c = (500, 300) if big else (int(rng.integers(1, 20)), int(rng.integers(1, 30)))
    scale = [1, 5, 50, 400, 900][case % 5]          # large values break a softmax that skips the max-subtraction
    return (rng.normal(size=(r, c)) * scale,)
'''),
    dict(fn="user_scores", ref='''import numpy as np

def clean(amounts):
    return np.clip(amounts, 0, None)

def per_user_totals(user_ids, amounts, n_users):
    totals = np.zeros(n_users)
    for u, x in zip(user_ids, amounts):
        totals[u] += x
    return totals

def normalize(totals):
    return totals / (totals.max() + 1e-9)

def user_scores(user_ids, amounts, n_users):
    """Total positive spend per user, scaled so the top user is about 1."""
    return normalize(per_user_totals(user_ids, clean(amounts), n_users))
''', gen='''def make_inputs(rng, big, case):
    n_users = 1000 if big else int(rng.integers(1, 20))
    n = 200000 if big else int(rng.integers(1, 300))
    return (rng.integers(0, n_users, size=n), rng.normal(size=n) * 50, n_users)
'''),
]
print([t["fn"] for t in KTASKS])

# %% [markdown]
# ### B2. The evaluator: correctness on 5 random inputs, then timing
# All of this runs in a subprocess with a timeout. Correctness uses `np.allclose` (relative 1e-5, absolute 1e-6) and requires the same shape. Timing takes the best of 3 runs on one large input. The profiler mode runs `cProfile` on the large input and returns the functions that used the most time.

# %%
KB_RUNNER = r'''
import copy, cProfile, json, os, pstats, sys, time, traceback
import numpy as np
mode, fn, ref_path, cand_path, gen_path = sys.argv[1:6]
def load(path):
    ns = {}
    exec(compile(open(path).read(), os.path.basename(path), "exec"), ns)
    return ns
gen = load(gen_path)["make_inputs"]
ref = load(ref_path)[fn]
res = {"correct": False, "speedup": 0.0, "error": None, "detail": None}
try:
    cand = load(cand_path)[fn]
    if mode == "profile":
        args = gen(np.random.default_rng(123), True, 0)
        prof = cProfile.Profile(); prof.enable(); cand(*args); prof.disable()
        st = pstats.Stats(prof).stats
        total = sum(v[2] for v in st.values()) or 1e-9
        rows = sorted(((k[2] if k[0] != "~" else "builtin " + k[2], os.path.basename(k[0]), v[1], v[2], v[3])
                       for k, v in st.items() if "disable" not in k[2]),
                      key=lambda r: -r[3])[:8]
        print(json.dumps({"total": total, "rows": rows})); sys.exit(0)
    for case in range(5):
        args = gen(np.random.default_rng(case), False, case)
        want, got = ref(*copy.deepcopy(args)), np.asarray(cand(*copy.deepcopy(args)), dtype=float)
        want = np.asarray(want, dtype=float)
        if got.shape != want.shape:
            res["detail"] = f"case {case}: shape {got.shape}, expected {want.shape}"; break
        if not np.allclose(got, want, rtol=1e-5, atol=1e-6):
            bad = ~np.isfinite(got)
            res["detail"] = (f"case {case}: {int(bad.sum())} non-finite values (inf/nan)" if bad.any() else
                             f"case {case}: max abs error {float(np.max(np.abs(got - want))):.3g}")
            break
    else:
        args = gen(np.random.default_rng(123), True, 0)
        def best(f):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); out = f(*copy.deepcopy(args)); ts.append(time.perf_counter() - t0)
            return min(ts), out
        t_ref, want = best(ref)
        t_new, got = best(cand)
        if np.asarray(got).shape == np.asarray(want).shape and np.allclose(got, want, rtol=1e-5, atol=1e-6):
            res.update(correct=True, speedup=t_ref / max(t_new, 1e-9), t_ref=t_ref, t_new=t_new)
        else:
            res["detail"] = "wrong values on the large input"
except Exception as e:
    res["error"] = traceback.format_exc().strip().splitlines()[-1][:300]
print(json.dumps(res))
'''
RUNNER_PATH = os.path.join(WORK, "kb_runner.py")
open(RUNNER_PATH, "w").write(KB_RUNNER)

def kb_run(task, cand_code, mode="eval", timeout=90):
    d = tempfile.mkdtemp(dir=WORK)
    paths = {}
    for name, src in (("ref", task["ref"]), ("cand", cand_code), ("gen", task["gen"])):
        paths[name] = os.path.join(d, f"{name}.py")
        open(paths[name], "w").write(src)
    out = None
    try:
        out = subprocess.run([sys.executable, "-I", RUNNER_PATH, mode, task["fn"], paths["ref"], paths["cand"], paths["gen"]],
                             capture_output=True, text=True, timeout=timeout)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"correct": False, "speedup": 0.0, "error": f"timed out after {timeout}s", "detail": None}
    except Exception:
        return {"correct": False, "speedup": 0.0, "error": "runner failed: " + ((out.stderr or "")[-200:] if out else ""), "detail": None}
    finally:
        shutil.rmtree(d, ignore_errors=True)

def fast_p(results, p):
    return sum(r["correct"] and r["speedup"] > p for r in results) / len(results)

# Test the evaluator before trusting it: the reference vs itself, a known-wrong version, a known-fast version.
self_check = kb_run(KTASKS[0], KTASKS[0]["ref"])
naive_softmax = "import numpy as np\ndef softmax_rows(M):\n    e = np.exp(M)\n    return e / e.sum(axis=1, keepdims=True)\n"
wrong_check = kb_run(KTASKS[2], naive_softmax)
fast_users = ("import numpy as np\ndef user_scores(user_ids, amounts, n_users):\n"
              "    t = np.bincount(user_ids, weights=np.clip(amounts, 0, None), minlength=n_users)\n    return t / (t.max() + 1e-9)\n")
fast_check = kb_run(KTASKS[3], fast_users)
print("reference vs itself:", self_check)
print("softmax without max-subtraction:", wrong_check)
print("bincount version of user_scores:", fast_check)

# %% [markdown]
# ### B3. The agent-system interface: scalar feedback vs. a profiler
# Wei, Nie et al. (2024/2025, *Improving Parallel Program Performance with LLM Optimizers via Agent-System Interfaces*) found that what the optimizer **sees** matters as much as the model. Their interface had two parts: a small language that limits what the agent can write, and **AutoGuide**, which turns raw system output into plain advice. With it, an LLM optimizer beat OpenTuner (which only gets a number) within 10 iterations.
#
# We compare two interfaces, 3 turns each (an attempt, then 2 rounds of feedback):
# - **Scalar:** the agent sees only "correct? speedup?", like a classic autotuner.
# - **Profiler + guide:** the agent also gets a `cProfile` table of the reference up front, and after each attempt a plain-language reading of the result: the error explained, which test case failed and how, and where its own code spends time.

# %%
def profile_text(task, code):
    p = kb_run(task, code, mode="profile")
    if "rows" not in p:
        return f"(profiler failed: {p.get('error')})", None
    lines = [f"{'function':40} {'calls':>8} {'own time s':>10} {'with callees s':>14}"]
    lines += [f"{f[:40]:40} {n:>8} {tt:>10.4f} {ct:>14.4f}" for f, _, n, tt, ct in p["rows"]]
    hot = p["rows"][0]
    return "\n".join(lines), (hot[0], hot[3] / p["total"], hot[2])

def guide(task, code, r):
    """A tiny AutoGuide: turn raw results into actionable advice."""
    if r.get("error"):
        return f"Your code crashed: {r['error']}. Fix this error first and keep the same function name."
    if not r["correct"]:
        return (f"Your code returns wrong results: {r['detail']}. Compare with the reference on that kind of input "
                "(edge cases, very large or very small values, window sizes, shapes).")
    table, hot = profile_text(task, code)
    tip = (f"Hot spot: `{hot[0]}` takes {hot[1]:.0%} of the time ({hot[2]} calls). If it loops over array elements in Python, "
           "replace the loop with whole-array NumPy operations." if hot else "")
    return f"Correct. Speedup {r['speedup']:.2f}x over the reference.\nProfile of your version:\n{table}\n{tip}"

def scalar_feedback(task, code, r):
    return f"Result: correct={r['correct']}, speedup={r['speedup']:.2f}x."

def optimize(task, interface, turns=3):
    intro = (f"Here is a slow Python function:\n```python\n{task['ref']}```\n"
             f"Write a faster version with the SAME function name `{task['fn']}` and the same arguments. Use NumPy. "
             "It must return the same values (within 1e-6) for any valid input. Return the complete code in one ```python block.")
    if interface == "profiler":
        intro += "\n\nProfiler output for the reference on a large input:\n" + profile_text(task, task["ref"])[0]
    msgs, best, history = [{"role": "user", "content": intro}], {"correct": False, "speedup": 0.0}, []
    for turn in range(turns):
        reply = chat(msgs, temperature=0.3, max_tokens=900).content or ""
        code = extract_code(reply)
        r = kb_run(task, code)
        if r["correct"] and r["speedup"] > best["speedup"]:
            best = r
        history.append(dict(best))                                # best-so-far after each turn
        if r["correct"] and r["speedup"] >= 20:                    # good enough: save calls
            history += [dict(best)] * (turns - turn - 1)
            break
        fb = guide(task, code, r) if interface == "profiler" else scalar_feedback(task, code, r)
        msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": fb + "\nWrite an improved version. Return the complete code in one ```python block."}]
    return history

RUNS = {}
for interface in ("scalar", "profiler"):
    RUNS[interface] = [optimize(t, interface) for t in KTASKS]     # up to 4 tasks x 3 turns = 12 calls per interface
    for t, h in zip(KTASKS, RUNS[interface]):
        print(f"{interface:8} {t['fn']:15} best speedup by turn: {[round(x['speedup'], 1) if x['correct'] else 'x' for x in h]}")

# %% [markdown]
# ### B4. fast_p, one-shot and after feedback

# %%
print(f"{'interface':10} {'turns':>5} {'fast_0':>7} {'fast_1':>7} {'fast_2':>7} {'fast_10':>8}")
FAST = {}
for interface, runs in RUNS.items():
    for n in (1, 3):
        res = [h[min(n, len(h)) - 1] for h in runs]
        FAST[(interface, n)] = {p: fast_p(res, p) for p in (0, 1, 2, 10)}
        print(f"{interface:10} {n:>5} " + " ".join(f"{FAST[(interface, n)][p]:>7.0%}" for p in (0, 1, 2)) + f" {FAST[(interface, n)][10]:>8.0%}")

# %% [markdown]
# What to notice: `fast_p` drops as `p` rises, the same shape as in KernelBench. Feedback turns usually help most by **fixing wrong answers** (KernelBench saw the same: execution feedback mostly fixed errors). Whether the profiler beats scalar feedback on 4 tasks is noisy; on `user_scores` the profiler points straight at `per_user_totals`, which is where the paper's "interpret the raw output" idea pays off.

# %%
check("each repo task is sound (hidden test fails on the bug, passes on the fix)", all(a and b for a, b in sound.values()))
check("evaluator: the reference matches itself", self_check["correct"])
check("evaluator: softmax without max-subtraction is caught as wrong", not wrong_check["correct"])
check("evaluator: a vectorized version is correct and faster", fast_check["correct"] and fast_check["speedup"] > 1)
check("at least one sampled edit fixes at least one repo (coverage > 0)", COVERAGE[N_EDITS] > 0)
check("test-vote selection is at least as good as picking at random", scores["votes"] >= scores["random"] - 1e-9)
check("the optimizer produced at least one correct, faster function", max(FAST[(i, 3)][1] for i in RUNS) > 0)

# %% [markdown]
# ## Your turn
# 1. **Serial compute.** CodeMonkeys let each attempt see test results and try again. Add one feedback round to `gen_edit`: run the attempt against its own model test, and if it fails, send the error back and ask for a new file. Does coverage@1 rise?
# 2. **Sharper fast_p.** Rerun B4 with `p = 5`. Which tasks still count? Why is a 1.0x threshold too easy for NumPy rewrites of Python loops (but hard on KernelBench, where the baseline is already optimized PyTorch)?
# 3. **A gaming check.** Write a "fast" `moving_average` that is only correct when `w == 25` (the size used for timing). Does our evaluator catch it? What does this say about checking on several random inputs?
#
# **Next:** a09

# %% [markdown]
# ## Solutions

# %%
# Solution 1: one round of execution feedback per attempt (up to 4 more calls; we try the first edit of each repo)
def gen_edit_with_feedback(ri, first):
    r = REPOS[ri]
    if not GEN_TESTS[ri]:
        return first
    ok, msg = run_test(r, first, GEN_TESTS[ri][0])
    if ok:
        return first
    reply = ask(edit_prompt(r) + f"\n\nYour previous attempt:\n```python\n{first}```\nfailed this test:\n```python\n{GEN_TESTS[ri][0]}```\n"
                f"Error:\n{msg}\nNote the test itself may be wrong. Return the complete fixed file in one ```python block.",
                temperature=0.3, max_tokens=900)
    return extract_code(reply)
before = sum(HIDDEN[ri][0] for ri in HIDDEN)
after = sum(run_test(REPOS[ri], gen_edit_with_feedback(ri, EDITS[ri][0]), REPOS[ri]["test"])[0] for ri in HIDDEN)
print(f"first attempt fixes {before}/{len(HIDDEN)} repos; after one feedback round: {after}/{len(HIDDEN)}")
# What to notice: model tests can be wrong, so feedback from them can also mislead. CodeMonkeys found that spending
# compute on more serial iterations or more parallel samples often gave similar coverage for the same cost.

# %%
# Solution 2: a higher speedup threshold (no new model calls)
for interface, runs in RUNS.items():
    res = [h[-1] for h in runs]
    print(interface, "fast_5:", f"{fast_p(res, 5):.0%}", [(t["fn"], round(r["speedup"], 1)) for t, r in zip(KTASKS, res) if r["correct"]])
# What to notice: replacing a Python loop with one NumPy call often gives 10-100x, so p=1 is easy here.
# KernelBench's baseline is PyTorch's own tuned kernels, so even p=1 is hard: one-shot, the best models matched PyTorch in under 20% of cases.

# %%
# Solution 3: a version that only works for the timing input's window size
cheat = ("import numpy as np\ndef moving_average(x, w):\n    k = np.ones(25) / 25\n    return np.convolve(x, k, mode='valid')\n")
print(kb_run(KTASKS[1], cheat))
# What to notice: it fails on the random small cases (other window sizes), so it scores as incorrect.
# If we had only checked the timing input, it would have looked perfect. Always check on varied random inputs.
