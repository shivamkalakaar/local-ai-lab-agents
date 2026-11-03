"""Small Python coding tasks with unit tests, plus a safe way to run model-written code.

Pure Python, no model calls. Hidden tests are generated with a fixed seed from the reference
solutions, so every run gets the same tasks and tests.

    from lal.codetasks import tasks, run_tests, extract_code, run_python
    t = tasks()[0]
    passed, total, feedback = run_tests(t["reference"], t)               # hidden tests
    passed, total, feedback = run_tests(code, t, public=True)            # only the public examples

Each task is a dict:
    id           short name
    prompt       what the model sees (signature, rules and the public examples)
    entry_point  name of the function to call
    public_tests [(args_tuple, expected), ...]  shown in the prompt; use them for feedback
    tests        [(args_tuple, expected), ...]  hidden tests (include the public ones); use them to grade
    reference    a correct solution (source code string)

Model code ALWAYS runs in a separate Python process with a timeout, never in this process.
"""
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SEED = 0

# ---------------------------------------------------------------- reference solutions
_REF = {}

_REF["is_palindrome"] = '''
def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
'''

_REF["second_largest"] = '''
def second_largest(nums):
    d = sorted(set(nums), reverse=True)
    return d[1] if len(d) >= 2 else None
'''

_REF["run_length_encode"] = '''
def run_length_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)
'''

_REF["merge_intervals"] = '''
def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
'''

_REF["roman_to_int"] = '''
def roman_to_int(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and v[c] < v[s[i + 1]]:
            total -= v[c]
        else:
            total += v[c]
    return total
'''

_REF["word_frequencies"] = '''
import re
def word_frequencies(text):
    out = {}
    for w in re.findall(r"[a-z]+", text.lower()):
        out[w] = out.get(w, 0) + 1
    return out
'''

_REF["rotate_matrix"] = '''
def rotate_matrix(m):
    return [list(row) for row in zip(*m[::-1])]
'''

_REF["days_in_month"] = '''
def days_in_month(year, month):
    if month == 2:
        leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31
'''

_REF["parse_duration"] = '''
import re
def parse_duration(s):
    m = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", s)
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + se
'''

_REF["format_money"] = '''
def format_money(cents):
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}${c // 100:,}.{c % 100:02d}"
'''

# ---------------------------------------------------------------- prompts and public examples
_SPEC = {
    "is_palindrome": ("is_palindrome(s: str) -> bool",
        "Return True if s reads the same forwards and backwards, ignoring case and ignoring every "
        "character that is not a letter or digit.",
        [(("A man, a plan, a canal: Panama",), True), (("hello",), False)]),
    "second_largest": ("second_largest(nums: list[int]) -> int | None",
        "Return the second largest DISTINCT value. Return None if there are fewer than two distinct values.",
        [(([3, 1, 4, 1, 5],), 4), (([7, 7],), None)]),
    "run_length_encode": ("run_length_encode(s: str) -> str",
        "Encode runs of the same character as the character followed by the run length. "
        "The empty string encodes to the empty string.",
        [(("aaabcc",), "a3b1c2"), (("",), "")]),
    "merge_intervals": ("merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
        "Merge overlapping intervals [start, end] and return them sorted by start. Intervals that "
        "touch (one ends where the next starts) also merge. The input may be unsorted.",
        [(([[1, 3], [2, 6], [8, 10]],), [[1, 6], [8, 10]]), (([[5, 6], [1, 5]],), [[1, 6]])]),
    "roman_to_int": ("roman_to_int(s: str) -> int",
        "Convert a valid Roman numeral (I, V, X, L, C, D, M, with subtractive forms like IV and CM) to an integer.",
        [(("XIV",), 14), (("MCMXCIV",), 1994)]),
    "word_frequencies": ("word_frequencies(text: str) -> dict[str, int]",
        "Count words. Lowercase the text first. A word is a maximal run of letters a-z; everything "
        "else (digits, punctuation, apostrophes, spaces) separates words.",
        [(("The cat and THE hat.",), {"the": 2, "cat": 1, "and": 1, "hat": 1}), (("don't",), {"don": 1, "t": 1})]),
    "rotate_matrix": ("rotate_matrix(m: list[list[int]]) -> list[list[int]]",
        "Return a NEW square matrix rotated 90 degrees clockwise. Do not change the input.",
        [(([[1, 2], [3, 4]],), [[3, 1], [4, 2]]), (([[1]],), [[1]])]),
    "days_in_month": ("days_in_month(year: int, month: int) -> int",
        "Return the number of days in the month (month is 1-12) using Gregorian leap-year rules.",
        [((2024, 2), 29), ((2023, 4), 30)]),
    "parse_duration": ("parse_duration(s: str) -> int",
        "Parse a duration like '1h30m', '45s' or '2h5s' into total seconds. Each of h, m, s appears "
        "at most once, in that order, and any of them may be missing. The empty string is 0.",
        [(("1h30m",), 5400), (("45s",), 45)]),
    "format_money": ("format_money(cents: int) -> str",
        "Format an amount in cents as dollars with a $ sign, comma thousands separators and exactly "
        "two decimals. Negative amounts start with '-$'.",
        [((123456,), "$1,234.56"), ((-5,), "-$0.05")]),
}


def _hidden_inputs(tid, rng):
    """Extra test inputs per task, drawn with a fixed seed. Expected values come from the reference."""
    letters = "abc"
    if tid == "is_palindrome":
        return [("",), ("Was it a car or a cat I saw?",), ("No 'x' in Nixon",), ("ab",), ("12321",), ("1a2",),
                ("Aa",), ("race a car",)]
    if tid == "second_largest":
        return [([5],), ([],), ([2, 1],), ([-1, -5, -3],), ([4, 4, 3, 3],)] + \
               [([rng.randint(-9, 9) for _ in range(rng.randint(1, 7))],) for _ in range(4)]
    if tid == "run_length_encode":
        rand = ["".join(rng.choice(letters) * rng.randint(1, 3) for _ in range(rng.randint(1, 5))) for _ in range(4)]
        return [("a",), ("abc",), ("zzzzzzzzzzzz",)] + [(r,) for r in rand]
    if tid == "merge_intervals":
        out = [([],), ([[1, 2], [2, 3]],), ([[1, 10], [2, 3]],), ([[4, 5]],)]
        for _ in range(4):
            iv = []
            for _ in range(rng.randint(2, 5)):
                a = rng.randint(0, 15)
                iv.append([a, a + rng.randint(0, 4)])
            out.append((iv,))
        return out
    if tid == "roman_to_int":
        return [("I",), ("IV",), ("IX",), ("XL",), ("CD",), ("MMXXV",), ("LVIII",), ("CMXCIX",)]
    if tid == "word_frequencies":
        return [("",), ("one two  two, three-three-three",), ("A1b2C3",), ("Hello hello HELLO!",)]
    if tid == "rotate_matrix":
        n3 = [[rng.randint(0, 9) for _ in range(3)] for _ in range(3)]
        n4 = [[rng.randint(0, 9) for _ in range(4)] for _ in range(4)]
        return [([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), (n3,), (n4,)]
    if tid == "days_in_month":
        return [(1900, 2), (2000, 2), (2023, 2), (2024, 1), (2024, 12), (2023, 11), (2100, 2), (2024, 9)]
    if tid == "parse_duration":
        return [("",), ("2h5s",), ("10m",), ("1h",), ("3h2m1s",), ("0s",)]
    if tid == "format_money":
        return [(0,), (99,), (100,), (100000000,), (-123456,), (1000,)]
    raise KeyError(tid)


def _reference_fn(tid):
    ns = {}
    exec(_REF[tid], ns)          # our own trusted reference code, not model output
    return ns[tid]


def tasks() -> list[dict]:
    """The 10 tasks, same every call (fixed seed)."""
    rng = random.Random(SEED)
    out = []
    for tid, (sig, rule, public) in _SPEC.items():
        fn = _reference_fn(tid)
        hidden = []
        for args in _hidden_inputs(tid, rng):
            args = args if isinstance(args, tuple) else (args,)
            hidden.append((args, fn(*json.loads(json.dumps(args)))))
        examples = "\n".join(f"    {tid}({', '.join(map(repr, a))}) == {e!r}" for a, e in public)
        prompt = (f"Write a Python function `{sig}`.\n{rule}\nExamples:\n{examples}\n"
                  "Use only the Python standard library.")
        out.append({"id": tid, "prompt": prompt, "entry_point": tid, "public_tests": list(public),
                    "tests": list(public) + hidden, "reference": _REF[tid].strip() + "\n"})
    return out


def task(tid: str) -> dict:
    return next(t for t in tasks() if t["id"] == tid)


# ---------------------------------------------------------------- running code safely
def extract_code(text: str) -> str:
    """Pull Python code out of a model reply: the last ```python block, else any ``` block, else the text."""
    text = text or ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def run_python(code: str, timeout: float = 5) -> tuple[str, str]:
    """Run code in a fresh Python process. Returns (stdout, stderr). Timeouts come back as stderr."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "snippet.py"
        path.write_text(code)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], capture_output=True, text=True,
                               timeout=timeout, cwd=d)
            return r.stdout[-4000:], r.stderr[-2000:]
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT after {timeout}s"


_HARNESS = '''
import json, copy
__results = []
for __args, __exp in __TESTS:
    try:
        __got = {fn}(*copy.deepcopy(__args))
        __ok = __got == __exp
        __results.append([__ok, repr(__args)[:120], repr(__exp)[:80], repr(__got)[:80], None])
    except Exception as __e:
        __results.append([False, repr(__args)[:120], repr(__exp)[:80], None, type(__e).__name__ + ": " + str(__e)[:120]])
print("__RESULTS__" + json.dumps(__results))
'''


def run_tests(code: str, task: dict, timeout: float = 5, public: bool = False) -> tuple[int, int, str]:
    """Run code plus the task's tests in a subprocess with a timeout.

    Returns (passed, total, feedback). feedback is short, plain text, ready to show a model:
    which inputs failed, what was expected and what came back (or the error).
    Use public=True to test only the public examples (for feedback during a repair loop),
    and the default (hidden tests) for grading.
    """
    tests = task["public_tests"] if public else task["tests"]
    total = len(tests)
    script = (code + "\n\n__TESTS = " + repr([(tuple(a), e) for a, e in tests]) + "\n"
              + _HARNESS.format(fn=task["entry_point"]))
    out, err = run_python(script, timeout)
    m = re.search(r"__RESULTS__(.*)", out)
    if not m:
        msg = (err.strip().splitlines() or ["no output"])[-1]
        if "TIMEOUT" in err:
            msg = f"the code took longer than {timeout}s (infinite loop?)"
        return 0, total, f"The code did not run: {msg}"
    try:
        results = json.loads(m.group(1))
    except Exception:
        return 0, total, "The code did not run: could not read test results."
    passed = sum(r[0] for r in results)
    lines = []
    for ok, args, exp, got, e in results:
        if ok:
            continue
        call = task["entry_point"] + (args[:-2] + ")" if args.endswith(",)") else args)   # (x,) -> (x)
        lines.append(f"- {call}: expected {exp}, " + (f"raised {e}" if e else f"got {got}"))
    fb = f"{passed}/{total} tests passed." + ("\nFailures:\n" + "\n".join(lines[:5]) if lines else "")
    return passed, total, fb
