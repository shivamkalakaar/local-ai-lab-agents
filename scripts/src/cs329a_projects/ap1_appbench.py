# %% [markdown]
# # ap1 · AppBench, rebuilt small: generate web apps and grade them automatically
# **Local AI Lab** · CS329A past project (Winter 2025) · build a mini benchmark for AI-generated web apps
#
# A web app is more than code that runs. A user must be able to click, type and see the right result.
# In this notebook you write five small app specs, let the local model build each app as one HTML file,
# and grade every app with an automatic checklist that clicks through the app's logic like a user would.
# Then you test the grader itself, the step most teams skip.

# %% [markdown]
# ## The original project
# **AppBench: Benchmarking AI-Generated Web Applications** by Ethan Hellman, Brendan McLaughlin, Abhinav Lalwani and Belinda Mo
# (Stanford CS329A, Winter 2025). [Paper](https://cs329a.stanford.edu/winter2025projects/appbench.pdf) ·
# [Poster](https://cs329a.stanford.edu/winter2025projects/appbenchposter.pdf)
#
# **Problem.** Code benchmarks check functions with unit tests. Nobody checked whether an AI-built web app is
# *usable*: does navigation work, does state survive a refresh, does the page handle errors?
#
# **Method.** 9 app requests from L0 (static page) to L4 (real-time collaborative whiteboard), spread over six
# axes: UI complexity, features, state management, API integration, cross-page navigation, data processing.
# For each request the authors wrote a checklist of **explicit** requirements (what the user asked for) and
# **implicit** ones (what a good app would also do, like a back button). They arranged the checklist as a
# **dependency graph**: you can't test "add to cart" before the product page opens. Each node becomes a scripted
# test built from shared tools: screenshot + multimodal LLM judging, an LLM-driven browser (Stagehand on
# Playwright), and network interception to test error states. Scores are weighted and scaled to 0–100.
#
# **Setup.** 7 builders: Replit, Bolt, Cursor Agent, Lovable, Claude Sonnet 3.7, GPT-4.5 and OpenHands CodeAct 2.1,
# each in one shot. Three humans scored every app without seeing the automatic scores.
#
# **Key results (as reported).**
# - Grader vs. humans: mean MSE 0.0302, mean MAE 0.1313, Pearson 0.8173, Spearman 0.6827.
# - Per-builder MSE ranged from 0.0160 (Service 4) to 0.0501 (CodeAct); per-task MSE from 0.0042 (l2-add-to-cart) to 0.0751 (l3-food-order).
# - All builder types scored above 75% up to level L2; scores fell at L3 and dropped sharply at L4.
#   2 of 7 builders could not produce a compiling app for one L3 and one L4 task.
# - One-shot models were "surprisingly competitive" with full agent products; the gap narrowed at higher levels.
#   Lovable and Bolt were the top builders.
#
# **Limitations they state.** Only 9 tasks, so no strong general claims; the graders need more calibration with UX experts.
#
# **Our mini version.** Their apps ran in a real browser. Colab has no browser, so each app must expose its logic
# as plain JavaScript functions on `window.app`. We check the page structure with Python's built-in HTML parser, and
# we run the app's own script in Node.js (which Colab has) inside a fake page, then call its functions. Same ideas:
# explicit vs. implicit checks, a dependency graph, weighted 0–100 scores, and "page refresh" tests for saved state.

# %%
import json, re, shutil, subprocess, tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from lal.llm import ask
from lal.checks import check

NODE = shutil.which("node")
print("node.js:", NODE or "not found (behavior tests fall back to weak text checks)")
OUT = Path("cs329a_projects/ap1_apps"); OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Read the page like a browser would (pure Python)
# `Page` collects tag counts, element ids, the text inside each tag, and the inline scripts.

# %%
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

class Page(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.tags, self.ids, self.text = Counter(), {}, defaultdict(str)
        self.scripts, self.stack, self._script = [], [], False
        self.attrs = []
        try:
            self.feed(html or "")
        except Exception as e:                      # very broken HTML: keep what we parsed
            print("parse warning:", e)

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        self.tags[tag] += 1
        self.attrs.append((tag, a))
        if a.get("id"):
            self.ids[a["id"]] = tag
        if tag == "script":
            self._script = not a.get("src")         # external scripts are not run
            if self._script:
                self.scripts.append("")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag == "script":
            self._script = False
        if tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass

    def handle_data(self, data):
        if self._script:
            self.scripts[-1] += data
            return
        for t in set(self.stack):
            self.text[t] += data + " "
        self.text["*"] += data + " "

    def has(self, tag, n=1):
        return self.tags[tag] >= n

    def text_has(self, tag, s):
        return s.lower() in re.sub(r"\s+", " ", self.text[tag]).lower()

p = Page("<html><head><title>Hi</title></head><body><header>Acme Labs</header><input id='x'><script>var a=1;</script></body></html>")
print(p.tags["input"], p.ids, p.text_has("header", "acme"), p.scripts)

# %% [markdown]
# ## 2. Run the app's own JavaScript in a fake page (Node.js, in a subprocess with a timeout)
# The fake page answers every DOM call with a harmless stand-in object, so the app's setup code doesn't crash.
# `localStorage` is real (a dictionary), so we can test saving. A **phase** is one page load: to test a refresh,
# we run phase 1, keep `localStorage`, and load the page again for phase 2. The value of the last expression
# in the last phase is the result.

# %%
HARNESS_JS = r"""
const vm = require('vm'), fs = require('fs');
const code = fs.readFileSync(process.argv[2], 'utf8');
const phases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const store = {};
const TEXTY = ['value','textContent','innerHTML','innerText','className','id','src','href','checked'];
function any() {                       // a stand-in for any DOM object
  const props = {};
  return new Proxy(function(){}, {
    get(t, p) {
      if (p === Symbol.toPrimitive) return () => '';
      if (p in props) return props[p];
      if (p === 'toString' || p === 'valueOf') return () => '';
      if (TEXTY.includes(p)) return '';
      if (p === 'length') return 0;
      if (p === 'then') return undefined;
      if (p === Symbol.iterator) return function* () {};
      if (p === 'children' || p === 'childNodes') return [];
      const v = any(); props[p] = v; return v;
    },
    set(t, p, v) { props[p] = v; return true; },
    apply() { return any(); },
    construct() { return any(); },
  });
}
const localStorage = {
  getItem: k => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: k => { delete store[k]; },
  clear: () => { for (const k in store) delete store[k]; },
  key: i => Object.keys(store)[i] ?? null,
  get length() { return Object.keys(store).length; },
};
let out = {load_error: null, error: null, value: null};
for (let i = 0; i < phases.length; i++) {
  const listeners = [];
  const document = any();
  document.readyState = 'loading';
  document.addEventListener = (ev, fn) => { if (typeof fn === 'function' && /load/i.test(ev)) listeners.push(fn); };
  const sb = {document, localStorage, console: {log(){}, error(){}, warn(){}, info(){}},
              alert(){}, confirm(){ return true; }, prompt(){ return ''; },
              setTimeout(){ return 0; }, clearTimeout(){}, setInterval(){ return 0; }, clearInterval(){},
              requestAnimationFrame(){ return 0; }, fetch(){ return new Promise(() => {}); },
              navigator: any(), location: any(), history: any()};
  sb.window = sb; sb.self = sb;
  sb.addEventListener = (ev, fn) => { if (typeof fn === 'function' && /load/i.test(ev)) listeners.push(fn); };
  const ctx = vm.createContext(sb);
  try {
    vm.runInContext(code, ctx, {timeout: 2000});
    for (const fn of listeners) fn.call(sb, any());
    if (typeof sb.onload === 'function') sb.onload(any());
  } catch (e) { out.load_error = String(e).slice(0, 200); }
  try {
    let v;
    for (const expr of phases[i]) v = vm.runInContext(expr, ctx, {timeout: 1000});
    out.value = (v === undefined) ? null : JSON.parse(JSON.stringify(v));
  } catch (e) { out.error = String(e).slice(0, 200); out.value = null; }
}
console.log(JSON.stringify(out));
"""
WORK = Path(tempfile.mkdtemp(prefix="ap1_"))
(WORK / "harness.js").write_text(HARNESS_JS)

def run_js(script: str, phases: list) -> dict:
    """Run the app's inline script in the fake page. Untrusted code: subprocess + timeout."""
    if not NODE:
        return {"skipped": True}
    d = Path(tempfile.mkdtemp(dir=WORK))
    (d / "app.js").write_text(script)
    (d / "phases.json").write_text(json.dumps(phases))
    try:
        r = subprocess.run([NODE, str(WORK / "harness.js"), str(d / "app.js"), str(d / "phases.json")],
                           capture_output=True, text=True, timeout=10)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"load_error": "timeout", "error": "timeout", "value": None}
    except Exception as e:
        return {"load_error": f"harness failed: {e}", "error": "harness failed", "value": None}

print(run_js("window.app = {double: x => 2*x}; localStorage.setItem('k','1');",
             [["app.double(21)"]]))
print(run_js("const app = {n: () => Number(localStorage.getItem('k') || 0), inc(){ localStorage.setItem('k', this.n()+1); }};",
             [["app.inc()", "app.inc()"], ["app.n()"]]))       # survives the "refresh"

# %% [markdown]
# ## 3. The checklist as a dependency graph
# Each check has a weight, a flag for **core** (explicitly asked) vs. **implicit** (a good app would do it),
# and a list of checks it depends on. If a parent fails, its children are skipped and score 0, the same
# "conditional test execution" AppBench uses so one missing button doesn't cause a pile of confusing failures.

# %%
def static(fn):
    return ("static", fn)

def js(phases, expected):
    return ("js", (phases, expected))

def same(got, want):
    if isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(got, bool):
        return abs(got - want) < 0.011
    return got == want

APP_LOADS = js([["typeof app === 'object' && app !== null"]], True)

SPECS = {
 "l0-about-us": dict(level=0, spec=
   "Design a static 'About Us' page for a company called Acme Labs. It needs a <header> with the company name, "
   "a <main> section with at least two paragraphs about the company, and a <footer> with the company address: "
   "12 Market Street, Springfield. Make it look good on phones too.",
   checks=[
     ("header names the company", 2, True, [], static(lambda p: p.text_has("header", "Acme Labs"))),
     ("main with 2+ paragraphs", 2, True, [], static(lambda p: p.has("main") and p.has("p", 2))),
     ("footer has the address", 2, True, [], static(lambda p: p.text_has("footer", "12 Market Street"))),
     ("has a <title>", 1, False, [], static(lambda p: p.has("title") and p.text["title"].strip() != "")),
     ("mobile viewport meta", 1, False, [], static(lambda p: any(t == "meta" and a.get("name") == "viewport" for t, a in p.attrs))),
   ]),
 "l1-theme-switcher": dict(level=1, spec=
   "Create a theme switcher page. Three buttons with ids btn-light, btn-dark and btn-blue switch the whole page "
   "between a light, a dark and a blue theme. Show a heading and a short paragraph so the theme is visible. "
   "Expose window.app = { setTheme(name), getTheme() }: setTheme('dark') applies the dark theme and getTheme() then "
   "returns 'dark'. The default theme is 'light'. Unknown theme names must be ignored. The buttons must call app.setTheme.",
   checks=[
     ("three theme buttons", 2, True, [], static(lambda p: all(i in p.ids for i in ("btn-light", "btn-dark", "btn-blue")))),
     ("heading and paragraph", 1, False, [], static(lambda p: (p.has("h1") or p.has("h2")) and p.has("p"))),
     ("app loads", 1, True, [], APP_LOADS),
     ("default theme is light", 2, True, ["app loads"], js([["app.getTheme()"]], "light")),
     ("setTheme('blue') works", 3, True, ["app loads"], js([["app.setTheme('blue')", "app.getTheme()"]], "blue")),
     ("unknown theme ignored", 2, False, ["setTheme('blue') works"], js([["app.setTheme('dark')", "app.setTheme('purple')", "app.getTheme()"]], "dark")),
   ]),
 "l1-tip-calculator": dict(level=1, spec=
   "Build a tip calculator. It has an input with id bill, an input with id tip-pct (tip percent), a button with id "
   "calc, and an element with id result that shows the tip and the total. Each input has a <label>. "
   "Expose window.app = { tip(bill, pct), total(bill, pct) }: tip returns bill*pct/100 rounded to 2 decimals, "
   "total returns bill + tip rounded to 2 decimals. If bill is negative or not a number, both return 0.",
   checks=[
     ("inputs, button, result", 2, True, [], static(lambda p: all(i in p.ids for i in ("bill", "tip-pct", "calc", "result")))),
     ("inputs have labels", 1, False, [], static(lambda p: p.has("label", 2))),
     ("app loads", 1, True, [], APP_LOADS),
     ("tip(50, 20) = 10", 2, True, ["app loads"], js([["app.tip(50, 20)"]], 10)),
     ("total(80, 15) = 92", 2, True, ["app loads"], js([["app.total(80, 15)"]], 92)),
     ("rounds to cents", 2, True, ["tip(50, 20) = 10"], js([["Math.abs(app.tip(33.33, 18) - 6) < 1e-9"]], True)),
     ("negative bill gives 0", 1, False, ["app loads"], js([["app.tip(-10, 20)"]], 0)),
   ]),
 "l2-food-order": dict(level=2, spec=
   "Build a single-page restaurant menu with exactly these items and prices: Burger $8.50, Fries $3.00, "
   "Soda $1.75, Salad $6.25. Each item has an add button. The bottom of the page lists the current order and shows "
   "the total in an element with id order-total. No login. Expose window.app = { add(name), remove(name), "
   "count(name), total() }: total() returns the order total as a number. remove takes away one unit and never "
   "goes below zero. Unknown item names are ignored.",
   checks=[
     ("menu shows all items and prices", 2, True, [], static(lambda p: all(p.text_has("*", s) for s in
                                                    ("Burger", "8.50", "Fries", "3.00", "Soda", "1.75", "Salad", "6.25")))),
     ("order total element", 1, True, [], static(lambda p: "order-total" in p.ids)),
     ("app loads", 1, True, [], APP_LOADS),
     ("2 Burgers + Soda = 18.75", 3, True, ["app loads"], js([["app.add('Burger')", "app.add('Burger')", "app.add('Soda')", "app.total()"]], 18.75)),
     ("remove takes one unit", 2, True, ["app loads"], js([["app.add('Fries')", "app.add('Fries')", "app.remove('Fries')", "app.count('Fries')"]], 1)),
     ("remove never below zero", 1, False, ["remove takes one unit"], js([["app.remove('Salad')", "app.count('Salad')"]], 0)),
     ("unknown item ignored", 1, False, ["2 Burgers + Soda = 18.75"], js([["app.add('Pizza')", "app.total()"]], 0)),
   ]),
 "l3-todo-saved": dict(level=3, spec=
   "Build a to-do list app with an input with id new-task, a button with id add-btn and a list with id task-list. "
   "Expose window.app = { addTask(text), toggle(index), remaining(), all() }. all() returns an array of "
   "{text, done} objects. toggle flips done for the task at that index. remaining() counts tasks that are not done. "
   "Ignore empty or whitespace-only text. Save the tasks in localStorage under the key 'todos' as JSON after every "
   "change, and load them when the page opens, so tasks survive a page refresh.",
   checks=[
     ("input, button, list", 2, True, [], static(lambda p: all(i in p.ids for i in ("new-task", "add-btn", "task-list")))),
     ("app loads", 1, True, [], APP_LOADS),
     ("add + toggle + remaining", 3, True, ["app loads"], js([["app.addTask('a')", "app.addTask('b')", "app.toggle(0)", "app.remaining()"]], 1)),
     ("saves JSON under 'todos'", 2, True, ["add + toggle + remaining"], js([["app.addTask('a')", "JSON.parse(localStorage.getItem('todos')).length"]], 1)),
     ("survives a refresh", 3, True, ["saves JSON under 'todos'"], js([["app.addTask('a')", "app.addTask('b')"], ["app.remaining()"]], 2)),
     ("ignores blank tasks", 1, False, ["app loads"], js([["app.addTask('   ')", "app.all().length"]], 0)),
   ]),
}
print({k: len(v["checks"]) for k, v in SPECS.items()})

# %%
def grade(name: str, html: str, verbose=False) -> dict:
    page = Page(html)
    script = "\n;\n".join(page.scripts)
    results, earned, total, core_earned, core_total = {}, 0, 0, 0, 0
    for cname, w, core, deps, (kind, arg) in SPECS[name]["checks"]:
        total += w; core_total += w if core else 0
        if any(not results.get(d, (False,))[0] for d in deps):
            results[cname] = (False, "skipped: depends on " + ", ".join(deps))
        elif kind == "static":
            try:
                ok = bool(arg(page))
            except Exception as e:
                ok = False
            results[cname] = (ok, "")
        else:
            phases, want = arg
            r = run_js(script, phases)
            if r.get("skipped"):                     # no node: weak fallback, function names must at least appear
                names = set(re.findall(r"app\.(\w+)\(", " ".join(sum(phases, []))))
                ok = bool(script) and all(re.search(rf"\b{n}\b", script) for n in names)
                results[cname] = (ok, "weak check (no node)")
            else:
                ok = r.get("error") is None and same(r.get("value"), want)
                why = "" if ok else f"got {r.get('value')!r}, want {want!r}" + (f"; error: {r['error']}" if r.get("error") else "") \
                      + (f"; page script error: {r['load_error']}" if r.get("load_error") else "")
                results[cname] = (ok, why)
        if results[cname][0]:
            earned += w; core_earned += w if core else 0
    out = {"score": round(100 * earned / total, 1), "core": round(100 * core_earned / max(core_total, 1), 1), "checks": results}
    if verbose:
        for c, (ok, why) in results.items():
            print(f"  {'✓' if ok else '✗'} {c}  {why}")
    return out

# %% [markdown]
# ## 4. Test the grader before you trust it
# AppBench's main claim is that its grader agrees with humans. We can't hire three humans here, so we do the
# cheaper first step: a **known-good** app must score 100, and a **known-broken** one must score low.

# %%
GOOD_TIP = """<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width"><title>Tips</title></head>
<body><h1>Tip calculator</h1>
<label for="bill">Bill</label><input id="bill" type="number">
<label for="tip-pct">Tip %</label><input id="tip-pct" type="number" value="15">
<button id="calc">Calculate</button><p id="result"></p>
<script>
const r2 = x => Math.round(x * 100) / 100;
window.app = {
  tip(bill, pct) { bill = Number(bill); if (!(bill >= 0)) return 0; return r2(bill * Number(pct) / 100); },
  total(bill, pct) { bill = Number(bill); if (!(bill >= 0)) return 0; return r2(bill + this.tip(bill, pct)); },
};
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('calc').addEventListener('click', () => {
    const b = document.getElementById('bill').value, p = document.getElementById('tip-pct').value;
    document.getElementById('result').textContent = `Tip ${app.tip(b, p)} · Total ${app.total(b, p)}`;
  });
});
</script></body></html>"""
BROKEN_TIP = GOOD_TIP.replace("<label", "<span").replace("bill * Number(pct) / 100", "bill * Number(pct)").replace('id="calc"', 'id="go"')
CRASHING_TIP = GOOD_TIP.replace("const r2", "undefinedFunction(); const r2")

ref = {n: grade("l1-tip-calculator", h) for n, h in [("good", GOOD_TIP), ("broken", BROKEN_TIP), ("crashes", CRASHING_TIP)]}
for n, g in ref.items():
    print(f"{n:8s} score {g['score']:5.1f}")
grade("l1-tip-calculator", BROKEN_TIP, verbose=True);

# %% [markdown]
# ## 5. Let the model build the five apps (one shot)
# Like the paper's "one-shot regime": same prompt shape for every task, one turn, no feedback.

# %%
BUILD = ("You are an expert front-end developer. Build this web app as ONE complete HTML file with all CSS and "
         "JavaScript inline. Do not load any external scripts, fonts or libraries.\n\nAPP REQUEST:\n{spec}\n\n"
         "Return only the file inside a ```html code block.")

def extract_html(text: str) -> str:
    m = re.search(r"```(?:html)?\s*(.*?)```", text or "", re.S | re.I)
    if m and "<" in m.group(1):
        return m.group(1).strip()
    m = re.search(r"(<!DOCTYPE.*?</html>|<html.*?</html>)", text or "", re.S | re.I)
    return m.group(1) if m else (text or "")

def build(name):
    try:
        return extract_html(ask(BUILD.format(spec=SPECS[name]["spec"]), temperature=0, max_tokens=3000))
    except Exception as e:
        print(name, "generation failed:", e)
        return ""

with ThreadPoolExecutor(2) as ex:
    apps = dict(zip(SPECS, ex.map(build, SPECS)))
one_shot = {}
for name, html in apps.items():
    (OUT / f"{name}.html").write_text(html)
    one_shot[name] = grade(name, html)
    print(f"\n{name} (L{SPECS[name]['level']}): score {one_shot[name]['score']} · core {one_shot[name]['core']}")
    grade(name, html, verbose=True)

# %% [markdown]
# ## 6. Does an "agent loop" help? Build → grade → fix once
# The paper found agent products only modestly better than one-shot models. Here the "agent" gets the names of its
# failed checks (not the hidden test code) and one chance to fix the file. Each fix is one more model call.

# %%
FIX = ("Here is a web app you built and the user tests it failed.\n\nAPP REQUEST:\n{spec}\n\nFAILED CHECKS:\n{fails}\n\n"
       "CURRENT FILE:\n```html\n{html}\n```\n\nReturn the complete corrected file inside a ```html code block.")

def fix(name):
    g = one_shot[name]
    fails = [f"- {c}: {why}" for c, (ok, why) in g["checks"].items() if not ok]
    if not fails:
        return apps[name]                            # nothing to fix, no call spent
    try:
        return extract_html(ask(FIX.format(spec=SPECS[name]["spec"], fails="\n".join(fails), html=apps[name][:12000]),
                                temperature=0, max_tokens=3000))
    except Exception as e:
        print(name, "fix failed:", e)
        return apps[name]

with ThreadPoolExecutor(2) as ex:
    fixed = dict(zip(SPECS, ex.map(fix, SPECS)))
with_fix = {n: grade(n, h) for n, h in fixed.items()}
print(f"{'task':20s} level  one-shot  after-fix")
for n in SPECS:
    print(f"{n:20s}  L{SPECS[n]['level']}    {one_shot[n]['score']:6.1f}    {with_fix[n]['score']:6.1f}")
mean = lambda d: sum(v["score"] for v in d.values()) / len(d)
print(f"mean                       {mean(one_shot):6.1f}    {mean(with_fix):6.1f}")

# %% [markdown]
# **What to notice.** Scores usually fall as the level goes up, as in the paper. Look at *which* checks fail:
# implicit checks (blank tasks, unknown items) and saved state across a refresh are the usual misses. A fix round
# with precise failure messages often helps a lot here, because our checks name the exact problem. In AppBench the
# agents got no such feedback, which is one reason their edge over one-shot was small.

# %%
check("grader: known-good app scores 100", ref["good"]["score"] == 100)
check("grader: broken app scores under 60", ref["broken"]["score"] < 60)
check("grader: crashing script fails the behavior checks", ref["crashes"]["core"] < 50)
check("all five apps were built and graded", len(one_shot) == 5 and all((OUT / f"{n}.html").exists() for n in SPECS))
check("the model's apps score at least 30/100 on average", mean(one_shot) >= 30)

# %% [markdown]
# ## Your turn
# 1. **Add an implicit check.** For `l2-food-order`, a good app would let the user clear the order. Add an implicit
#    check that `app.clear()` (if it exists) sets the total to 0. How many of the model's apps pass a check nobody asked for?
# 2. **Calibrate against a human (you).** Open two or three apps in `cs329a_projects/ap1_apps/` (in Colab, use
#    `IPython.display.HTML`). Score each 0–100 by hand, then compute the Spearman correlation with the grader.
#    AppBench reported 0.68 with three humans. Where do you and the grader disagree?
# 3. **Add an LLM judge.** AppBench also used a model to judge screenshots. Ask the local model to score each
#    HTML file 0–10 against its spec, and compare its ranking with the checklist grader. Which one would you trust, and why?

# %% [markdown]
# ## Solutions

# %%
# 1. An implicit "clear order" check. It depends on "app loads". Notice: the spec never asked for clear(),
#    so most apps fail it. That is exactly what implicit checks measure: going beyond the literal request.
SPECS["l2-food-order"]["checks"].append(
    ("can clear the order", 1, False, ["app loads"],
     js([["app.add('Burger')", "typeof app.clear === 'function' ? (app.clear(), app.total()) : -1"]], 0)))
g = grade("l2-food-order", fixed["l2-food-order"], verbose=True)
print("food-order score with the new check:", g["score"])

# %%
# 2. Spearman correlation between your scores and the grader's. Replace my_scores with your own numbers.
def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x)
        i = 0
        while i < len(x):                                  # average ranks for ties
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b); n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    sd = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / sd if sd else 0.0

names = list(SPECS)
grader_scores = [with_fix[n]["score"] for n in names]
my_scores = [90, 70, 80, 60, 50]                           # <- your hand scores, same order as `names`
print(dict(zip(names, grader_scores)))
print("Spearman(you, grader) =", round(spearman(my_scores, grader_scores), 3))

# %%
# 3. A text-only LLM judge (5 calls). Notice how often it gives high scores to apps whose logic is broken:
#    a judge that reads code can't "click", which is why AppBench simulated real interaction.
def judge(name):
    reply = ask(f"APP REQUEST:\n{SPECS[name]['spec']}\n\nHTML FILE:\n```html\n{fixed[name][:8000]}\n```\n\n"
                "Score how well this file meets the request, 0 to 10. End with a line 'SCORE: <number>'.", max_tokens=400)
    m = re.findall(r"SCORE:\s*(\d+(?:\.\d+)?)", reply or "")
    return float(m[-1]) * 10 if m else 50.0
with ThreadPoolExecutor(2) as ex:
    judge_scores = list(ex.map(judge, names))
for n, a, b in zip(names, grader_scores, judge_scores):
    print(f"{n:20s} grader {a:5.1f}   judge {b:5.1f}")
print("Spearman(judge, grader) =", round(spearman(judge_scores, grader_scores), 3))
