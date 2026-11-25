// Screenshot a page with headless Chrome over the DevTools protocol (no dependencies; Node 22+).
//   node scripts/shot.mjs <url> <out.png> [waitMs] [jsToRunBeforeShot]
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";

const [url, out, waitMs = "8000", click] = process.argv.slice(2);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9333;
const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${port}`, "--window-size=1600,1000",
  "--force-device-scale-factor=2", "--hide-scrollbars", "--user-data-dir=/tmp/studio-shot-profile", "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 40 && !target; i++) {
  await sleep(250);
  try {
    const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    target = list.find((t) => t.type === "page");
  } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
const logs = [];
ws.addEventListener("message", (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg.result); pending.delete(msg.id); }
  if (msg.method === "Runtime.consoleAPICalled") logs.push(msg.params.args.map((a) => a.value ?? a.description).join(" "));
  if (msg.method === "Runtime.exceptionThrown") logs.push("EXCEPTION " + msg.params.exceptionDetails.text + " " + (msg.params.exceptionDetails.exception?.description ?? ""));
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });

await send("Runtime.enable");
await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 2, mobile: false });
await send("Page.navigate", { url });
await sleep(Number(waitMs));
if (click) {
  await send("Runtime.evaluate", { expression: click });
  await sleep(2500);
}
const state = await send("Runtime.evaluate", { expression: "document.querySelector('footer')?.innerText", returnByValue: true });
const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(out, Buffer.from(shot.data, "base64"));
console.log("status bar:", state?.result?.value);
if (logs.length) console.log("console:\n" + logs.slice(0, 10).join("\n"));
ws.close();
chrome.kill();
process.exit(0);
