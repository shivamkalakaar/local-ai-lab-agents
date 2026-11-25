#!/usr/bin/env bash
# Agent Studio: start the agent server and the web UI, then open the browser.
#   ./start.sh                     # default model (best in the bake-off)
#   STUDIO_MODEL=google/gemma-4-31b-it ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
if [ -d "$ROOT/harness/cs329z_hw1" ]; then HARNESS="$ROOT/harness"; else HARNESS="$ROOT/../hw1"; fi
if [ -f "$ROOT/studio/server.py" ]; then SERVER="$ROOT/studio/server.py"; else SERVER="$ROOT/server.py"; fi
WEB="$ROOT/web"

say() { printf "\033[36m▸\033[0m %s\n" "$*"; }
die() { printf "\033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
if ! command -v node >/dev/null || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 18 ]; then
  [ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh" && (nvm use 22 >/dev/null 2>&1 || nvm use --lts >/dev/null 2>&1) || true
fi
command -v node >/dev/null && [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 18 ] || die "Node.js 18+ is required: https://nodejs.org"

# API key: harness/.env, or OPENROUTER_API_KEY in the environment.
if [ ! -f "$HARNESS/.env" ]; then
  [ -n "${OPENROUTER_API_KEY:-}" ] || die "Set OPENROUTER_API_KEY (https://openrouter.ai/keys), or put it in $HARNESS/.env"
  printf "OPENROUTER_API_KEY=%s\nCS329Z_BUDGET_USD=2\n" "$OPENROUTER_API_KEY" > "$HARNESS/.env"
  say "wrote $HARNESS/.env (spending cap \$2; edit CS329Z_BUDGET_USD to change it)"
fi

say "preparing the data (first run only)"
(cd "$HARNESS" && [ -f data/emails/emails.jsonl ] || uv run python data/download.py)

say "installing and building the web UI (first run only)"
(cd "$WEB" && { [ -d node_modules ] || npm install --no-audit --no-fund; } && { [ -d .next ] || npm run build; })

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

say "starting the agent server on http://127.0.0.1:8765"
(cd "$HARNESS" && uv run python "$SERVER") &
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:8765/api/health >/dev/null && break; sleep 0.5; done

say "starting the UI on http://localhost:3000"
(cd "$WEB" && npm run start >/dev/null) &
for _ in $(seq 1 40); do curl -sf http://localhost:3000 >/dev/null && break; sleep 0.5; done
command -v open >/dev/null && open http://localhost:3000 || true
say "running. Press Ctrl-C to stop both."
wait
