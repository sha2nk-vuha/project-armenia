#!/usr/bin/env bash
#
# Start the Anomaly Inspector (backend + frontend) for local use.
#
# On first run this creates the Python virtualenv, installs backend and
# frontend dependencies, then launches both servers. Subsequent runs reuse
# the existing venv / node_modules and start immediately.
#
# Usage:
#   ./start.sh            # set up if needed, then run backend + frontend
#   ./start.sh --setup    # only install dependencies, do not start servers
#
# Open http://localhost:5173 once both servers report ready. Press Ctrl-C to stop.

set -euo pipefail

# Resolve the directory this script lives in, so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

setup_only=false
if [[ "${1:-}" == "--setup" ]]; then
  setup_only=true
fi

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: '$1' is not installed or not on PATH." >&2
    exit 1
  fi
}

require "$PYTHON_BIN"
require npm

# ── Backend setup ─────────────────────────────────────────────────────────
if [[ ! -d "$BACKEND_DIR/venv" ]]; then
  echo "==> Creating Python virtualenv (backend/venv)"
  "$PYTHON_BIN" -m venv "$BACKEND_DIR/venv"
  echo "==> Installing backend dependencies"
  "$BACKEND_DIR/venv/bin/pip" install --upgrade pip >/dev/null
  "$BACKEND_DIR/venv/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
else
  echo "==> Backend venv present (skipping install)"
fi

# ── Frontend setup ────────────────────────────────────────────────────────
if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "==> Installing frontend dependencies"
  (cd "$FRONTEND_DIR" && npm install)
else
  echo "==> Frontend node_modules present (skipping install)"
fi

if [[ "$setup_only" == true ]]; then
  echo "==> Setup complete."
  exit 0
fi

# ── Run both servers ──────────────────────────────────────────────────────
pids=()
cleanup() {
  echo
  echo "==> Shutting down…"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "==> Starting backend on http://$BACKEND_HOST:$BACKEND_PORT"
(
  cd "$BACKEND_DIR"
  exec venv/bin/uvicorn main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) &
pids+=("$!")

echo "==> Starting frontend (Vite dev server) on http://localhost:5173"
(
  cd "$FRONTEND_DIR"
  exec npm run dev
) &
pids+=("$!")

echo "==> Both servers launching. Open http://localhost:5173 — press Ctrl-C to stop."
wait
