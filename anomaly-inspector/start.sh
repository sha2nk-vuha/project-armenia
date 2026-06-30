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

# The pinned backend dependencies (onnxruntime==1.18.0, numpy==1.26.4, …) only
# ship wheels for Python 3.10–3.12. Pick a compatible interpreter automatically
# unless the caller forces one via PYTHON_BIN. Note that a conda "base" env or a
# bare `python3` is often 3.13+, which has no matching wheels.
pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    local ver
    ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
    case "$ver" in
      3.10|3.11|3.12) echo "$candidate"; return ;;
    esac
  done
  return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
  echo "ERROR: No compatible Python found (need 3.10, 3.11, or 3.12)." >&2
  echo "       Install one of those, or set PYTHON_BIN to a compatible interpreter." >&2
  exit 1
fi

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

# Vite 8 requires Node >= 20.19. A stale system Node (e.g. /usr/bin/node 12) will
# fail with "syntax error, unexpected token '.'" on modern JS. If nvm is present,
# load it and switch to a compatible version automatically.
NODE_MIN_MAJOR=20
node_major() { node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0; }

if [[ "$(node_major)" -lt "$NODE_MIN_MAJOR" ]] && [[ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]]; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # Relax strict mode: nvm's functions can return non-zero in ways that would
  # trip `set -e`, and we pass --no-use so sourcing doesn't inherit this
  # script's positional args (e.g. --setup) into nvm's auto-use parser.
  set +eu
  # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh" --no-use
  nvm use --lts >/dev/null 2>&1 || nvm use default >/dev/null 2>&1 || nvm use node >/dev/null 2>&1
  set -eu
fi

require node
require npm

if [[ "$(node_major)" -lt "$NODE_MIN_MAJOR" ]]; then
  echo "ERROR: Node $(node --version) is too old; Vite needs Node >= ${NODE_MIN_MAJOR}.19." >&2
  echo "       Install a newer Node (e.g. 'nvm install --lts') and re-run." >&2
  exit 1
fi

# ── Backend setup ─────────────────────────────────────────────────────────
if [[ ! -d "$BACKEND_DIR/venv" ]]; then
  echo "==> Creating Python virtualenv (backend/venv) using $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
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
