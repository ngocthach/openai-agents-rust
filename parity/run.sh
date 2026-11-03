#!/usr/bin/env bash
# Parity harness runner: runs a scenario in the Python SDK and in the Rust port against the
# local capture server, then compares the recorded requests. See docs/parity-harness.md.
#
#   parity/run.sh setup                 install the Python SDK into the uv cache environment
#   parity/run.sh capture <name|all>    record and diff request payloads (no API calls)
#   parity/run.sh live <name|all>       phase 4: real API runs (not wired yet)
set -euo pipefail

RS_REPO=/Users/ngocthach/myGit/openai-agents-rust
PY_REPO=/Users/ngocthach/myGit/openai-agents-python
PARITY="$RS_REPO/parity"
OUT="$PARITY/out"

# The Python environment lives outside the checkout; the Rust artifacts outside the repo.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/oars-build/py-venv}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/oars-build/parity}"
PY_RUN=(uv run --project "$PY_REPO" --python 3.13 --no-sync python)

usage() {
  echo "usage: parity/run.sh setup | capture <scenario|all> | live <scenario|all>" >&2
  exit 2
}

setup() {
  uv sync --project "$PY_REPO" --python 3.13 --no-dev
  "${PY_RUN[@]}" -c 'import agents, sys; print("agents", agents.__file__, "python", sys.version.split()[0])'
}

# Scenario names for a mode: every directory under parity/scenarios except `live`, or the
# directories under parity/scenarios/live.
scenarios() {
  local mode=$1 name=$2
  if [ "$name" != all ]; then
    echo "$name"
  elif [ "$mode" = capture ]; then
    ls "$PARITY/scenarios" | grep -v '^live$'
  else
    ls "$PARITY/scenarios/live"
  fi
}

SERVER_PID=""
PORT=""
start_server() {
  mkdir -p "$OUT"
  python3 "$PARITY/capture_server.py" --port 0 --out "$OUT" --scenarios "$PARITY/scenarios" \
    > "$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  local i
  for i in $(seq 1 50); do
    PORT=$(sed -n 's/^PORT=//p' "$OUT/server.log" 2>/dev/null | head -1)
    if [ -n "$PORT" ]; then break; fi
    sleep 0.1
  done
  if [ -z "$PORT" ]; then
    echo "capture server did not start:" >&2
    cat "$OUT/server.log" >&2
    exit 1
  fi
}
stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

# Run one capture scenario on both sides. Tracing is switched off except for the scenario
# that captures the trace export itself.
capture_one() {
  local s=$1
  rm -rf "${OUT:?}/$s"
  mkdir -p "$OUT/$s"
  local envs=(OPENAI_API_KEY=sk-parity-dummy "PARITY_OUT=$OUT/$s")
  if [ "$s" != tracing_export ]; then
    envs+=(OPENAI_AGENTS_DISABLE_TRACING=1)
  fi
  (
    cd "$PY_REPO"
    env "${envs[@]}" "OPENAI_BASE_URL=http://127.0.0.1:$PORT/$s/py/v1" \
      "${PY_RUN[@]}" "$PARITY/scenarios/$s/py.py"
  )
  (
    cd "$RS_REPO"
    env "${envs[@]}" "OPENAI_BASE_URL=http://127.0.0.1:$PORT/$s/rs/v1" OPENAI_AGENTS_PARITY_CAPTURE=1 \
      cargo test -q --all-features --test parity_capture "$s" -- --exact --nocapture
  )
}

cmd=${1:-}
case "$cmd" in
  setup)
    setup
    ;;
  capture)
    [ $# -eq 2 ] || usage
    failed=""
    start_server
    for s in $(scenarios capture "$2"); do
      capture_one "$s"
      if ! python3 "$PARITY/compare.py" "$s" --out "$OUT" --waivers "$PARITY/waivers.json" \
           --scenarios "$PARITY/scenarios"; then
        failed="$failed $s"
      fi
    done
    stop_server
    if [ -n "$failed" ]; then
      echo "PARITY FAILED:$failed"
      exit 1
    fi
    ;;
  live)
    [ $# -eq 2 ] || usage
    # The real key comes from the repo's .env and is only ever exported into the two child
    # processes; nothing below prints it.
    if [ ! -f "$RS_REPO/.env" ]; then
      echo "live mode needs $RS_REPO/.env with OPENAI_API_KEY" >&2
      exit 1
    fi
    set -a
    # shellcheck disable=SC1091
    . "$RS_REPO/.env"
    set +a
    export PARITY_MODEL="${PARITY_MODEL:-gpt-6-luna}"
    export OPENAI_AGENTS_PARITY_LIVE=1
    failed=""
    for s in $(scenarios live "$2"); do
      rm -rf "${OUT:?}/live/$s"
      mkdir -p "$OUT/live/$s"
      export PARITY_OUT="$OUT/live/$s"
      ( cd "$PY_REPO" && "${PY_RUN[@]}" "$PARITY/scenarios/live/$s/py.py" )
      ( cd "$RS_REPO" && cargo test -q --all-features --test parity_live "$s" -- --exact --nocapture --test-threads=1 )
      if ! python3 "$PARITY/compare_live.py" "$s" --out "$OUT/live" --scenarios "$PARITY/scenarios/live"; then
        failed="$failed $s"
      fi
    done
    if [ -n "$failed" ]; then
      echo "PARITY LIVE FAILED:$failed"
      exit 1
    fi
    ;;
  *)
    usage
    ;;
esac
