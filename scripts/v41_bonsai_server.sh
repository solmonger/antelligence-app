#!/usr/bin/env bash
# Start / stop / check Ternary Bonsai 8B on the Mac (goal 4.5). Metal, PrismML llama.cpp `prism` branch.
#   scripts/v41_bonsai_server.sh start|stop|status
# --alias carries the pinned revision: LlamaCppBackend checks the served name (Identity.served_name).
# The template ends the generation prompt with an empty think block itself, so prefill_empty_think stays False.
set -euo pipefail
ROOT=${V41_ROOT:-/Volumes/WD_BLACK/antelligence-v4.1}
HF_ID=prism-ml/Ternary-Bonsai-8B-gguf
REV=c2aefbeb4b24469cd11579c3384b990404c17a30
PORT=${V41_BONSAI_PORT:-8521}
BIN=$ROOT/llama-prism/build/bin/llama-server
GGUF=$ROOT/hf/hub/models--prism-ml--Ternary-Bonsai-8B-gguf/snapshots/$REV/Ternary-Bonsai-8B-Q2_0_g64.gguf
LOG=$ROOT/llama-server-$PORT.log
PIDFILE=$ROOT/llama-server-$PORT.pid
case "${1:-}" in
  start)
    [ -x "$BIN" ] && [ -f "$GGUF" ] || { echo "missing $BIN or $GGUF" >&2; exit 1; }
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then echo "port $PORT is busy" >&2; exit 1; fi
    nohup "$BIN" -m "$GGUF" -c 32768 -ngl 99 --jinja --host 127.0.0.1 --port "$PORT" --alias "$HF_ID@$REV" >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    for _ in $(seq 1 60); do
      curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "up on :$PORT (pid $(cat "$PIDFILE"))"; exit 0; }
      sleep 1
    done
    echo "did not become healthy; see $LOG" >&2; exit 1 ;;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"; echo stopped ;;
  status)
    curl -fs "http://127.0.0.1:$PORT/health" && echo ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
