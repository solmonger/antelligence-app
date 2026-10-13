#!/usr/bin/env bash
# v4.1 pod-side launcher (goal 4.5): two vLLM servers on one L40S plus d1-3B in its own uv venv.
#
#   Qwen3.5-9B FP8  :8100  --gpu-memory-utilization 0.42   (deferral target)
#   Qwen3.5-4B FP8  :8101  --gpu-memory-utilization 0.24
#   d1-3B (shim)    :8102  gets the remaining memory (transformers, trust_remote_code, pinned revision)
#
# Servers start one after another, because vLLM sizes its KV cache from the free memory it sees at start-up.
# Tuning knobs (env, all optional): V41_GPU_UTIL_9B, V41_GPU_UTIL_4B, V41_MAX_MODEL_LEN, V41_MAX_NUM_SEQS,
# V41_MAX_CUDAGRAPH_CAPTURE_SIZE (lower it if a hybrid-layer assertion fires). The final values are printed so the
# run log records them. The API key comes from the environment (VLLM_API_KEY) and is never echoed.
#
# Served names carry the revision ("<hf id>@<sha>"): vLLM cannot otherwise report which commit it loaded, and the
# backends (antelligence/providers/decision_backends.py) check identity against that name.
set -euo pipefail

: "${VLLM_API_KEY:?VLLM_API_KEY must be set in the environment}"

Q9_ID="RedHatAI/Qwen3.5-9B-FP8-dynamic"
Q9_REV="790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"
Q4_ID="RedHatAI/Qwen3.5-4B-FP8-dynamic"
Q4_REV="397b7ba47a99b3221ebfc0cfc5a279118cb733ad"
D1_REV="051bcc464b01b9f92942b364d9586b0ef5912432"

GPU_UTIL_9B="${V41_GPU_UTIL_9B:-0.42}"
GPU_UTIL_4B="${V41_GPU_UTIL_4B:-0.24}"
MAX_MODEL_LEN="${V41_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${V41_MAX_NUM_SEQS:-32}"
CUDAGRAPH_SIZE="${V41_MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
D1_VENV="${V41_D1_VENV:-/opt/d1-venv}"
D1_SCRIPT="${V41_D1_SCRIPT:-/opt/v41/v41_d1_server.py}"
READY_TIMEOUT_S="${V41_READY_TIMEOUT_S:-900}"

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

# --language-model-only only if this vLLM build lists it.
LANG_ONLY=()
if { vllm serve --help=all 2>&1 || vllm serve --help 2>&1 || true; } | grep -q -- '--language-model-only'; then
  LANG_ONLY=(--language-model-only)
fi

CUDAGRAPH=()
if [[ -n "$CUDAGRAPH_SIZE" ]]; then CUDAGRAPH=(--max-cudagraph-capture-size "$CUDAGRAPH_SIZE"); fi

wait_ready() {  # name port pid
  local name="$1" port="$2" pid="$3" waited=0
  until curl -fsS -o /dev/null -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${port}/v1/models"; do
    kill -0 "$pid" 2>/dev/null || { echo "FATAL: $name exited before it was ready" >&2; exit 1; }
    (( waited += 5 )) || true
    (( waited < READY_TIMEOUT_S )) || { echo "FATAL: $name not ready after ${READY_TIMEOUT_S}s" >&2; exit 1; }
    sleep 5
  done
  echo "ready: $name on :$port"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
}

serve_qwen() {  # name hf_id revision gpu_util port
  local name="$1" id="$2" rev="$3" util="$4" port="$5"
  echo "start: $name $id@$rev gpu-memory-utilization=$util max-model-len=$MAX_MODEL_LEN"
  vllm serve "$id" \
    --revision "$rev" \
    --served-model-name "${id}@${rev}" \
    --host 0.0.0.0 --port "$port" \
    --gpu-memory-utilization "$util" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --seed 0 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    ${LANG_ONLY[@]+"${LANG_ONLY[@]}"} ${CUDAGRAPH[@]+"${CUDAGRAPH[@]}"} &
  pids+=("$!")
  wait_ready "$name" "$port" "$!"
}

serve_qwen qwen3.5-9b "$Q9_ID" "$Q9_REV" "$GPU_UTIL_9B" 8100
serve_qwen qwen3.5-4b "$Q4_ID" "$Q4_REV" "$GPU_UTIL_4B" 8101

# d1-3B: its own venv, because transformers >= 5.14 may clash with vLLM's own pin.
if [[ ! -x "$D1_VENV/bin/python" ]]; then
  uv venv "$D1_VENV"
  uv pip install --python "$D1_VENV/bin/python" "transformers>=5.14" torch pillow
fi
echo "start: d1-3B revision $D1_REV"
V41_D1_TOKEN="$VLLM_API_KEY" "$D1_VENV/bin/python" "$D1_SCRIPT" --host 0.0.0.0 --port 8102 --revision "$D1_REV" &
pids+=("$!")
d1_pid="$!"
waited=0
until curl -fsS -o /dev/null "http://127.0.0.1:8102/health"; do
  kill -0 "$d1_pid" 2>/dev/null || { echo "FATAL: d1 exited before it was ready" >&2; exit 1; }
  (( waited += 5 )) || true
  (( waited < READY_TIMEOUT_S )) || { echo "FATAL: d1 not ready after ${READY_TIMEOUT_S}s" >&2; exit 1; }
  sleep 5
done
echo "ready: d1-3B on :8102"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
echo "final: gpu_util_9b=$GPU_UTIL_9B gpu_util_4b=$GPU_UTIL_4B max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS cudagraph=${CUDAGRAPH_SIZE:-default}"

wait -n "${pids[@]}"
echo "FATAL: a server exited" >&2
exit 1
