# Mac-local models (goal section 4.5)

Everything lives on the external SSD `/Volumes/WD_BLACK/antelligence-v4.1/`. Smoke run 2026-10-13, Apple Silicon Mac,
synthetic decision requests only (no MRI data). Smoke driver: `scripts/v41_mac_smoke.py`.

## Ternary Bonsai 8B (Bonsai arm)

| Item | Value |
|---|---|
| Runtime | PrismML llama.cpp fork, branch `prism`, https://github.com/PrismML-Eng/llama.cpp |
| Fork commit | `e8fd209c975c80aa7fb36a8423571ff3b1b7e955` (`/props` build_info `b51-e8fd209`) |
| Build | cmake 4.4.4 (from a pip venv on the SSD), `-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release`, target `llama-server` |
| Model | `prism-ml/Ternary-Bonsai-8B-gguf`, file `Ternary-Bonsai-8B-Q2_0_g64.gguf` |
| HF revision (full) | `c2aefbeb4b24469cd11579c3384b990404c17a30` (resolved from the HF commit list; it is the head commit) |
| File size / sha256 | 2,310,125,920 bytes; `e17b298d84ee78797916ae5c2ecc8211469cc65cccfe3080cd9a9bb503fbc55e` |
| Launcher | `scripts/v41_bonsai_server.sh start|stop|status`: `-c 32768 -ngl 99 --jinja --port 8521`, plus `--host 127.0.0.1` and `--alias <repo>@<revision>` |

`--alias` is required: `LlamaCppBackend` checks the served name against `Identity.served_name`, and without it the server
reports the file path. Run it as `LlamaCppBackend(url, "prism-ml/Ternary-Bonsai-8B-gguf", <revision>)`.

**Thinking.** The chat template has no `enable_thinking` switch, but its generation prompt already ends with
`<think>\n\n</think>\n\n`. No thinking text appeared (a plain chat reply was clean). Leave `prefill_empty_think=False`; setting
it would add a second empty block.

**Smoke (1 request, JSON-schema over `answer_schema`, target/release/amount).** Wall 1.0 s (server 881 ms, 217 prompt
tokens, 24 generated). Logprobs came back: `prob_method="logprobs"`, `degraded=False`, all labels single tokens. Identity
check passed. Bonsai latency is reported separately from the pod workers (different hardware).

## d1-3B on MPS (edge latency)

| Item | Value |
|---|---|
| Model / revision | `LiquidAI/d1-3B` @ `051bcc464b01b9f92942b364d9586b0ef5912432` |
| Runtime | transformers 5.14.1, torch 2.14.1, torchvision 0.29.1 in `/Volumes/WD_BLACK/antelligence-v4.1/d1-venv`; device `mps`, bf16 |
| Server | `HF_HOME=<ssd>/hf HF_HUB_OFFLINE=1 d1-venv/bin/python -I scripts/v41_d1_server.py --port 8522` |

**Remote-code audit.** All six files (`api.py`, `runner.py`, `prompt.py`, `hybrid.py`, `lfm2_vl.py`, `modeling_d1.py`) were read in
full before `trust_remote_code` was enabled. No network calls, subprocess, `exec`/`eval`, or file writes. The only I/O is
`AutoTokenizer`/`AutoProcessor.from_pretrained(<model path>)` (hub reads) and `torch.compile`, which only runs with
`compile=True` (off). The CUDA flash-attention ops are gated off on MPS (`_fused`).

**Fix.** The remote `engine` loads the tokenizer and processor from `model.name_or_path`. The shim used the repo name, so
those resolved against `main`, not the pin, and failed offline (500). `load_model` now loads everything from the pinned
snapshot dir. The shim also prints 500 tracebacks to stderr (server log only; the client still gets the bare error name).

**Smoke (5 requests through `D1Backend`, 64x64 FLAIR+T1c synthetic patch -> 2 PNGs each, 3 questions).**
Per-request wall: 0.72, 0.30, 0.57, 0.38, 0.50 s (server 602, 293, 519, 376, 483 ms; the first request includes warm-up;
the first-ever call was 6.8 s). 332 input tokens, 0 output. `system_one` returned probabilities for all three types
(choice `target`, noul `release`, score `amount`), each summing to 1. Images were accepted. `prob_method="system_one"`.
