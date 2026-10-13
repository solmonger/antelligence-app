"""Static checks of scripts/v41_pod_serve.sh and the v4_pod worker payload. Nothing is run."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from scripts import v4_pod

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "v41_pod_serve.sh"


def test_pod_serve_script_flags():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0
    # two pinned Qwen3.5 FP8 servers with the specified memory fractions
    assert 'Q9_ID="RedHatAI/Qwen3.5-9B-FP8-dynamic"' in text and 'Q9_REV="790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"' in text
    assert 'Q4_ID="RedHatAI/Qwen3.5-4B-FP8-dynamic"' in text and 'Q4_REV="397b7ba47a99b3221ebfc0cfc5a279118cb733ad"' in text
    assert 'GPU_UTIL_9B="${V41_GPU_UTIL_9B:-0.42}"' in text and 'GPU_UTIL_4B="${V41_GPU_UTIL_4B:-0.24}"' in text
    assert len(re.findall(r"^\s*vllm serve ", text, re.M)) == 1  # one launcher function, called twice
    assert text.count("serve_qwen qwen3.5-") == 2
    for flag in ("--revision", "--served-model-name", "--gpu-memory-utilization", "--reasoning-parser qwen3", "--seed 0"):
        assert flag in text, flag
    assert """--default-chat-template-kwargs '{"enable_thinking": false}'""" in text
    assert re.search(r'--max-model-len "\$MAX_MODEL_LEN"', text) and 'MAX_MODEL_LEN="${V41_MAX_MODEL_LEN:-4096}"' in text
    assert '"${id}@${rev}"' in text  # the revision rides in the served name (identity check)
    # --language-model-only only when `vllm serve --help` lists it
    assert re.search(r"vllm serve --help.*\| grep -q -- '--language-model-only'", text)
    assert re.search(r"grep -q -- '--language-model-only'; then\n\s+LANG_ONLY=\(--language-model-only\)\n\s*fi", text)
    assert re.search(r"^\s*LANG_ONLY=\(--language-model-only\)", text, re.M) and "LANG_ONLY=()" in text  # empty by default
    assert "--max-cudagraph-capture-size" in text and "V41_MAX_CUDAGRAPH_CAPTURE_SIZE" in text
    # one at a time, each waited for, then d1 in its own uv venv with the pinned revision
    assert text.index("serve_qwen qwen3.5-9b") < text.index("serve_qwen qwen3.5-4b") < text.index('uv venv "$D1_VENV"')
    assert 'D1_REV="051bcc464b01b9f92942b364d9586b0ef5912432"' in text and '--revision "$D1_REV"' in text
    assert 'uv pip install --python "$D1_VENV/bin/python" "transformers==${D1_TRANSFORMERS}"' in text
    assert 'D1_TRANSFORMERS="5.14.1"' in text and "transformers>=" not in text.split("D1_TRANSFORMERS=")[1]
    assert "v41_d1_server.py" in text
    # the key is read from the environment and never echoed
    assert ': "${VLLM_API_KEY:?' in text and not re.search(r"echo[^\n]*VLLM_API_KEY", text)


def test_push_scripts_commands_carry_paths_only():
    cmds = v4_pod.push_commands("203.0.113.5", 2222, identity="/home/u/.ssh/pod_key")
    assert [c[0] for c in cmds] == ["scp", "scp", "ssh"]
    assert all("-P" in c or "-p" in c for c in cmds) and cmds[0][-1].startswith("root@203.0.113.5:/opt/v41/")
    assert cmds[-1][-1].endswith("touch /opt/v41/.pushed") and "mv /opt/v41/serve.sh.part /opt/v41/serve.sh" in cmds[-1][-1]
    out, ran = [], []
    class R:
        returncode = 0
    assert v4_pod.cmd_push_scripts("203.0.113.5", 2222, "/home/u/.ssh/pod_key", out.append, lambda c: ran.append(c) or R()) == 0
    assert len(ran) == 3 and not any("pod_key" in line for line in out)  # the key path is not echoed
    class Bad:
        returncode = 1
    assert v4_pod.cmd_push_scripts("h", 22, None, out.append, lambda c: Bad()) == 1 and len(cmds) == 3


def test_workers_pod_payload_embeds_launcher_and_keeps_budget_path():
    payload = v4_pod.deploy_payload("v41-workers", "secret", "NVIDIA L40S", "tok")
    cmd = payload["dockerStartCmd"][0]
    assert payload["env"] == {"VLLM_API_KEY": "secret"} and "secret" not in cmd
    assert len(cmd) < 8000 and "base64" not in cmd  # the launcher is pushed, not embedded
    assert "until [ -f /opt/v41/.pushed ]" in cmd and "bash /opt/v41/serve.sh" in cmd and "[redacted]" in cmd
    assert {"8100/http", "8101/http", "8102/http"} <= set(payload["ports"])
    assert v4_pod.served_name("v41-workers") == "RedHatAI/Qwen3.5-9B-FP8-dynamic@790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"
    assert "v41-workers" not in v4_pod.MODELS  # existing single-model pins are untouched
