"""scripts/v41_pod_serve.sh and the v4_pod worker payload: hash-pinned launcher fetch. No network, no pod."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess

from pathlib import Path

import pytest

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


def head_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=v4_pod.ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_sha256(commit, path):
    out = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=v4_pod.ROOT, capture_output=True, check=True).stdout
    return hashlib.sha256(out).hexdigest()


def test_push_scripts_is_gone():
    assert not hasattr(v4_pod, "push_commands") and not hasattr(v4_pod, "cmd_push_scripts")
    with pytest.raises(SystemExit):
        v4_pod.main(["push-scripts", "--host", "h", "--port", "1"])


def test_workers_pod_payload_embeds_hashes_and_keeps_budget_path():
    commit = head_sha()
    pin = v4_pod.scripts_pin(commit)
    assert pin.hashes["serve.sh"] == git_sha256(commit, "scripts/v41_pod_serve.sh")
    assert pin.hashes["v41_d1_server.py"] == git_sha256(commit, "scripts/v41_d1_server.py")
    payload = v4_pod.deploy_payload("v41-workers", "secret", "NVIDIA L40S", "tok", pin)
    cmd = payload["dockerStartCmd"][0]
    assert payload["env"] == {"VLLM_API_KEY": "secret"} and "secret" not in cmd
    assert len(cmd) < 8000 and "base64" not in cmd and ".pushed" not in cmd
    for h in pin.hashes.values():
        assert h in cmd
    base = f"https://raw.githubusercontent.com/solmonger/antelligence-app/{commit}/scripts/"
    assert base + "v41_pod_serve.sh" in cmd and base + "v41_d1_server.py" in cmd
    assert "curl" in cmd and "urllib" in cmd and "bash /opt/v41/serve.sh" in cmd and "[redacted]" in cmd
    assert cmd.index("http.server 8001") < cmd.index("fetch_verified")  # log port is up before the fetch
    assert {"8100/http", "8101/http", "8102/http", "8001/http"} <= set(payload["ports"])
    assert v4_pod.served_name("v41-workers") == "RedHatAI/Qwen3.5-9B-FP8-dynamic@790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"
    assert "v41-workers" not in v4_pod.MODELS  # existing single-model pins are untouched


def test_workers_payload_requires_pin():
    with pytest.raises(v4_pod.ScriptsCommitError):
        v4_pod.deploy_payload("v41-workers", "secret", "NVIDIA L40S", "tok")


def test_scripts_pin_validation():
    for bad in ("", "abc123", "Z" * 40, head_sha().upper(), head_sha()[:39]):
        with pytest.raises(v4_pod.ScriptsCommitError):
            v4_pod.scripts_pin(bad)
    with pytest.raises(v4_pod.ScriptsCommitError, match="not found"):
        v4_pod.scripts_pin("0" * 40)


# --- run the generated fetch fragment against a local fixture -----------------------


def fixture(tmp_path, tamper=False):
    commit = "a" * 40
    srv = tmp_path / "srv" / commit / "scripts"
    srv.mkdir(parents=True)
    files = {"v41_pod_serve.sh": b"#!/bin/bash\necho serve\n", "v41_d1_server.py": b"print('d1')\n"}
    hashes = {"serve.sh": hashlib.sha256(files["v41_pod_serve.sh"]).hexdigest(),
              "v41_d1_server.py": hashlib.sha256(files["v41_d1_server.py"]).hexdigest()}
    if tamper:
        files["v41_d1_server.py"] = b"print('evil')\n"
    for n, data in files.items():
        (srv / n).write_bytes(data)
    pin = v4_pod.ScriptsPin(commit, hashes)
    dest, log = tmp_path / "opt", tmp_path / "workers.log"
    frag = v4_pod.fetch_fragment(pin, str(dest), str(log), base=(tmp_path / "srv").as_uri())
    return frag, dest, log


def run_frag(frag, path=None):
    env = {**os.environ, **({"PATH": path} if path else {})}
    return subprocess.run(["bash", "-c", frag + "\necho REACHED_SERVE"], capture_output=True, text=True, env=env)


def test_fragment_downloads_and_verifies(tmp_path):
    frag, dest, log = fixture(tmp_path)
    r = run_frag(frag)
    assert r.returncode == 0 and "REACHED_SERVE" in r.stdout, r.stderr
    assert (dest / "serve.sh").read_bytes().startswith(b"#!/bin/bash") and not list(dest.glob("*.part"))
    assert log.read_text().count("verified ") == 2


def test_fragment_refuses_tampered_file(tmp_path):
    frag, dest, log = fixture(tmp_path, tamper=True)
    r = run_frag(frag)
    assert r.returncode != 0 and "REACHED_SERVE" not in r.stdout
    assert "HASH MISMATCH: v41_d1_server.py" in log.read_text() and "refusing to run" in log.read_text()
    assert not (dest / "v41_d1_server.py").exists() and not list(dest.glob("*.part"))


def test_fragment_falls_back_to_urllib_without_curl(tmp_path):
    frag, dest, log = fixture(tmp_path)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("python3", "bash", "mkdir", "mv", "rm", "sleep", "echo"):
        found = shutil.which(tool)
        if found:
            (bindir / tool).symlink_to(found)
    assert shutil.which("curl", path=str(bindir)) is None
    r = run_frag(frag, str(bindir))
    assert r.returncode == 0 and "REACHED_SERVE" in r.stdout, r.stderr + log.read_text()
    assert (dest / "v41_d1_server.py").exists()


def test_fragment_fails_on_missing_file(tmp_path):
    frag, dest, log = fixture(tmp_path)
    (tmp_path / "srv" / ("a" * 40) / "scripts" / "v41_d1_server.py").unlink()
    r = run_frag(frag.replace("sleep 3", "sleep 0"))
    assert r.returncode != 0 and "FETCH FAILED" in log.read_text()
