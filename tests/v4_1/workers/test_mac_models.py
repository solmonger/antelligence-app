"""Mac-local model setup (docs/v4.1/MAC-MODELS.md): d1 loads from the pinned snapshot dir; Bonsai launcher flags."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _shim():
    spec = importlib.util.spec_from_file_location("v41_d1_server_mac", REPO / "scripts" / "v41_d1_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_d1_snapshot_is_requested_at_the_pinned_revision(monkeypatch):
    shim = _shim()
    import huggingface_hub

    seen = {}

    def fake(repo, **kw):
        seen.update(repo=repo, **kw)
        return "/cache/snapshots/" + kw["revision"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert shim.snapshot_path(shim.D1_REVISION).endswith(shim.D1_REVISION)
    assert seen == {"repo": "LiquidAI/d1-3B", "revision": "051bcc464b01b9f92942b364d9586b0ef5912432"}


def test_d1_shim_logs_500_tracebacks_server_side_only():
    text = (REPO / "scripts" / "v41_d1_server.py").read_text()
    assert "traceback.print_exc()" in text and 'self._send(500, {"error": type(exc).__name__})' in text


def test_bonsai_launcher_flags_and_syntax():
    script = REPO / "scripts" / "v41_bonsai_server.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    for flag in ("-c 32768", "-ngl 99", "--jinja", "--port", "--alias", "c2aefbeb4b24469cd11579c3384b990404c17a30",
                 "Ternary-Bonsai-8B-Q2_0_g64.gguf"):
        assert flag in text
    assert "prefill" in text  # documents why no empty-think prefill is needed
