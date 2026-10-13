"""scripts/v41_pod.py with mock transports only: no network, no real key."""
from __future__ import annotations

import json
import stat
import subprocess

import pytest

import httpx

from scripts import v41_pod, v4_pod

SHA = subprocess.run(["git", "rev-parse", "HEAD"], cwd=v4_pod.ROOT, capture_output=True, text=True).stdout.strip()
FAKE_KEY = "rpa_FAKEKEY_do_not_print_123456"


def make(tmp_path, quote_price=1.0, log=None):
    log = [] if log is None else log

    def handler(req: httpx.Request) -> httpx.Response:
        log.append((req.method, str(req.url), req.content.decode() if req.content else ""))
        if req.url.host == "api.runpod.io":
            return httpx.Response(200, json={"data": {"gpuTypes": [{
                "id": "NVIDIA L40S", "securePrice": quote_price, "communityPrice": 0.7,
                "lowestPrice": {"stockStatus": "High", "availableGpuCounts": [1, 2]}}]}})
        if req.method == "POST":
            return httpx.Response(201, json={"id": "pod1", "costPerHr": quote_price})
        if req.method == "DELETE":
            return httpx.Response(200)
        return httpx.Response(200, json={"desiredStatus": "RUNNING"})

    paths = v4_pod.Paths(state=tmp_path / "ledger" / "v4.1-pod-state.json",
                         costs=tmp_path / "ledger" / "v4.1-pod-costs.jsonl")
    return httpx.MockTransport(handler), paths, log


def run(argv, transport, paths):
    out: list = []
    rc = v41_pod.main(argv, paths=paths, transport=transport, api_key=FAKE_KEY, echo=out.append)
    return rc, out


def test_quote_uses_graphql_readonly(tmp_path):
    transport, paths, log = make(tmp_path)
    rc, out = run(["quote"], transport, paths)
    assert rc == 0 and len(log) == 1
    method, url, body = log[0]
    assert method == "POST" and url == v4_pod.GRAPHQL_URL
    query = json.loads(body)["query"]
    assert query.lstrip().startswith("query") and "mutation" not in query.lower()
    data = json.loads(out[0])
    assert data["ok"] and data["secure_per_hr"] == 1.0 and data["community_per_hr"] == 0.7
    assert data["availability"] == "High"
    assert not paths.state.exists() and not paths.costs.exists()


def test_up_refuses_over_v41_budget(tmp_path):
    transport, paths, log = make(tmp_path, quote_price=1.0)
    v4_pod.append_cost(paths.costs, v4_pod.cost_row("start", "old", 9.0, "2026-10-13T00:00:00+00:00"))
    v4_pod.append_cost(paths.costs, v4_pod.cost_row("stop", "old", 9.0, "2026-10-13T01:00:00+00:00"))
    rc, out = run(["up", "v41-workers", "--scripts-commit", SHA], transport, paths)  # 9.00 spent + 1.00 > 9.50
    assert rc == 1 and any("hard stop" in line and "9.50" in line for line in out)
    assert not any(m == "POST" and "rest.runpod.io" in u for m, u, _ in log)
    assert not paths.state.exists()


def test_ledger_path_is_v41(tmp_path):
    assert v41_pod.v41_paths().costs.name == "v4.1-pod-costs.jsonl"
    assert v41_pod.v41_paths().state.name == "v4.1-pod-state.json"
    assert v41_pod.HARD_STOP_USD == 9.50
    transport, paths, _ = make(tmp_path)
    rc, _ = run(["up", "v41-workers", "--scripts-commit", SHA, "--auto-down-after-min", "30"], transport, paths)
    assert rc == 0
    assert stat.S_IMODE(paths.state.stat().st_mode) == 0o600
    rows = [json.loads(l) for l in paths.costs.read_text().splitlines()]
    assert rows[0]["event"] == "start" and rows[0]["image"] == v4_pod.VLLM_IMAGE
    state = json.loads(paths.state.read_text())
    assert state["image"] == v4_pod.VLLM_IMAGE and state["auto_down_deadline_utc"]
    rc, out = run(["status"], transport, paths)
    assert any("auto-down deadline" in l for l in out)
    assert v4_pod.HARD_STOP_USD == 2.75  # v4 bindings restored after the command


def test_down_idempotent(tmp_path):
    transport, paths, _ = make(tmp_path)
    run(["up", "v41-workers", "--scripts-commit", SHA], transport, paths)
    assert run(["down"], transport, paths)[0] == 0
    assert run(["down"], transport, paths)[0] == 0
    rows = [json.loads(l) for l in paths.costs.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["start", "stop"] and rows[1]["image"]
    assert not paths.state.exists()


def test_cost_warns_after_deadline(tmp_path):
    transport, paths, _ = make(tmp_path)
    run(["up", "v41-workers", "--scripts-commit", SHA, "--auto-down-after-min", "30"], transport, paths)
    state = json.loads(paths.state.read_text())
    state["auto_down_deadline_utc"] = "2020-01-01T00:00:00+00:00"
    v4_pod.save_state(paths.state, state)
    assert any("WARNING" in l for l in run(["cost"], transport, paths)[1])


def test_unpinned_image_refused(tmp_path):
    transport, paths, log = make(tmp_path)
    rc, _ = run(["up", "v41-workers", "--scripts-commit", SHA, "--image", "vllm/vllm-openai:latest"], transport, paths)
    assert rc == 2 and not log


def test_key_never_printed(tmp_path, capsys):
    transport, paths, _ = make(tmp_path)
    for argv in (["quote"], ["up", "v41-workers", "--scripts-commit", SHA, "--auto-down-after-min", "5"], ["status"], ["cost"], ["down"]):
        v41_pod.main(argv, paths=paths, transport=transport, api_key=FAKE_KEY)
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.out + captured.err
    assert "api_secret" not in captured.out


def test_up_requires_scripts_commit(tmp_path):
    transport, paths, log = make(tmp_path)
    with pytest.raises(SystemExit):
        run(["up", "v41-workers"], transport, paths)
    assert not log


@pytest.mark.parametrize("bad", ["abc", "g" * 40, "0" * 40])
def test_up_refuses_bad_scripts_commit_before_any_call(tmp_path, bad):
    transport, paths, log = make(tmp_path)
    rc, out = run(["up", "v41-workers", "--scripts-commit", bad], transport, paths)
    assert rc == 2 and any("refused" in l for l in out) and not log and not paths.state.exists()


def test_up_records_commit_and_hashes_in_state_ledger_and_payload(tmp_path):
    transport, paths, log = make(tmp_path)
    rc, _ = run(["up", "v41-workers", "--scripts-commit", SHA], transport, paths)
    assert rc == 0
    hashes = v4_pod.scripts_pin(SHA).hashes
    state = json.loads(paths.state.read_text())
    assert state["scripts_commit"] == SHA and state["scripts_sha256"] == hashes
    assert state["log_url"].endswith("/workers.log")
    start = json.loads(paths.costs.read_text().splitlines()[0])
    assert start["scripts_commit"] == SHA and start["scripts_sha256"] == hashes
    post = [b for m, u, b in log if m == "POST" and "rest.runpod.io" in u][0]
    cmd = json.loads(post)["dockerStartCmd"][0]
    assert SHA in cmd and all(h in cmd for h in hashes.values()) and len(cmd) < 8000
