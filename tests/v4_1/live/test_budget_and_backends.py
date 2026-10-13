"""Budget hard stops, ledgered paid calls, and the production backend builders (no network)."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
from pathlib import Path

import pytest

from antelligence.experiments import v41_budget, v41_prod
from antelligence.experiments.v41_harness import BONSAI, D1, FW, Q4, Q9
from tests.v4_1.workers.helpers import BRIEF, Recorder, make_request

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
FAKE_POD_SECRET = "podsecret_FAKE_0123456789"
FAKE_NOUS = "nous_FAKE_KEY_abcdef"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_nous_stops_at_7_50(tmp_path):
    usage = tmp_path / "u.jsonl"
    write_rows(usage, [{"backend": "nous_fw", "model": "deepseek/deepseek-v4.1-flash", "cost_usd": 7.49}])
    assert v41_budget.check_or_refuse("nous", usage=usage, pods=None) == pytest.approx(7.49)
    write_rows(usage, [{"backend": "nous_fw", "model": "deepseek/x", "cost_usd": 7.49}, {"backend": "nous", "model": "d", "cost_usd": 0.01}])
    with pytest.raises(v41_budget.BudgetRefused, match="nous hard stop"):
        v41_budget.check_or_refuse("nous", usage=usage, pods=None)


def test_pod_stops_at_24_and_a_running_pod_keeps_billing(tmp_path):
    pods = tmp_path / "p.jsonl"
    start = (NOW - timedelta(hours=23.5)).isoformat()
    write_rows(pods, [{"event": "start", "pod_id": "p1", "utc": start, "rate_per_hr": 1.0}])
    assert v41_budget.check_or_refuse("runpod", usage=None, pods=pods, now=NOW) == pytest.approx(23.5)
    with pytest.raises(v41_budget.BudgetRefused, match="runpod hard stop"):
        v41_budget.check_or_refuse("runpod", usage=None, pods=pods, now=NOW + timedelta(minutes=40))
    write_rows(pods, [{"event": "start", "pod_id": "p1", "utc": start, "rate_per_hr": 1.0},
                      {"event": "stop", "pod_id": "p1", "utc": (NOW - timedelta(hours=1)).isoformat(), "rate_per_hr": 1.0}])
    assert v41_budget.check_or_refuse("runpod", usage=None, pods=pods, now=NOW + timedelta(days=3)) == pytest.approx(22.5)


def test_preflight_probe_rows_without_cost_are_priced(tmp_path):
    usage = tmp_path / "u.jsonl"
    write_rows(usage, [{"provider": "nous", "model": "deepseek/deepseek-v4.1-flash",
                        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}])
    assert v41_budget.spend_now(usage, None)["nous"] == pytest.approx(0.106 + 0.513)


def test_unknown_provider_is_a_key_error(tmp_path):
    with pytest.raises(KeyError):
        v41_budget.check_or_refuse("openai", usage=None, pods=None)


class Spy:
    model = "m"

    def __init__(self):
        self.calls = 0

    def describe(self):
        return {"backend": "spy"}

    async def decide(self, request):
        self.calls += 1
        raise AssertionError("must not be reached")


def test_guard_refuses_before_the_backend_is_called(tmp_path):
    usage = tmp_path / "u.jsonl"
    write_rows(usage, [{"backend": "nous_fw", "model": "deepseek/x", "cost_usd": 7.6}])
    spy = Spy()
    g = v41_prod.GuardedDecision(spy, v41_prod.BudgetGuard("nous", usage, tmp_path / "none.jsonl"))
    with pytest.raises(v41_budget.BudgetRefused):
        asyncio.run(g.decide(make_request("m")))
    assert spy.calls == 0


def test_guarded_chat_refuses(tmp_path):
    usage = tmp_path / "u.jsonl"
    write_rows(usage, [{"backend": "nous", "model": "deepseek/x", "cost_usd": 8.0}])

    class Inner:
        called = False

        def describe(self):
            return {}

        async def complete(self, r):
            Inner.called = True

    g = v41_prod.GuardedChat(Inner(), v41_prod.BudgetGuard("nous", usage, tmp_path / "none.jsonl"))
    with pytest.raises(v41_budget.BudgetRefused):
        asyncio.run(g.complete(object()))
    assert not Inner.called


# --- builders ---------------------------------------------------------------------------------------------------


def pod_state(tmp_path, base="https://pod123-8100.proxy.runpod.net/v1"):
    p = tmp_path / "ledger" / "v4.1-pod-state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pod_id": "pod123", "api_secret": FAKE_POD_SECRET, "base_url": base}))
    return p


def test_pod_endpoint_urls_and_secret_hidden_in_repr(tmp_path):
    pod = v41_prod.read_pod_endpoint(pod_state(tmp_path))
    assert pod.urls == {Q9: "https://pod123-8100.proxy.runpod.net", Q4: "https://pod123-8101.proxy.runpod.net",
                        D1: "https://pod123-8102.proxy.runpod.net"}
    assert FAKE_POD_SECRET not in repr(pod) and FAKE_POD_SECRET not in str(pod)
    with pytest.raises(v41_prod.ProdConfigError, match="not found"):
        v41_prod.read_pod_endpoint(tmp_path / "missing.json")
    with pytest.raises(v41_prod.ProdConfigError, match=":8100"):
        v41_prod.read_pod_endpoint(pod_state(tmp_path, "https://pod123-9999.proxy.runpod.net/v1"))


def test_nous_key_read_from_env_file_only(tmp_path, monkeypatch):
    env = tmp_path / "hermes.env"
    env.write_text(f"OTHER=1\nNOUS_API_KEY={FAKE_NOUS}\n")
    assert v41_prod.read_nous_key(env) == FAKE_NOUS
    monkeypatch.setenv(v41_prod.NOUS_ENV_FILE_VAR, str(env))        # the override variable
    assert v41_prod.read_nous_key() == FAKE_NOUS
    import os
    assert "NOUS_API_KEY" not in os.environ                        # returned to the caller only
    for bad in ("OTHER=1\n", "NOUS_API_KEY=\n", "NOUS_API_KEY=   \n"):
        env.write_text(bad)
        with pytest.raises(v41_prod.ProdConfigError, match="NOUS_API_KEY not available") as ei:
            v41_prod.read_nous_key(env)
        assert FAKE_NOUS not in str(ei.value)
    with pytest.raises(v41_prod.ProdConfigError, match="NOUS_API_KEY not available"):
        v41_prod.read_nous_key(tmp_path / "missing.env")


def test_nothing_reads_hermes_auth_json():
    """Hermes's login token is off limits (decisions/2026-10-13-nous-dedicated-key.md)."""
    root = Path(v41_prod.__file__).resolve().parents[2]
    files = list((root / "antelligence").rglob("*.py")) + list((root / "scripts").glob("v41_*"))
    hits = [str(f.relative_to(root)) for f in files if f.is_file() and "auth.json" in f.read_text(errors="ignore")]
    assert not hits, hits


def test_build_live_workers_pins_and_only_builds_what_is_needed(tmp_path):
    w = v41_prod.build_live_workers(pod_state=pod_state(tmp_path), usage_ledger=tmp_path / "u.jsonl",
                                    pod_costs=tmp_path / "pc.jsonl", cache_dir=tmp_path / "c", needed={D1, Q9, Q4, BONSAI},
                                    nous_key=lambda: pytest.fail("Nous key must not be read when FW is not needed"))
    assert set(w) == {D1, Q9, Q4, BONSAI}
    assert w[Q9].model == "RedHatAI/Qwen3.5-9B-FP8-dynamic@790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"
    assert w[Q4].model.startswith("RedHatAI/Qwen3.5-4B-FP8-dynamic@397b7ba4")
    assert w[BONSAI].model.startswith("prism-ml/Ternary-Bonsai-8B-gguf@")
    only_bonsai = v41_prod.build_live_workers(pod_state=tmp_path / "no-pod.json", needed={BONSAI}, usage_ledger=tmp_path / "u.jsonl",
                                              cache_dir=None, nous_key=lambda: pytest.fail("no key"))
    assert set(only_bonsai) == {BONSAI}                      # a Mac-only run never touches the pod state


def test_pinned_ids_match_the_pod_serve_script():
    text = (v41_prod.REPO / "scripts/v41_pod_serve.sh").read_text()
    for value in (v41_prod.Q9_ID, v41_prod.Q9_REV, v41_prod.Q4_ID, v41_prod.Q4_REV):
        assert value in text
    revs = (v41_prod.REPO / "config/v4_1/model_revisions.json").read_text()
    assert v41_prod.BONSAI_REV in revs


def test_every_paid_fw_call_writes_a_ledger_row_with_cost(tmp_path):
    reply = {"model": "deepseek/deepseek-v4.1-flash", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
        "target": {"A": 0.5, "B": 0.3, "C": 0.2}, "release": 0.9, "amount": {"0": 0.0, "1": 0.1, "2": 0.6, "3": 0.2, "4": 0.1}})}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
    rec = Recorder(lambda r: httpx.Response(200, json=reply))
    usage = tmp_path / "u.jsonl"
    w = v41_prod.build_live_workers(needed={FW}, usage_ledger=usage, pod_costs=tmp_path / "pc.jsonl", cache_dir=None,
                                    nous_key=lambda: FAKE_NOUS, transport=rec.transport())
    req = make_request(w[FW].model)
    resp = asyncio.run(w[FW].backend.decide(req))
    rows = [json.loads(l) for l in usage.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["backend"] == "nous_fw" and rows[0]["cost_usd"] > 0
    assert rows[0]["input_tokens"] == 1000 and resp.cost_usd == pytest.approx(rows[0]["cost_usd"])
    assert FAKE_NOUS not in usage.read_text()
    # at the stop the next call is refused and writes no row
    write_rows(usage, [{"backend": "nous_fw", "model": "deepseek/x", "cost_usd": 7.50}])
    n = len(rec.requests)
    with pytest.raises(v41_budget.BudgetRefused):
        asyncio.run(w[FW].backend.decide(req))
    assert len(rec.requests) == n


def test_queen_is_ledgered_guarded_and_cached(tmp_path):
    q = v41_prod.build_live_queen(usage_ledger=tmp_path / "u.jsonl", pod_costs=tmp_path / "pc.jsonl", cache_dir=tmp_path / "c",
                                  nous_key=lambda: FAKE_NOUS)
    from antelligence.providers.wrappers import Cached

    assert isinstance(q, Cached)
    assert isinstance(q.inner, v41_prod.GuardedChat) and q.inner.guard.provider == "nous"
    assert q.inner.inner.role == "queen" and q.inner.inner.writer.path == tmp_path / "u.jsonl"
