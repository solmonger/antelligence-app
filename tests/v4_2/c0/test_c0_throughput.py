"""C0 throughput tests: seeding, the sleeping fake backend, the dev-only gate, lag monitor, phase timers, attribution,
the sign-key cache (identical output) and that profiled episodes end at T=120 with identical metrics however they run.
Synthetic phantom only: no MRI data, no network, no keys."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from antelligence.experiments import v42_fake, v42_profile
from antelligence.experiments.v41_fake import FakeWorkerBackend
from antelligence.kernel import board as board_mod
from antelligence.worlds.tumor_mri import WorldConfig
from tests.v4_1.world.phantom import make_phantom

REPO = Path(__file__).resolve().parents[3]
DEV_CASE = json.loads((REPO / "config/v4_1/split.json").read_text())["dev_a"][0]
EVAL_CASE = json.loads((REPO / "config/v4_1/split.json").read_text())["eval"][0]


def table(scale: float = 1.0) -> v42_fake.LatencyTable:
    q = [round(1.0 + 0.1 * i, 3) for i in range(101)]            # 1.0 .. 11.0 s
    return v42_fake.LatencyTable({"d1": q, "qwen3.5-4b": q, "qwen3.5-9b": [x * 2 for x in q]})


def phantom_loader(case: str, variant: str = "full"):
    return make_phantom(flair=True), None


def cfg(**kw) -> v42_profile.ProfileConfig:
    base = dict(repo_root=str(REPO), cases_root="unused", latency_path="", time_scale=0.0, anchors=True)
    base.update(kw)
    return v42_profile.ProfileConfig(**base)


def harness(c: v42_profile.ProfileConfig, tmp_path: Path, tbl=None, timer=None):
    h = v42_profile.build_harness(c, timer, tmp_path, load_case=phantom_loader)
    if tbl is not None:
        h.workers = v42_fake.build_sleeping_workers(tbl, c.time_scale)
    return h


# ---- seeding (INTERFACES section 1) -------------------------------------------------------------------------

def test_seed64_is_the_interfaces_rule():
    expect = int.from_bytes(hashlib.sha256(b"c0_latency|d1|abc|3").digest()[:8], "big")
    assert v42_fake.seed64("c0_latency", "d1", "abc", 3) == expect
    a, b = v42_fake.rng("x", 1).random(4), v42_fake.rng("x", 1).random(4)
    assert np.array_equal(a, b) and not np.array_equal(a, v42_fake.rng("x", 2).random(4))


# ---- sleeping backend -----------------------------------------------------------------------------------------

def _request(h, model="d1"):
    from antelligence.kernel.decision import DecisionRequest, Question
    return DecisionRequest(model=h.workers[model].model, state="s", seed=1,
                           questions={"look": Question("noul", "q", criteria={"true": "a", "false": "b"})})


def test_latency_draw_is_seeded_bounded_and_scaled(tmp_path):
    t = table()
    w1 = v42_fake.build_sleeping_workers(t, 1.0)["d1"].backend
    w2 = v42_fake.build_sleeping_workers(t, 0.5)["d1"].backend
    h = harness(cfg(), tmp_path)
    req = _request(h)
    d1, d2 = w1.latency_s(req), w2.latency_s(req)
    assert 1.0 <= d1 <= 11.0 and d1 == w1.latency_s(req)          # inside the table, repeatable
    assert d2 == pytest.approx(d1 / 2)                            # time scale only multiplies
    req2 = dataclasses.replace(req, seed=2, state="other")
    assert w1.latency_s(req2) != d1                               # different request hash, different draw


def test_sleeping_backend_awaits_and_answers_like_the_fake(tmp_path):
    t = table()
    sb = v42_fake.build_sleeping_workers(t, 0.001)["d1"].backend
    plain = FakeWorkerBackend("d1")
    h = harness(cfg(), tmp_path)
    req = _request(h)
    t0 = time.perf_counter()
    resp = asyncio.run(sb.decide(req))
    assert time.perf_counter() - t0 >= sb.latency_s(req) * 0.9
    ref = asyncio.run(plain.decide(req))
    assert resp.answers == ref.answers and resp.cites == ref.cites
    assert resp.timing["network_ms"] == int(sb.latency_s(req) * 1000)
    assert sb.inner.requests == []                                 # the fake's request list is not left to grow


def test_latency_table_rejects_a_decreasing_row():
    with pytest.raises(ValueError):
        v42_fake.LatencyTable({"d1": [2.0, 1.0]}, [0.0, 1.0])


# ---- gate: development cases only -----------------------------------------------------------------------------

def test_dev_gate_refuses_eval_and_unknown_cases(tmp_path):
    gate = v42_fake.dev_only_gate(v42_profile.dev_cases(REPO))
    assert gate(DEV_CASE) == "dev"
    for bad in (EVAL_CASE, "NOT_A_CASE"):
        with pytest.raises(PermissionError):
            gate(bad)
    h = harness(cfg(), tmp_path)
    with pytest.raises(PermissionError):
        asyncio.run(h.run("SW-B", EVAL_CASE, 1))


# ---- lag monitor and phase timers -----------------------------------------------------------------------------

def test_lag_monitor_sees_a_blocked_loop():
    async def go():
        m = v42_profile.LagMonitor(0.01)
        m.start()
        await asyncio.sleep(0.05)
        time.sleep(0.12)                      # CPU work in the loop thread
        await asyncio.sleep(0.05)
        return await m.stop()
    s = asyncio.run(go())
    assert s["max_ms"] >= 80 and s["samples"] >= 3


def test_phase_timer_is_exclusive():
    t = v42_profile.PhaseTimer(ticks=10)

    def inner():
        time.sleep(0.03)

    def outer():
        time.sleep(0.02)
        w_inner()
        time.sleep(0.02)

    w_inner, w_outer = t.wrap(inner, "inner"), t.wrap(outer, "outer")
    w_outer()
    assert t.total["inner"] == pytest.approx(0.03, abs=0.015)
    assert t.total["outer"] == pytest.approx(0.04, abs=0.015)      # outer excludes inner
    assert t.calls["inner"] == 1 and t.calls["outer"] == 1


def test_instrument_restores_everything():
    from antelligence.worlds.tumor_mri.physics import DrugField
    before = DrugField.__dict__["step"]
    with v42_profile.instrument(v42_profile.PhaseTimer()):
        assert DrugField.__dict__["step"] is not before
    assert DrugField.__dict__["step"] is before


# ---- fix: cached signing key ----------------------------------------------------------------------------------

def test_sign_id_cache_is_bit_identical_to_parsing_each_time():
    for i in range(3):
        seed, eid = hashlib.sha256(f"s{i}".encode()).digest(), hashlib.sha256(f"e{i}".encode()).hexdigest()
        legacy = Ed25519PrivateKey.from_private_bytes(seed).sign(bytes.fromhex(eid)).hex()
        assert board_mod.sign_id(seed, eid) == legacy == board_mod.sign_id(bytearray(seed), eid)
    a, b = hashlib.sha256(b"a").digest(), hashlib.sha256(b"b").digest()
    assert board_mod.sign_id(a, "00" * 32) != board_mod.sign_id(b, "00" * 32)      # keyed by seed


# ---- episodes: T = 120, determinism, identical metrics however they run ---------------------------------------

@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("c0")
    h = harness(cfg(), tmp)
    res = asyncio.run(h.run("SW-B", DEV_CASE, 1))
    return res, v42_profile.fingerprint(res)


def test_episode_ends_at_tick_120_and_the_board_is_longer(baseline):
    res, fp = baseline
    assert res.run_log["ticks"] == 120 == fp["ticks"] == WorldConfig.from_yaml(str(REPO / "config/v4_1/world.yaml")).ticks
    assert res.bundle["metrics"]["ticks"] == 120
    assert res.cell["board_size"] > 120                 # the v4.1 "about tick 200" counted board entries, not ticks
    assert res.cell["board_size"] >= 200


def test_same_seed_same_metrics(baseline, tmp_path):
    res2 = asyncio.run(harness(cfg(), tmp_path).run("SW-B", DEV_CASE, 1))
    assert v42_profile.fingerprint(res2) == baseline[1]


def test_sleeping_does_not_change_metrics(baseline, tmp_path):
    c = cfg(time_scale=0.0005)
    res = asyncio.run(harness(c, tmp_path, tbl=table()).run("SW-B", DEV_CASE, 1))
    assert v42_profile.fingerprint(res) == baseline[1]


def test_phase_timers_do_not_change_metrics(baseline, tmp_path):
    timer = v42_profile.PhaseTimer()
    with v42_profile.instrument(timer):
        res = asyncio.run(harness(cfg(), tmp_path, timer=timer).run("SW-B", DEV_CASE, 1))
    assert v42_profile.fingerprint(res) == baseline[1]
    assert {"board_append", "world_diffusion", "brief_build", "conformal_router", "decision_entries", "observation",
            "mock_anchor", "bundle_write", "queen_step", "look_png_render"} <= set(timer.total)
    assert timer.calls["board_append"] == res.cell["board_size"] + 0 or timer.calls["board_append"] >= res.cell["board_size"]
    assert sum(timer.per_tick["board_append"]) == pytest.approx(timer.total["board_append"])


def test_concurrent_episodes_equal_sequential_ones(baseline, tmp_path):
    """Four episodes at once in one process give each case the metrics it gives alone (fixed seeds, fake backends)."""
    h = harness(cfg(time_scale=0.0003), tmp_path, tbl=table())
    results = asyncio.run(h.run_many([("SW-B", DEV_CASE, 1)] * 1 + [("SW-B", DEV_CASE, 2)] * 3, concurrency=4))
    assert not [r for r in results if isinstance(r, BaseException)]
    assert v42_profile.fingerprint(results[0]) == baseline[1]
    seq = asyncio.run(harness(cfg(), tmp_path / "s").run("SW-B", DEV_CASE, 2))
    assert all(v42_profile.fingerprint(r)["trace_hash"] == v42_profile.fingerprint(seq)["trace_hash"] for r in results[1:])


def test_board_off_and_scripted_arms_also_stable(tmp_path):
    for arm in ("SW-noBoard", "S", "S-prime"):
        c = cfg(arm=arm, anchors=False)
        a = asyncio.run(harness(c, tmp_path / f"{arm}a").run(arm, DEV_CASE, 1))
        b = asyncio.run(harness(c, tmp_path / f"{arm}b").run(arm, DEV_CASE, 1))
        assert v42_profile.fingerprint(a) == v42_profile.fingerprint(b)
        assert a.run_log["ticks"] == 120


# ---- attribution ----------------------------------------------------------------------------------------------

def _row(rnd, squad, calls, cf=None):
    return {"kind": "decision", "round": rnd, "squad": squad, "calls": calls, "counterfactual": cf or {}, "audit": None}


def _call(worker, net=0, server=0, queue=0, cached=False):
    return {"worker": worker, "network_ms": net, "server_ms": server, "queue_ms": queue, "cached": cached}


def test_attribution_critical_path_and_shares(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("v42_attribute", REPO / "scripts/v42_attribute.py")
    attr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attr)
    # calibration: parallel calls, the slowest is the squad's wait; the slowest squad is the round's wait
    rows = [_row(0, 0, [_call("qwen3.5-9b", net=4000)], {"d1": {"call": _call("d1", net=1000, server=100)}}),
            _row(0, 1, [_call("qwen3.5-9b", net=9000, queue=500)], {"d1": {"call": _call("d1", net=2000)}}),
            _row(1, 0, [_call("qwen3.5-9b", net=3000)], {})]
    ep = attr.episode({"calibration": True, "wall_s": 20.0}, rows, use_cached=False)
    assert ep["modelled_wait_s"] == pytest.approx(9.5 + 3.0)
    assert ep["components_s"] == {"server": 0.0, "network": pytest.approx(12.0), "queue": pytest.approx(0.5)}
    assert ep["remainder_s"] == pytest.approx(20.0 - 12.5)
    # a cached call waited nothing in that run
    cached = [_row(0, 0, [_call("qwen3.5-9b", net=9000, cached=True)])]
    assert attr.episode({"calibration": True, "wall_s": 5.0}, cached, use_cached=False)["modelled_wait_s"] == 0.0
    assert attr.episode({"calibration": True, "wall_s": 5.0}, cached, use_cached=True)["modelled_wait_s"] == pytest.approx(9.0)
    # router mode: d1 first, then the slowest of the rest
    router = [_row(0, 0, [_call("d1", net=2000)], {"qwen3.5-9b": {"call": _call("qwen3.5-9b", net=6000)},
                                                    "qwen3.5-4b": {"call": _call("qwen3.5-4b", net=3000)}})]
    ep = attr.episode({"calibration": False, "wall_s": 10.0}, router, use_cached=True)
    assert ep["modelled_wait_s"] == pytest.approx(8.0) and ep["whatif_overlap_wait_s"] == pytest.approx(6.0)


def test_latency_table_from_logs_is_monotone(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("v42_attribute", REPO / "scripts/v42_attribute.py")
    attr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attr)
    run = tmp_path / "calib1" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "run_log.json").write_text(json.dumps({"calibration": True, "wall_s": 1.0, "arm": "SW-B", "ticks": 120, "rounds": 24}))
    rows = [_row(0, s, [_call("qwen3.5-9b", net=1000 + 100 * s)]) for s in range(8)]
    (run / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    doc = attr.latency_distribution(tmp_path)
    q = doc["models"]["qwen3.5-9b"]["q"]
    assert len(q) == 101 and q == sorted(q) and q[0] == 1.0 and q[-1] == 1.7
    v42_fake.LatencyTable({m: v["q"] for m, v in doc["models"].items()}, doc["quantiles"])      # loads cleanly
