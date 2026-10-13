"""Split-conformal deferral: 20% target, empty set defers, deterministic 10% audit, defer-rate alarm band, freezing."""
from __future__ import annotations

import math

import numpy as np
import pytest

from antelligence.experiments.v41_conformal import (ALARM_BAND, ConformalDeferral, ConformalError, ConformalThresholds,
                                                    DeferralMonitor, DevRecord, audit_coverage, audit_selected,
                                                    choose_alpha, conformal_quantile, defer_rate_alarm,
                                                    freeze_thresholds, prediction_set)

TARGETS = ("A", "B", "C", "S", "X")
LEVELS = ("1", "2", "3", "4", "5")


def synthetic(n: int, seed: int, agree: float = 0.9, scale: tuple = (1.5, 4.5)) -> list:
    """d1 probabilities of varying sharpness; the deferral target agrees with d1's argmax ``agree`` of the time."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        probs, labels = {}, {}
        for q, names in (("target", TARGETS), ("release", ("yes", "no")), ("amount", LEVELS)):
            logits = rng.normal(0, rng.uniform(*scale), len(names))
            p = np.exp(logits) / np.exp(logits).sum()
            probs[q] = dict(zip(names, p.tolist()))
            top = int(np.argmax(p))
            labels[q] = names[top] if rng.random() < agree else names[int(rng.integers(len(names)))]
        out.append(DevRecord(probs, labels))
    return out


def rate(records, qhat_by_q) -> dict:
    return {q: np.mean([len(prediction_set(r.probs[q], qhat_by_q[q])) != 1 for r in records]) for q in qhat_by_q}


def test_conformal_target_20pct_on_synthetic():
    dev = synthetic(2000, seed=1)
    th = freeze_thresholds(dev, arm="main", target_model="qwen3.5-9b")
    for q in ("target", "release", "amount"):
        info = th.per_question[q]
        assert info["n"] == 2000 and 0.0 < info["alpha"] < 1.0
        assert info["dev_defer_rate"] == pytest.approx(0.20, abs=0.02) and info["on_target"], (q, info)
    # the dev rate is what the thresholds reproduce on the dev records, and it generalises to fresh draws
    fresh = synthetic(4000, seed=2)
    got = rate(fresh, {q: th.qhat(q) for q in ("target", "release", "amount")})
    for q, r in got.items():
        assert r == pytest.approx(0.20, abs=0.04), (q, r)
    assert th.joint_dev_defer_rate >= max(th.per_question[q]["dev_defer_rate"] for q in th.per_question) - 1e-9
    # a d1 that is rarely sure cannot reach 20% (sets are never singletons): the nearest rate is reported and flagged
    poor = freeze_thresholds(synthetic(2000, seed=1, scale=(0.1, 0.5)), arm="main", target_model="t")
    assert not all(v["on_target"] for v in poor.per_question.values())
    assert all(v["dev_defer_rate"] > 0.2 for v in poor.per_question.values() if not v["on_target"])
    # quantile arithmetic: rank = ceil((n+1)(1-alpha)); beyond n -> +inf (every label in the set)
    assert conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.5) == 0.3
    assert math.isinf(conformal_quantile([0.1, 0.2], 0.1))
    with pytest.raises(ConformalError):
        conformal_quantile([], 0.1)
    with pytest.raises(ConformalError):
        choose_alpha([], "target")


def test_empty_set_defers():
    # d1 is unsure about everything: top probability 0.4. With a tight qhat (< 0.6) no label qualifies -> empty set.
    flat = {"target": {"A": 0.4, "B": 0.3, "C": 0.2, "S": 0.05, "X": 0.05}, "release": {"yes": 0.5, "no": 0.5},
            "amount": {l: 0.2 for l in LEVELS}}
    th = ConformalThresholds("main", "qwen3.5-9b",
                             {q: {"alpha": 0.5, "qhat": 0.1, "n": 10, "dev_defer_rate": 0.2} for q in flat}, 0.2)
    assert prediction_set(flat["target"], 0.1) == ()
    res = ConformalDeferral(th).decide(flat)
    assert res.deferred and res.empty and all(len(s) == 0 for s in res.sets.values())
    # a confident, singleton answer does not defer
    sharp = {"target": {"A": 0.97, "B": 0.01, "C": 0.01, "S": 0.005, "X": 0.005}, "release": {"yes": 0.98, "no": 0.02},
             "amount": {"1": 0.01, "2": 0.01, "3": 0.96, "4": 0.01, "5": 0.01}}
    ok = ConformalDeferral(th).decide(sharp)
    assert not ok.deferred and ok.sets["target"] == ("A",) and ok.sets["amount"] == ("3",)
    # more than one label also defers
    wide = ConformalThresholds("main", "t", {q: {"alpha": 0.1, "qhat": 0.8, "n": 10, "dev_defer_rate": 0.2} for q in flat}, 0.2)
    assert ConformalDeferral(wide).decide(flat).deferred


def test_freeze_roundtrip_and_tamper(tmp_path):
    th = freeze_thresholds(synthetic(300, seed=3), arm="bonsai", target_model="bonsai-8b")
    path = tmp_path / "thresholds.json"
    digest = th.save(path)
    again = ConformalThresholds.load(path, expect_sha256=digest)
    assert again.sha256 == digest and again.arm == "bonsai" and again.per_question == th.per_question
    inf_th = ConformalThresholds("main", "t", {q: {"alpha": 0.01, "qhat": math.inf, "n": 5, "dev_defer_rate": 1.0}
                                               for q in ("target", "release", "amount")}, 1.0)
    inf_th.save(tmp_path / "inf.json")
    assert math.isinf(ConformalThresholds.load(tmp_path / "inf.json").qhat("target"))
    path.write_text(path.read_text().replace('"alpha":', '"alpha":0.5,"x":', 1))
    with pytest.raises(ConformalError):
        ConformalThresholds.load(path)
    with pytest.raises(ConformalError):
        ConformalThresholds.load(tmp_path / "inf.json", expect_sha256="0" * 64)


def test_audit_sample_deterministic_10pct():
    keys = [f"run-1|{squad}|{rnd}" for squad in range(8) for rnd in range(1250)]  # 10,000 decisions
    picked = [audit_selected(k) for k in keys]
    assert picked == [audit_selected(k) for k in keys]  # deterministic
    assert sum(picked) / len(keys) == pytest.approx(0.10, abs=0.01)
    assert picked != [audit_selected(k, salt="other") for k in keys]  # salt changes the sample
    assert not any(audit_selected(k, rate=0.0) for k in keys[:200]) and all(audit_selected(k, rate=1.0) for k in keys[:200])
    cov = audit_coverage([({"target": ("A",), "release": ("yes",), "amount": ("3",)}, {"target": "A", "release": "yes", "amount": "3"}),
                          ({"target": ("A",), "release": ("yes",), "amount": ("3",)}, {"target": "B", "release": "yes", "amount": "3"})])
    assert cov == {"target": 0.5, "release": 1.0, "amount": 1.0, "all": 0.5}
    assert audit_coverage([]) == {}


def test_defer_alarm_band():
    assert ALARM_BAND == (0.05, 0.40)
    assert defer_rate_alarm(10, 100) is None and defer_rate_alarm(5, 100) is None and defer_rate_alarm(40, 100) is None
    low, high = defer_rate_alarm(4, 100), defer_rate_alarm(41, 100)
    assert low["kind"] == "defer_rate" and low["value_ppm"] == 40_000
    assert high["value_ppm"] == 410_000 and len(high["note"]) <= 140
    assert set(low) == {"kind", "value_ppm", "note"}  # INTERFACES section 4 alarm payload, integers only
    assert defer_rate_alarm(0, 0) is None
    mon = DeferralMonitor()
    for i in range(20):
        mon.record(i < 10)  # 50% deferred
    assert mon.finish()["value_ppm"] == 500_000
