"""Bonsai calibration mode (goal 4.5, step 9): Bonsai 8B drives, d1 and Qwen3.5-4B answer counterfactually, the rows are
what ``v41_fit.samples_from(bonsai=True)`` reads, and the fitted files drive a SW-Bonsai run. Fakes only."""
import asyncio
import json

import pytest

from antelligence.experiments import v41_fit
from antelligence.experiments.v41_conformal import ConformalThresholds
from antelligence.experiments.v41_harness import (BONSAI, D1, Q4, Q9, ROUTER_MODELS_BY_ARM, HarnessError, KeyVault,
                                                  V41Harness)
from antelligence.experiments.v41_router import KNNRouter
from tests.v4_1.harness.conftest import CASE_DEV, CASE_EVAL, build_rig, fake_workers, small_cfg
from tests.v4_1.harness.test_harness import run

SEEDS = (1, 2, 3)


def split_file(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"dev_a": [CASE_DEV], "dev_b": [], "eval": [CASE_EVAL], "reserve": []}))
    return p


def calibrate(tmp_path, arm="SW-Bonsai", seeds=SEEDS):
    rig = build_rig(tmp_path, vault=KeyVault())
    return rig, [run(rig, arm, CASE_DEV, s, calibration=True) for s in seeds]


def test_bonsai_drives_and_the_9b_is_not_called(tmp_path):
    rig, (res, *_) = calibrate(tmp_path, seeds=(1,))
    assert set(rig.calls) == {D1, Q4, BONSAI}            # never Q9
    assert res.board.header["arm"] == "SW-Bonsai:calibration"
    assert res.run_log["calibration"] is True and res.bundle["spec"]["calibration"] is True
    for r in res.decision_log:
        assert r["backend"] == BONSAI and r["router_pick"] == BONSAI and r["deferred"] == 0 and r["escalated"] == 0
        assert set(r["counterfactual"]) == {D1, Q4}
        assert r["sprime"] and len(r["features"]) == 11
        assert [c["worker"] for c in r["calls"]] == [BONSAI]
    assert {e.payload["backend"] for e in res.board.by_type("decision")} == {BONSAI}


def test_check_ready_needs_d1_q4_bonsai_and_no_9b(tmp_path):
    full = fake_workers()
    only = {k: full[k] for k in (D1, Q4, BONSAI)}
    run(build_rig(tmp_path / "ok", workers=only), "SW-Bonsai", calibration=True)         # no Q9 needed
    for gone in (D1, Q4, BONSAI):
        w = {k: v for k, v in only.items() if k != gone}
        with pytest.raises(HarnessError, match="needs workers"):
            run(build_rig(tmp_path / f"no-{gone}", workers=w), "SW-Bonsai", calibration=True)
    # SW-B calibration still needs the 9B and not Bonsai
    run(build_rig(tmp_path / "b", workers={k: full[k] for k in (D1, Q4, Q9)}), "SW-B", calibration=True)


def test_calibration_scope_stays_narrow(tmp_path):
    with pytest.raises(HarnessError):
        run(build_rig(tmp_path / "e"), "SW-Bonsai", CASE_EVAL, calibration=True)
    for arm in ("SW-D", "SW-staticQueen", "FW", "S"):
        with pytest.raises(HarnessError):
            run(build_rig(tmp_path / arm), arm, calibration=True)
    with pytest.raises(HarnessError):       # counterfactual-all is not combined with calibration
        run(build_rig(tmp_path / "cf"), "SW-Bonsai", calibration=True, log_counterfactuals="all")


def test_fit_end_to_end_then_a_run_with_the_fitted_files(tmp_path):
    _, results = calibrate(tmp_path / "calib")
    split = split_file(tmp_path)
    conf, rout = tmp_path / "conformal_bonsai.json", tmp_path / "router_bonsai.json"
    with pytest.raises(v41_fit.FitError, match="--bonsai"):                          # Bonsai logs need --bonsai
        v41_fit.run_fit([tmp_path / "calib" / "out"], tag="t", split_file=split)
    rep = v41_fit.run_fit([tmp_path / "calib" / "out"], tag="b", bonsai=True, split_file=split, out_conformal=conf,
                          out_router=rout, report_path=tmp_path / "fit.json")
    assert rep["variant"] == "bonsai" and rep["inputs"]["usable_decisions"] == sum(len(r.decision_log) for r in results)
    assert conf.is_file() and rout.is_file()
    th, router = ConformalThresholds.load(conf), KNNRouter.load(rout)
    assert tuple(router.models) == ROUTER_MODELS_BY_ARM["SW-Bonsai"]
    assert th.target_model == BONSAI and rep["router"]["models"] == list(ROUTER_MODELS_BY_ARM["SW-Bonsai"])
    assert v41_fit.DEFERRAL_CANDIDATES[True] == (BONSAI,)
    assert {m for m in (th.model_for(q) for q in ("target", "release", "amount")) if m} <= {BONSAI}
    # a fake SW-Bonsai run with the fitted files passes _check_ready and runs
    h = V41Harness(load_case=build_rig(tmp_path / "x").harness.load_case, gate=lambda c: "dev",
                   workers=fake_workers(), queen_provider=build_rig(tmp_path / "y").harness.queen_provider,
                   world_config=small_cfg(), bonsai_router=router, bonsai_thresholds=th, router=None, thresholds=None,
                   allow_unanchored=True, keys=KeyVault(), out_dir=tmp_path / "live")
    res = asyncio.run(h.run("SW-Bonsai", CASE_DEV, 7))
    assert res.board.header["arm"] == "SW-Bonsai" and res.decision_log
    assert {r["router_pick"] for r in res.decision_log} <= set(ROUTER_MODELS_BY_ARM["SW-Bonsai"])


def test_fit_arm_consistency(tmp_path):
    split = json.loads(split_file(tmp_path).read_text())
    _, _ = calibrate(tmp_path / "bon", seeds=(1,))
    rig = build_rig(tmp_path / "b")
    run(rig, "SW-B", CASE_DEV, 1, calibration=True)
    run(rig, "SW-noBoard", CASE_DEV, 1, calibration=True)
    (bon,) = v41_fit.discover([tmp_path / "bon" / "out"])
    others = v41_fit.discover([tmp_path / "b" / "out"])
    assert len(others) == 2
    with pytest.raises(v41_fit.FitError, match="--bonsai"):
        v41_fit.load_run(bon, split)
    assert v41_fit.load_run(bon, split, bonsai=True).arm == "SW-Bonsai"
    for d in others:
        assert v41_fit.load_run(d, split).arm in ("SW-B", "SW-noBoard")
        with pytest.raises(v41_fit.FitError, match="SW-Bonsai"):
            v41_fit.load_run(d, split, bonsai=True)
    # eval cases stay refused in every mode
    eval_split = {**split, "dev_a": [], "eval": [CASE_DEV]}
    for flag, d in ((True, bon), (False, others[0])):
        with pytest.raises(v41_fit.FitError, match="eval"):
            v41_fit.load_run(d, eval_split, bonsai=flag)
