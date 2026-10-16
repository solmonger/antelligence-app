"""v4.31 Step D aggregator on fabricated rows (no case data): structure, the bootstrap, and the placement-free conservation bound."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments import v431_phaseA as PA
from antelligence.worlds.tumor_mri import physical as P
from tests.v4_1.world.phantom import make_phantom

REPO = Path(__file__).resolve().parents[2]


def _agg():
    spec = importlib.util.spec_from_file_location("v431_aggregate", REPO / "scripts" / "v431_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)           # type: ignore[union-attr]
    return mod


def _row(case, seed, scale, crop):
    rng = np.random.default_rng(seed)
    dose = rng.lognormal(np.log(scale), 1.0, crop.truth.shape)
    th = list(PA.THRESHOLDS_UMOL_H_L)
    return {"arm": "S", "plan_thr": None, "case": case, "seed": seed, "split": "dev_a", "n_truth_voxels": int(crop.truth.sum()),
            "outside_crop_truth_voxels": 0, "curves": PA.evaluate_dose(dose, crop.truth, None, 0, th),
            "dose_truth": {"median": 1.0, "p10": 0.1, "p90": 10.0, "max_all": 99.0}, "metrics": {"coverage": 0.5, "released_total": 61.6, "time_to_50_coverage": 7},
            "off_target": {b: {"off_target_share": 0.1, "off_target_mg": 6.0, "on_target_mg": 55.0} for b in PA.BANDS},
            "ledger": {"lost_boundary_mg": 0.0, "closure_residual_mg": 1e-5}, "release_voxel_share": 0.12, "n_releases": 30}


def test_curve_block_structure_and_ci_brackets_the_mean():
    A = _agg()
    crop = make_phantom(36)
    rows = [_row(f"C{i}", s, 1000.0 * (1 + i), crop) for i in range(6) for s in (9, 10)]
    blk = A.curve_block(rows)
    assert blk["n_cases"] == 6 and blk["n_runs"] == 12
    for b in PA.BANDS:
        for k in A.KINDS:
            m, lo, hi = (np.array(blk[b][k][x]) for x in ("mean", "ci95_lo", "ci95_hi"))
            assert len(m) == len(PA.THRESHOLDS_UMOL_H_L) and np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12)
    # the mean over cases of a nonincreasing curve is nonincreasing
    assert np.all(np.diff(blk["nominal"]["plain"]["mean"]) <= 1e-12)


def test_scalar_block_in_mg():
    A = _agg()
    crop = make_phantom(36)
    sb = A.scalar_block([_row("C0", 9, 1000.0, crop)], units=True)
    assert sb["off_target_mg"]["nominal"] == 6.0 and sb["release_voxel_share_of_exposure"] == 0.12


@pytest.mark.parametrize("k", ["low", "high"])
def test_conservation_bound_is_the_exposure_budget_over_threshold(k):
    A = _agg()
    crop = make_phantom(36)
    rows = [_row("C0", 9, 1000.0, crop)]
    p = P.PhysicalParams.carmustine_wafer(k)
    th = list(PA.THRESHOLDS_UMOL_H_L)
    b = A.bound_curve(rows, k, th)
    e_total = 61.6 * 1297.7237 / (p.k_clear_s * 8.0)
    n = rows[0]["n_truth_voxels"]
    assert b["E_total_umol_h_L_voxels"] == pytest.approx(e_total, rel=1e-5)
    i = th.index(1.0e5)
    assert b["mean_over_cases"][i] == pytest.approx(min(1.0, e_total / (n * 1.0e5)), rel=1e-6)   # e_total uses a rounded conversion constant
    assert b["mean_over_cases"][0] == 1.0 and b["mean_over_cases"][-1] < b["mean_over_cases"][i]
    # 1 mg*s/mm^3 = 1297.72 umol*h/L, so the bound at a 1e3 threshold is trivially 1 for a phantom-sized tumour, then falls as 1/T
    assert b["mean_over_cases"][th.index(1.0e3)] == 1.0


def test_legacy_grid_contains_the_frozen_dose_threshold_exactly():
    assert PA.LEGACY_D_T == 5.0 and PA.LEGACY_D_T in PA.THRESHOLDS_LEGACY
    assert list(PA.THRESHOLDS_LEGACY) == sorted(PA.THRESHOLDS_LEGACY)


def test_conservation_check_flags_an_impossible_coverage():
    A = _agg()
    th = list(PA.THRESHOLDS_UMOL_H_L)
    p = P.PhysicalParams.carmustine_wafer("high")
    e_total = p.budget_mg * p.exposure_to_umol_h_L / (p.k_clear_s * p.voxel_volume_mm3)
    ok = {"n_truth_voxels": 1000, "outside_crop_truth_voxels": 0, "curves": {"nominal": {"plain": [min(1.0, 0.5 * e_total / (1000 * t)) for t in th]}}}
    bad = {"n_truth_voxels": 1000, "outside_crop_truth_voxels": 0, "curves": {"nominal": {"plain": [1.0] * len(th)}}}
    assert A.conservation_check([ok], "high", th)["violations"] == 0
    assert A.conservation_check([bad], "high", th)["violations"] > 0
