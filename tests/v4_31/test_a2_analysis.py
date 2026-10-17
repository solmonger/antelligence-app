"""v4.31 A.2: constants, the corrected gap test, the cavity reservoir and the tissue-depth consequence. No case data."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from antelligence.experiments import v431_a2 as A2
from antelligence.worlds.tumor_mri import physical as P

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)           # type: ignore[union-attr]
    return mod


def test_marked_levels_carry_their_flags_and_the_outcome_caveat_is_in_every_report_text():
    assert A2.KILL_BAND_UMOL_H_L == (7.0, 50.0) and A2.KILL_ONE_LOG_UMOL_H_L == 23.5 and A2.KILL_THRESHOLDS == (7.0, 23.5, 50.0)
    for w in ("DERIVED BY FITTING", "Single study", "one rat line", "not cleanly proportional", "band, never a line"):
        assert w in A2.KILL_FLAG
    assert A2.CLINICAL_MEAN_UMOL_H_L == 262.0 and A2.CLINICAL_RANGE_UMOL_H_L == (55.0, 651.0)
    for w in ("cavity fluid, not tissue", "not a bar", "Ohnishi"):
        assert w in A2.CLINICAL_FLAG
    assert "transport is" in A2.POTENCY_STATEMENT and abs(A2.CLINICAL_MEAN_UMOL_H_L / A2.KILL_ONE_LOG_UMOL_H_L - 11.1) < 0.1
    for w in ("0.393", "0.185", "n = 15", "not shown", "never been shown to predict a patient outcome"):
        assert w in A2.OUTCOME_CAVEAT
    assert "CALIBRATED" in A2.CALIBRATION_STATEMENT and "no carmustine tissue" in A2.CALIBRATION_STATEMENT and "unphysical" in A2.MOTION_STATEMENT


def test_gap_consistency_matches_the_goals_example_and_flags_real_instability():
    ok = A2.gap_consistency(0.204, 0.239, 0.132)                                  # the goal's example: +0.20, moving between +0.13 and +0.24
    assert ok["holds"] and ok["same_sign"] and ok["ratio"] < 3
    assert not A2.gap_consistency(0.1, -0.02, 0.05)["holds"] and A2.gap_consistency(0.1, -0.02, 0.05)["reason"] == "changes sign"
    c = A2.gap_consistency(0.019, 0.055, 0.010)
    assert not c["holds"] and "changes order" in c["reason"]
    assert not A2.gap_consistency(0.001, 0.002, 0.001)["holds"]                    # all below the floor: not resolved
    assert A2.gap_consistency(-0.2, -0.24, -0.13)["holds"]                         # a consistent negative gap also holds


def test_reservoir_closed_form_matches_a_numerical_ode_and_conserves_mass():
    chk = _load("v431_a2_checks")
    M, V, k = 61.6, 14.1, A2.K_CAVITY_PER_H
    def S(t):                                         # mg/h, the published law
        return 0.6 * M / 10.0 if t < 10 else (0.4 * M / 120.0 if t < 130 else 0.0)
    sol = solve_ivp(lambda t, y: [S(t) / V - k * y[0]], (0, 400), [0.0], max_step=0.05, dense_output=True, rtol=1e-9, atol=1e-12)
    for t in (1.0, 9.0, 24.0, 129.0, 168.0, 336.0):
        want = float(sol.sol(t)[0]) * 1000.0 * 1000.0 / A2.CARMUSTINE_MW                # mg/mL -> umol/L
        assert chk.reservoir(M, V, k, [t])[f"{t:g}h"] == pytest.approx(want, rel=2e-3)
    # all the drug eventually empties through k_cavity: AUC(0..inf) = M (umol) / (V (L) * k (/h))
    auc = chk.reservoir_auc_umol_h_L(M, V, k, t_end_h=4000.0)
    assert auc == pytest.approx(M * 1000.0 / A2.CARMUSTINE_MW / (V / 1000.0) / k, rel=2e-3)


def test_the_reservoir_cannot_reproduce_the_human_series_with_any_cavity_volume():
    """The finding, pinned: the published release into a reservoir RISES for days; the measured series FALLS from 1 h. The shape, not just the level, is wrong."""
    chk = _load("v431_a2_checks")
    c = chk.part_c()
    assert c["shape"]["model_day1_over_1h_any_volume"] > 5 and c["shape"]["measured_day1_over_1h"] < 0.4
    assert c["shape"]["mass_balance_first_hour"]["volume_that_would_give_3.26_umol_per_L_L"] > 1.0                 # litres, not a cavity
    for k in ("declared_r10mm", "declared_r15mm", "declared_r20mm"):
        r = c["rows"][k]["ratio_to_target"]
        assert r["1 h"] > 100 and r["day 1"] > 1000 and r["day 7"] > 1000
    assert c["rows"]["best_fit_volume"]["volume_mL"] > 1.0e4 and c["rows"]["best_fit_volume"]["rms_log10_residual"] > 0.3
    assert c["tissue_series_exists"] is False and "never been measured" in c["tissue_note"]


def test_tissue_depth_at_the_kill_band_follows_l_ln_wall_over_threshold():
    chk = _load("v431_a2_checks")
    d = chk.part_d()["rows"]["ref"]
    assert d["L_mm"] == pytest.approx(0.505, abs=1e-3)
    assert d["depth_mm"]["clinical mean (262)"]["to_7"] == pytest.approx(d["L_mm"] * math.log(262 / 7), rel=1e-9)
    assert d["depth_mm"]["clinical mean (262)"]["to_23.5"] == pytest.approx(1.22, abs=0.02) and d["depth_mm"]["clinical mean (262)"]["to_7"] == pytest.approx(1.83, abs=0.02)
    assert chk.part_d()["rows"]["low"]["depth_mm"]["clinical mean (262)"]["to_7"] > 3 * d["depth_mm"]["clinical mean (262)"]["to_7"]


def test_checks_profile_table_covers_both_geometries_and_only_k_ref_is_accepted():
    chk = _load("v431_a2_checks")
    a = chk.part_a()["sets"]
    assert a["ref"]["profile_check"]["accepted"] and not a["low"]["profile_check"]["accepted"] and not a["high"]["profile_check"]["accepted"]
    for k in a:
        assert set(a[k]["geometries"]) == {"planar (wafer experiments)", "point source"}
    assert a["ref"]["geometries"]["planar (wafer experiments)"]["x10_mm"] == pytest.approx(1.163, abs=0.002)
    b = chk.part_b()["shares"]["ref"]
    assert b["2mm"]["release_voxel_share_centred_source"] == pytest.approx(0.69, abs=0.01) and b["1mm"]["release_voxel_share_centred_source"] == pytest.approx(0.34, abs=0.01)


def test_the_corrected_phase_a_gap_table_replaces_the_old_column():
    agg = _load("v431_aggregate_a2")
    res = REPO / "docs/research/v4.31-20261016/phaseA/stepD/results.json"
    md = agg.phase_a_gap_table(res)
    assert "keeps sign and order?" in md and "gap larger than the band?" not in md.split("The old column")[1].split("replaced")[1]
    assert "ABSOLUTE LEVEL" in md and "does not inherit a level" in md
    row = [ln for ln in md.splitlines() if ln.startswith("| 1000 |")][0]
    assert "+0.204" in row and "+0.239" in row and "+0.132" in row and "**yes**" in row      # the goal's example: low k, T = 1e3
