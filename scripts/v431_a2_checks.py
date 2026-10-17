#!/usr/bin/env python
"""v4.31 A.2 checks: writes docs/research/v4.31-20261016/phaseA2/checks_a2.json. $0, no case data, no model.

    python scripts/v431_a2_checks.py [--out docs/research/v4.31-20261016/phaseA2/checks_a2.json]

* A  the profile test for all three clearances in BOTH geometries (planar = the wafer experiments' geometry; point source), and the sanity floor.
* B  the release-voxel exposure share at 2, 1 and 0.5 mm for all three clearances, and the old stepper's delivery for k_ref (diagnostic only).
* C  the cavity as a RESERVOIR: the wafer releases (60% in 10 h, 40% over the next 120 h: the published law, NOT extended) into a cavity compartment that empties at
     k_cavity = 0.010 /h; the tissue sees the cavity as a boundary source. Target: the measured human series (697 ng/mL at 1 h, 220 at day 1, at least 176 through day 7, 14.9 at
     day 14, 0.28 at day 30). The ONLY thing fitted is the cavity volume (declared 4, 14 and 34 mL, plus a continuous single-parameter fit reported as such). What is left over is a
     finding. Human brain TISSUE concentrations after wafer implantation have never been measured, so there is no tissue series to check against.
* D  what the clinical cavity exposure means for tissue at each clearance: with the cavity as a boundary source the exposure falls as exp(-x / L), so the depth at which it drops to the
     kill band follows. A consequence of the model, not a measurement.

Entry and motion are unchanged and unphysical; nothing here says anything about delivery in a patient.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v431_a2 as A2  # noqa: E402
from antelligence.worlds.tumor_mri import physical as P  # noqa: E402
from antelligence.worlds.tumor_mri.physics import DrugField  # noqa: E402

OUT_DEFAULT = REPO / "docs" / "research" / "v4.31-20261016" / "phaseA2" / "checks_a2.json"
SETS = ("ref", "low", "high")
RADII_MM = (10.0, 15.0, 20.0)                      # the declared cavity geometry assumption: 4, 14 and 34 mL
MG_TO_UMOL = 1000.0 / A2.CARMUSTINE_MW


def part_a() -> Dict[str, Any]:
    out: Dict[str, Any] = {"published_carmustine_x10_mm": P.PUBLISHED_CARMUSTINE_X10_MM, "factor": P.PROFILE_FACTOR, "sets": {}, "legacy_world_L_mm": math.sqrt(2.0 / -math.log1p(-0.02))}
    for k in SETS:
        p = P.PhysicalParams.carmustine_wafer(k)
        c = p.profile_check()
        out["sets"][k] = {"k_clear_s": p.k_clear_s, "L_mm": p.L_mm, "role": p.role, "profile_check": c, "sanity_floor_0.4_2.1": bool(P.L_BAND_MM[0] <= p.L_mm <= P.L_BAND_MM[1]),
                          "decay_per_tick": p.decay, "diffuse_sigma_mm": p.diffuse_sigma_mm,
                          "geometries": {"planar (wafer experiments)": {"x10_mm": c["planar_x10_mm"], "vs_rat_1.0": c["ratio_planar_to_rat"], "vs_rabbit_under_1.2": c["ratio_planar_to_rabbit_bound"]},
                                         "point source": {"x10_mm": c["point_source_x10_mm"], "vs_rat_1.0": c["ratio_point_to_rat"], "vs_rabbit_under_1.2": c["ratio_point_to_rabbit_bound"]}}}
    return out


def part_b() -> Dict[str, Any]:
    out: Dict[str, Any] = {"shares": {}, "stepper_ref": {}}
    D = P.D_STAR_CM2_S * P.MM2_PER_CM2
    for k in SETS:
        p = P.PhysicalParams.carmustine_wafer(k)
        row = {}
        for sp in (2.0, 1.0, 0.5):
            ps = p.with_voxel(sp)
            share_centred = P.release_voxel_share(sp, D, p.k_clear_s, 10, 5)
            tb = P.build_tables(ps, n_sub=5, corner=sp < 2.0)
            last = tb.cum[-1].astype(np.float64)
            f = ps.voxel_volume_mm3 * ps.k_clear_s / ps.exposure_to_umol_h_L
            row[f"{sp:g}mm"] = {"release_voxel_share_centred_source": share_centred, "largest_single_voxel_share_at_the_actual_placement": float(last.max() * f),
                                "placement": "corner of 8 voxels" if sp < 2.0 else "voxel centre", "table_radius_vox": tb.radius_vox}
        out["shares"][k] = row
    p = P.PhysicalParams.carmustine_wafer("ref")
    grid = 81
    c = grid // 2
    f = DrugField((grid,) * 3, release_sigma_vox=0.2, diffuse_sigma_vox=p.diffuse_sigma_vox, decay=p.decay)
    f.deposit((c, c, c), 1.0)
    for _ in range(p.ticks):
        f.step()
    tot_ana = (1.0 - math.exp(-p.k_clear_s * p.ticks * p.tick_seconds)) / p.k_clear_s
    out["stepper_ref"] = {"k_dt": p.k_clear_s * p.tick_seconds, "total_exposure_ratio_stepper_over_analytic": float(f.dose.sum()) * p.tick_seconds / tot_ana,
                          "closed_form_total_ratio": (math.exp(-p.k_clear_s * p.tick_seconds) * p.k_clear_s * p.tick_seconds) / p.decay,
                          "note": "Diagnostic only; nothing tuned. The analytic kernel is the scorer."}
    return out


# --- C: the cavity as a reservoir -------------------------------------------------------------------------------------------------------------------------

def _phases(M_mg: float):
    return [(0.0, 10.0, 0.6 * M_mg / 10.0), (10.0, 130.0, 0.4 * M_mg / 120.0)]          # (t0 h, t1 h, mg/h): the published release law, not extended


def reservoir(M_mg: float, V_mL: float, k_cav_per_h: float, times_h: List[float]) -> Dict[str, float]:
    """Cavity concentration (umol/L) of a well-mixed reservoir: dC/dt = S(t)/V - k_cav C, with the release law above. Closed form (piecewise exponentials)."""
    out = {}
    for t in times_h:
        c = 0.0
        for t0, t1, S in _phases(M_mg):
            if t <= t0:
                continue
            te = min(t, t1)
            c += S / (V_mL * k_cav_per_h) * (math.exp(-k_cav_per_h * (t - te)) - math.exp(-k_cav_per_h * (t - t0)))     # mg/mL
        out[f"{t:g}h"] = c * 1000.0 * MG_TO_UMOL                                                                        # mg/mL -> mg/L -> umol/L
    return out


def reservoir_auc_umol_h_L(M_mg: float, V_mL: float, k_cav_per_h: float, t_end_h: float = 720.0) -> float:
    f = lambda t: reservoir(M_mg, V_mL, k_cav_per_h, [t])[f"{t:g}h"]       # noqa: E731
    val, _ = integrate.quad(f, 0.0, t_end_h, limit=400, points=[10.0, 130.0], epsabs=0.0, epsrel=1e-8)
    return float(val)


def _score(V_mL: float, M_mg: float, k: float) -> float:
    """Log10 residual over the five target points (the 'at least 176' day-7 point is a lower bound: no penalty when the model is above it)."""
    res = 0.0
    for _, t, ng, kind in A2.HUMAN_SERIES:
        c = reservoir(M_mg, V_mL, k, [t])[f"{t:g}h"]
        tgt = A2.ng_per_ml_to_umol_per_L(ng)
        r = math.log10(max(c, 1e-300) / tgt)
        res += (min(r, 0.0) if kind == "lower" else r) ** 2
    return res


def part_c() -> Dict[str, Any]:
    M = P.BUDGET_CAP_MG
    times = [1.0, 24.0, 168.0, 336.0, 720.0]
    series = [{"label": lab, "t_h": t, "ng_per_mL": ng, "umol_per_L": A2.ng_per_ml_to_umol_per_L(ng), "kind": kind} for lab, t, ng, kind in A2.HUMAN_SERIES]
    out: Dict[str, Any] = {
        "model": ("A cavity RESERVOIR: the wafer releases 61.6 mg (60% uniformly over the first 10 h, 40% uniformly over the next 120 h: the published law, Dang 1996 and Fleming & Saltzman 2002; "
                  "the release tail is NOT extended) into a well-mixed cavity compartment that empties at k_cavity = 0.010 /h (every segment of the human series falls in 0.0086-0.0113 /h). "
                  "The tissue sees the cavity as a boundary source (one-way coupling: the tissue does not draw the cavity down)."),
        "k_cavity_per_h": A2.K_CAVITY_PER_H, "k_cavity_range_per_h": list(A2.K_CAVITY_RANGE_PER_H), "series": series,
        "fitted": "ONLY the cavity volume. Declared volumes 4.2, 14.1 and 33.5 mL (radius 10, 15 and 20 mm, Phase A's geometry assumption), plus one continuous single-parameter fit reported as such.",
        "tissue_series_exists": False,
        "tissue_note": "Human brain TISSUE concentrations after wafer implantation have never been measured, so there is no tissue series to check the tissue side against.", "rows": {}}
    for a in RADII_MM:
        V = 4.0 / 3.0 * math.pi * a ** 3 / 1000.0
        pred = reservoir(M, V, A2.K_CAVITY_PER_H, times)
        out["rows"][f"declared_r{a:g}mm"] = {"volume_mL": V, "predicted_umol_per_L": pred, "ratio_to_target": {s["label"]: pred[f"{s['t_h']:g}h"] / s["umol_per_L"] for s in series},
                                             "auc_0_30d_umol_h_L": reservoir_auc_umol_h_L(M, V, A2.K_CAVITY_PER_H)}
    fit = minimize_scalar(lambda lv: _score(math.exp(lv), M, A2.K_CAVITY_PER_H), bounds=(math.log(1.0), math.log(2.0e5)), method="bounded")
    Vf = math.exp(fit.x)
    pf = reservoir(M, Vf, A2.K_CAVITY_PER_H, times)
    out["rows"]["best_fit_volume"] = {"volume_mL": Vf, "volume_L": Vf / 1000.0, "predicted_umol_per_L": pf, "rms_log10_residual": math.sqrt(fit.fun / len(series)),
                                      "ratio_to_target": {s["label"]: pf[f"{s['t_h']:g}h"] / s["umol_per_L"] for s in series},
                                      "auc_0_30d_umol_h_L": reservoir_auc_umol_h_L(M, Vf, A2.K_CAVITY_PER_H),
                                      "note": "a single continuous fit of the volume to the five points; a volume of this size is not a resection cavity"}
    # what the shape says, independent of any volume: the measured series FALLS from 1 h to day 1 while the release is still running
    c1, c24 = reservoir(M, 14.1, A2.K_CAVITY_PER_H, [1.0, 24.0]).values()
    out["shape"] = {"model_day1_over_1h_any_volume": c24 / c1, "measured_day1_over_1h": A2.HUMAN_SERIES[1][2] / A2.HUMAN_SERIES[0][2],
                    "model_day14_over_day7_any_volume": reservoir(M, 14.1, A2.K_CAVITY_PER_H, [336.0])["336h"] / reservoir(M, 14.1, A2.K_CAVITY_PER_H, [168.0])["168h"],
                    "measured_day14_over_day7_at_most": A2.HUMAN_SERIES[3][2] / A2.HUMAN_SERIES[2][2],
                    "mass_balance_first_hour": {"mg_released": 0.6 * M / 10.0, "umol_released": 0.6 * M / 10.0 * MG_TO_UMOL,
                                                "volume_that_would_give_3.26_umol_per_L_L": 0.6 * M / 10.0 * MG_TO_UMOL / A2.ng_per_ml_to_umol_per_L(697.0)}}      # umol / (umol/L) = litres
    return out


def part_d() -> Dict[str, Any]:
    """With the cavity as a boundary source and the tissue loss fast against the cavity's time scale, the tissue exposure at depth x from the wall is the wall exposure times exp(-x / L)."""
    out: Dict[str, Any] = {"definition": "depth (mm) at which the exposure falls to a threshold: x = L ln(wall exposure / threshold); planar wall, tissue at quasi-steady state "
                                         "(1/k is seconds to hours against a cavity series of days). A consequence of the model, not a measurement.", "rows": {}}
    for k in SETS:
        p = P.PhysicalParams.carmustine_wafer(k)
        rows = {}
        for lab, w in (("clinical low (55)", A2.CLINICAL_RANGE_UMOL_H_L[0]), ("clinical mean (262)", A2.CLINICAL_MEAN_UMOL_H_L), ("clinical high (651)", A2.CLINICAL_RANGE_UMOL_H_L[1])):
            rows[lab] = {f"to_{t:g}": max(0.0, p.L_mm * math.log(w / t)) for t in A2.KILL_THRESHOLDS}
        out["rows"][k] = {"L_mm": p.L_mm, "depth_mm": rows}
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    doc = {"schema": "v431_a2_checks", "A": part_a(), "B": part_b(), "C": part_c(), "D": part_d(),
           "kill_band": {"umol_h_L": list(A2.KILL_BAND_UMOL_H_L), "half": A2.KILL_HALF_UMOL_H_L, "one_log": A2.KILL_ONE_LOG_UMOL_H_L, "flag": A2.KILL_FLAG},
           "clinical": {"mean": A2.CLINICAL_MEAN_UMOL_H_L, "range": list(A2.CLINICAL_RANGE_UMOL_H_L), "flag": A2.CLINICAL_FLAG, "potency": A2.POTENCY_STATEMENT},
           "caveat": A2.OUTCOME_CAVEAT, "calibration": A2.CALIBRATION_STATEMENT, "motion": A2.MOTION_STATEMENT}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
