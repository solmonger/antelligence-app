#!/usr/bin/env python
"""v4.31 Phase A checks (goal Steps A, B and C): writes docs/research/v4.31-20261016/phaseA/checks.json. $0, no case data, no model.

    python scripts/v431_phaseA_checks.py [--out docs/research/v4.31-20261016/phaseA/checks.json]

* A  parameter block with every source, the standing acceptance test (L for both ends of the clearance pair), the tick trade-off table.
* B  the three kernel checks: (1) exact vs a numeric time integral, (2) against the published wafer profiles, (3) the stepped DrugField
     against the analytic kernel, DIAGNOSTIC ONLY (nothing is tuned to close the gap), plus the sub-sampling convergence and the release
     voxel's share of the exposure.
* C  the human in-cavity cross-check: what these parameters predict for 61.6 mg released as 60% in 10 h and 40% over 5 days, against the
     measured 2.5-3.0 umol/L at 1 h, above 0.6 umol/L through day 7, nearly undetectable by day 14. A mismatch is a finding, not a failure;
     no parameter is adjusted.

Entry and motion are unchanged and unphysical here; nothing in this file says anything about delivery in a patient.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import integrate, special

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.worlds.tumor_mri import physical as P  # noqa: E402
from antelligence.worlds.tumor_mri.physics import DrugField  # noqa: E402

# quad reports a roundoff warning when the cavity concentration is ~1e-35 umol/L (days after the release ended); the value is correct to its own scale
warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)

OUT_DEFAULT = REPO / "docs" / "research" / "v4.31-20261016" / "phaseA" / "checks.json"
PAIR = ("low", "high")

# Published figures the checks compare against (grounding report, transport ledger and Verification log).
WAFER_X10_MM = {"Fung 1996, rat brain, carmustine from polymer pellets, days 3-14": 1.0,
                "Strasser 1995, rabbit, carmustine (reported as 'under 1.2 mm', taken at 1.2)": 1.2,
                "Strasser 1995, rabbit, 70 kDa dextran (not carmustine)": 1.7}
HUMAN_CAVITY = {"t_1h_umol_L": (2.5, 3.0), "through_day7_above_umol_L": 0.6, "day14": "nearly undetectable"}


# --- A -------------------------------------------------------------------------------------------------------------------------


def part_a() -> Dict[str, Any]:
    out: Dict[str, Any] = {"sets": {}}
    for k in PAIR:
        p = P.PhysicalParams.carmustine_wafer(k)
        out["sets"][k] = {**p.to_dict(), "L_mm": p.L_mm, "in_stated_band_0.4_2.0": P.L_BAND_STATED_MM[0] <= p.L_mm <= P.L_BAND_STATED_MM[1],
                          "in_enforced_band": P.L_BAND_MM[0] <= p.L_mm <= P.L_BAND_MM[1]}
    out["band"] = {"stated_mm": list(P.L_BAND_STATED_MM), "enforced_mm": list(P.L_BAND_MM), "reason": P.L_BAND_REASON}
    out["legacy_world_L_mm"] = math.sqrt(2.0 / -math.log1p(-0.02))       # sqrt(D dt / (k dt)) with the v4.3 constants: sigma 1 voxel, decay 0.02
    out["tick_tradeoff"] = P.tick_tradeoff(P.PhysicalParams.carmustine_wafer("low"))
    # How long an episode must run for a release to deliver (nearly) all of its exposure. The goal says "essentially complete within about 10 ticks".
    D = P.D_STAR_CM2_S * P.MM2_PER_CM2
    sat: Dict[str, Any] = {}
    for k in PAIR:
        p = P.PhysicalParams.carmustine_wafer(k)
        row: Dict[str, Any] = {"total_exposure_fraction_after_ticks": {str(n): -math.expm1(-p.k_clear_s * n * p.tick_seconds) for n in (5, 10, 20, 40, 120)}}
        for frac in (0.9, 0.99, 0.999):
            row[f"ticks_to_{frac}_of_total"] = int(math.ceil(-math.log(1.0 - frac) / (p.k_clear_s * p.tick_seconds)))
        row["far_field_fraction_after_10_ticks"] = {f"{r:g}mm": float(P.point_exposure_T(r, D, p.k_clear_s, 10 * p.tick_seconds) / P.point_exposure_inf(r, D, p.k_clear_s))
                                                    for r in (2.0, 4.0, 6.0, 10.0)}
        sat[k] = row
    out["saturation"] = sat
    return out


# --- B -------------------------------------------------------------------------------------------------------------------------


def part_b_exact() -> List[Dict[str, Any]]:
    D = P.D_STAR_CM2_S * P.MM2_PER_CM2
    rows = []
    for k_name, k in (("low", P.K_CLEAR_LOW_S), ("high", P.K_CLEAR_HIGH_S)):
        for r in (0.5, 1.0, 2.0, 5.0):
            num = P.numeric_time_integral(r, D, k)
            cf = float(P.point_exposure_inf(r, D, k))
            rows.append({"k": k_name, "r_mm": r, "closed_form": cf, "numeric_time_integral": num, "rel_error": abs(cf - num) / num,
                         "pass_1e-6": abs(cf - num) / num < 1e-6})
    return rows


def part_b_voxel_exact() -> List[Dict[str, Any]]:
    """The voxel-averaged kernel against a brute-force 6^3 Gauss-Legendre average of the NUMERIC time integral (a voxel off the source)."""
    D = P.D_STAR_CM2_S * P.MM2_PER_CM2
    rows = []
    x, w = np.polynomial.legendre.leggauss(6)
    w = w / 2.0
    for k_name, k in (("low", P.K_CLEAR_LOW_S), ("high", P.K_CLEAR_HIGH_S)):
        for off in ((3, 2, 1), (4, 0, 0), (6, 1, 1)):
            got = float(P.voxel_average_kernel(np.array([off]), 2.0, D, k, math.inf, 6)[0])
            tot = 0.0
            for i in range(6):
                for j in range(6):
                    for l in range(6):
                        pt = np.array(off) * 2.0 + 1.0 * np.array([x[i], x[j], x[l]])
                        tot += w[i] * w[j] * w[l] * P.numeric_time_integral(float(np.linalg.norm(pt)), D, k)
            rows.append({"k": k_name, "offset_vox": list(off), "voxel_averaged": got, "brute_force": tot, "rel_error": abs(got - tot) / tot})
    return rows


def part_b_convergence() -> Dict[str, Any]:
    D = P.D_STAR_CM2_S * P.MM2_PER_CM2
    out: Dict[str, Any] = {}
    for k_name, k in (("low", P.K_CLEAR_LOW_S), ("high", P.K_CLEAR_HIGH_S)):
        rows = []
        for off in ((1, 0, 0), (1, 1, 0), (2, 0, 0), (3, 2, 1), (5, 0, 0), (0, 0, 0)):
            ref = P.reference_voxel_average(off, 2.0, D, k, math.inf, n_ref=40 if off != (0, 0, 0) else 96)
            errs = {f"n{n}": abs(float(P.voxel_average_kernel(np.array([off]), 2.0, D, k, math.inf, n)[0]) - ref) / ref for n in (3, 5, 7, 9)}
            rows.append({"offset_vox": list(off), "reference": ref, "rel_error_vs_n": errs})
        out[k_name] = {"rows": rows, "production_n_sub": 5, "release_voxel_n_sub": P.SOURCE_VOXEL_N_SUB,
                       "note": "the release voxel (offset 0,0,0) always uses SOURCE_VOXEL_N_SUB; its 'n3..n9' columns show the plain count for comparison"}
        out[k_name]["release_voxel_share_of_total_exposure"] = P.release_voxel_share(2.0, D, k, 10, 5)
    return out


def part_b_wafer() -> Dict[str, Any]:
    out: Dict[str, Any] = {"definition": "distance at which exposure falls to 10% of its value 0.5 mm from the source (point source, exp(-r/L)/r)",
                           "published_mm": WAFER_X10_MM, "pass_factor": 1.5, "rows": {}}
    for k in PAIR:
        p = P.PhysicalParams.carmustine_wafer(k)
        x10 = P.x10_distance_mm(p.L_mm)
        planar = P.planar_x10_mm(p.L_mm)
        out["rows"][k] = {"L_mm": p.L_mm, "point_source_x10_mm": x10, "planar_source_x10_mm": planar,
                          "ratio_to_published_point": {name: x10 / v for name, v in WAFER_X10_MM.items()},
                          "ratio_to_published_planar": {name: planar / v for name, v in WAFER_X10_MM.items()}}
    out["goal_expected_point_source_range_mm"] = [1.6, 2.2]
    carm = [v for n, v in WAFER_X10_MM.items() if "carmustine" in n and "dextran" not in n]
    out["verdict"] = {k: {"point_source_within_1.5x_of_all_carmustine": all(1 / 1.5 <= out["rows"][k]["point_source_x10_mm"] / v <= 1.5 for v in carm),
                          "planar_within_1.5x_of_all_carmustine": all(1 / 1.5 <= out["rows"][k]["planar_source_x10_mm"] / v <= 1.5 for v in carm)}
                      for k in PAIR}
    return out


def part_b_stepper(grid: int = 81) -> Dict[str, Any]:
    """DIAGNOSTIC ONLY. The existing DrugField with the derived sigma and decay, one near-point release, 120 ticks, against the analytic kernel.
    Nothing is tuned to close the gap."""
    out: Dict[str, Any] = {}
    for k in PAIR:
        p = P.PhysicalParams.carmustine_wafer(k)
        c = grid // 2
        f = DrugField((grid,) * 3, release_sigma_vox=0.2, diffuse_sigma_vox=p.diffuse_sigma_vox, decay=p.decay)
        f.deposit((c, c, c), 1.0)
        for _ in range(p.ticks):
            f.step()
        # stepper: mass*ticks per voxel -> concentration*time: / voxel volume (mg/mm^3), * tick seconds (s) -> mg*s/mm^3 -> umol*h/L
        step_expo = f.dose / p.voxel_volume_mm3 * p.tick_seconds * p.exposure_to_umol_h_L
        tb = P.build_tables(p, n_sub=5)
        w = min(tb.radius_vox, c)
        ana = tb.cumulative(p.ticks).astype(np.float64)
        sl = slice(tb.radius_vox - w, tb.radius_vox + w + 1)
        ana = ana[sl, sl, sl]
        st = step_expo[c - w:c + w + 1, c - w:c + w + 1, c - w:c + w + 1]
        axis = [{"offset_vox": o, "stepper": float(st[w + o, w, w]), "analytic": float(ana[w + o, w, w]),
                 "ratio_stepper_over_analytic": float(st[w + o, w, w] / ana[w + o, w, w])} for o in (0, 1, 2, 3, 5, 8) if o <= w]
        # total exposure integral: analytic is (1 - exp(-kT))/k per mg; the stepper sums the concentration at tick ENDS only
        tot_ana = (1.0 - math.exp(-p.k_clear_s * p.ticks * p.tick_seconds)) / p.k_clear_s
        tot_step = float(f.dose.sum()) / p.voxel_volume_mm3 * p.tick_seconds * p.voxel_volume_mm3      # integral over the grid (mg*s per mg released)
        mask = ana > 1e-3 * ana.max()
        ratio = st[mask] / ana[mask]
        iso = {}
        for thr in (10.0, 100.0, 1000.0, 10000.0):                 # radius (voxels, along an axis) where 7.7 mg exceeds thr
            a_line, s_line = 7.7 * ana[w:, w, w], 7.7 * st[w:, w, w]
            ra = float(np.argmax(a_line < thr)) if (a_line < thr).any() else float(len(a_line))
            rs = float(np.argmax(s_line < thr)) if (s_line < thr).any() else float(len(s_line))
            iso[f"{thr:g}"] = {"analytic_radius_vox": ra, "stepper_radius_vox": rs}
        out[k] = {"sigma_vox": p.diffuse_sigma_vox, "decay": p.decay, "axis": axis,
                  "total_exposure_ratio_stepper_over_analytic": tot_step / tot_ana,
                  "closed_form_total_ratio": (math.exp(-p.k_clear_s * p.tick_seconds) * p.k_clear_s * p.tick_seconds) / p.decay,
                  "median_cell_ratio": float(np.median(ratio)), "p05_cell_ratio": float(np.percentile(ratio, 5)),
                  "p95_cell_ratio": float(np.percentile(ratio, 95)), "cells_compared": int(mask.sum()),
                  "isodose_radius_vox_for_7.7mg": iso}
    out["note"] = ("Diagnostic only. The stepper sums the concentration at tick ENDS, so it misses the first tick's exposure and has first-order time error "
                   "when k*dt is not small (0.22 low, 1.08 high). Nothing was tuned to close the gap.")
    return out


# --- C: human in-cavity cross-check ----------------------------------------------------------------------------------------------


def _ball_center_conc(tau: float, a: float, D: float) -> float:
    """Concentration (per unit initial uniform concentration c0) at the centre of a ball of radius a, tau seconds after an instantaneous
    uniform ball release into an unbounded medium, no loss: erf(a / (2 sqrt(D tau))) - (a / sqrt(pi D tau)) exp(-a^2 / (4 D tau))."""
    if tau <= 0.0:
        return 1.0
    x = a / (2.0 * math.sqrt(D * tau))
    return float(special.erf(x) - (2.0 * x / math.sqrt(math.pi)) * math.exp(-x * x))


def _ball_mean_conc(tau: float, a: float, D: float) -> float:
    """Mean concentration over the ball (per unit c0) tau seconds after an instantaneous uniform ball release, no loss."""
    if tau <= 0.0:
        return 1.0
    s = 2.0 * math.sqrt(D * tau)

    def c(r: float) -> float:
        if r < 1e-9:
            return _ball_center_conc(tau, a, D)
        return 0.5 * (special.erf((a - r) / s) + special.erf((a + r) / s)) - (math.sqrt(D * tau / math.pi) / r) * (
            math.exp(-((r - a) ** 2) / (s * s)) - math.exp(-((r + a) ** 2) / (s * s)))

    val, _ = integrate.quad(lambda r: 3.0 * r * r * c(r) / a ** 3, 0.0, a, epsabs=0.0, epsrel=1e-10, limit=200)
    return float(val)


def cavity_prediction(p: P.PhysicalParams, radius_mm: float, times_h: List[float], *, fast_frac: float = 0.6, fast_hours: float = 10.0,
                      slow_hours: float = 120.0, mean: bool = False) -> Dict[str, float]:
    """In-cavity concentration (umol/L) at ``times_h`` from the whole budget released into a ball cavity of radius ``radius_mm``:
    ``fast_frac`` uniformly over the first ``fast_hours`` and the rest uniformly over the next ``slow_hours`` (60% / 10 h, 40% / 5 days).
    Convolution of the release rate with the ball Green's function times exp(-k tau). Free-space tissue model; no CSF, no flushing."""
    D, k = p.d_star_mm2_s, p.k_clear_s
    vol = 4.0 / 3.0 * math.pi * radius_mm ** 3                               # mm^3
    M = p.budget_mg
    phases = [(0.0, fast_hours * 3600.0, fast_frac * M), (fast_hours * 3600.0, (fast_hours + slow_hours) * 3600.0, (1.0 - fast_frac) * M)]
    g = _ball_mean_conc if mean else _ball_center_conc
    out = {}
    for th in times_h:
        t = th * 3600.0
        mg_per_mm3 = 0.0
        for t0, t1, m in phases:
            lo, hi = t0, min(t1, t)
            if hi <= lo:
                continue
            rate = m / (t1 - t0)                                               # mg/s
            val, _ = integrate.quad(lambda ts: rate * g(t - ts, radius_mm, D) * math.exp(-k * (t - ts)) / vol, lo, hi,
                                    epsabs=0.0, epsrel=1e-9, limit=200)
            mg_per_mm3 += val
        out[f"{th:g}h"] = mg_per_mm3 * 1.0e6 * 1000.0 / p.molar_mass_g_mol     # mg/mm^3 -> mg/L -> umol/L
    return out


def part_c() -> Dict[str, Any]:
    times = [1.0, 6.0, 10.0, 24.0, 72.0, 120.0, 168.0, 336.0]            # 1 h, ..., day 5, day 7, day 14
    out: Dict[str, Any] = {"human": HUMAN_CAVITY, "release_schedule": "61.6 mg BCNU: 60% uniformly over the first 10 h, 40% uniformly over the next 120 h (5 days)",
                           "geometry_assumption": ("The resection-cavity volume is NOT in the grounding report. A ball of radius 10, 15 and 20 mm (4, 14 and 34 mL) is "
                                                   "a declared GEOMETRY ASSUMPTION used only to turn a mass into a concentration; it is not a physics parameter and is not tuned."),
                           "model": "free-space tissue (D*, k), instantaneous ball Green's function convolved with the release rate; no CSF, no flushing", "rows": {}}
    for k in PAIR:
        p = P.PhysicalParams.carmustine_wafer(k)
        rows = {}
        for a in (10.0, 15.0, 20.0):
            cen = cavity_prediction(p, a, times)
            mean = cavity_prediction(p, a, times, mean=True)
            rows[f"radius_{a:g}mm"] = {"volume_mL": 4.0 / 3.0 * math.pi * a ** 3 / 1000.0, "centre_umol_L": cen, "mean_umol_L": mean,
                                       "ratio_to_human_1h_low_2.5": mean["1h"] / 2.5, "ratio_to_human_1h_high_3.0": mean["1h"] / 3.0,
                                       "day7_over_0.6": mean["168h"] / 0.6, "day14_umol_L": mean["336h"]}
        out["rows"][k] = rows
    # model-free mass balance for the first hour: released mass in 1 h / (cavity volume) vs the measured level
    first_h_mg = P.BUDGET_CAP_MG * 0.6 / 10.0
    out["first_hour_mass_balance"] = {"mg_released_in_first_hour": first_h_mg, "umol": first_h_mg * 1000.0 / P.BCNU_MOLAR_MASS_G_MOL,
                                      "umol_per_L_if_all_stayed_in_14mL": first_h_mg * 1000.0 / P.BCNU_MOLAR_MASS_G_MOL / 0.014,
                                      "ratio_to_measured_2.5_to_3.0": [first_h_mg * 1000.0 / P.BCNU_MOLAR_MASS_G_MOL / 0.014 / 3.0, first_h_mg * 1000.0 / P.BCNU_MOLAR_MASS_G_MOL / 0.014 / 2.5]}
    # the effective terminal decay constant the human data imply (grounding report: ~0.0103 /h) against the model's two ends
    out["terminal_decay"] = {"human_per_h": 0.0103, "human_per_s": 0.0103 / 3600.0, "model_low_per_s": P.K_CLEAR_LOW_S, "model_high_per_s": P.K_CLEAR_HIGH_S,
                             "model_low_over_human": P.K_CLEAR_LOW_S / (0.0103 / 3600.0), "model_high_over_human": P.K_CLEAR_HIGH_S / (0.0103 / 3600.0),
                             "source": "grounding report: carmustine in the human resection cavity decays in its terminal phase at ~0.0103 /h (fitted to the paper's series)"}
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    doc = {"schema": "v431_phaseA_checks", "A": part_a(), "B": {"exact": part_b_exact(), "voxel_exact": part_b_voxel_exact(),
                                                              "convergence": part_b_convergence(), "wafer": part_b_wafer(), "stepper": part_b_stepper()},
           "C": part_c()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
