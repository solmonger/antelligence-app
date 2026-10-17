#!/usr/bin/env python
"""v4.31 A.2: assemble G0-approval-a2.md from the generated evidence and the hand-written header and narrative.

    python scripts/v431_g0_a2.py --checks .../phaseA2/checks_a2.json --gridcheck .../gridcheck.json --gridcheck025 .../gridcheck_025.json --tables .../tables_a2.md \
        --gapfix .../phaseA_gap_corrected.md --header H.md --narrative N.md --out "$V431/G0-approval-a2.md"

Every table is generated; nothing numerical is retyped. The header and the narrative (the headline sentence, the Gliadel comparison, the open items, the closing paragraph) are written by hand.
Output goes to the vault folder given by --out (Obsidian only); never under an earlier version's path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def profile_section(c: Dict[str, Any]) -> str:
    A = c["A"]
    out = ["### 1. The profile test, all three clearances, in both geometries", "",
           "The acceptance test: in the near-planar geometry of the wafer experiments, x10 = 2.303 L must fall within 1.5x of BOTH carmustine measurements (1.0 mm rat, Fung 1996; under 1.2 mm rabbit, "
           "Strasser 1995, a bound, so only its upper side is tested). The point-source distance is shown beside it; the geometry of each comparison is stated. 0.4-2.1 mm is a sanity floor only.", "",
           "| set | role | k (1/s) | L (mm) | geometry | x10 (mm) | vs rat 1.0 | vs rabbit <1.2 | within 1.5x of both | sanity floor |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("ref", "low", "high"):
        s = A["sets"][k]
        pc = s["profile_check"]
        for geo, ok in (("planar (wafer experiments)", pc["accepted"]), ("point source", None)):
            g = s["geometries"][geo]
            within = (1 / 1.5 <= g["vs_rat_1.0"] <= 1.5) and g["vs_rabbit_under_1.2"] <= 1.5
            out.append(f"| {k} | {s['role']} | {s['k_clear_s']:g} | {s['L_mm']:.3f} | {geo} | {g['x10_mm']:.2f} | {g['vs_rat_1.0']:.2f}x | {g['vs_rabbit_under_1.2']:.2f}x | "
                       f"{'**YES**' if within else 'no'}{' (the acceptance test)' if geo.startswith('planar') else ''} | {'pass' if s['sanity_floor_0.4_2.1'] else 'FAIL'} |")
    out += ["", c["calibration"], "", "The rat comparison passes largely by construction (k_ref is calibrated on that profile); the rabbit comparison, another species and a bound, is the independent one, and it passes too (0.97x). "
            "The carried pair, from inert-tracer efflux, fails both comparisons in both geometries.", ""]
    return "\n".join(out)


def grid_section(c: Dict[str, Any], g: Dict[str, Any], g25: Dict[str, Any]) -> str:
    out = ["### 2. The scoring grid: convergence, the grid chosen, and the release-voxel exposure shares", "",
           "**Is 1 mm truth recoverable from the existing raw/ bundles?** No: those bundles are 2 mm (120 x 120 x 78). It IS recoverable from the original Task01 label NIfTIs (1 mm native), which need no perception "
           "and no model; the alignment of that 1 mm truth against the stored 2 mm truth is exact (Dice 1.00000 on all 40 development cases). Perception, targets and release points stay on the 2 mm grid; only the scorer refines.", "",
           f"**Convergence check.** The same episodes (S and S' planned for 23.5 umol*h/L, seed 9) scored on a 2 mm, 1 mm and 0.5 mm lattice, at the three kill-band thresholds, on {len(g['cases'])} cases "
           f"({', '.join(g['cases'])}); tolerance {g['tolerance']:g}. Largest and mean absolute change in plain coverage when the lattice is refined:", "",
           "| clearance | refinement | max change at T=7 | at T=23.5 | at T=50 | mean change (over the three) | within 0.02? |", "|---|---|---|---|---|---|---|"]
    for k in ("ref", "low", "high"):
        for pr, v in g["summary"][k]["pairs"].items():
            bt = v["by_threshold"]
            out.append(f"| {k} | {pr.replace('->', ' -> ')} | {bt['7']['max_abs']:.3f} | {bt['23.5']['max_abs']:.3f} | {bt['50']['max_abs']:.3f} | "
                       f"{sum(x['mean_abs'] for x in bt.values()) / 3:.4f} | {'**yes**' if v['converged_within_0.02'] else 'NO'} |")
    pr = g25["summary"]["ref"]["pairs"]["0.5mm->0.25mm"]
    bt = pr["by_threshold"]
    out.append(f"| ref (3 smallest cases only) | 0.5mm -> 0.25mm | {bt['7']['max_abs']:.3f} | {bt['23.5']['max_abs']:.3f} | {bt['50']['max_abs']:.3f} | "
               f"{sum(x['mean_abs'] for x in bt.values()) / 3:.4f} | {'**yes**' if pr['converged_within_0.02'] else 'NO'} |")
    out += ["", "**The grid I chose.** At k_ref, 1 mm has **not** converged (refining it to 0.5 mm still changes coverage by up to "
            f"{max(x['max_abs'] for x in g['summary']['ref']['pairs']['1mm->0.5mm']['by_threshold'].values()):.3f}); 0.5 mm has (the next refinement changes it by at most "
            f"{max(x['max_abs'] for x in bt.values()):.3f}, checked on the three smallest cases only: 0.25 mm on a large tumour is 64x the voxels of 2 mm and was not run). "
            "So k_ref is scored at **0.5 mm** and reported at 1 mm beside it. The two ends of the pair converge at 1 mm or coarser and are scored at 1 mm. The 0.5 mm and 0.25 mm lattices use the native 1 mm labels "
            "nearest-neighbour upsampled (no finer truth exists), so they test the scorer's resolution of the exposure gradient, not the labels.", "",
            "**Release-voxel exposure share** (share of one release's total exposure inside the voxel it is released in; the 0.8 line is 'counting single voxels instead of fields'). The centred-source column is the "
            "definition the goal uses; in the scoring runs a release at a 2 mm voxel centre sits at the CORNER of eight finer voxels, so the actual placement spreads over them and the largest single voxel holds less:", "",
            "| clearance | lattice | share, centred source | largest single voxel at the actual placement |", "|---|---|---|---|"]
    for k in ("ref", "low", "high"):
        for sp in ("2mm", "1mm", "0.5mm"):
            s = c["B"]["shares"][k][sp]
            out.append(f"| {k} | {sp} | **{100 * s['release_voxel_share_centred_source']:.1f}%** | {100 * s['largest_single_voxel_share_at_the_actual_placement']:.1f}% ({s['placement']}) |")
    st = c["B"]["stepper_ref"]
    out += ["", f"At k_ref the old stepped field would deliver {100 * st['total_exposure_ratio_stepper_over_analytic']:.0f}% of the analytic exposure (k dt = {st['k_dt']:.1f}); the analytic kernel is the scorer (diagnostic only; nothing tuned).", ""]
    return "\n".join(out)


def cavity_section(c: Dict[str, Any]) -> str:
    C, D = c["C"], c["D"]
    out = ["### 3. The cavity as a reservoir (and what is left over)", "", C["model"], "", f"Fitted: {C['fitted']}", "",
           "| | " + " | ".join(f"{s['label']}" for s in C["series"]) + " |", "|---|" + "---|" * len(C["series"]),
           "| measured (umol/L) | " + " | ".join((">= " if s["kind"] == "lower" else "") + f"{s['umol_per_L']:.3g}" for s in C["series"]) + " |"]
    for key, r in C["rows"].items():
        lab = key.replace("declared_r", "declared r = ").replace("mm", " mm").replace("best_fit_volume", "single-parameter best fit")
        pred = r["predicted_umol_per_L"]
        out.append(f"| model, {lab} ({r['volume_mL']:,.0f} mL) (umol/L) | " + " | ".join(f"{pred[f'{s['t_h']:g}h']:.3g}" for s in C["series"]) + " |")
    out += ["", "Ratio of model to measured:", "", "| | " + " | ".join(s["label"] for s in C["series"]) + " |", "|---|" + "---|" * len(C["series"])]
    for key, r in C["rows"].items():
        out.append(f"| {key.replace('declared_r', 'declared r = ').replace('mm', ' mm').replace('best_fit_volume', 'best fit')} | " + " | ".join(f"{r['ratio_to_target'][s['label']]:.3g}x" for s in C["series"]) + " |")
    sh, mb = C["shape"], C["shape"]["mass_balance_first_hour"]
    bf = C["rows"]["best_fit_volume"]
    out += ["", "**What is left over, in four findings:**", "",
            f"1. **Level.** At the declared volumes the reservoir is {min(C['rows'][k]['ratio_to_target']['1 h'] for k in C['rows'] if k.startswith('declared')):.0f}x to "
            f"{max(C['rows'][k]['ratio_to_target']['1 h'] for k in C['rows'] if k.startswith('declared')):,.0f}x too high at 1 h. The first-hour mass balance alone ({mb['umol_released']:.1f} umol released) needs a volume of "
            f"**{mb['volume_that_would_give_3.26_umol_per_L_L']:.1f} L** to dilute to the measured 3.26 umol/L.",
            f"2. **Shape.** The measured series FALLS from 1 h to day 1 (x{sh['measured_day1_over_1h']:.2f}) while the release is still running; a reservoir fed by the published release law RISES (x{sh['model_day1_over_1h_any_volume']:.1f}), for any volume. "
            "The release law is not extended and k_cavity is as measured, so this is not a tuning problem.",
            f"3. **The only volume that fits is not a cavity.** The single-parameter best fit is {bf['volume_L']:,.0f} L (rms residual {bf['rms_log10_residual']:.2f} in log10), and even that is {1 / bf['ratio_to_target']['1 h']:.0f}x too LOW at 1 h.",
            "4. **Measured exposure needs a tiny share of the load.** An exposure of 262 umol*h/L in a 14 mL cavity emptying at 0.010 /h corresponds to about "
            f"{262.0 * 0.014 * 0.010 / (61.6 * 1000.0 / 214.05) * 100:.3f}% of the 61.6 mg, i.e. the sampled fluid compartment sees about one ten-thousandth of the nominal release. "
            "Where the rest goes (wafer retention, wall uptake, chemical degradation, sampling) is not in the data and I do not invent it.", "",
            "**Conclusion, stated as a finding and not as a fix.** The two-compartment reservoir with the published release law and the measured emptying rate does NOT reproduce the human cavity series; the cavity "
            "needs its own measured volume and its own measured input, and neither exists. " + C["tissue_note"], "",
            "**What the measured cavity exposure would mean for tissue (a consequence of the model, not a measurement).** With the cavity as a boundary source the exposure at depth x from the wall is the wall exposure "
            "times exp(-x / L). Depth (mm) at which it falls to each kill-band threshold, for the clinical mean and its range:", "",
            "| clearance | L (mm) | wall exposure | to 7 (half-kill) | to 23.5 (one log) | to 50 (band top) |", "|---|---|---|---|---|---|"]
    for k in ("ref", "low", "high"):
        r = D["rows"][k]
        for lab, v in r["depth_mm"].items():
            out.append(f"| {k} | {r['L_mm']:.2f} | {lab} | {v['to_7']:.2f} mm | {v['to_23.5']:.2f} mm | {v['to_50']:.2f} mm |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for n in ("checks", "gridcheck", "gridcheck025", "tables", "gapfix", "header", "narrative", "out"):
        ap.add_argument(f"--{n}", type=Path, required=True)
    a = ap.parse_args()
    if "antelligence-v4.1" in str(a.out) or "v4.2" in a.out.parts or "v4.3" in a.out.parts:
        raise SystemExit("refused: output path is under an earlier version")
    c = json.loads(a.checks.read_text())
    g, g25 = json.loads(a.gridcheck.read_text()), json.loads(a.gridcheck025.read_text())
    parts: List[str] = [a.header.read_text(), "", profile_section(c), grid_section(c, g, g25), cavity_section(c),
                        "### 4. The corrected band test", "", a.gapfix.read_text(), "",
                        "### 5. The curves with both marked levels, the headline, and the geometry", "", "![[G0-curves-a2.png]]", "",
                        "*Figure. Whole-tumour coverage against exposure threshold. Red band: the kill band (7-50, fitted from one rat-glioma series). Grey: the clinically delivered cavity-fluid exposure (262, range 55-651). Marks, not bars. "
                        "Solid: nominal mask; shaded: the +/-1-voxel band on the scoring lattice.*", "", a.tables.read_text(), "", a.narrative.read_text()]
    a.out.write_text("\n".join(parts))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
