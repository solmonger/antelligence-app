#!/usr/bin/env python
"""v4.31 A.2 aggregator: merges the runner's shard files into results_a2.json (deterministic) from the SCORED curves (the fine lattice), not the world's 2 mm ones.

    python scripts/v431_aggregate_a2.py --runs /Volumes/WD_BLACK/antelligence-v4.31/runs --out docs/research/v4.31-20261016/phaseA2 [--commit HASH]

Descriptive only: no bar, no verdict. Every coverage number carries its threshold, its scoring lattice and its +/-1-voxel erosion/dilation band (on THAT lattice). Intervals are
case-clustered bootstrap (10,000 resamples, seeded) of the mean over the 40 development cases (seeds 9 and 10 averaged within a case), context not a test.

Sets: ref_0.5mm (PRIMARY: 1 mm has not converged at k_ref, 0.5 mm has), ref_1mm (the same runs on the native label grid, for comparison), low_1mm, high_1mm (converged at 1 mm).
The corrected gap test (does the oracle-minus-script gap keep its sign and its order across the nominal, eroded and dilated masks?) replaces the old 'gap larger than the band?' column.

Entry and motion are unchanged and unphysical; nothing here is a claim about delivery in a patient.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v431_a2 as A2  # noqa: E402
from antelligence.experiments import v431_phaseA as PA  # noqa: E402
from antelligence.worlds.tumor_mri import physical as P  # noqa: E402

_spec = importlib.util.spec_from_file_location("v431_aggregate", REPO / "scripts" / "v431_aggregate.py")
AG = importlib.util.module_from_spec(_spec)
sys.modules["v431_aggregate"] = AG
_spec.loader.exec_module(AG)          # type: ignore[union-attr]

BANDS = PA.BANDS
PLAN = (1.0, 7.0, 10.0, 23.5, 50.0, 100.0, 262.0, 1000.0, 1.0e4, 1.0e5, 1.0e6)
# label -> (run folder, clearance set, scoring-grid key, scoring spacing mm)
SETS = {"ref_0.5mm": ("a2_ref", "ref", "0.5mm", 0.5), "ref_1mm": ("a2_ref", "ref", "1mm", 1.0), "low_1mm": ("a2_low", "low", "1mm", 1.0), "high_1mm": ("a2_high", "high", "1mm", 1.0)}
HEADLINE = (7.0, 23.5, 50.0, 262.0)


def scored_rows(rows: List[Dict[str, Any]], grid: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        s = r["scored"][grid]
        out.append({**r, "curves": s["curves"], "n_truth_voxels": s["n_truth_voxels"], "outside_crop_truth_voxels": s["outside_crop_truth_voxels"]})
    return out


def at(curve: List[float], th: List[float], t: float) -> float:
    return float(np.interp(math.log10(t), [math.log10(x) for x in th], curve))


def geometry_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    g = [r["scored"]["1mm"]["geometry"] for r in rows if "1mm" in r["scored"]]
    keys = ["depth_lt_25mm", "pia_lt_5mm", "ependyma_lt_10mm_proxy"]
    out: Dict[str, Any] = {"n_runs": len(g), "rule_breaking_mass_share_mean": {}, "exposure_share_mean": {}}
    for k in keys:
        v = [x["rule_breaking_mass_share"][k] for x in g if x["rule_breaking_mass_share"].get(k) is not None]
        out["rule_breaking_mass_share_mean"][k] = float(np.mean(v)) if v else None
    for k in ("within_5mm_of_pia", "within_10mm_of_ependyma_proxy"):
        v = [x["exposure_share"][k] for x in g if x["exposure_share"].get(k) is not None]
        out["exposure_share_mean"][k] = float(np.mean(v)) if v else None
    out["ventricle_proxy_mL_min_median_max"] = [float(np.min([x["ventricle_proxy_mL"] for x in g])), float(np.median([x["ventricle_proxy_mL"] for x in g])),
                                                float(np.max([x["ventricle_proxy_mL"] for x in g]))]
    return out


def build(runs: Path, commit: Optional[str]) -> Dict[str, Any]:
    th = list(PA.THRESHOLDS_UMOL_H_L)
    doc: Dict[str, Any] = {"schema": "v431_a2_results", "code_commit": commit, "seeds": [9, 10], "thresholds_umol_h_L": th, "plan_thresholds": list(PLAN), "headline_thresholds": list(HEADLINE),
                           "statement": A2.MOTION_STATEMENT, "caveat": A2.OUTCOME_CAVEAT, "calibration": A2.CALIBRATION_STATEMENT, "kill_flag": A2.KILL_FLAG, "clinical_flag": A2.CLINICAL_FLAG,
                           "potency": A2.POTENCY_STATEMENT, "old_gap_column_note": A2.OLD_COLUMN_NOTE, "primary": "ref_0.5mm", "sets": {}}
    cache: Dict[str, List[Dict[str, Any]]] = {}
    for label, (folder, k, grid, sp) in SETS.items():
        if folder not in cache:
            cache[folder] = AG.load(runs, folder)
        allrows = cache[folder]
        rows = [r for r in allrows if "scored" in r and grid in r["scored"]]
        if not rows:
            continue
        phys = P.PhysicalParams.carmustine_wafer(k)
        blk: Dict[str, Any] = {"k_set": k, "role": phys.role, "physical": phys.to_dict(), "scoring_grid_mm": sp, "config_sha": rows[0]["config_sha"]}
        srows = scored_rows(rows, grid)
        s_rows = [r for r in srows if r["arm"] == "S"]
        blk["S"] = {"curves": AG.curve_block(s_rows), **{k_: v for k_, v in AG.scalar_block([r for r in rows if r["arm"] == "S"], True).items()}}
        blk["bound"] = AG.bound_curve(s_rows, k, th, sp)
        blk["conservation_check_all_runs"] = AG.conservation_check(srows, k, th, sp)
        blk["Sp_plan"] = {}
        for pt in PLAN:
            rp = [r for r in srows if r["arm"] == "Sp" and r["plan_thr"] == pt]
            if rp:
                blk["Sp_plan"][f"{pt:g}"] = {"curves": AG.curve_block(rp), **AG.scalar_block([r for r in rows if r["arm"] == "Sp" and r["plan_thr"] == pt], True)}
        # headline and gap test: S' planned for T, scored at T, against S at T, per mask
        head, gaps = {}, {}
        for t in sorted(set(HEADLINE) | set(PLAN)):
            key = f"{t:g}"
            if key not in blk["Sp_plan"]:
                continue
            per = {}
            for b in BANDS:
                s_v = at(blk["S"]["curves"][b]["plain"]["mean"], th, t)
                p_v = at(blk["Sp_plan"][key]["curves"][b]["plain"]["mean"], th, t)
                per[b] = {"S": s_v, "Sp_planned_for_T": p_v, "gap": p_v - s_v}
            gaps[key] = {"by_mask": per, "consistency": A2.gap_consistency(per["nominal"]["gap"], per["eroded"]["gap"], per["dilated"]["gap"])}
            if t in HEADLINE:
                head[key] = per
        blk["headline_coverage"] = head
        blk["gap_test"] = gaps
        if label == "ref_0.5mm" or label == "ref_1mm":
            blk["geometry_1mm"] = {"S": geometry_block([r for r in rows if r["arm"] == "S"]),
                                   **{f"Sp_plan{pt:g}": geometry_block([r for r in rows if r["arm"] == "Sp" and r["plan_thr"] == pt]) for pt in (7.0, 23.5, 50.0, 262.0, 1000.0)
                                      if any(r["arm"] == "Sp" and r["plan_thr"] == pt for r in rows)}}
        else:
            blk["geometry_1mm"] = {"S": geometry_block([r for r in rows if r["arm"] == "S"]),
                                   **{f"Sp_plan{pt:g}": geometry_block([r for r in rows if r["arm"] == "Sp" and r["plan_thr"] == pt]) for pt in (23.5, 262.0)
                                      if any(r["arm"] == "Sp" and r["plan_thr"] == pt for r in rows)}}
        blk["n_runs"] = len(rows)
        doc["sets"][label] = blk
    return doc


def phase_a_gap_table(results_a: Path) -> str:
    """The Phase A Step D gap table, regenerated with the corrected test (from the committed Phase A results.json, which is not modified)."""
    d = json.loads(results_a.read_text())
    th = d["thresholds_units"]
    ix = lambda t: min(range(len(th)), key=lambda j: abs(math.log10(th[j]) - math.log10(t)))       # noqa: E731
    out = ["### Phase A Step D, the gap column corrected", "", A2.OLD_COLUMN_NOTE, "",
           "Does the paired gap (S' planned for T, scored at T, minus S at T; plain coverage) keep its sign and its order (max/min of the three magnitudes <= 3) across the nominal, eroded and dilated masks?", ""]
    for lab, name in (("low", "low k (1.2e-4 /s)"), ("high", "high k (6.0e-4 /s)")):
        u = d["units"][lab]
        out += [f"#### {name}", "", "| T | gap, nominal | gap, eroded | gap, dilated | keeps sign and order? | why |", "|---|---|---|---|---|---|"]
        for t in (1, 10, 100, 1000, 10000, 100000, 1000000):
            key = f"{t:g}"
            if key not in u["Sp_plan"]:
                continue
            j = ix(t)
            g = {b: u["Sp_plan"][key]["curves"][b]["plain"]["mean"][j] - u["S"]["curves"][b]["plain"]["mean"][j] for b in ("nominal", "eroded", "dilated")}
            c = A2.gap_consistency(g["nominal"], g["eroded"], g["dilated"])
            out.append(f"| {t:g} | {g['nominal']:+.3f} | {g['eroded']:+.3f} | {g['dilated']:+.3f} | **{'yes' if c['holds'] else 'no'}** | {c['reason']} |")
        out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--phase-a-results", type=Path, default=REPO / "docs/research/v4.31-20261016/phaseA/stepD/results.json")
    a = ap.parse_args(argv)
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents and a.out.resolve() != allowed:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    a.out.mkdir(parents=True, exist_ok=True)
    doc = build(a.runs, a.commit)
    res = a.out / "results_a2.json"
    res.write_text(json.dumps(doc, indent=1, sort_keys=True))
    if a.phase_a_results.exists():
        (a.out / "phaseA_gap_corrected.md").write_text(phase_a_gap_table(a.phase_a_results))
    print(f"wrote {res}  sha256={AG.sha256_file(res)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
