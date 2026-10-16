#!/usr/bin/env python
"""v4.31 A.2: the grid-convergence check. Coverage of the same episodes scored on a 2 mm, 1 mm and 0.5 mm lattice, at the three kill-band thresholds.

    python scripts/v431_gridcheck.py [--out docs/research/v4.31-20261016/phaseA2/gridcheck.json]

Cases: six development cases (one small, one large). Arms S and S' (planned for 23.5 umol*h/L), seed 9, all three clearances. The 2 mm score is the world's own field; 1 mm is the
native label grid; 0.5 mm uses the 1 mm labels nearest-neighbour upsampled (no finer truth exists), so 1 mm -> 0.5 mm tests the SCORER's resolution of the exposure gradient, not the labels.
The grid at which coverage has converged to within 0.02 is stated; if 1 mm has not converged at k_ref, 0.5 mm is the grid to use. $0, no model, no key, development cases only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v431_phaseA as PA  # noqa: E402
from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, WorldConfig, oracle_for, run_episode  # noqa: E402
from antelligence.worlds.tumor_mri.fine_grid import load_fine_case  # noqa: E402
from antelligence.worlds.tumor_mri.scoring import release_log, score_dose  # noqa: E402
from antelligence.worlds.tumor_mri.store_adapter import load_case  # noqa: E402

CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
FINE_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.31/data/cases_1mm_a2")
CASES = ("BRATS_373", "BRATS_406", "BRATS_167", "BRATS_183", "BRATS_016", "BRATS_034")      # smallest to largest tumour
THRESHOLDS = (7.0, 23.5, 50.0)
TOL = 0.02


def cov_at(dose, truth, thr):
    return float((dose[truth] >= thr).sum() / max(int(truth.sum()), 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "docs/research/v4.31-20261016/phaseA2/gridcheck.json")
    ap.add_argument("--spacings", default="1,0.5", help="scoring lattices (mm) after the world's own 2 mm; consecutive pairs are compared")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--ks", default="ref,low,high")
    a = ap.parse_args()
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    rows: List[Dict[str, Any]] = []
    spacings = [float(x) for x in a.spacings.split(",")]
    ks, cases = a.ks.split(","), a.cases.split(",")
    for k in ks:
        for case in cases:
            lc = load_case(case, root=CASES_ROOT, repo_root=REPO)
            fine = load_fine_case(case, root=FINE_ROOT, repo_root=REPO)
            for arm in ("S", "Sp"):
                doc = yaml.safe_load((REPO / "config/v4_31" / f"world_units_k_{k}.yaml").read_text())["world"]
                doc["physics"] = {**doc["physics"], "policy_threshold_umol_h_L": 23.5}
                cfg = WorldConfig.from_dict(doc)
                w = TumorMRIWorld(lc.crop, cfg, seed=9, regions=lc.regions, scorer_channels=lc.scorer_channels)
                pol = oracle_for(w) if arm == "Sp" else None
                run_episode(w, (lambda x: pol) if pol else (lambda x: ScriptS()))
                rel = release_log(w.field)
                params = cfg.physical_params()
                row: Dict[str, Any] = {"k": k, "case": case, "arm": arm, "n_truth_2mm": int(lc.crop.truth.sum()), "coverage": {}}
                d2 = w.field.dose
                row["coverage"]["2mm"] = {f"{t:g}": cov_at(d2, lc.crop.truth, t) for t in THRESHOLDS}
                for sp in spacings:
                    truth, _, outside = fine.at_spacing(sp)
                    dose = score_dose(rel, params, fine, sp)
                    row["coverage"][f"{sp:g}mm"] = {f"{t:g}": cov_at(dose, truth[0], t) for t in THRESHOLDS}
                    del dose
                rows.append(row)
                print(k, case, arm, {g: [round(v, 3) for v in c.values()] for g, c in row["coverage"].items()}, flush=True)
    summary: Dict[str, Any] = {}
    grids = ["2mm"] + [f"{sp:g}mm" for sp in spacings]
    for k in ks:
        sel = [r for r in rows if r["k"] == k]
        pairs = {}
        for g0, g1 in zip(grids[:-1], grids[1:]):
            per = {}
            for t in THRESHOLDS:
                key = f"{t:g}"
                diffs = [abs(r["coverage"][g0][key] - r["coverage"][g1][key]) for r in sel]
                per[key] = {"max_abs": float(max(diffs)), "mean_abs": float(np.mean(diffs))}
            pairs[f"{g0}->{g1}"] = {"by_threshold": per, "converged_within_0.02": all(v["max_abs"] <= TOL for v in per.values())}
        first = next((p.split("->")[0] for p, v in pairs.items() if v["converged_within_0.02"]), None)
        summary[k] = {"pairs": pairs, "first_grid_whose_refinement_changes_coverage_by_at_most_0.02": first}
    doc = {"schema": "v431_gridcheck", "tolerance": TOL, "thresholds": list(THRESHOLDS), "cases": cases, "spacings_mm": spacings, "arms": ["S", "Sp (planned for 23.5)"], "seed": 9,
           "note": "0.5 mm truth is the native 1 mm labels nearest-neighbour upsampled: 1 mm -> 0.5 mm tests the scorer's resolution of the exposure gradient, not the labels",
           "rows": rows, "summary": summary}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps({k: {p: v["converged_within_0.02"] for p, v in s_["pairs"].items()} for k, s_ in summary.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
