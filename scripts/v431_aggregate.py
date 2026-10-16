#!/usr/bin/env python
"""v4.31 Phase A Step D aggregator: merges the runner's shard files into one deterministic results.json (and the markdown tables for G0).

    python scripts/v431_aggregate.py --runs /Volumes/WD_BLACK/antelligence-v4.31/runs \
        --out docs/research/v4.31-20261016/phaseA/stepD [--commit HASH]

Descriptive only: NO bar, NO pass or fail, NO verdict. Every number carries its +/-1-voxel erosion/dilation band. 95% intervals are case-clustered
bootstrap (10,000 resamples, seeded) of the MEAN over the 40 development cases (each case averages seeds 9 and 10), reported as context, not as a test.

Also computes the placement-free conservation bound on coverage: whatever the placement or motion, the exposure integrated over all space is exactly
M/k per release, so at most M * conv / (k * V_voxel * T) voxels can reach threshold T, i.e. coverage(T) <= min(1, E_total / (N_truth * T)) with
E_total = 61.6 mg * 1297.7 (umol*h/L per mg*s/mm^3) / (k * 8 mm^3). It ignores that exposure outside the tumour is wasted, so it is an upper bound only.

Entry and motion are unchanged and unphysical; nothing here says anything about delivery in a patient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v431_phaseA as PA  # noqa: E402
from antelligence.worlds.tumor_mri import physical as P  # noqa: E402

BANDS = PA.BANDS
KINDS = ("plain", "w35", "w50")
DECADES = [float(10.0 ** e) for e in range(-1, 8)]


def load(runs: Path, label: str) -> List[Dict[str, Any]]:
    rows = []
    for f in sorted((runs / label).glob("runs_shard*of*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (r["arm"], r["plan_thr"] if r["plan_thr"] is not None else -1.0, r["case"], r["seed"]))
    return rows


def _case_mean(rows: List[Dict[str, Any]], getter) -> Dict[str, float]:
    """Average a scalar/array over the seeds of each case -> {case: value}."""
    acc: Dict[str, List[Any]] = {}
    for r in rows:
        acc.setdefault(r["case"], []).append(np.asarray(getter(r), dtype=np.float64))
    return {c: np.mean(v, axis=0) for c, v in sorted(acc.items())}


def _boot_ci(per_case: np.ndarray, n: int = 10000, seed: int = 20261016) -> List[float]:
    """95% case-clustered bootstrap interval of the mean of ``per_case`` (cases on axis 0)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_case), size=(n, len(per_case)))
    means = per_case[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5, axis=0)) if means.ndim == 1 else np.percentile(means, 2.5, axis=0).tolist(),
            float(np.percentile(means, 97.5, axis=0)) if means.ndim == 1 else np.percentile(means, 97.5, axis=0).tolist()]


def curve_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean / median / interval of every curve over the cases (seeds averaged within a case)."""
    out: Dict[str, Any] = {"n_runs": len(rows), "n_cases": len({r["case"] for r in rows})}
    for b in BANDS:
        out[b] = {}
        for k in KINDS:
            pc = _case_mean(rows, lambda r: r["curves"][b][k])
            arr = np.stack([pc[c] for c in sorted(pc)])
            lo, hi = _boot_ci(arr)
            out[b][k] = {"mean": arr.mean(axis=0).tolist(), "median": np.median(arr, axis=0).tolist(), "ci95_lo": lo, "ci95_hi": hi}
    return out


def scalar_block(rows: List[Dict[str, Any]], units: bool) -> Dict[str, Any]:
    def mean_of(getter) -> float:
        vals = np.array([float(getter(r)) for r in rows])
        return float(vals.mean())
    out: Dict[str, Any] = {
        "truth_dose_median_mean": mean_of(lambda r: r["dose_truth"]["median"]), "truth_dose_p10_mean": mean_of(lambda r: r["dose_truth"]["p10"]),
        "truth_dose_p90_mean": mean_of(lambda r: r["dose_truth"]["p90"]), "released_total_mean": mean_of(lambda r: r["metrics"]["released_total"]),
        "off_target_share": {b: mean_of(lambda r, b=b: r["off_target"][b]["off_target_share"]) for b in BANDS},
        "coverage_at_policy_threshold_mean": mean_of(lambda r: r["metrics"]["coverage"]),
        "time_to_50_coverage_reached_share": float(np.mean([r["metrics"]["time_to_50_coverage"] is not None for r in rows]))}
    if units:
        out["off_target_mg"] = {b: mean_of(lambda r, b=b: r["off_target"][b]["off_target_mg"]) for b in BANDS}
        out["lost_boundary_mg_exposure_left_the_crop"] = mean_of(lambda r: r["ledger"]["lost_boundary_mg"])
        out["release_voxel_share_of_exposure"] = mean_of(lambda r: r["release_voxel_share"])
        out["release_voxel_share_max"] = float(max(r["release_voxel_share"] for r in rows))
        out["n_releases_mean"] = mean_of(lambda r: r["n_releases"])
        out["closure_residual_mg_max_abs"] = float(max(abs(r["ledger"]["closure_residual_mg"]) for r in rows))
    else:
        out["lost_boundary_share"] = mean_of(lambda r: r["metrics"]["lost_boundary_share"])
    return out


def bound_curve(rows: List[Dict[str, Any]], k_set: str, thresholds: List[float]) -> Dict[str, Any]:
    """Placement-free conservation bound, averaged over the cases (see the module docstring). Nominal mask, plain coverage."""
    p = P.PhysicalParams.carmustine_wafer(k_set)
    e_total = p.budget_mg * p.exposure_to_umol_h_L / (p.k_clear_s * p.voxel_volume_mm3)        # umol*h/L summed over voxels
    per_case = {}
    for r in rows:
        n = r["n_truth_voxels"] + r["outside_crop_truth_voxels"]
        per_case[r["case"]] = n
    arr = np.array([[min(1.0, e_total / (n * t)) for t in thresholds] for n in per_case.values()])
    return {"E_total_umol_h_L_voxels": e_total, "mean_over_cases": arr.mean(axis=0).tolist(),
            "mean_exposure_per_truth_voxel_if_perfectly_spread": float(np.mean([e_total / n for n in per_case.values()]))}


def build(runs: Path, commit: Optional[str]) -> Dict[str, Any]:
    leg = load(runs, "legacy")
    doc: Dict[str, Any] = {"schema": "v431_phaseA_stepD", "code_commit": commit, "seeds": [9, 10],
                           "statement": ("Descriptive only: no bar, no verdict. Entry and motion are UNCHANGED and UNPHYSICAL: squads still walk through tissue "
                                         "one voxel per tick carrying their whole dose. Nothing here is a claim about delivery in a patient."),
                           "thresholds_units": list(PA.THRESHOLDS_UMOL_H_L), "thresholds_legacy": list(PA.THRESHOLDS_LEGACY),
                           "plan_thresholds_units": list(PA.PLAN_THRESHOLDS_UMOL_H_L), "bands": list(BANDS),
                           "weights": {"ET": PA.W_ET, "label2_local": PA.W_LABEL2_LOCAL, "oedema": list(PA.W_OEDEMA),
                                       "outside_crop": "uncovered, at the case's mean weight"}}
    if leg:
        doc["legacy"] = {"config_sha": leg[0]["config_sha"], "D_t_arbitrary_units": 5.0}
        for arm in ("S", "Sp"):
            rs = [r for r in leg if r["arm"] == arm]
            doc["legacy"][arm] = {"curves": curve_block(rs), **scalar_block(rs, False)}
    doc["units"] = {}
    for k in ("low", "high"):
        rows = load(runs, f"units_{k}")
        if not rows:
            continue
        blk: Dict[str, Any] = {"physical": P.PhysicalParams.carmustine_wafer(k).to_dict(), "config_sha": rows[0]["config_sha"]}
        rs = [r for r in rows if r["arm"] == "S"]
        blk["S"] = {"curves": curve_block(rs), **scalar_block(rs, True)}
        blk["bound"] = bound_curve(rs, k, list(PA.THRESHOLDS_UMOL_H_L))
        th = list(PA.THRESHOLDS_UMOL_H_L)
        blk["Sp_plan"] = {}
        for pt in PA.PLAN_THRESHOLDS_UMOL_H_L:
            rp = [r for r in rows if r["arm"] == "Sp" and r["plan_thr"] == pt]
            if rp:
                blk["Sp_plan"][f"{pt:g}"] = {"curves": curve_block(rp), **scalar_block(rp, True)}
        # S' attainable ceiling: planned for T, scored at T (the diagonal), per band and kind, plus the planned-by-scored matrix (plain, nominal)
        diag: Dict[str, Any] = {}
        for b in BANDS:
            diag[b] = {}
            for kind in KINDS:
                diag[b][kind] = {f"{pt:g}": blk["Sp_plan"][f"{pt:g}"]["curves"][b][kind]["mean"][th.index(pt)]
                                 for pt in PA.PLAN_THRESHOLDS_UMOL_H_L if f"{pt:g}" in blk["Sp_plan"]}
        blk["Sp_diagonal_mean"] = diag
        blk["Sp_matrix_plain_nominal_mean"] = {f"{pt:g}": blk["Sp_plan"][f"{pt:g}"]["curves"]["nominal"]["plain"]["mean"] for pt in PA.PLAN_THRESHOLDS_UMOL_H_L
                                               if f"{pt:g}" in blk["Sp_plan"]}
        # per-case table at the decades: S, S' diagonal (plain, nominal) averaged over the two seeds
        per_case = []
        for c in sorted({r["case"] for r in rs}):
            row: Dict[str, Any] = {"case": c, "split": next(r["split"] for r in rs if r["case"] == c),
                                   "n_truth_voxels": next(r["n_truth_voxels"] for r in rs if r["case"] == c)}
            row["S_plain_nominal"] = {f"{t:g}": float(np.mean([r["curves"]["nominal"]["plain"][th.index(t)] for r in rs if r["case"] == c])) for t in DECADES}
            row["Sp_diag_plain_nominal"] = {}
            for pt in PA.PLAN_THRESHOLDS_UMOL_H_L:
                sel = [r for r in rows if r["arm"] == "Sp" and r["plan_thr"] == pt and r["case"] == c]
                if sel:
                    row["Sp_diag_plain_nominal"][f"{pt:g}"] = float(np.mean([r["curves"]["nominal"]["plain"][th.index(pt)] for r in sel]))
            per_case.append(row)
        blk["per_case"] = per_case
        doc["units"][k] = blk
    return doc


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commit", default=None)
    a = ap.parse_args(argv)
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents and a.out.resolve() != allowed:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    doc = build(a.runs, a.commit)
    a.out.mkdir(parents=True, exist_ok=True)
    res = a.out / "results.json"
    res.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"wrote {res}  sha256={sha256_file(res)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
