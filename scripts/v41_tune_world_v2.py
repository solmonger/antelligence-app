#!/usr/bin/env python3
"""Tune drug_scale_k and D_t for drug_mode per_perceived_volume (total_drug = k x V_perceived) with S and S' on DEV only.

Dev = dev_a + dev_b of config/v4_1/split.json (eval and reserve are refused before anything loads). Coverage per case is
the mean over --seeds (default 1,2). Targets (goal 4.2): strict S' 0.70-0.80, S 0.30-0.50; widened S' 0.60-0.85, S 0.20-0.55.
If neither band is met the point nearest the band centres is reported (`no_feasible_point`). The share of cases with
S' - S >= 0.15 is reported but is NOT a gate (operator, 2026-10-13: the "on >= 80% of cases" condition was dropped).

Dose and the per-case budget are both linear in k, so behaviour depends only on D_t / k: the (k, D_t) plane is a one-
dimensional family. Both are searched (several D_t share episodes through the ratio) and the result is reported at D_t = 5.

Extra dev-only diagnostics for G1: drug needed for 50% / 70% median coverage (S and S'), gap by truth-volume bin.

  python scripts/v41_tune_world_v2.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antelligence.worlds.tumor_mri.config import WorldConfig  # noqa: E402
from antelligence.worlds.tumor_mri.tuning import (STRICT, WIDENED, SplitError, assert_dev_only, case_stats_for_case,  # noqa: E402
                                                  drug_for_coverage, episode_coverages_for_case, gap_by_truth_bin,
                                                  grid_search, init_worker, load_split, timed_episode_coverages_for_case)

CANON_DT = 5.0
HEADER = """# v4.1 tumor_mri world constants (world_tuning_v2: drug scales with perceived volume).
# Per case: total_drug = drug_scale_k x V_perceived, V_perceived = sum of calibrated p_tumour over the crop (voxels);
# perception only, never truth. Decision: vault antelligence/v4.1/decisions/2026-10-13-drug-budget-scales-with-perceived-volume.md
# k and D_t tuned by scripts/v41_tune_world_v2.py on the 40 DEV cases (dev_a + dev_b), scripts S and S' only, seeds {seeds},
# through the L1 store adapter. Status: {status} ({band} band). Result file: {result}
#   median S' coverage {sprime:.3f}, median S coverage {s:.3f}, S' - S >= 0.15 on {gap:.0%} of cases.
# {note}
# Only D_t / drug_scale_k matters (dose and budget are both linear in k); D_t is pinned at 5 and k carries the tuning.
# No `total_drug` here: the loader rejects it in per_perceived_volume mode (the budget is per case).
"""
NOTES = {
    "ok": "Strict band met.",
    "fallback_widened_band": "Strict band missed; the widened band (S' 0.60-0.85, S 0.20-0.55) is met.",
    "no_feasible_point": "NO GRID POINT MEETS EVEN THE WIDENED MEDIANS. Best-effort point; raise at G1.",
}


def _pool_map(pool, workers):
    def run(fn, arg_tuples):
        if not arg_tuples:
            return []
        cols = list(zip(*arg_tuples))
        return list(pool.map(fn, *cols, chunksize=max(1, len(arg_tuples) // (workers * 8))))
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-dir", type=Path, default=ROOT / "data/v4_1/cases")
    ap.add_argument("--split", type=Path, default=ROOT / "config/v4_1/split.json")
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: every dev case in the store)")
    ap.add_argument("--config", type=Path, default=ROOT / "config/v4_1/world.yaml")
    ap.add_argument("--out", type=Path, default=None, help="world yaml to write (default: overwrite --config)")
    ap.add_argument("--result", type=Path, default=ROOT / "docs/research/v4.1-20261013/world_tuning_v2.json")
    ap.add_argument("--k-grid", default="", help="comma-separated k values (default: 24 geometric 0.02..20)")
    ap.add_argument("--dt-grid", default="2.5,5,10", help="comma-separated D_t values")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--refine", type=int, default=14)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)

    split = load_split(args.split)
    ids = sorted(c for c in args.cases.split(",") if c) if args.cases else sorted(
        p.parent.name for p in args.cases_dir.glob("*/case.npz") if p.parent.name in split["dev"])
    try:
        assert_dev_only(ids, split)
    except SplitError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    if not ids:
        print("REFUSED: no dev cases to tune on", file=sys.stderr)
        return 3
    seeds = [int(x) for x in args.seeds.split(",") if x]
    base = (WorldConfig.from_yaml(args.config) if args.config.exists() else WorldConfig()).with_(
        drug_mode="per_perceived_volume", drug_scale_k=1.0)
    ks = [float(x) for x in args.k_grid.split(",")] if args.k_grid else [float(x) for x in np.geomspace(0.02, 20, 24)]
    dts = [float(x) for x in args.dt_grid.split(",")]
    cases = {c: c for c in ids}

    cache: dict = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.cases_dir), str(ROOT))) as pool:
        mp = _pool_map(pool, args.workers)
        kw = dict(drug_field="drug_scale_k", seeds=seeds, evaluate=episode_coverages_for_case, map_fn=mp, cache=cache)
        res = grid_search(cases, base, ks, dts, **kw)
        print(f"stage 1: {res['status']} best={res['best']}", flush=True)
        if args.refine > 0:  # refine along the D_t = CANON_DT line around the best equivalent k
            best = res["best"]
            k_eq = CANON_DT * best["drug_scale_k"] / best["dose_threshold"]
            eq = sorted({CANON_DT * p["drug_scale_k"] / p["dose_threshold"] for p in res["points"]})
            i = min(range(len(eq)), key=lambda j: abs(np.log(eq[j] / k_eq)))
            fine = [float(x) for x in np.geomspace(eq[max(i - 1, 0)], eq[min(i + 1, len(eq) - 1)], args.refine)]
            ks = sorted(set(ks) | set(fine))
            res = grid_search(cases, base, ks, sorted(set(dts) | {CANON_DT}), **kw)
            print(f"stage 2: {res['status']} best={res['best']}", flush=True)
        best = res["best"]
        # canonical report point: D_t = 5, k with the same D_t / k ratio (identical behaviour)
        k_star = CANON_DT * best["drug_scale_k"] / best["dose_threshold"]
        cfg = base.with_(drug_scale_k=k_star, dose_threshold=CANON_DT)
        stats = dict(zip(ids, pool.map(case_stats_for_case, ids)))
        jobs = [(c, cfg, sd) for c in ids for sd in seeds]
        timed = list(pool.map(timed_episode_coverages_for_case, *zip(*jobs), chunksize=1))
        # determinism: rerun two cases from scratch a third time
        two = ids[:2]
        again = [pool.submit(timed_episode_coverages_for_case, c, cfg, sd) for c in two for sd in seeds]
        again = [f.result() for f in again]
    ratio = round(best["dose_threshold"] / best["drug_scale_k"], 9)
    pc = res["per_case"]
    drift = max(abs(t[0] - pc[c]["s_by_seed"][seeds.index(sd)]) + abs(t[1] - pc[c]["sprime_by_seed"][seeds.index(sd)])
                for (c, _, sd), t in zip(jobs, timed))
    drift2 = max(abs(a[0] - pc[c]["s_by_seed"][seeds.index(sd)]) + abs(a[1] - pc[c]["sprime_by_seed"][seeds.index(sd)])
                 for (c, sd), a in zip([(c, sd) for c in two for sd in seeds], again))
    sec_s, sec_sp = [t[2] for t in timed], [t[3] for t in timed]
    per_case = [{"case": c, "split": "dev_a" if c in split["dev_a"] else "dev_b", **stats[c],
                 "total_drug": round(k_star * stats[c]["v_perceived"], 3),
                 **{k: [round(x, 6) for x in v] if isinstance(v, list) else round(v, 6) for k, v in pc[c].items()},
                 "gap": round(pc[c]["sprime"] - pc[c]["s"], 6)} for c in ids]
    gaps = np.array([p["gap"] for p in per_case])
    # drug efficiency from the grid, expressed as D_t = 5 equivalent k (unique ratios)
    curve_s, curve_sp = {}, {}
    for p in res["points"]:
        k_eq = CANON_DT * p["drug_scale_k"] / p["dose_threshold"]
        curve_s[round(k_eq, 9)], curve_sp[round(k_eq, 9)] = p["median_s"], p["median_sprime"]
    med_v = float(np.median([p["v_perceived"] for p in per_case]))
    eff = {}
    for name, curve in (("S", curve_s), ("Sprime", curve_sp)):
        for tgt in (0.5, 0.7):
            k = drug_for_coverage(list(curve.items()), tgt)
            eff[f"{name}_median_cov_{int(tgt * 100)}"] = None if k is None else {
                "k": k, "median_case_total_drug": k * med_v}
    doc = {
        "status": res["status"], "band": res["band"], "targets": {"strict": STRICT, "widened": WIDENED},
        "drug_mode": "per_perceived_volume", "seeds": seeds, "n_cases": len(ids), "dev_only": True,
        "chosen": {"drug_scale_k": k_star, "dose_threshold": CANON_DT, "ratio_dt_over_k": CANON_DT / k_star},
        "dt_k_degeneracy": "dose and budget are both linear in k, so only D_t / k matters; every (k, D_t) with the same ratio is identical",
        "summary": {"median_s": best["median_s"], "median_sprime": best["median_sprime"], "gap_ge_0.15_fraction": best["gap_frac"],
                    "median_gap": float(np.median(gaps)), "min_gap": float(gaps.min()),
                    "mean_s": float(np.mean([p["s"] for p in per_case])), "mean_sprime": float(np.mean([p["sprime"] for p in per_case])),
                    "median_v_perceived": med_v},
        "gap_by_truth_volume_bin": gap_by_truth_bin(per_case),
        "drug_efficiency": {"definition": "smallest k (drug per perceived voxel, D_t=5 equivalent) at which the dev median "
                            "coverage reaches the level; log-linear interpolation on the grid. median_case_total_drug = k x median V_perceived.",
                            **eff},
        "seconds_per_episode": {"script_S_mean": float(np.mean(sec_s)), "oracle_Sprime_mean": float(np.mean(sec_sp)),
                                "oracle_Sprime_max": float(np.max(sec_sp)), "n_episodes": len(timed)},
        "rerun_max_abs_drift_all": float(drift), "rerun_2cases_max_abs_drift": float(drift2), "rerun_cases": two,
        "per_case": per_case,
        "grid": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()} for p in res["points"]],
        "notes": "Per-case coverage is the mean over seeds; per-seed values are in s_by_seed / sprime_by_seed. Aggregates and "
                 "public MSD case ids only. truth_vox is the in-crop WT voxel count (scorer side, report-only).",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(doc, indent=1) + "\n")
    out = args.out or args.config
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = args.result.resolve().relative_to(ROOT)
    except ValueError:
        rel = args.result
    out.write_text(HEADER.format(seeds=",".join(map(str, seeds)), status=res["status"], band=res["band"], result=rel,
                                 sprime=best["median_sprime"], s=best["median_s"], gap=best["gap_frac"], note=NOTES[res["status"]])
                   + yaml.safe_dump({"world": cfg.to_dict()}, sort_keys=False))
    print(json.dumps({k: doc[k] for k in ("status", "band", "chosen", "summary", "gap_by_truth_volume_bin", "drug_efficiency",
                                          "seconds_per_episode", "rerun_max_abs_drift_all", "rerun_2cases_max_abs_drift")}, indent=1))
    return 0 if res["status"] == "ok" else (1 if res["status"] == "fallback_widened_band" else 2)


if __name__ == "__main__":
    sys.exit(main())
