#!/usr/bin/env python
"""Realised uptake report (v4.3 B1): the mean of the foraging uptake field u per sub-region, level and case.

    python scripts/v43_uptake_report.py [--out docs/research/v4.3-20261015/uptake/realised_u.json] [--cases-root DIR] [--limit N]

For every development case (dev_a and dev_b of config/v4_1/split.json) and each of low, mid, high it builds the field from the
case's own scorer-side sub-region channels and records, per case and pooled over all cases:

* the mean u over the rim (ET), the core (TC without ET), the edema (WT without TC) and the whole tumour (WT), with voxel counts;
* the share of tumour voxels clipped at the floor 0.05 and at the ceiling 1.0 (whole tumour, and per sub-region).

AGGREGATES ONLY: no volume, no crop, no mask and no per-voxel value is written. It reads the case store (development cases only,
through the gated loader) and runs no episode, no arm and no model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402
from antelligence.experiments.v42_harness import load_dev_case  # noqa: E402
from antelligence.worlds.tumor_mri.foraging import U_MAX, U_MIN, UPTAKE_LEVELS, foraging_uptake, subregion_masks  # noqa: E402

DEFAULT_OUT = REPO / "docs" / "research" / "v4.3-20261015" / "uptake" / "realised_u.json"
DEFAULT_CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
PARTS = ("rim", "core", "edema", "wt")


def case_stats(u: np.ndarray, wt: np.ndarray, tc: np.ndarray, et: np.ndarray) -> Dict[str, Any]:
    """Sums (not means) per part, so the pooled mean over cases is voxel-weighted: n, sum_u, n_low, n_high."""
    parts = dict(subregion_masks(wt, tc, et))
    parts["wt"] = np.asarray(wt, dtype=bool)
    out: Dict[str, Any] = {}
    for name in PARTS:
        m = parts[name]
        v = u[m]
        out[name] = {"n": int(m.sum()), "sum_u": float(v.sum()), "n_clipped_low": int((v <= U_MIN).sum()),
                     "n_clipped_high": int((v >= U_MAX).sum())}
    return out


def summarise(stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in PARTS:
        s = stats[name]
        n = s["n"]
        out[name] = {"n_voxels": n, "mean_u": (s["sum_u"] / n) if n else None,
                     "share_clipped_at_0.05": (s["n_clipped_low"] / n) if n else None,
                     "share_clipped_at_1.0": (s["n_clipped_high"] / n) if n else None}
    return out


def pooled(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot = {name: {"n": 0, "sum_u": 0.0, "n_clipped_low": 0, "n_clipped_high": 0} for name in PARTS}
    for c in per_case:
        for name in PARTS:
            for k in tot[name]:
                tot[name][k] += c[name][k]
    return summarise(tot)


def case_means(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Unweighted mean over cases of each case's mean u (and its min and max), per part; cases with an empty part are skipped."""
    out: Dict[str, Any] = {}
    for name in PARTS:
        vals = [c[name]["sum_u"] / c[name]["n"] for c in per_case if c[name]["n"]]
        out[name] = {"cases": len(vals), "mean_of_case_means": float(np.mean(vals)) if vals else None,
                     "min_case_mean": float(np.min(vals)) if vals else None, "max_case_mean": float(np.max(vals)) if vals else None}
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT if DEFAULT_CASES_ROOT.is_dir() else None)
    ap.add_argument("--limit", type=int, default=None, help="first N cases of each split (smoke)")
    args = ap.parse_args(argv)
    if args.cases_root is None:
        print("no case store found; pass --cases-root", file=sys.stderr)
        return 2
    split = load_checked_split(REPO)
    cases: Dict[str, List[str]] = {k: list(split[k])[:args.limit] if args.limit else list(split[k]) for k in ("dev_a", "dev_b")}
    levels: Dict[str, Any] = {}
    skipped: List[Dict[str, str]] = []
    stats: Dict[str, Dict[str, Dict[str, Any]]] = {lvl: {} for lvl in UPTAKE_LEVELS}
    for group, ids in cases.items():
        for cid in ids:
            try:
                case = load_dev_case(cid, root=args.cases_root)
                ch = case.scorer_channels
                if not ch or ch.get("tc") is None or ch.get("et") is None:
                    raise ValueError("TC / ET channels missing")
                wt, tc, et = case.crop.truth, ch["tc"], ch["et"]
            except Exception as exc:                      # a case that cannot be read is reported, never guessed
                skipped.append({"case": cid, "group": group, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            for lvl in UPTAKE_LEVELS:
                stats[lvl][cid] = {"group": group, **case_stats(foraging_uptake(wt, tc, et, cid, lvl), wt, tc, et)}
    for lvl, preset in UPTAKE_LEVELS.items():
        rows = stats[lvl]
        per_case = {cid: {"group": r["group"], **summarise(r)} for cid, r in rows.items()}
        levels[lvl] = {
            "preset": dict(preset),
            "n_cases": len(rows),
            "overall_voxel_weighted": pooled(list(rows.values())),
            "overall_mean_of_case_means": case_means(list(rows.values())),
            "by_group": {g: pooled([r for r in rows.values() if r["group"] == g]) for g in cases if any(r["group"] == g for r in rows.values())},
            "per_case": per_case,
        }
    doc = {
        "what": "realised mean uptake u per sub-region, from the foraging field (antelligence/worlds/tumor_mri/foraging.py); aggregates only",
        "clip_bounds": [U_MIN, U_MAX],
        "sub_regions": {"rim": "ET & WT", "core": "TC & ~ET & WT", "edema": "WT & ~TC"},
        "split_sha256": sha256_file(REPO / "config" / "v4_1" / "split.json"),
        "foraging_py_sha256": sha256_file(REPO / "antelligence" / "worlds" / "tumor_mri" / "foraging.py"),
        "cases": {g: len(ids) for g, ids in cases.items()},
        "skipped": skipped,
        "levels": levels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, args.out)
    print(f"{'level':<6} {'rim':>7} {'core':>7} {'edema':>7} {'wt':>7}   (voxel-weighted mean u, {sum(len(v) for v in cases.values())} cases, {len(skipped)} skipped)")
    for lvl, d in levels.items():
        o = d["overall_voxel_weighted"]
        print(f"{lvl:<6} " + " ".join(f"{o[p]['mean_u']:7.3f}" if o[p]["mean_u"] is not None else f"{'-':>7}" for p in PARTS))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
