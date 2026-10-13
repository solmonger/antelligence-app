#!/usr/bin/env python3
"""Tune config/v4_1/world.yaml total_drug and dose_threshold (D_t) with scripts S and S' on DEV cases only.

Targets (goal section 4.2): median S' coverage 0.70-0.80, median S 0.30-0.50, S' - S >= 0.15 on >= 80% of cases;
if no grid point meets them, a widened band is tried and the result is marked `fallback_widened_band`.
The script refuses any case that is not in the dev split (default config/v4_1/split.json; dev = dev_a + dev_b; eval and reserve refused).

  python scripts/v41_tune_world.py --cases-dir data/v4_1/perception --split config/v4_1/split.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antelligence.worlds.tumor_mri.config import WorldConfig  # noqa: E402
from antelligence.worlds.tumor_mri.data import Crop  # noqa: E402
from antelligence.worlds.tumor_mri.tuning import SplitError, assert_dev_only, grid_search, load_split  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-dir", type=Path, default=ROOT / "data/v4_1/perception", help="dir of <case_id>.npz crops")
    ap.add_argument("--split", type=Path, default=ROOT / "config/v4_1/split.json")
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: every dev case with a crop)")
    ap.add_argument("--config", type=Path, default=ROOT / "config/v4_1/world.yaml")
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite --config")
    ap.add_argument("--drug-grid", default="", help="comma-separated total_drug values (default: 25 geometric 25..25000)")
    ap.add_argument("--dt-grid", default="5", help="comma-separated D_t values")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    split = load_split(args.split)
    if args.cases:
        ids = [c for c in args.cases.split(",") if c]
    else:
        ids = sorted(p.stem for p in args.cases_dir.glob("*.npz") if p.stem in split.get("dev", set()))
    try:
        assert_dev_only(ids, split)
    except SplitError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    if not ids:
        print("REFUSED: no dev cases to tune on", file=sys.stderr)
        return 3
    crops = {c: Crop.from_npz(args.cases_dir / f"{c}.npz", case_id=c) for c in ids}
    base = WorldConfig.from_yaml(args.config) if args.config.exists() else WorldConfig()
    drugs = [float(x) for x in args.drug_grid.split(",")] if args.drug_grid else [float(x) for x in np.geomspace(25, 25000, 25)]
    dts = [float(x) for x in args.dt_grid.split(",")]
    res = grid_search(crops, base, drugs, dts, seed=args.seed)
    out = args.out or args.config
    doc = {"world": base.to_dict(), "tuning": {k: res[k] for k in ("status", "band", "best", "n_points", "cases")}}
    if res["status"] != "no_feasible_point":
        doc["world"] = base.with_(total_drug=res["best"]["total_drug"], dose_threshold=res["best"]["dose_threshold"]).to_dict()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"{res['status']}: {res['best']}")
    return 0 if res["status"] == "ok" else (1 if res["status"] == "fallback_widened_band" else 2)


if __name__ == "__main__":
    sys.exit(main())
