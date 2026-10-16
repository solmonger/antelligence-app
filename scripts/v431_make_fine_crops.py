#!/usr/bin/env python
"""v4.31 A.2: build the 1 mm SCORING arrays (truth, brain, FLAIR, T1; no perception) for the 40 DEVELOPMENT cases (see antelligence/worlds/tumor_mri/fine_grid.py).

    python scripts/v431_make_fine_crops.py [--cases BRATS_016,...] [--out /Volumes/WD_BLACK/antelligence-v4.31/data/cases_1mm]

Reads (read-only) the MSD Task01 NIfTIs and the 2 mm perception store from /Volumes/WD_BLACK/antelligence-v4.1/data_v4_1; writes only under
/Volumes/WD_BLACK/antelligence-v4.31/data/. Never writes MRI into git. Development cases only: anything else is refused. No model is run, no key is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.perception import msd  # noqa: E402
from antelligence.worlds.tumor_mri import fine_grid as FG  # noqa: E402

SRC = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="dev")
    ap.add_argument("--out", type=Path, default=FG.DEFAULT_FINE_ROOT)
    a = ap.parse_args()
    if Path("/Volumes/WD_BLACK/antelligence-v4.31/data").resolve() not in a.out.resolve().parents and a.out.resolve() != Path("/Volumes/WD_BLACK/antelligence-v4.31/data").resolve():
        raise SystemExit(f"refused: {a.out} is not under /Volumes/WD_BLACK/antelligence-v4.31/data")
    sp = json.loads((REPO / "config" / "v4_1" / "split.json").read_text())
    dev = sp["dev_a"] + sp["dev_b"]
    cases = dev if a.cases == "dev" else a.cases.split(",")
    bad = [c for c in cases if c not in dev]
    if bad:
        raise SystemExit(f"refused: {bad} is not a development case")
    root = SRC / "Task01_BrainTumour"
    ds = msd.read_dataset_json(root)
    ldef = msd.label_definition(ds)
    for c in cases:
        case = msd.load_case(root, c)
        if any(abs(z - 1.0) > 1e-6 for z in case.spacing_mm):
            raise SystemExit(f"{c}: spacing {case.spacing_mm} is not 1 mm")
        with np.load(SRC / "cases" / c / "case.npz") as z:
            truth2 = z["truth"]
            shape2 = tuple(int(v) for v in z["p_tumour"].shape)
            origin = [int(v) for v in z["origin_vox"]]
        arr = FG.make_fine_arrays(case.image, case.label, case.channels, lambda lab: msd.truth_channels(lab, ldef), truth2, origin, shape2)
        if arr["agreement"] < 0.99:
            raise SystemExit(f"{c}: the 1 mm truth sampled back onto the 2 mm grid agrees with the stored 2 mm truth only to Dice {arr['agreement']:.4f}: grids are misaligned")
        path = FG.save_fine_case(c, arr, a.out)
        print(f"{c}: crop {tuple(arr['brain_mask'].shape)} WT voxels {int(arr['truth'][0].sum())} outside {arr['outside']} agreement Dice {arr['agreement']:.5f} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
