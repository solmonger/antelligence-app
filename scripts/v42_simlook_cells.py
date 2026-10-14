#!/usr/bin/env python
"""Build the SimLook cell table and d1's reliability by distance band from the v4.1 SW-B calibration boards (goal 5.1).

Reads (read-only): <calib>/cells.jsonl (run_id -> case, arm), <calib>/export/<run_id>.jsonl (board; `evidence` entries with
payload pos_vox / look_p_ppm / p_tumour_ppm), <calib>/runs/<run_id>/decisions.jsonl. Labels each look with scorer-side
whole-tumour truth from the dev-case store. Development cases only: an evaluation case is refused.

A board look is kept only if the decision row of the same run, squad and k-th round has an equal, non-null look_p_ppm
(a failed look stores the scan's p instead). Every drop is counted by reason.

Writes under --out (default docs/research/v4.2-20261014/simlook/): cells_v4_2.json (counts + quantiles, no MRI) and
reliability.json (aggregates by distance band). No per-voxel data, no MRI, no sub-region truth is written.

    python scripts/v42_simlook_cells.py [--calib DIR] [--cases-root DIR] [--out DIR] [--limit-cases N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.worlds.tumor_mri.simlook import (BANDS, build_cell_table, distance_band_index, save_table,  # noqa: E402
                                                   scan_band_index)

DEFAULT_CALIB = Path("/Volumes/WD_BLACK/antelligence-v4.1/runs/calib1")
DEFAULT_CASES = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
DEFAULT_OUT = REPO / "docs/research/v4.2-20261014/simlook"
ARM = "SW-B"


def _jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def board_looks(calib: Path, run_id: str, drops: Counter) -> List[Tuple[int, Tuple[int, int, int], int, int]]:
    """Verified looks of one run: (squad, pos_vox, look_p_ppm, scan_p_ppm)."""
    rows = _jsonl(calib / "export" / f"{run_id}.jsonl")
    head = rows[0]
    if head.get("arm") != f"{ARM}:calibration" or head.get("split") != "dev":
        raise SystemExit(f"{run_id}: header arm/split {head.get('arm')}/{head.get('split')} is not a dev {ARM} calibration run")
    dec = {(d["squad"], d["round"]): d for d in _jsonl(calib / "runs" / run_id / "decisions.jsonl")}
    per_squad: Dict[int, int] = defaultdict(int)
    out = []
    for e in rows[1:]:
        p = e.get("payload", {})
        if e.get("type") != "evidence" or "look_p_ppm" not in p or not str(e["author"]["agent"]).startswith("squad-"):
            continue
        sq = int(p["squad"])
        k = per_squad[sq]
        per_squad[sq] += 1
        d = dec.get((sq, k))
        if d is None:
            drops["no_decision_row"] += 1
        elif d.get("error"):
            drops["decision_error"] += 1
        elif d.get("look_p_ppm") is None:
            drops["decision_look_null"] += 1
        elif int(d["look_p_ppm"]) != int(p["look_p_ppm"]):
            drops["value_mismatch"] += 1
        elif int(d["tick"]) != int(e["tick"]):
            drops["tick_mismatch"] += 1
        else:
            out.append((sq, tuple(int(c) for c in p["pos_vox"]), int(p["look_p_ppm"]), int(p["p_tumour_ppm"])))
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> Optional[float]:
    """P(score_pos > score_neg) + 0.5 P(tie), by ranks. None if a side is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def reliability(band: np.ndarray, look: np.ndarray, scan: np.ndarray) -> Dict[str, Any]:
    """d1 by distance band. Label is fixed within a band (inside bands are tumour), so per-band calibration is the gap between
    the mean look and the observed rate; AUROC compares a band's looks with the opposite-label looks at a comparable
    distance (edge: in_0_4 vs out_0_4; mid: in_gt4 vs out_4_10; far: in_gt4 vs out_gt10), plus overall."""
    lab = (band >= 3).astype(float)
    per_band = {}
    for i, name in enumerate(BANDS):
        m = band == i
        if not m.any():
            per_band[name] = {"n": 0}
            continue
        rate = float(lab[m].mean())
        per_band[name] = {"n": int(m.sum()), "label_rate": rate, "mean_look": float(look[m].mean()),
                          "mean_scan_p": float(scan[m].mean()), "calibration_gap_look": float(look[m].mean() - rate),
                          "calibration_gap_scan": float(scan[m].mean() - rate), "brier_look": float(((look[m] - rate) ** 2).mean()),
                          "brier_scan": float(((scan[m] - rate) ** 2).mean()), "accuracy_at_0.5": float(((look[m] >= 0.5) == (rate == 1.0)).mean())}

    def pair(pos_b: int, neg_b: int) -> Dict[str, Any]:
        p, n = band == pos_b, band == neg_b
        return {"positive": BANDS[pos_b], "negative": BANDS[neg_b], "auroc_look": auroc(look[p], look[n]), "auroc_scan": auroc(scan[p], scan[n])}

    bins = np.minimum((look * 10).astype(int), 9)
    curve = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int((bins == b).sum()),
              "mean_look": float(look[bins == b].mean()) if (bins == b).any() else None,
              "observed_tumour_rate": float(lab[bins == b].mean()) if (bins == b).any() else None} for b in range(10)]
    ece = float(sum(c["n"] * abs(c["mean_look"] - c["observed_tumour_rate"]) for c in curve if c["n"]) / len(look))
    return {"per_band": per_band, "auroc_by_zone": {"edge": pair(3, 2), "mid": pair(4, 1), "far": pair(4, 0)},
            "auroc_overall": {"look": auroc(look[lab == 1], look[lab == 0]), "scan": auroc(scan[lab == 1], scan[lab == 0])},
            "reliability_curve_overall": curve, "ece_overall": ece}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--cases-root", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit-cases", type=int, default=0, help="debug: first N runs only (the output is then NOT the committed table)")
    a = ap.parse_args(argv)

    from antelligence.worlds.tumor_mri.store_adapter import load_case  # gate: dev cases only

    runs = [c for c in _jsonl(a.calib / "cells.jsonl") if c.get("arm") == ARM]
    if a.limit_cases:
        runs = runs[:a.limit_cases]
    drops: Counter = Counter()
    bands, scans, looks, scan_p = [], [], [], []
    cases = []
    n_board = 0
    for c in runs:
        before = sum(drops.values())
        kept = board_looks(a.calib, c["run_id"], drops)
        n_board += len(kept) + sum(drops.values()) - before
        loaded = load_case(c["case"], root=a.cases_root)   # raises GateError for any non-dev case
        if loaded.split != "dev":
            raise SystemExit(f"{c['case']}: not a development case")
        band_vol = distance_band_index(loaded.crop.truth, loaded.crop.voxel_mm)
        cases.append(c["case"])
        for _sq, pos, look_ppm, scan_ppm in kept:
            bands.append(int(band_vol[pos]))
            scans.append(int(scan_band_index(np.array([scan_ppm / 1e6]))[0]))
            looks.append(look_ppm)
            scan_p.append(scan_ppm)
    b, s, v, sp = (np.asarray(x) for x in (bands, scans, looks, scan_p))
    src = hashlib.sha256(b"".join(sorted(f"{c['run_id']}".encode() for c in runs))).hexdigest()
    meta = {"arm": ARM, "runs": len(runs), "cases": len(set(cases)), "split": "dev", "board_looks_seen": n_board,
            "dropped": dict(drops), "kept": int(len(v)), "run_ids_sha256": src,
            "note": "simulated sensor built from d1's recorded looks (v4.1 step-7 SW-B calibration boards); values are already "
                    "temperature-calibrated; no MRI, no per-voxel truth."}
    table = build_cell_table(b, s, v, source="boards", meta=meta)
    rel = {"source": "boards", "n_looks": int(len(v)), "dropped": dict(drops), "cases": len(set(cases)),
           "definitions": {"inside_depth": "mm to nearest non-tumour voxel", "outside_distance": "mm to nearest tumour voxel",
                           "look": "d1 look p (temperature-calibrated)", "scan": "perception p_tumour at the look"},
           **reliability(b, v / 1e6, sp / 1e6)}
    a.out.mkdir(parents=True, exist_ok=True)
    save_table(table, a.out / "cells_v4_2.json")
    (a.out / "reliability.json").write_text(json.dumps(rel, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"kept": int(len(v)), "seen": n_board, "dropped": dict(drops), "merges": len(table["merge_log"]),
                      "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
