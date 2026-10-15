#!/usr/bin/env python
"""Fit and check the combined scan-and-look estimate c = P(tumour | scan p, look) (v4.3, workstream A1).

Fit mode (default). Reads the verified d1 looks of the v4.1 SW-B calibration boards (read-only; `board_looks` is imported by
file path from the v4.2 SimLook script, never run), labels each look with the case's whole-tumour truth at the look's voxel
(development cases only), fits the combined estimate on dev-A looks, and reports dev-A (in sample) and dev-B (held out):
n, ECE and AUROC for the combined estimate, the scan alone and the look alone, plus the five distance bands and the drop
counters. Writes coefficients.json and calibration.json under --out (default docs/research/v4.3-20261015/combined/).
No per-voxel data, no MRI and no sub-region truth is written.

    python scripts/v43_combined_fit.py [--calib DIR] [--cases-root DIR] [--out DIR]

Circularity mode. SimLook draws from the same looks the fit used. Reads the S-solo bundles of a run directory
(RUN_DIR/S-solo/<case>/seed<N>/bundle.json), takes each look from extra.colony.reading_logs, the scan p from the case crop
and the truth from the case crop, and writes circularity.json: does the combined estimate still beat the scan on simulated looks?

    python scripts/v43_combined_fit.py --circularity RUN_DIR --cases-root DIR --out DIR [--coefficients FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.worlds.tumor_mri import combined_fit as cf  # noqa: E402

DEFAULT_CALIB = Path("/Volumes/WD_BLACK/antelligence-v4.1/runs/calib1")
DEFAULT_CASES = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
DEFAULT_OUT = REPO / "docs/research/v4.3-20261015/combined"
SPLIT_FILE = REPO / "config/v4_1/split.json"
V42_BOARD_SCRIPT = REPO / "scripts" / "v42_simlook_cells.py"
ARM = "SW-B"
BAND_NAMES = ["out_gt10", "out_4_10", "out_0_4", "in_0_4", "in_gt4"]
# each band is compared with the opposite-label band at a comparable distance (same pairing as the v4.2 reliability table)
BAND_PARTNER = {0: 4, 1: 4, 2: 3, 3: 2, 4: 1}
ZONES = {"edge": (3, 2), "mid": (4, 1), "far": (4, 0)}   # (tumour band, non-tumour band)
FORBIDDEN_OUT = ("config/v4_1", "config/v4_2", "docs/research/v4.1-20261013", "docs/research/v4.2-20261014")


def _jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _refuse_forbidden_out(out: Path) -> None:
    resolved = out.resolve()
    for bad in FORBIDDEN_OUT:
        if resolved == (REPO / bad).resolve() or (REPO / bad).resolve() in resolved.parents:
            raise SystemExit(f"refusing to write under {bad}")
    for root in ("/Volumes/WD_BLACK/antelligence-v4.1", "/Volumes/WD_BLACK/antelligence-v4.2"):
        if resolved == Path(root) or Path(root) in resolved.parents:
            raise SystemExit(f"refusing to write under {root}")


def import_board_looks() -> Callable:
    """`board_looks` from the v4.2 SimLook script, loaded by file path (importing it does not run its main)."""
    spec = importlib.util.spec_from_file_location("_simlook_cells_by_path", V42_BOARD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.board_looks


def split_sets(split_file: Path = SPLIT_FILE) -> Dict[str, set]:
    doc = json.loads(split_file.read_text())
    return {"dev_a": set(doc["dev_a"]), "dev_b": set(doc["dev_b"])}


# --- metrics ---------------------------------------------------------------------------------------------------------------

def _score_block(score: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    return {"ece": cf.ece(score, y), "auroc": cf.auroc(score, y)}


def metrics(scan: np.ndarray, look: np.ndarray, y: np.ndarray, coef: Sequence[float], clip: Sequence[float] = cf.CLIP) -> Dict[str, Any]:
    """n, ECE and AUROC of the combined estimate, the scan alone and the look alone. Probabilities in [0, 1]."""
    if len(y) == 0:
        return {"n": 0}
    comb = cf.predict(coef, scan, look, clip[0], clip[1])
    return {"n": int(len(y)), "positive_rate": float(np.mean(y)), "combined": _score_block(comb, y),
            "scan": _score_block(scan, y), "look": _score_block(look, y)}


def band_metrics(scan: np.ndarray, look: np.ndarray, band: np.ndarray, coef: Sequence[float], clip: Sequence[float] = cf.CLIP) -> Dict[str, Any]:
    """Per distance band: n, ECE of each score (the label is constant inside a band, so ECE is the calibration gap), and AUROC of
    the band's scores against the opposite-label partner band. Also the edge / mid / far zone pairs."""
    comb = cf.predict(coef, scan, look, clip[0], clip[1]) if len(band) else np.array([])
    scores = {"combined": comb, "scan": scan, "look": look}
    lab = (band >= 3).astype(float)

    def pair_auroc(pos_b: int, neg_b: int) -> Dict[str, Any]:
        p, n = band == pos_b, band == neg_b
        out: Dict[str, Any] = {"positive": BAND_NAMES[pos_b], "negative": BAND_NAMES[neg_b]}
        for name, s in scores.items():
            out[name] = cf.auroc(np.concatenate([s[p], s[n]]), np.concatenate([np.ones(p.sum()), np.zeros(n.sum())]))
        return out

    per_band = {}
    for i, name in enumerate(BAND_NAMES):
        m = band == i
        if not m.any():
            per_band[name] = {"n": 0}
            continue
        entry: Dict[str, Any] = {"n": int(m.sum()), "label_rate": float(lab[m].mean()), "partner": BAND_NAMES[BAND_PARTNER[i]]}
        for sname, s in scores.items():
            entry[f"ece_{sname}"] = cf.ece(s[m], lab[m])
        pos_b, neg_b = (i, BAND_PARTNER[i]) if i >= 3 else (BAND_PARTNER[i], i)
        pa = pair_auroc(pos_b, neg_b)
        entry["auroc_vs_partner"] = {k: pa[k] for k in scores}
        per_band[name] = entry
    return {"per_band": per_band, "auroc_by_zone": {z: pair_auroc(*pair) for z, pair in ZONES.items()}}


# --- fit mode --------------------------------------------------------------------------------------------------------------

def gather(calib: Path, cases_root: Path, split_file: Path = SPLIT_FILE, repo_root: Path = REPO,
           board_looks: Optional[Callable] = None) -> Dict[str, Any]:
    """Verified, labelled looks of every dev-A / dev-B SW-B calibration run."""
    from antelligence.experiments.v42_harness import assert_dev_case, load_dev_case  # dev-only gate

    board_looks = board_looks or import_board_looks()
    sets = split_sets(split_file)
    runs = [c for c in _jsonl(calib / "cells.jsonl") if c.get("arm") == ARM]
    truth_cache: Dict[str, Tuple[np.ndarray, float]] = {}
    drops = {"dev_a": Counter(), "dev_b": Counter()}
    seen = Counter()
    rows: List[dict] = []
    for c in runs:
        case, run_id = c["case"], c["run_id"]
        assert_dev_case(case, repo_root)
        split = "dev_a" if case in sets["dev_a"] else "dev_b" if case in sets["dev_b"] else None
        if split is None:
            raise SystemExit(f"{case}: not in dev_a or dev_b")
        if case not in truth_cache:
            from antelligence.worlds.tumor_mri.simlook import distance_band_index
            loaded = load_dev_case(case, root=cases_root, repo_root=repo_root)
            truth_cache[case] = (distance_band_index(loaded.crop.truth, loaded.crop.voxel_mm), loaded.crop.truth)
        band_vol, truth = truth_cache[case]
        before = sum(drops[split].values())
        kept = board_looks(calib, run_id, drops[split])
        seen[split] += len(kept) + sum(drops[split].values()) - before
        for sq, pos, look_ppm, scan_ppm in kept:
            if not all(0 <= int(p) < s for p, s in zip(pos, truth.shape)):
                drops[split]["pos_outside_crop"] += 1
                continue
            rows.append({"run": run_id, "case": case, "split": split, "squad": int(sq), "pos": tuple(int(p) for p in pos),
                         "look_ppm": int(look_ppm), "scan_ppm": int(scan_ppm), "label": int(truth[pos]), "band": int(band_vol[pos])})
    return {"rows": rows, "drops": {k: dict(v) for k, v in drops.items()}, "seen": dict(seen), "runs": len(runs)}


def _arrays(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([r["scan_ppm"] for r in rows], dtype=np.float64) / 1e6, np.array([r["look_ppm"] for r in rows], dtype=np.float64) / 1e6,
            np.array([r["label"] for r in rows], dtype=np.float64), np.array([r["band"] for r in rows], dtype=np.int64))


def inputs_sha256(rows: Sequence[dict]) -> str:
    keyed = sorted([r["run"], r["squad"], list(r["pos"]), r["look_ppm"], r["scan_ppm"]] for r in rows)
    return hashlib.sha256(json.dumps(keyed, separators=(",", ":")).encode()).hexdigest()


def run_fit(data: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(coefficients.json body, calibration.json body) from gathered rows. Fits on dev-A only."""
    rows = data["rows"]
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("dev_a", "dev_b")}
    a = by_split["dev_a"]
    if not a or len({r["label"] for r in a}) < 2:
        raise SystemExit("dev-A has no looks of one class; cannot fit")
    scan, look, y, _ = _arrays(a)
    coef, info = cf.fit_logistic(cf.design(scan, look), y)
    coef_doc = cf.coefficients_doc(coef, split="dev_a", n_looks=len(a), n_cases=len({r["case"] for r in a}),
                                   ridge=info["ridge"], inputs_sha256=inputs_sha256(a))
    out: Dict[str, Any] = {"schema": "combined_calibration_v4_3", "coef": coef_doc["coef"], "fit_info": info,
                           "fit_split": "dev_a", "label": "whole-tumour truth at the look's voxel",
                           "definitions": {"scan": "perception p_tumour at the look (ppm / 1e6)", "look": "d1 look p, temperature-calibrated (ppm / 1e6)",
                                           "combined": "logistic on [1, logit scan, logit look, logit scan * logit look], clip 0.001 to 0.999",
                                           "ece": "10 equal-width bins on the predicted probability",
                                           "band_auroc": "band scores vs the opposite-label partner band; zones as in the v4.2 reliability table"},
                           "runs": data["runs"], "board_looks_seen": data["seen"], "dropped": data["drops"], "splits": {}}
    for s, rs in by_split.items():
        sc, lk, yy, bb = _arrays(rs)
        out["splits"][s] = {"cases": len({r["case"] for r in rs}), "in_sample": s == "dev_a",
                            "overall": metrics(sc, lk, yy, coef) if len(rs) else {"n": 0},
                            "by_band": band_metrics(sc, lk, bb, coef) if len(rs) else {}}
    return coef_doc, out


def fit_main(a: argparse.Namespace) -> int:
    _refuse_forbidden_out(a.out)
    data = gather(a.calib, a.cases_root)
    coef_doc, cal = run_fit(data)
    cf.save_json(coef_doc, a.out / "coefficients.json")
    cf.save_json(cal, a.out / "calibration.json")
    print(json.dumps({"coef": coef_doc["coef"], "fit": coef_doc["fit"], "dropped": cal["dropped"], "out": str(a.out),
                      "dev_a": cal["splits"]["dev_a"]["overall"], "dev_b": cal["splits"]["dev_b"]["overall"]}, indent=1))
    return 0


# --- circularity mode ------------------------------------------------------------------------------------------------------

def default_case_loader(cases_root: Path, repo_root: Path = REPO) -> Callable[[str], Tuple[np.ndarray, np.ndarray]]:
    from antelligence.experiments.v42_harness import load_dev_case

    def load(case: str) -> Tuple[np.ndarray, np.ndarray]:
        crop = load_dev_case(case, root=cases_root, repo_root=repo_root).crop
        return np.asarray(crop.p_tumour), np.asarray(crop.truth, dtype=bool)
    return load


def run_circularity(run_dir: Path, coef_doc: Mapping[str, Any], load_case: Callable[[str], Tuple[np.ndarray, np.ndarray]],
                    sets: Mapping[str, set]) -> Dict[str, Any]:
    """Combined vs scan vs look on the looks recorded in the S-solo bundles of ``run_dir``. ``load_case(case)`` returns
    ``(p_tumour, truth)`` on the crop grid. A case outside dev-A and dev-B is refused before anything is loaded."""
    coef, clip = coef_doc["coef"], coef_doc["clip"]
    bundles = sorted((run_dir / "S-solo").glob("*/seed*/bundle.json"))
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    drops: Counter = Counter()
    recs: List[Tuple[str, Tuple[int, int, int], float, float, int]] = []
    for path in bundles:
        case = path.parent.parent.name
        split = "dev_a" if case in sets["dev_a"] else "dev_b" if case in sets["dev_b"] else None
        if split is None:
            raise SystemExit(f"{case}: not a dev-A or dev-B case; refused")
        if case not in cache:
            cache[case] = load_case(case)
        p_map, truth = cache[case]
        colony = ((json.loads(path.read_text()).get("extra") or {}).get("colony")) or {}
        for log in (colony.get("reading_logs") or {}).values():
            for e in log:
                if e.get("kind") != "look":
                    continue
                if e.get("value") is None or e.get("pos_vox") is None:
                    drops["look_without_value_or_pos"] += 1
                    continue
                pos = tuple(int(c) for c in e["pos_vox"])
                if len(pos) != p_map.ndim or not all(0 <= c < s for c, s in zip(pos, p_map.shape)):
                    drops["pos_outside_crop"] += 1
                    continue
                recs.append((split, case, pos, float(e["value"]), float(p_map[pos]), int(truth[pos])))
    result: Dict[str, Any] = {"schema": "circularity_v4_3", "run_dir_bundles": len(bundles), "dropped": dict(drops),
                              "coefficients_fit": dict(coef_doc.get("fit", {})),
                              "note": "dev-A looks are in the sample the coefficients were fit on; dev-B is held out."}

    def block(sel: Sequence[tuple]) -> Dict[str, Any]:
        if not sel:
            return {"n": 0, "n_unique_voxels": 0}
        look = np.array([r[3] for r in sel]); scan = np.array([r[4] for r in sel]); y = np.array([r[5] for r in sel], dtype=np.float64)
        return {**metrics(scan, look, y, coef, clip), "n_unique_voxels": len({(r[1], r[2]) for r in sel})}

    result["overall"] = block(recs)
    for s in ("dev_a", "dev_b"):
        result[s] = block([r for r in recs if r[0] == s])
    return result


def circularity_main(a: argparse.Namespace) -> int:
    _refuse_forbidden_out(a.out)
    coef_doc = cf.load_coefficients(a.coefficients)
    res = run_circularity(a.circularity, coef_doc, default_case_loader(a.cases_root), split_sets())
    cf.save_json(res, a.out / "circularity.json")
    print(json.dumps({"overall": res["overall"], "out": str(a.out)}, indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--cases-root", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--circularity", type=Path, default=None, metavar="RUN_DIR", help="circularity mode: read S-solo bundles under RUN_DIR")
    ap.add_argument("--coefficients", type=Path, default=DEFAULT_OUT / "coefficients.json", help="circularity mode: coefficients file")
    a = ap.parse_args(argv)
    return circularity_main(a) if a.circularity else fit_main(a)


if __name__ == "__main__":
    raise SystemExit(main())
