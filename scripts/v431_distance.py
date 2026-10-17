#!/usr/bin/env python
"""v4.31 A.2: how far is each tumour voxel from its nearest release point? (post hoc, on the release logs the runs already stored: no new runs)

    python scripts/v431_distance.py --runs /Volumes/WD_BLACK/antelligence-v4.31/runs --out docs/research/v4.31-20261016/phaseA2

Alongside the coverage curve: the distribution of the distance from each tumour voxel to its nearest release, and the fraction of TUMOUR VOLUME within 0.2 mm and within 0.5 mm of a release.
0.2 mm is the published design target for lethal coverage by distributed delivery (Hauert, Berman, Nagpal & Bhatia 2013, Nano Today 8:566-576); 0.5 mm is the wafer-calibrated drug reach
(L_ref = 0.505 mm). The two bracket the length scale the whole problem turns on. Releases are the logged 2 mm voxel centres placed on the 1 mm lattice (x, y exactly; z snapped to the nearest
corner), tumour = native 1 mm whole-tumour truth, development cases only. Measure only. Entry and motion are unchanged and unphysical.

Writes distance_to_release.json (deterministic) and distance_tables.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.worlds.tumor_mri import geometry as GEO  # noqa: E402
from antelligence.worlds.tumor_mri.fine_grid import load_fine_case  # noqa: E402

FOLDERS = {"ref": "a2_ref", "low": "a2_low", "high": "a2_high"}
FINE_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.31/data/cases_1mm_a2")
PLANS = (7.0, 23.5, 50.0, 262.0)


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big") % (2 ** 32)


def load_rows(runs: Path, folder: str) -> List[Dict[str, Any]]:
    rows = []
    for f in sorted((runs / folder).glob("runs_shard*of*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (r["arm"], r["plan_thr"] if r["plan_thr"] is not None else -1.0, r["case"], r["seed"]))
    return rows


def boot_ci(per_case: np.ndarray, n: int = 10000, seed: int = 20261016) -> List[float]:
    rng = np.random.default_rng(seed)
    means = per_case[rng.integers(0, len(per_case), size=(n, len(per_case)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents and a.out.resolve() != allowed:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    fine_cache: Dict[str, Any] = {}
    doc: Dict[str, Any] = {"schema": "v431_distance_to_release", "design_target_mm": GEO.DESIGN_TARGET_MM, "wafer_reach_mm": GEO.WAFER_REACH_MM, "radii_mm": list(GEO.FRACTION_RADII_MM),
                           "bins_mm": [float(e) if np.isfinite(e) else None for e in GEO.DISTANCE_BINS_MM],
                           "design_target_source": "Hauert, Berman, Nagpal & Bhatia 2013, Nano Today 8:566-576: the published design target for lethal coverage by distributed delivery is 0.2 mm",
                           "wafer_reach_source": "L_ref = sqrt(D*/k_ref) = 0.505 mm (calibrated to carmustine's wafer penetration profiles)", "sets": {}}
    for k, folder in FOLDERS.items():
        rows = [r for r in load_rows(a.runs, folder) if "scored" in r and "1mm" in r["scored"] and "releases" in r["scored"]["1mm"]]
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            key = "S" if r["arm"] == "S" else f"Sp_plan{r['plan_thr']:g}"
            if r["arm"] == "S" or r["plan_thr"] in PLANS:
                groups.setdefault(key, []).append(r)
        res: Dict[str, Any] = {}
        for key, rs in sorted(groups.items()):
            per_case: Dict[str, List[Dict[str, Any]]] = {}
            for r in rs:
                if r["case"] not in fine_cache:
                    fine_cache[r["case"]] = load_fine_case(r["case"], root=FINE_ROOT, repo_root=REPO)
                fine = fine_cache[r["case"]]
                rel = r["scored"]["1mm"]["releases"]
                lat = fine.release_lattice([x[0] for x in rel], 1.0).astype(np.float64)
                st = GEO.tumour_release_distance_stats(fine.truth[0], lat, 1.0, n_samples=4000, seed=_seed(k, key, r["case"], r["seed"]))
                per_case.setdefault(r["case"], []).append(st)
            cases = sorted(per_case)
            m = lambda f: np.array([np.mean([f(s) for s in per_case[c]]) for c in cases])          # noqa: E731
            fr = {r_: m(lambda s, r_=r_: s["fraction_within_mm"][r_]) for r_ in (f"{x:g}" for x in GEO.FRACTION_RADII_MM)}
            hist = np.mean([m(lambda s, i=i: s["hist_fraction"][i]) for i in range(len(GEO.DISTANCE_BINS_MM) - 1)], axis=1)
            res[key] = {"n_cases": len(cases), "n_runs": len(rs),
                        "fraction_of_tumour_volume_within_mm": {r_: {"mean": float(v.mean()), "median": float(np.median(v)), "ci95_mean": boot_ci(v), "max": float(v.max())} for r_, v in fr.items()},
                        "hist_fraction_mean": hist.tolist(),
                        "distance_mm": {q: float(m(lambda s, q=q: s["quantiles_mm"][q]).mean()) for q in ("min", "p10", "median", "mean", "p90", "max")},
                        "n_release_points_mean": float(m(lambda s: s["n_release_points"]).mean()),
                        "tumour_volume_cm3_median": float(np.median(m(lambda s: s["tumour_volume_mm3"])) / 1000.0)}
        doc["sets"][k] = res
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "distance_to_release.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    # tables
    lab = ["0-0.5", "0.5-1", "1-2", "2-3", "3-5", "5-7.5", "7.5-10", "10-15", "15-20", "20-30", "30-50", ">50"]
    md = ["<!-- generated by scripts/v431_distance.py from the stored release logs; do not edit by hand -->", "",
          "#### Distance from each tumour voxel to its nearest release point (1 mm lattice; releases at their 2 mm voxel centres; whole-tumour truth; mean over 40 cases)", "",
          "0.2 mm is the published design target for lethal coverage by distributed delivery (Hauert, Berman, Nagpal & Bhatia 2013, Nano Today 8:566-576); 0.5 mm is the wafer-calibrated drug reach "
          "(L = 0.505 mm). The two bracket the length scale the whole problem turns on.", "",
          "| clearance | arm | releases | median distance | 90th percentile | within 0.2 mm (design target) | within 0.5 mm (drug reach) | within 1 mm | within 2 mm |", "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: f"{100 * v['mean']:.3f}% (95% {100 * v['ci95_mean'][0]:.3f}-{100 * v['ci95_mean'][1]:.3f})"      # noqa: E731

    def nm(key: str) -> str:
        return key.replace("Sp_plan", "S' planned for ")

    for k, res in doc["sets"].items():
        for key, r in res.items():
            fw = r["fraction_of_tumour_volume_within_mm"]
            md.append(f"| {k} | {nm(key)} | {r['n_release_points_mean']:.0f} | {r['distance_mm']['median']:.1f} mm | {r['distance_mm']['p90']:.1f} mm | {f(fw['0.2'])} | {f(fw['0.5'])} | "
                      f"{100 * fw['1']['mean']:.2f}% | {100 * fw['2']['mean']:.2f}% |")
    md += ["", "Histogram of the distance (share of tumour volume per band, mean over cases), k_ref:", "", "| arm | " + " | ".join(lab) + " |", "|---|" + "---|" * len(lab)]
    for key, r in doc["sets"].get("ref", {}).items():
        md.append(f"| {nm(key)} | " + " | ".join(f"{100 * x:.1f}%" for x in r["hist_fraction_mean"]) + " |")
    md.append("")
    (a.out / "distance_tables.md").write_text("\n".join(md))
    print(f"wrote {a.out / 'distance_to_release.json'}  sha256={hashlib.sha256((a.out / 'distance_to_release.json').read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
