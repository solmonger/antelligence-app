#!/usr/bin/env python
"""World tuning for v4.2 (goal 9.1). Arms S-solo and S' ONLY; never a sharing arm. Development-A cases, seed 1.

    python scripts/v42_tune.py --out runs/tune [--cases BRATS_016,BRATS_406,...] [--workers 4] [--ticks N]

Grid (12 points, at most 24): u_core {0.3,0.5,0.7} x f3_decay {0.04,0.08} x f3_share {0.1,0.2}. F2 Source B (cases without
TC / ET channels) uses f2_share 0.2, fixed, so the grid does not grow. For every point and case it runs S-solo and S'.
Selection: among points with median S-solo in [0.25, 0.55] and median S' in [0.60, 0.85], the one with the largest share of
cases with S' - S-solo >= 0.15 (ties: larger median gap, then grid order). If none qualifies: report that, print the table,
flag it. It does NOT re-tune drug_scale_k and it does NOT write config/v4_2/world.yaml; it prints the proposed values.

Resumable: ``episodes.jsonl`` gets one line per finished episode; a rerun skips those. ``tune_results.json`` is rewritten at
the end (and on every run) with every point.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v42_drive as drive  # noqa: E402
from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402
from antelligence.experiments.v42_harness import EvalRefused  # noqa: E402

TUNE_ARMS = ("S-solo", "S'")
U_CORE = (0.3, 0.5, 0.7)
F3_DECAY = (0.04, 0.08)
F3_SHARE = (0.1, 0.2)
MAX_POINTS = 24
GAP = 0.15
SOLO_BAND = (0.25, 0.55)
PRIME_BAND = (0.60, 0.85)
SEED = 1


class TuneArmRefused(ValueError):
    """Tuning runs S-solo and S' only; a sharing arm is refused."""


def grid() -> List[Dict[str, float]]:
    pts = [{"u_core": u, "f3_decay": d, "f3_share": s} for u, d, s in itertools.product(U_CORE, F3_DECAY, F3_SHARE)]
    if len(pts) > MAX_POINTS:
        raise ValueError(f"grid has {len(pts)} points; at most {MAX_POINTS}")
    return pts


def check_arms(arms: Sequence[str]) -> None:
    bad = [a for a in arms if a not in TUNE_ARMS]
    if bad:
        raise TuneArmRefused(f"tune mode runs {TUNE_ARMS} only; refused: {bad}")


def check_cases(cases: Sequence[str], split_doc: Optional[Mapping[str, Any]] = None) -> None:
    doc = split_doc or load_checked_split(REPO)
    for c in cases:
        if c not in doc["dev_a"]:
            raise EvalRefused(f"case {c!r} is not in dev-A; tuning uses dev-A cases only")


def make_tasks(points: Sequence[Mapping[str, float]], cases: Sequence[str], arms: Sequence[str], *, config: Path,
               cases_root: Optional[Path], ticks: Optional[int], done: set) -> List[Dict[str, Any]]:
    check_arms(arms)
    tasks = []
    for i, p in enumerate(points):
        for case in cases:
            for arm in arms:
                if (i, arm, case) in done:
                    continue
                tasks.append({"kind": "episode", "point": i, "arm": arm, "case_id": case, "seed": SEED, "config": str(config),
                              "cases_root": str(cases_root) if cases_root else None, "ticks": ticks, "board": False,
                              "facts": {"f2": p["u_core"], "f3": (p["f3_decay"], p["f3_share"])}})
    return tasks


def select(points: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ok = [p for p in points if p["complete"] and p["qualifies"]]
    if not ok:
        return {"selected": None, "flag": "NO POINT QUALIFIES: medians outside the bands for every complete point. "
                "drug_scale_k is NOT re-tuned automatically; read the table and decide."}
    best = max(ok, key=lambda p: (p["share_gap_ge_0.15"], p["median_gap"], -p["index"]))
    return {"selected": best["index"], "flag": None}


def summarise(points: Sequence[Mapping[str, float]], cases: Sequence[str], results: Mapping[tuple, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, p in enumerate(points):
        per_case = []
        for c in cases:
            a, b = results.get((i, "S-solo", c)), results.get((i, "S'", c))
            if a is not None and b is not None:
                per_case.append({"case_id": c, "s_solo": a["coverage"], "s_prime": b["coverage"], "gap": b["coverage"] - a["coverage"],
                                 "s_solo_s": a["timing_s"]["wall"], "s_prime_s": b["timing_s"]["wall"]})
        complete = len(per_case) == len(cases) and bool(cases)
        row: Dict[str, Any] = {"index": i, **p, "n_cases": len(per_case), "complete": complete, "per_case": per_case}
        if per_case:
            ms, mp = statistics.median(r["s_solo"] for r in per_case), statistics.median(r["s_prime"] for r in per_case)
            row.update({"median_s_solo": ms, "median_s_prime": mp, "median_gap": statistics.median(r["gap"] for r in per_case),
                        "share_gap_ge_0.15": sum(r["gap"] >= GAP - 1e-12 for r in per_case) / len(per_case),
                        "solo_in_band": SOLO_BAND[0] <= ms <= SOLO_BAND[1], "prime_in_band": PRIME_BAND[0] <= mp <= PRIME_BAND[1]})
            row["qualifies"] = row["solo_in_band"] and row["prime_in_band"]
        else:
            row["qualifies"] = False
        out.append(row)
    return out


def load_done(path: Path) -> Dict[tuple, Dict[str, Any]]:
    res: Dict[tuple, Dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                                   # a torn last line from a killed run: redo that episode
                res[(r["point"], r["arm"], r["case_id"])] = r
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", default="", help="comma list of dev-A case ids (default: all 30 of dev-A)")
    ap.add_argument("--arms", default=",".join(TUNE_ARMS), help="S-solo and S' only; anything else is refused")
    ap.add_argument("--workers", type=int, default=4, help=f"process pool size, at most {drive.MAX_WORKERS}")
    ap.add_argument("--config", type=Path, default=drive.DEFAULT_CONFIG)
    ap.add_argument("--cases-root", type=Path, default=drive.DEFAULT_CASES_ROOT if drive.DEFAULT_CASES_ROOT.is_dir() else None)
    ap.add_argument("--ticks", type=int, default=None, help="override episode length (dry runs only)")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    try:
        check_arms(arms)
        cases = [c.strip() for c in a.cases.split(",") if c.strip()] or list(load_checked_split(REPO)["dev_a"])
        check_cases(cases)
        drive.guard_dev_cases(cases)
    except (TuneArmRefused, EvalRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    points = grid()
    a.out.mkdir(parents=True, exist_ok=True)
    ep = a.out / "episodes.jsonl"
    results = load_done(ep)
    tasks = make_tasks(points, cases, arms, config=a.config, cases_root=a.cases_root, ticks=a.ticks, done=set(results))
    print(f"{len(points)} points x {len(cases)} cases x {len(arms)} arms; {len(tasks)} episodes to run, {len(results)} already done", flush=True)
    with ep.open("a") as fh:
        def on_result(task: Mapping[str, Any], r: Dict[str, Any]) -> None:
            r = {**r, "point": task["point"]}
            results[(task["point"], task["arm"], task["case_id"])] = r
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
            fh.flush()
            print(f"  point {task['point']:2d} {task['arm']:7s} {task['case_id']:10s} coverage {r['coverage']:.3f}  {r['timing_s']['wall']:.2f}s", flush=True)
        drive.run_pool(tasks, a.workers, on_result)
    rows = summarise(points, cases, results)
    sel = select(rows)
    wall = [r["timing_s"]["wall"] for r in results.values()]
    doc = {"seed": SEED, "arms": list(TUNE_ARMS), "cases": cases, "bands": {"s_solo": SOLO_BAND, "s_prime": PRIME_BAND, "gap": GAP},
           "points": rows, **sel, "episode_wall_s": {"n": len(wall), "mean": sum(wall) / len(wall) if wall else None,
                                                     "max": max(wall) if wall else None},
           "proposed": None if sel["selected"] is None else {k: rows[sel["selected"]][k] for k in ("u_core", "f3_decay", "f3_share")},
           "note": "config/v4_2/world.yaml is NOT written by this script."}
    (a.out / "tune_results.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print("\npoint  u_core decay share  med S-solo  med S'   gap>=.15  qualifies")
    for r in rows:
        if r["n_cases"]:
            print(f"{r['index']:5d}  {r['u_core']:.1f}   {r['f3_decay']:.2f}  {r['f3_share']:.1f}   {r['median_s_solo']:.3f}      {r['median_s_prime']:.3f}   "
                  f"{r['share_gap_ge_0.15']:.2f}      {r['qualifies']}{'' if r['complete'] else '  (incomplete)'}")
    if sel["selected"] is None:
        print(f"\n{sel['flag']}")
    else:
        print(f"\nproposed (NOT written to config): {doc['proposed']}  [point {sel['selected']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
