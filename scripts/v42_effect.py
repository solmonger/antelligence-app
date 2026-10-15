#!/usr/bin/env python
"""Message-effect study (goal 8, M1's ``effect_study``) over a verdict output directory.

    python scripts/v42_effect.py --run-dir runs/verdict [--workers 4] [--runs 10] [--items 30]

For S-marks and S-talk: take the baselines (``bundle.json`` of each completed episode), pick 10 runs by M1's seeded sample,
and for up to 30 marks / messages of each run re-run the episode with that item removed (K1 removal replay). Reports the
effect rate (the receiver's next action changed), the end-of-run coverage delta and the spam rate. Writes
``<run-dir>/effect.json`` and adds ``effect`` to ``<run-dir>/results.json`` when it exists. Development cases only; the
config and fact settings of the baseline are taken from ``results.json`` meta (the dry-run overrides are re-applied).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v42_drive as drive  # noqa: E402
from antelligence.experiments.v42_harness import EvalRefused  # noqa: E402
from antelligence.experiments.v42_metrics import effect_summary, sample_runs  # noqa: E402
from antelligence.experiments.v42_results import split_of  # noqa: E402
from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402

EFFECT_ARMS = ("S-marks", "S-talk")


def baselines(run_dir: Path, arm: str) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for rec_path in sorted((run_dir / arm).glob("*/seed*/record.json")):
        d = rec_path.parent
        if not json.loads(rec_path.read_text()).get("complete"):
            continue
        case, seed = d.parent.name, int(d.name[4:])
        colony = (json.loads((d / "bundle.json").read_text()).get("extra") or {}).get("colony")
        if colony is not None:
            out[(case, seed)] = colony
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4, help=f"at most {drive.MAX_WORKERS}")
    ap.add_argument("--config", type=Path, default=None, help="default: the config recorded in results.json, else the repo's")
    ap.add_argument("--cases-root", type=Path, default=drive.DEFAULT_CASES_ROOT if drive.DEFAULT_CASES_ROOT.is_dir() else None)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    res_path = a.run_dir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.is_file() else None
    meta = (results or {}).get("meta", {})
    ov = meta.get("dry_overrides") or {}
    config = a.config or Path(meta.get("config", drive.DEFAULT_CONFIG))
    doc = load_checked_split(REPO)
    base = {arm: baselines(a.run_dir, arm) for arm in EFFECT_ARMS}
    try:
        for arm in EFFECT_ARMS:
            for case, _ in base[arm]:
                split_of(doc, case)
                drive.guard_dev_cases([case])
    except EvalRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    tasks = []
    for arm in EFFECT_ARMS:
        for key in sample_runs(arm, list(base[arm]), a.runs):
            tasks.append({"kind": "effect", "arm": arm, "case_id": key[0], "seed": key[1], "bundle": base[arm][key],
                          "config": str(config), "ticks": ov.get("ticks"), "board": True, "max_items": a.items,
                          "cases_root": str(a.cases_root) if a.cases_root else None,
                          "facts": {"f2": ov.get("f2"), "f3": ov.get("f3")}})
    print(f"{len(tasks)} runs to replay", flush=True)
    rows: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in EFFECT_ARMS}
    unapplied = {arm: 0 for arm in EFFECT_ARMS}
    nruns = {arm: 0 for arm in EFFECT_ARMS}

    def on_result(task: Mapping[str, Any], r: Dict[str, Any]) -> None:
        arm = task["arm"]
        rows[arm] += r["rows"]
        unapplied[arm] += r["unapplied"]
        nruns[arm] += 1
        print(f"  {arm:8s} {task['case_id']:10s} seed {task['seed']}: {len(r['rows'])} items replayed, {r['unapplied']} unapplied", flush=True)

    drive.run_pool(tasks, a.workers, on_result)
    effect = {arm: {"arm": arm, "runs": nruns[arm], "unapplied": unapplied[arm], "rows": rows[arm], **effect_summary(rows[arm])}
              for arm in EFFECT_ARMS}
    (a.run_dir / "effect.json").write_text(json.dumps(effect, indent=1, sort_keys=True, default=str))
    if results is not None:
        results["effect"] = {arm: {k: v for k, v in e.items() if k != "rows"} for arm, e in effect.items()}
        res_path.write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
    for arm, e in effect.items():
        print(f"{arm}: runs {e['runs']} items {e['n_items']} effect_rate {e['effect_rate']} mean_delta {e['mean_delta']} spam {e['spam_rate']} unapplied {e['unapplied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
