#!/usr/bin/env python
"""Message-effect study (v4.2 section 8, v4.3 paths) over a verdict output directory.

    python scripts/v43_effect.py --run-dir runs/verdict-mid --out runs/effect-mid [--workers 4] [--runs 10] [--items 30]

For S-marks and S-talk: take the baselines (``bundle.json`` of each completed episode under ``--run-dir``), pick 10 runs by the
seeded sample, and for up to 30 marks / messages of each run re-run the episode with that one item removed (K1 removal replay).
Reports the effect rate (the receiver's next action changed), the end-of-run coverage delta and the spam rate, and, new in
v4.3, the effect rate of the UPTAKE marks (``good_uptake`` and ``slow_uptake``) apart from the other marks and the messages.
Writes ``effect.json`` under ``--out`` (required); ``--run-dir`` is only read. The config, level and dry overrides of the
baseline are taken from its ``results.json`` meta. Development cases only. Needs the freeze file (it replays sharing arms).
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

from antelligence.experiments import v43_arms  # noqa: E402
from antelligence.experiments import v43_drive as drive  # noqa: E402
from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402
from antelligence.experiments.v42_harness import EvalRefused  # noqa: E402
from antelligence.experiments.v42_results import split_of  # noqa: E402
from antelligence.experiments.v43_freeze import DEFAULT_FREEZE_FILE, FreezeRefused, freeze_needed, require_frozen  # noqa: E402
from antelligence.experiments.v43_metrics import effect_by_group, effect_summary, sample_runs, tag_mark_kinds  # noqa: E402
from antelligence.worlds.tumor_mri.combined import DEFAULT_COEFFICIENTS_PATH  # noqa: E402

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
    ap.add_argument("--run-dir", required=True, type=Path, help="verdict output directory (read only)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4, help=f"at most {drive.MAX_WORKERS}")
    ap.add_argument("--config", type=Path, default=None, help="default: the config recorded in results.json, else the repo's")
    ap.add_argument("--coefficients", type=Path, default=None, help="default: the file recorded in results.json, else the repo's")
    ap.add_argument("--cases-root", type=Path, default=drive.DEFAULT_CASES_ROOT if drive.DEFAULT_CASES_ROOT.is_dir() else None)
    ap.add_argument("--freeze-file", type=Path, default=DEFAULT_FREEZE_FILE)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    v43_arms.register_all()
    try:
        drive.assert_out_allowed(a.out)
    except drive.OutRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if a.workers > drive.MAX_WORKERS:
        print(f"refused: at most {drive.MAX_WORKERS} workers", file=sys.stderr)
        return 2
    res_path = a.run_dir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.is_file() else None
    meta = (results or {}).get("meta", {})
    ov = meta.get("dry_overrides") or {}
    config = a.config or Path(meta.get("config", drive.DEFAULT_CONFIG))
    coefficients = a.coefficients or Path(meta.get("coefficients", DEFAULT_COEFFICIENTS_PATH))
    level = meta.get("level", "mid")
    doc = load_checked_split(REPO)
    base = {arm: baselines(a.run_dir, arm) for arm in EFFECT_ARMS}
    try:
        for arm in EFFECT_ARMS:
            for case, _ in base[arm]:
                split_of(doc, case)
                drive.guard_dev_cases([case])
        if freeze_needed(EFFECT_ARMS, on_dev_case=any(base[arm] for arm in EFFECT_ARMS)):
            require_frozen(a.freeze_file)
    except (EvalRefused, FreezeRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    tasks = []
    for arm in EFFECT_ARMS:
        for key in sample_runs(arm, list(base[arm]), a.runs):
            tasks.append({"kind": "effect", "arm": arm, "case_id": key[0], "seed": key[1], "bundle": base[arm][key],
                          "config": str(config), "ticks": ov.get("ticks"), "board": True, "max_items": a.items,
                          "uptake_level": level, "coefficients": str(coefficients), "freeze_file": str(a.freeze_file),
                          "cases_root": str(a.cases_root) if a.cases_root else None})
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
    effect = {}
    for arm in EFFECT_ARMS:
        marks = {k: v.get("marks", []) for k, v in base[arm].items()}
        tagged = tag_mark_kinds(rows[arm], marks)
        effect[arm] = {"arm": arm, "level": level, "runs": nruns[arm], "unapplied": unapplied[arm], "rows": tagged,
                       "by_group": effect_by_group(tagged), **effect_summary(tagged)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "effect.json").write_text(json.dumps(effect, indent=1, sort_keys=True, default=str))
    for arm, e in effect.items():
        up = e["by_group"]["uptake_marks"]
        print(f"{arm}: runs {e['runs']} items {e['n_items']} effect_rate {e['effect_rate']} mean_delta {e['mean_delta']} spam {e['spam_rate']} "
              f"unapplied {e['unapplied']}; uptake marks: items {up['n_items']} effect_rate {up['effect_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
