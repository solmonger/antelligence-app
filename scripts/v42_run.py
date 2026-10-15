#!/usr/bin/env python
"""Run v4.2 colony arms on development cases (Q1). No model, no network, no key: scripts only.

    python scripts/v42_run.py --out runs/q1 --arms S,S-solo,S-marks,S-talk,S-marks-quiet,"S'" \\
        --cases BRATS_016,BRATS_406 --seeds 1,2 [--ticks 120] [--f2 0.5] [--f3 0.04,0.1]
    python scripts/v42_run.py --out runs/q1-smoke --smoke [--cases BRATS_016] [--ticks 60]

* Development cases only. An evaluation, reserve or unknown case id exits with status 2 before anything is loaded.
* Resumable: an episode whose ``record.json`` exists is skipped. ``bundle.json`` is written first and ``record.json`` (the
  completion marker) last, both atomically, so a killed run never leaves a half-written episode that looks finished.
* Layout: ``<out>/<arm>/<case>/seed<N>/{bundle.json, record.json, cards/ant-NN.json}`` (the final name cards).
* ``--phantom [PATH]`` adds the synthetic phantom (a smoke fixture, case id ``phantom``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from antelligence.experiments.v42_harness import (EvalRefused, V42Harness, assert_dev_case, bundle_of,  # noqa: E402
                                                  load_dev_case, phantom_case, resolve_arm)
import antelligence.experiments.v42_arms  # noqa: E402,F401  (registers the control arms)
from antelligence.kernel.colony.episode import ColonyConfig  # noqa: E402
from antelligence.worlds.tumor_mri import WorldConfig  # noqa: E402
from antelligence.worlds.tumor_mri.config import FactsConfig  # noqa: E402

DEFAULT_CONFIG = REPO / "config" / "v4_2" / "world.yaml"
DEFAULT_CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
DEFAULT_PHANTOM = REPO / "tests" / "v4_1" / "world" / "fixtures" / "phantom.npz"
ALL_ARMS = "S,S-solo,S-marks,S-talk,S-marks-quiet,S'"
CLI_ALIASES = {"S-prime": "S'"}                  # command line only; the arm is S' everywhere else


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def episode_dir(out: Path, arm: str, case: str, seed: int) -> Path:
    return out / arm / case / f"seed{seed}"


def done(d: Path) -> bool:
    try:
        return bool(json.loads((d / "record.json").read_text()).get("complete"))
    except (OSError, ValueError):
        return False


def load_configs(path: Path):
    doc = yaml.safe_load(path.read_text())
    return WorldConfig.from_dict(doc["world"]), ColonyConfig.from_dict({**doc.get("colony", {}), "quiet_rule": False})


def facts_for(base: FactsConfig, args: argparse.Namespace, has_channels: bool) -> FactsConfig:
    f = replace(base, f1_simlook=not args.no_f1)
    if args.f2 is not None:
        f = replace(f, f2=True, u_core=args.f2, f2_source="A" if has_channels else "B", f2_share=0.0 if has_channels else 0.2)
    if args.f3 is not None:
        decay, share = args.f3
        f = replace(f, f3=True, f3_decay=decay, f3_share=share)
    return f


def run_one(h: V42Harness, arm: str, case, seed: int, out: Path) -> Dict[str, Any]:
    d = episode_dir(out, resolve_arm(arm).name, case.case_id, seed)
    if done(d):
        return {"skipped": True, "arm": resolve_arm(arm).name, "case_id": case.case_id, "seed": seed}
    rec = h.run_episode(arm, case, seed)
    atomic_write(d / "bundle.json", json.dumps(bundle_of(rec), sort_keys=True))
    for ant_id, card in (rec.get("cards_final") or {}).items():
        atomic_write(d / "cards" / f"{ant_id}.json", json.dumps(card, sort_keys=True))
    slim = {k: v for k, v in rec.items() if k != "colony"}
    slim.update({"bundle": "bundle.json", "complete": True})
    atomic_write(d / "record.json", json.dumps(slim, sort_keys=True, default=str))
    return rec


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default=ALL_ARMS, help="comma list; S-prime is accepted as an alias of S'")
    ap.add_argument("--cases", default="", help="comma list of development case ids")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--phantom", nargs="?", const=str(DEFAULT_PHANTOM), default=None, help="add the synthetic phantom")
    ap.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT if DEFAULT_CASES_ROOT.is_dir() else None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ticks", type=int, default=None, help="override the episode length (smoke runs)")
    ap.add_argument("--no-f1", action="store_true", help="do not inject SimLook (colony arms then refuse to run)")
    ap.add_argument("--f2", type=float, default=None, metavar="U_CORE", help="turn F2 on, u_core in {0.3,0.5,0.7}")
    ap.add_argument("--f3", type=lambda s: tuple(float(x) for x in s.split(",")), default=None, metavar="DECAY,SHARE")
    ap.add_argument("--no-board", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="every arm once on one dev case and the phantom; prints wall times")
    return ap.parse_args(argv)


def dev_cases_available(root: Optional[Path]) -> List[str]:
    from antelligence.experiments.v41_gate import load_checked_split
    doc = load_checked_split(REPO)
    base = root if root is not None else Path("data/v4_1/cases")
    return [c for c in doc["dev_a"] + doc["dev_b"] if (base / c / "case.npz").is_file()]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    arms = [resolve_arm(CLI_ALIASES.get(a.strip(), a.strip())).name for a in args.arms.split(",") if a.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    try:
        for c in cases:
            assert_dev_case(c)                                     # refuse evaluation / reserve / unknown ids up front
    except EvalRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.smoke and not cases:
        avail = dev_cases_available(args.cases_root)
        cases = avail[:1]
        if not cases:
            print("no development case found under the cases root; running the phantom only", file=sys.stderr)
        if args.phantom is None:
            args.phantom = str(DEFAULT_PHANTOM)
    world_cfg, colony_cfg = load_configs(args.config)
    results: List[Dict[str, Any]] = []
    todo = [(c, False) for c in cases] + ([("phantom", True)] if args.phantom else [])
    for case_id, is_phantom in todo:
        if is_phantom:
            case = phantom_case(Path(args.phantom))
            cfg = WorldConfig(total_drug=400.0, dose_threshold=6.0, ticks=args.ticks or 60)
        else:
            case = load_dev_case(case_id, root=args.cases_root)
            cfg = world_cfg.with_(ticks=args.ticks) if args.ticks else world_cfg
        cfg = cfg.with_(facts=facts_for(cfg.facts, args, case.scorer_channels is not None))
        h = V42Harness(cfg, colony_cfg, with_board=not args.no_board)
        for arm in arms:
            for seed in seeds:
                t0 = time.perf_counter()
                rec = run_one(h, arm, case, seed, args.out)
                if rec.get("skipped"):
                    print(f"skip   {arm:14s} {case.case_id:10s} seed {seed} (done)")
                    continue
                wall = time.perf_counter() - t0
                results.append({"arm": rec["arm"], "case_id": case.case_id, "seed": seed, "coverage": rec["metrics"]["coverage"],
                                "timing_s": rec["timing_s"], "episode_wall_s": round(wall, 3), "ticks": cfg.ticks})
                print(f"ran    {rec['arm']:14s} {case.case_id:10s} seed {seed}  coverage {rec['metrics']['coverage']:.3f}  "
                      f"wall {wall:6.2f}s (run {rec['timing_s']['run']:.2f}s, simlook {rec['timing_s']['simlook']:.2f}s)")
    if args.smoke:
        atomic_write(args.out / "smoke.json", json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
