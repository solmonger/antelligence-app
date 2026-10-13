#!/usr/bin/env python
"""Run v4.1 episodes (goal 4.7): one run bundle, board export, anchors and cell record per (arm, case, seed).

  .venv/bin/python scripts/v41_run.py --config run.json --out-dir runs --cells cells.jsonl \\
      --run SW-B:BRATS_001:1 --run SW-noBoard:BRATS_001:1 --concurrency 4

``--config`` names the live backends, queen and cache (see ``antelligence/experiments/v41_live.py``); API keys are read
from the environment variables the config names, nowhere else. The gate (``v41_gate``) runs before every episode; an
evaluation case needs the committed preregistration, its confirmed anchor receipt and the G1 approval file.
``--calibration`` is Phase 0 step 7 (Qwen3.5-9B drives, d1 and Qwen3.5-4B answer counterfactually, development only).
``--anchors mock`` anchors to in-process mock OTS/Rekor/chain (plumbing and tests); live anchors are wired by the lead at
the plumbing run, because they need the chain signer.

Exit code 1 if any episode failed; failed runs are written to the cells file with an ``error`` field.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v41_live  # noqa: E402
from antelligence.experiments.v41_conformal import ConformalThresholds  # noqa: E402
from antelligence.experiments.v41_harness import (ARM_SPECS, GateConfig, KeyVault, V41Harness,  # noqa: E402
                                                  append_cell, error_cell, make_gate)
from antelligence.experiments.v41_router import KNNRouter  # noqa: E402
from antelligence.kernel.policies.v41_harness import load_harness  # noqa: E402
from antelligence.provenance.outbox import ProvenanceOutbox  # noqa: E402
from antelligence.worlds.tumor_mri import WorldConfig  # noqa: E402


def _git_sha() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    sha = out.stdout.strip()
    return sha if len(sha) == 40 else "0" * 40


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="ARM:CASE:SEED")
    ap.add_argument("--config", required=True, help="live backends JSON (v41_live)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", help="append one cell per run to this JSONL file")
    ap.add_argument("--loader", default=v41_live.DEFAULT_LOADER, help="module:function, load_case(case_id) -> (Crop, regions)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--anchors", choices=("none", "mock"), default="none")
    ap.add_argument("--world-yaml", default=str(REPO / "config/v4_1/world.yaml"))
    ap.add_argument("--promoted-harness", help="harness.yaml with the promoted edits (default: the base harness)")
    ap.add_argument("--router", help="frozen kNN router JSON")
    ap.add_argument("--thresholds", help="frozen conformal thresholds JSON")
    ap.add_argument("--bonsai-router")
    ap.add_argument("--bonsai-thresholds")
    ap.add_argument("--look-temperature", type=float, default=1.0)
    ap.add_argument("--ledger-dir", help="a ledger/ directory for run keys (needed to replay later)")
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--prereg", default=str(REPO / "docs/research/v4.1-20261013/PREREGISTRATION.md"))
    ap.add_argument("--anchor-receipt", default=str(REPO / "docs/research/v4.1-20261013/chain/prereg-receipt.json"))
    ap.add_argument("--g1", default=str(REPO / "docs/research/v4.1-20261013/G1-approval.md"))
    return ap


def build_harness(args: argparse.Namespace, get_env: Callable[[str], Optional[str]]) -> V41Harness:
    cfg = v41_live.load_config(args.config)
    runs = [r.split(":") for r in args.run]
    arms = {r[0] for r in runs}
    if not arms <= set(ARM_SPECS):
        raise SystemExit(f"unknown arm(s): {sorted(arms - set(ARM_SPECS))}")
    loader = v41_live.resolve_loader(args.loader)
    workers = v41_live.build_workers(cfg, get_env)
    needs_queen = any(ARM_SPECS[a].uses_queen for a in arms)
    anchorer_factory = None
    if args.anchors == "mock":
        from antelligence.anchoring.tamper import Lab
        lab = Lab()
        anchorer_factory = lambda board, akey, out_dir: lab.anchorer(board, akey, out_dir)  # noqa: E731
    harness_doc = load_harness(args.promoted_harness) if args.promoted_harness else None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return V41Harness(
        load_case=loader, gate=make_gate(GateConfig(Path(args.repo_root), Path(args.prereg), Path(args.anchor_receipt),
                                                    Path(args.g1))),
        workers=workers, queen_provider=v41_live.build_queen_provider(cfg, get_env) if needs_queen else None,
        world_config=WorldConfig.from_yaml(args.world_yaml), promoted_harness=harness_doc,
        router=KNNRouter.load(args.router) if args.router else None,
        bonsai_router=KNNRouter.load(args.bonsai_router) if args.bonsai_router else None,
        thresholds=ConformalThresholds.load(args.thresholds) if args.thresholds else None,
        bonsai_thresholds=ConformalThresholds.load(args.bonsai_thresholds) if args.bonsai_thresholds else None,
        look_temperature=args.look_temperature, anchorer_factory=anchorer_factory,
        allow_unanchored=args.anchors == "none", outbox=ProvenanceOutbox(str(out / "outbox.sqlite")),
        keys=KeyVault(args.ledger_dir), out_dir=out, code_sha=_git_sha(), hardware=cfg.get("hardware"))


def main(argv: Optional[Sequence[str]] = None, *, get_env: Callable[[str], Optional[str]] = os.environ.get,
         harness: Optional[V41Harness] = None) -> int:
    args = parser().parse_args(argv)
    runs = []
    for spec in args.run:
        arm, case, seed = spec.rsplit(":", 2)
        runs.append((arm, case, int(seed)))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    h = harness or build_harness(args, get_env)
    results = asyncio.run(h.run_many(runs, concurrency=args.concurrency, calibration=args.calibration))
    failed = 0
    for (arm, case, seed), res in zip(runs, results):
        if isinstance(res, BaseException):
            failed += 1
            print(f"FAILED {arm} {case} {seed}: {type(res).__name__}: {res}", file=sys.stderr)
            cell: Any = error_cell(arm, case, seed, "dev", res)
        else:
            cell = res.cell
            print(f"{arm} {case} seed {seed}: coverage {cell['coverage']:.3f} off-target {cell['off_target']:.3f} "
                  f"board {cell['board_size']} anchors {cell['anchors']} -> {res.run_dir}")
        if args.cells:
            append_cell(args.cells, cell)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
