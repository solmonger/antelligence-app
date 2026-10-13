#!/usr/bin/env python
"""Run v4.1 episodes (goal 4.7): one run bundle, board export, anchors and cell record per (arm, case, seed).

  .venv/bin/python scripts/v41_run.py --config run.json --out-dir runs --cells cells.jsonl \\
      --run SW-B:BRATS_001:1 --run SW-noBoard:BRATS_001:1 --concurrency 4

``--config`` names the live backends, queen and cache (see ``antelligence/experiments/v41_live.py``); API keys are read
from the environment variables the config names, nowhere else. The gate (``v41_gate``) runs before every episode; an
evaluation case needs the committed preregistration, its confirmed anchor receipt and the G1 approval file.
``--calibration`` is Phase 0 step 7 (Qwen3.5-9B drives, d1 and Qwen3.5-4B answer counterfactually, development only).
``--anchors mock`` anchors to in-process mock OTS/Rekor/chain (plumbing and tests). ``--anchors live`` is the real thing:
OTS public calendars, Rekor (rekor.sigstore.dev), AnchorRegistry (chain 300), BoardLog for SW-D and ReputationRegistry for
SW-Byz-chain, over the persistent queue in ledger/. It WRITES TO PUBLIC NETWORKS and needs the chain signer in the
environment (chain.config.get_private_key). ``--backends live`` builds the pod, Nous and Bonsai backends (no --config):
the pod URL and secret come from ledger/v4.1-pod-state.json, the Nous key is read in-process, every paid call is ledgered
in ledger/v4.1-usage.jsonl and refused past Nous $7.50 / RunPod $9.50.

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

from antelligence.anchoring import live  # noqa: E402
from antelligence.providers.redact import scrub  # noqa: E402
from antelligence.experiments import v41_fake, v41_live, v41_prod  # noqa: E402
from antelligence.experiments.v41_conformal import ConformalThresholds  # noqa: E402
from antelligence.experiments.v41_harness import (ARM_SPECS, D1, ROUTER_MODELS_BY_ARM, GateConfig, KeyVault,  # noqa: E402
                                                  V41Harness, append_cell, error_cell, make_gate)
from antelligence.experiments.v41_router import KNNRouter  # noqa: E402
from antelligence.experiments.v41_split import load_split, which_split  # noqa: E402
from antelligence.kernel.policies.v41_harness import load_harness  # noqa: E402
from antelligence.provenance.outbox import ProvenanceOutbox  # noqa: E402
from antelligence.worlds.tumor_mri import WorldConfig  # noqa: E402


def _git_sha() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    sha = out.stdout.strip()
    return sha if len(sha) == 40 else "0" * 40


def _split_of(args: argparse.Namespace, case: str) -> str:
    """The case's real split from the committed split file ('dev' or 'eval'; reserve and unknown ids count as 'eval',
    the stricter side, since the gate refuses them anyway)."""
    try:
        split = which_split(load_split(Path(args.repo_root) / "config/v4_1/split.json"), case)
        return "dev" if split == "dev" else "eval"
    except (KeyError, OSError, ValueError):
        return "eval"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="ARM:CASE:SEED")
    ap.add_argument("--config", help="backends JSON (v41_live); not used with --backends live")
    ap.add_argument("--backends", choices=("config", "live", "fake"), default="config",
                    help="'config' builds backends from --config; 'live' from the pod state file, Nous auth and Bonsai URL; "
                         "'fake' = offline deterministic fakes (identity 'fake', dry plumbing runs, no network)")
    ap.add_argument("--pod-state", default=str(v41_prod.POD_STATE))
    ap.add_argument("--bonsai-url", default=v41_prod.DEFAULT_BONSAI_URL, help="llama.cpp server for the Bonsai arm")
    ap.add_argument("--reply-cache", default=str(v41_prod.CACHE_DIR),
                    help="episode reply cache for --backends live (worker replies and queen replies, keyed by request hash)")
    ap.add_argument("--usage-ledger", default=str(v41_prod.USAGE_LEDGER))
    ap.add_argument("--pod-costs", default=str(v41_prod.POD_COSTS))
    ap.add_argument("--chain-queue", default=str(live.DEFAULT_QUEUE))
    ap.add_argument("--anchor-archive", default=str(live.DEFAULT_ARCHIVE), help="aggregated anchors.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", help="append one cell per run to this JSONL file")
    ap.add_argument("--loader", default=v41_live.DEFAULT_LOADER, help="module:function, load_case(case_id) -> (Crop, regions)")
    ap.add_argument("--cases-root", help="L1 store (default data/v4_1/cases); the missing_flair variant reads <root>_missing_flair")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--anchors", choices=("none", "mock", "live"), default="none")
    ap.add_argument("--allow-unanchored", action="store_true",
                    help="let anchored arms (SW-B, FW, SW-D, SW-Byz-*) run without anchors; refused by default")
    ap.add_argument("--world-yaml", default=str(REPO / "config/v4_1/world.yaml"))
    ap.add_argument("--promoted-harness", help="harness.yaml with the promoted edits (default: the base harness)")
    ap.add_argument("--router", help="frozen kNN router JSON; 'none' = Phase 0 bootstrap (all to Qwen3.5-9B, no deferral); "
                                     "'fake' = seeded random kNN + mid thresholds (only with --backends fake)")
    ap.add_argument("--thresholds", help="frozen conformal thresholds JSON")
    ap.add_argument("--bonsai-router")
    ap.add_argument("--bonsai-thresholds")
    ap.add_argument("--look-temperature", type=float, default=None,
                    help="explicit override of d1's look temperature; default = frozen config/v4_1/look_temperature.json "
                         "(sha256-verified). Refused on eval cases unless equal to the frozen value")
    ap.add_argument("--ledger-dir", help="a ledger/ directory for run keys (needed to replay later; default ledger/ with --anchors live)")
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--prereg", default=str(REPO / "docs/research/v4.1-20261013/PREREGISTRATION.md"))
    ap.add_argument("--anchor-receipt", default=str(REPO / "docs/research/v4.1-20261013/chain/prereg-receipt.json"))
    ap.add_argument("--g1", default=str(REPO / "docs/research/v4.1-20261013/G1-approval.md"))
    return ap


def _routers(args: argparse.Namespace):
    """(router, thresholds, bonsai_router, bonsai_thresholds) from the flags; 'none' / 'fake' build in-process bootstraps."""
    def pick(spec: Optional[str], thr_path: Optional[str], bonsai: bool):
        if spec == "none":
            return v41_fake.null_bootstrap(bonsai)
        if spec == "fake":
            return v41_fake.fake_bootstrap(bonsai)
        return (KNNRouter.load(spec) if spec else None,
                ConformalThresholds.load(thr_path) if thr_path else None)
    router, thresholds = pick(args.router, args.thresholds, False)
    if args.bonsai_router in (None, "none", "fake") and args.router in ("none", "fake"):
        bonsai_router, bonsai_thresholds = pick(args.bonsai_router or args.router, None, True)
    else:
        bonsai_router, bonsai_thresholds = pick(args.bonsai_router, args.bonsai_thresholds, True)
    return router, thresholds, bonsai_router, bonsai_thresholds


def build_harness(args: argparse.Namespace, get_env: Callable[[str], Optional[str]]) -> V41Harness:
    live_backends = args.backends == "live"
    fake_backends = args.backends == "fake"
    if not live_backends and not fake_backends and not args.config:
        raise SystemExit("--config is required unless --backends live or fake")
    if args.router == "fake" and not fake_backends:
        raise SystemExit("--router fake is only for --backends fake")
    cfg = {} if (live_backends or fake_backends) else v41_live.load_config(args.config)
    runs = [r.split(":") for r in args.run]
    arms = {r[0] for r in runs}
    if not arms <= set(ARM_SPECS):
        raise SystemExit(f"unknown arm(s): {sorted(arms - set(ARM_SPECS))}")
    unanchored = sorted(a for a in arms if ARM_SPECS[a].anchored) if args.anchors == "none" else []
    if unanchored and not args.allow_unanchored:
        raise SystemExit(f"arms {unanchored} must be anchored: pass --anchors mock (or the live wiring), "
                         "or --allow-unanchored to run them without anchors on purpose")
    if args.loader == v41_live.DEFAULT_LOADER:
        from antelligence.worlds.tumor_mri.store_adapter import make_harness_loader
        loader = make_harness_loader(root=args.cases_root, repo_root=args.repo_root, prereg_path=args.prereg,
                                     anchor_receipt=args.anchor_receipt, g1_file=args.g1)
    else:
        loader = v41_live.resolve_loader(args.loader)
    needs_queen = any(ARM_SPECS[a].uses_queen for a in arms)
    if fake_backends:
        workers = v41_fake.build_fake_workers()
        queen = v41_fake.build_fake_queen() if needs_queen else None
    elif live_backends:
        needed = {m for a in arms if ARM_SPECS[a].kind in ("workers", "fw") for m in ROUTER_MODELS_BY_ARM[a]}
        if any(ARM_SPECS[a].kind in ("workers", "fw") for a in arms):
            needed.add(D1)  # every swarm arm, FW included, takes a d1 local look each round (goal 4.5)
        workers = v41_prod.build_live_workers(pod_state=args.pod_state, bonsai_url=args.bonsai_url, needed=needed,
                                              usage_ledger=args.usage_ledger, pod_costs=args.pod_costs,
                                              cache_dir=Path(args.reply_cache))
        queen = v41_prod.build_live_queen(usage_ledger=args.usage_ledger, pod_costs=args.pod_costs,
                                       cache_dir=Path(args.reply_cache)) if needs_queen else None
    else:
        workers = v41_live.build_workers(cfg, get_env)
        queen = v41_live.build_queen_provider(cfg, get_env) if needs_queen else None
    anchorer_factory = post_mode_factory = reputation_client = None
    if args.anchors == "live":
        args.live_anchoring = live.build_live_anchoring(queue_path=args.chain_queue, archive_path=args.anchor_archive)
        anchorer_factory = args.live_anchoring.anchorer_factory
        post_mode_factory = args.live_anchoring.post_mode_factory
        reputation_client = args.live_anchoring.reputation_client
    if args.anchors == "mock":
        from antelligence.anchoring.tamper import Lab
        lab = args.mock_lab = Lab()
        anchorer_factory = lambda board, akey, out_dir: lab.anchorer(board, akey, out_dir)  # noqa: E731
        post_mode_factory, reputation_client = v41_fake.MockPostMode, v41_fake.MockReputationClient()
    harness_doc = load_harness(args.promoted_harness) if args.promoted_harness else None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    router, thresholds, bonsai_router, bonsai_thresholds = _routers(args)
    return V41Harness(
        load_case=loader, gate=make_gate(GateConfig(Path(args.repo_root), Path(args.prereg), Path(args.anchor_receipt),
                                                    Path(args.g1))),
        workers=workers, queen_provider=queen,
        world_config=WorldConfig.from_yaml(args.world_yaml), promoted_harness=harness_doc,
        router=router, bonsai_router=bonsai_router, thresholds=thresholds, bonsai_thresholds=bonsai_thresholds,
        look_temperature=args.look_temperature, anchorer_factory=anchorer_factory,
        post_mode_factory=post_mode_factory, reputation_client=reputation_client,
        allow_unanchored=args.allow_unanchored, outbox=ProvenanceOutbox(str(out / "outbox.sqlite")),
        backends_kind=args.backends if args.backends in ("live", "fake") else "config", keys=KeyVault(args.ledger_dir), out_dir=out, code_sha=_git_sha(), hardware=cfg.get("hardware") or ({"d1": "L40S", "qwen3.5-9b": "L40S", "qwen3.5-4b": "L40S", "bonsai-8b": "Mac"} if live_backends else None)
        or ({n: "fake" for n in ("d1", "qwen3.5-9b", "qwen3.5-4b", "bonsai-8b", "fw")} if fake_backends else None))


def main(argv: Optional[Sequence[str]] = None, *, get_env: Callable[[str], Optional[str]] = os.environ.get,
         harness: Optional[V41Harness] = None) -> int:
    args = parser().parse_args(argv)
    if args.anchors == "live" and not args.ledger_dir:
        args.ledger_dir = str(REPO / "ledger")        # live runs must keep their keys: the tamper suite needs them
    args.live_anchoring = None
    args.mock_lab = None
    runs = []
    for spec in args.run:
        arm, case, seed = spec.rsplit(":", 2)
        runs.append((arm, case, int(seed)))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    h = harness or build_harness(args, get_env)
    print(f"look temperature {h.look_temperature} ({h.look_temperature_source})")
    try:
        results = asyncio.run(h.run_many(runs, concurrency=args.concurrency, calibration=args.calibration))
    finally:
        if args.live_anchoring is not None:
            print(f"chain queue at exit: {args.live_anchoring.close()}")
    if args.mock_lab is not None:
        report = v41_fake.verify_mock_exports(args.mock_lab, args.out_dir, results)
        print(f"mock verifier: {report.get('verdict')} (ok={report.get('ok')}) -> {args.out_dir}/verify-mock-report.json")
    failed = 0
    for (arm, case, seed), res in zip(runs, results):
        if isinstance(res, BaseException):
            failed += 1
            print(f"FAILED {arm} {case} {seed}: {scrub(f'{type(res).__name__}: {res}')}", file=sys.stderr)
            cell: Any = error_cell(arm, case, seed, _split_of(args, case), res,
                              backends_kind=h.backends_kind, router_kind=h.router_kind)
        else:
            cell = res.cell
            print(f"{arm} {case} seed {seed}: coverage {cell['coverage']:.3f} off-target {cell['off_target']:.3f} "
                  f"board {cell['board_size']} anchors {cell['anchors']} -> {res.run_dir}")
        if args.cells:
            append_cell(args.cells, cell)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
