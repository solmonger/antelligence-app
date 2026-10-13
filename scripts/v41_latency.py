#!/usr/bin/env python
"""v4.1 latency benchmark (goal 4.7): 200 development decisions per backend at concurrency 1 and 8, board mode A against
mode B, and anchor latency per outside system. Writes docs/research/v4.1-20261013/latency/results.json.

  .venv/bin/python scripts/v41_latency.py --config run.json --n-cases 5 --out latency/results.json --anchors mock

Development cases only (the first ``--n-cases`` of dev_a in the committed split). The states are scripted S episodes
rendered as typed requests; the only model traffic is the benchmark itself, to the endpoints in ``--config``. Results
from this script are the headline latency numbers; latency logged in evaluation is "under load" and secondary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v41_latency, v41_live  # noqa: E402
from antelligence.experiments.v41_split import load_split  # noqa: E402
from antelligence.worlds.tumor_mri import WorldConfig  # noqa: E402

RESULTS = REPO / "docs/research/v4.1-20261013/latency/results.json"


def main(argv: Optional[Sequence[str]] = None, *, get_env: Callable[[str], Optional[str]] = os.environ.get,
         workers: Optional[Mapping] = None, loader: Optional[Callable] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="live backends JSON (v41_live); not needed when called with workers=")
    ap.add_argument("--loader", default=v41_live.DEFAULT_LOADER)
    ap.add_argument("--split", default=str(REPO / "config/v4_1/split.json"))
    ap.add_argument("--n-cases", type=int, default=5)
    ap.add_argument("--n", type=int, default=v41_latency.N_DECISIONS)
    ap.add_argument("--world-yaml", default=str(REPO / "config/v4_1/world.yaml"))
    ap.add_argument("--anchors", choices=("none", "mock"), default="none")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    dev = load_split(Path(args.split))["dev_a"][: args.n_cases]
    load_case = loader or v41_live.resolve_loader(args.loader)
    if workers is None:
        if not args.config:
            ap.error("--config is required")
        workers = v41_live.build_workers(v41_live.load_config(args.config), get_env)
    cfg = WorldConfig.from_yaml(args.world_yaml)
    requests = {name: v41_latency.harvest_requests(load_case, dev, w.model, args.n, cfg) for name, w in workers.items()}
    result = {"study": "v4.1", "kind": "latency", "split": "dev", "cases": dev, "n_decisions": args.n,
              "backends": asyncio.run(v41_latency.bench_all({n: w.backend for n, w in workers.items()}, requests)),
              "board_modes": v41_latency.bench_board_modes()}
    if args.anchors == "mock":
        from antelligence.anchoring.tamper import Lab
        lab = Lab()
        result["anchors"] = {"note": "mock transports: plumbing only, not network latency",
                             **v41_latency.bench_anchors(lambda b, k, d: lab.anchorer(b, k, d))}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
