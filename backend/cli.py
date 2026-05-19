"""cli.py — unified command-line interface for Antelligence simulation tools.

Subcommands
-----------
simulate    Run a single simulation and write metrics to JSON.
benchmark   Run multiple simulations, aggregate stats, write report to JSON.
leaderboard Fetch and display the on-chain leaderboard.

Usage
-----
    antelligence simulate --steps 100 --bots 10 --output results.json
    antelligence benchmark --runs 5 --output benchmark.json
    antelligence leaderboard --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure backend package is importable when invoked as a script.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Import at module level so tests can patch backend.cli.TumorNanobotModel
try:
    from nanobot_simulation import TumorNanobotModel
except ImportError:
    try:
        from backend.nanobot_simulation import TumorNanobotModel  # type: ignore[no-redef]
    except ImportError:
        TumorNanobotModel = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run a single simulation."""
    import random
    import numpy as np

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    voxel_size = 10.0
    domain_size = float(args.grid_size) * voxel_size
    tumor_radius = min(domain_size * 0.33, 200.0)

    print(f"[simulate] bots={args.bots}, steps={args.steps}, grid={args.grid_size}×{args.grid_size}", file=sys.stderr)
    model = TumorNanobotModel(
        n_nanobots=args.bots,
        domain_size=domain_size,
        voxel_size=voxel_size,
        tumor_radius=tumor_radius,
    )

    for step in range(args.steps):
        model.step()
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(f"  step {step + 1}/{args.steps}", file=sys.stderr)

    metrics = dict(model.metrics)
    stats = model.geometry.get_tumor_statistics()
    total = max(1, stats.get("total_cells", 1))
    living = stats.get("living_cells", total)
    metrics["kill_rate"] = (total - living) / total
    metrics["step_count"] = model.step_count

    result = {
        "config": {
            "num_bots": args.bots,
            "steps": args.steps,
            "grid_size": args.grid_size,
            "seed": args.seed,
        },
        "metrics": metrics,
    }

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[simulate] results written to {args.output}")
    else:
        print(json.dumps(result, indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run multiple simulations and aggregate statistics."""
    import random
    import numpy as np

    results = []
    steps = getattr(args, "steps", 50)
    grid_size = getattr(args, "grid_size", 30)
    bots = getattr(args, "bots", 5)
    voxel_size = 10.0
    domain_size = float(grid_size) * voxel_size
    tumor_radius = min(domain_size * 0.33, 200.0)

    print(f"[benchmark] runs={args.runs}, steps={steps}, bots={bots}", file=sys.stderr)

    for i in range(args.runs):
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        model = TumorNanobotModel(
            n_nanobots=bots,
            domain_size=domain_size,
            voxel_size=voxel_size,
            tumor_radius=tumor_radius,
        )
        for _ in range(steps):
            model.step()
        stats = model.geometry.get_tumor_statistics()
        total = max(1, stats.get("total_cells", 1))
        living = stats.get("living_cells", total)
        kill_rate = (total - living) / total
        results.append({"run": i, "seed": seed, "kill_rate": kill_rate, **model.metrics})
        print(f"  run {i + 1}/{args.runs}: kill_rate={kill_rate:.4f}", file=sys.stderr)

    kill_rates = [r["kill_rate"] for r in results]
    summary = {
        "runs": args.runs,
        "mean_kill_rate": sum(kill_rates) / len(kill_rates),
        "min_kill_rate": min(kill_rates),
        "max_kill_rate": max(kill_rates),
        "results": results,
    }

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[benchmark] results written to {args.output}")
    else:
        print(json.dumps(summary, indent=2))


def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Display the on-chain leaderboard."""
    try:
        from chain.leaderboard import get_leaderboard, format_leaderboard_table

        entries = get_leaderboard(limit=args.limit)
        print(format_leaderboard_table(entries))
    except Exception as exc:  # noqa: BLE001
        # Graceful fallback when blockchain is unavailable.
        print(f"[leaderboard] could not reach on-chain data: {exc}")
        print("[leaderboard] No entries to display (dry-run / offline mode).")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="antelligence",
        description="Antelligence nanobot simulation CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- simulate ---
    p_sim = sub.add_parser("simulate", help="Run a single simulation")
    p_sim.add_argument("--steps", type=int, default=100, help="Number of simulation steps (default: 100)")
    p_sim.add_argument("--bots", type=int, default=10, help="Number of nanobots (default: 10)")
    p_sim.add_argument("--grid-size", dest="grid_size", type=int, default=60, help="Grid size N×N (default: 60)")
    p_sim.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p_sim.add_argument("--output", type=str, default=None, help="Output JSON file (default: stdout)")
    p_sim.set_defaults(func=cmd_simulate)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Run multiple simulations and aggregate stats")
    p_bench.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    p_bench.add_argument("--steps", type=int, default=50, help="Steps per run (default: 50)")
    p_bench.add_argument("--bots", type=int, default=5, help="Nanobots per run (default: 5)")
    p_bench.add_argument("--grid-size", dest="grid_size", type=int, default=30)
    p_bench.add_argument("--output", type=str, default=None, help="Output JSON file (default: stdout)")
    p_bench.set_defaults(func=cmd_benchmark)

    # --- leaderboard ---
    p_lb = sub.add_parser("leaderboard", help="Display the on-chain leaderboard")
    p_lb.add_argument("--limit", type=int, default=10, help="Number of entries to show (default: 10)")
    p_lb.set_defaults(func=cmd_leaderboard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
