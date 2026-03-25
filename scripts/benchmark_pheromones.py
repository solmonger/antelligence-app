#!/usr/bin/env python3
"""Benchmark: nanobots with pheromones vs without.

Runs the tumor simulation under both conditions and compares kill rates.
This validates that the pheromone signaling system provides measurable benefit.

Usage:
    python3 scripts/benchmark_pheromones.py [--steps 100] [--seeds 3] [--bots 5]
"""

import argparse
import json
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from nanobot_simulation import TumorNanobotModel
from biofvm import (
    create_trail_pheromone,
    create_alarm_pheromone,
    create_recruitment_pheromone,
)


def run_simulation(n_steps, n_bots, tumor_radius, seed, with_pheromones):
    """Run a simulation and return results.

    Args:
        n_steps: Number of simulation steps
        n_bots: Number of nanobots
        tumor_radius: Tumor radius in µm
        seed: Random seed for reproducibility
        with_pheromones: Whether to enable pheromone substrates

    Returns:
        Dict with kill_rate, kills, total_cells, deliveries, runtime_s
    """
    np.random.seed(seed)
    import random
    random.seed(seed)

    model = TumorNanobotModel(
        domain_size=400.0,
        voxel_size=20.0,
        n_nanobots=n_bots,
        tumor_radius=tumor_radius,
        agent_type="heuristic",
        with_queen=False,
        use_llm_queen=False,
    )

    if with_pheromones:
        create_trail_pheromone(model.microenv)
        create_alarm_pheromone(model.microenv)
        create_recruitment_pheromone(model.microenv)

    total_cells = len(model.geometry.tumor_cells)
    start = time.time()

    for step in range(n_steps):
        model.step()

    runtime = time.time() - start
    living = len(model.geometry.get_living_cells())
    kills = total_cells - living
    kill_rate = (kills / total_cells * 100) if total_cells > 0 else 0

    total_deliveries = sum(bot.deliveries_made for bot in model.nanobots)
    total_drug = sum(bot.total_drug_delivered for bot in model.nanobots)

    return {
        "kills": kills,
        "total_cells": total_cells,
        "kill_rate": round(kill_rate, 2),
        "deliveries": total_deliveries,
        "total_drug": round(total_drug, 2),
        "living_cells": living,
        "runtime_s": round(runtime, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Pheromone benchmark")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps per run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--bots", type=int, default=5, help="Number of nanobots")
    parser.add_argument("--radius", type=float, default=100.0, help="Tumor radius (µm)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    results = {"with_pheromones": [], "without_pheromones": []}

    for seed in range(args.seeds):
        for label, enabled in [("without_pheromones", False), ("with_pheromones", True)]:
            if not args.json:
                print(f"Running seed={seed}, pheromones={'ON' if enabled else 'OFF'}...", end=" ", flush=True)
            result = run_simulation(
                n_steps=args.steps,
                n_bots=args.bots,
                tumor_radius=args.radius,
                seed=seed,
                with_pheromones=enabled,
            )
            result["seed"] = seed
            results[label].append(result)
            if not args.json:
                print(f"kill_rate={result['kill_rate']}%, deliveries={result['deliveries']}, {result['runtime_s']}s")

    # Compute averages
    summary = {}
    for label in ["without_pheromones", "with_pheromones"]:
        runs = results[label]
        avg_kill = np.mean([r["kill_rate"] for r in runs])
        avg_deliveries = np.mean([r["deliveries"] for r in runs])
        avg_drug = np.mean([r["total_drug"] for r in runs])
        summary[label] = {
            "avg_kill_rate": round(avg_kill, 2),
            "avg_deliveries": round(avg_deliveries, 2),
            "avg_total_drug": round(avg_drug, 2),
            "runs": len(runs),
        }

    improvement = summary["with_pheromones"]["avg_kill_rate"] - summary["without_pheromones"]["avg_kill_rate"]
    summary["improvement_pct"] = round(improvement, 2)
    summary["pheromones_better"] = improvement > 0

    if args.json:
        print(json.dumps({"ok": True, "summary": summary, "runs": results}, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS ({args.seeds} seeds, {args.steps} steps, {args.bots} bots)")
        print(f"{'='*60}")
        print(f"Without pheromones: {summary['without_pheromones']['avg_kill_rate']}% kill rate, {summary['without_pheromones']['avg_deliveries']} deliveries")
        print(f"With pheromones:    {summary['with_pheromones']['avg_kill_rate']}% kill rate, {summary['with_pheromones']['avg_deliveries']} deliveries")
        print(f"Improvement:        {improvement:+.2f}%")
        print(f"Pheromones better:  {summary['pheromones_better']}")


if __name__ == "__main__":
    main()
