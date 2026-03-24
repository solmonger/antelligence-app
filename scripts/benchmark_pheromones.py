#!/usr/bin/env python3
"""Benchmark: nanobots with pheromones vs without pheromones.

Runs two simulations with identical parameters:
1. Pheromones ENABLED (trail, alarm, recruitment influence nanobot movement)
2. Pheromones DISABLED (nanobots use only oxygen gradient + random walk)

Compares: kill rate, drug delivery efficiency, time to first kill, coverage.

Usage:
    python3 scripts/benchmark_pheromones.py [--steps 500] [--nanobots 10] [--seeds 3]
"""

import sys
import os
import types
import argparse
import json
import time

# Mock external modules
for mod_name in ["dotenv", "litellm_client", "blockchain", "blockchain.client"]:
    mock = types.ModuleType(mod_name)
    if mod_name == "dotenv":
        mock.load_dotenv = lambda: None
    if mod_name == "litellm_client":
        mock.create_client = lambda *a, **kw: None
    if mod_name == "blockchain.client":
        mock.w3 = None
        mock.acct = None
        mock.tumor_intel_contract = None
        mock.TUMOR_INTEL_CONTRACT_ADDRESS = None
    sys.modules[mod_name] = mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
from backend.nanobot_simulation import TumorNanobotModel
from backend.tumor_environment import CellPhase


def run_simulation(n_steps, n_nanobots, seed, pheromones_enabled):
    """Run a single simulation and return metrics."""
    np.random.seed(seed)

    model = TumorNanobotModel(
        domain_size=400.0,
        voxel_size=10.0,
        n_nanobots=n_nanobots,
        tumor_radius=150.0,
        agent_type="Rule-Based",
        with_queen=False,
    )

    if not pheromones_enabled:
        # Zero out pheromone chemotaxis weights
        for bot in model.nanobots:
            bot.chemotaxis_weights["trail"] = 0.0
            bot.chemotaxis_weights["alarm"] = 0.0
            bot.chemotaxis_weights["recruitment"] = 0.0
            bot.chemotaxis_weights["chemokine_signal"] = 0.0
            bot.chemotaxis_weights["toxicity_signal"] = 0.0

    initial_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    first_kill_step = None

    start_time = time.time()
    for step_i in range(n_steps):
        model.step()

        alive_now = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
        if first_kill_step is None and alive_now < initial_alive:
            first_kill_step = step_i

    elapsed = time.time() - start_time

    # Collect metrics
    final_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    kills = initial_alive - final_alive
    kill_rate = kills / initial_alive if initial_alive > 0 else 0

    total_deliveries = sum(b.deliveries_made for b in model.nanobots)
    total_drug = sum(b.total_drug_delivered for b in model.nanobots)

    apoptotic = sum(1 for c in model.geometry.tumor_cells
                    if c.phase == CellPhase.APOPTOTIC)
    necrotic = sum(1 for c in model.geometry.tumor_cells
                   if c.phase == CellPhase.NECROTIC)

    return {
        "seed": seed,
        "pheromones": pheromones_enabled,
        "steps": n_steps,
        "initial_cells": initial_alive,
        "final_alive": final_alive,
        "kills": kills,
        "kill_rate": round(kill_rate, 4),
        "apoptotic": apoptotic,
        "necrotic": necrotic,
        "deliveries": total_deliveries,
        "total_drug_ug": round(total_drug, 2),
        "first_kill_step": first_kill_step,
        "elapsed_sec": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Pheromone benchmark")
    parser.add_argument("--steps", type=int, default=300, help="Simulation steps")
    parser.add_argument("--nanobots", type=int, default=10, help="Number of nanobots")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    results = []
    seeds = list(range(42, 42 + args.seeds))

    for seed in seeds:
        for pheromones in [True, False]:
            label = "WITH pheromones" if pheromones else "WITHOUT pheromones"
            print(f"\n{'='*60}")
            print(f"  Seed {seed} | {label} | {args.steps} steps | {args.nanobots} bots")
            print(f"{'='*60}")

            result = run_simulation(args.steps, args.nanobots, seed, pheromones)
            results.append(result)

            print(f"  Kills: {result['kills']}/{result['initial_cells']} "
                  f"({result['kill_rate']*100:.1f}%)")
            print(f"  Deliveries: {result['deliveries']} | "
                  f"Drug: {result['total_drug_ug']} µg")
            print(f"  First kill at step: {result['first_kill_step']}")
            print(f"  Time: {result['elapsed_sec']}s")

    # Summary
    with_ph = [r for r in results if r["pheromones"]]
    without_ph = [r for r in results if not r["pheromones"]]

    avg_kill_with = np.mean([r["kill_rate"] for r in with_ph])
    avg_kill_without = np.mean([r["kill_rate"] for r in without_ph])
    avg_del_with = np.mean([r["deliveries"] for r in with_ph])
    avg_del_without = np.mean([r["deliveries"] for r in without_ph])

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({args.seeds} seeds, {args.steps} steps, {args.nanobots} bots)")
    print(f"{'='*60}")
    print(f"  Avg kill rate WITH pheromones:    {avg_kill_with*100:.1f}%")
    print(f"  Avg kill rate WITHOUT pheromones: {avg_kill_without*100:.1f}%")
    improvement = ((avg_kill_with - avg_kill_without) / max(avg_kill_without, 0.001)) * 100
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Avg deliveries WITH:    {avg_del_with:.0f}")
    print(f"  Avg deliveries WITHOUT: {avg_del_without:.0f}")

    if args.json:
        print(json.dumps({
            "results": results,
            "summary": {
                "avg_kill_rate_with": round(avg_kill_with, 4),
                "avg_kill_rate_without": round(avg_kill_without, 4),
                "improvement_pct": round(improvement, 1),
                "avg_deliveries_with": round(avg_del_with, 1),
                "avg_deliveries_without": round(avg_del_without, 1),
            }
        }, indent=2))


if __name__ == "__main__":
    main()
