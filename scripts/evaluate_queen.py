#!/usr/bin/env python3
"""Evaluate Queen policy vs fixed-policy baseline.

Runs paired simulations (queen-guided vs fixed) across multiple seeds
and measures improvement in kill rate. Phase 4 success gate: ≥10%
improvement across ≥5 seeds.

Usage:
    python3 scripts/evaluate_queen.py [--seeds 5] [--steps 300] [--nanobots 10] [--epoch-interval 50]
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
from backend.nanobot_simulation import TumorNanobotModel, NanobotState
from backend.tumor_environment import CellPhase
from backend.queen_policy import QueenPolicy, WorkerParams, SwarmMetrics, apply_params_to_nanobot


def collect_metrics(model, step, prev_kills, prev_deliveries):
    """Collect current swarm metrics."""
    living = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    initial = len(model.geometry.tumor_cells)
    killed = initial - living
    hypoxic = sum(1 for c in model.geometry.get_living_cells() if c.phase == CellPhase.HYPOXIC)
    total_deliveries = sum(b.deliveries_made for b in model.nanobots)
    total_drug = sum(b.total_drug_delivered for b in model.nanobots)
    avg_payload = np.mean([b.drug_payload for b in model.nanobots])

    state_counts = {}
    for bot in model.nanobots:
        state_counts[bot.state] = state_counts.get(bot.state, 0) + 1

    return SwarmMetrics(
        step=step,
        total_living_cells=living,
        cells_killed_this_epoch=killed - prev_kills,
        deliveries_this_epoch=total_deliveries - prev_deliveries,
        drug_delivered_this_epoch=total_drug,
        avg_nanobot_payload=avg_payload,
        nanobots_searching=state_counts.get(NanobotState.SEARCHING, 0),
        nanobots_delivering=state_counts.get(NanobotState.DELIVERING, 0),
        nanobots_returning=state_counts.get(NanobotState.RETURNING, 0),
        hypoxic_cell_count=hypoxic,
        kill_rate=killed / initial if initial > 0 else 0,
    ), killed, total_deliveries


def run_with_queen(n_steps, n_nanobots, seed, epoch_interval):
    """Run simulation with queen episodic policy."""
    np.random.seed(seed)
    model = TumorNanobotModel(
        domain_size=400.0, voxel_size=10.0, n_nanobots=n_nanobots,
        tumor_radius=150.0, agent_type="Rule-Based", with_queen=False,
    )
    queen = QueenPolicy(epoch_interval=epoch_interval)
    initial_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)

    prev_kills = 0
    prev_deliveries = 0
    start = time.time()

    for step_i in range(n_steps):
        if queen.should_update(step_i):
            metrics, prev_kills, prev_deliveries = collect_metrics(
                model, step_i, prev_kills, prev_deliveries
            )
            params = queen.update(metrics)
            for bot in model.nanobots:
                apply_params_to_nanobot(bot, params)

        model.step()

    elapsed = time.time() - start
    final_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    kills = initial_alive - final_alive
    kill_rate = kills / initial_alive if initial_alive > 0 else 0
    deliveries = sum(b.deliveries_made for b in model.nanobots)

    return {
        "mode": "queen",
        "seed": seed,
        "kill_rate": round(kill_rate, 4),
        "kills": kills,
        "initial_cells": initial_alive,
        "deliveries": deliveries,
        "elapsed": round(elapsed, 2),
        "epochs": queen.epoch_count,
        "queen_summary": queen.get_epoch_summary(),
    }


def run_fixed_policy(n_steps, n_nanobots, seed):
    """Run simulation with fixed default parameters (no queen)."""
    np.random.seed(seed)
    model = TumorNanobotModel(
        domain_size=400.0, voxel_size=10.0, n_nanobots=n_nanobots,
        tumor_radius=150.0, agent_type="Rule-Based", with_queen=False,
    )
    initial_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    start = time.time()

    for _ in range(n_steps):
        model.step()

    elapsed = time.time() - start
    final_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    kills = initial_alive - final_alive
    kill_rate = kills / initial_alive if initial_alive > 0 else 0
    deliveries = sum(b.deliveries_made for b in model.nanobots)

    return {
        "mode": "fixed",
        "seed": seed,
        "kill_rate": round(kill_rate, 4),
        "kills": kills,
        "initial_cells": initial_alive,
        "deliveries": deliveries,
        "elapsed": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Queen policy evaluation harness")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--nanobots", type=int, default=10)
    parser.add_argument("--epoch-interval", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        fixed = run_fixed_policy(args.steps, args.nanobots, seed)
        queen = run_with_queen(args.steps, args.nanobots, seed, args.epoch_interval)

        results.append({"seed": seed, "fixed": fixed, "queen": queen})

        delta = queen["kill_rate"] - fixed["kill_rate"]
        pct = (delta / max(fixed["kill_rate"], 0.001)) * 100
        print(f"  Fixed:  {fixed['kill_rate']*100:.1f}% kill rate, {fixed['deliveries']} deliveries")
        print(f"  Queen:  {queen['kill_rate']*100:.1f}% kill rate, {queen['deliveries']} deliveries ({queen['epochs']} epochs)")
        print(f"  Delta:  {delta*100:+.1f}pp ({pct:+.1f}%)")

    # Summary
    fixed_rates = [r["fixed"]["kill_rate"] for r in results]
    queen_rates = [r["queen"]["kill_rate"] for r in results]
    avg_fixed = np.mean(fixed_rates)
    avg_queen = np.mean(queen_rates)
    improvement = ((avg_queen - avg_fixed) / max(avg_fixed, 0.001)) * 100

    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY ({args.seeds} seeds, {args.steps} steps)")
    print(f"{'='*60}")
    print(f"  Avg fixed kill rate:  {avg_fixed*100:.1f}%")
    print(f"  Avg queen kill rate:  {avg_queen*100:.1f}%")
    print(f"  Improvement:          {improvement:+.1f}%")
    print(f"  SUCCESS GATE (≥10%):  {'PASS' if improvement >= 10 else 'FAIL'}")

    # Per-seed breakdown
    print(f"\n  Per-seed results:")
    for r in results:
        d = r["queen"]["kill_rate"] - r["fixed"]["kill_rate"]
        print(f"    Seed {r['seed']}: fixed={r['fixed']['kill_rate']*100:.1f}%, queen={r['queen']['kill_rate']*100:.1f}%, delta={d*100:+.1f}pp")

    if args.json:
        print(json.dumps({
            "config": {
                "seeds": args.seeds,
                "steps": args.steps,
                "nanobots": args.nanobots,
                "epoch_interval": args.epoch_interval,
            },
            "results": results,
            "summary": {
                "avg_fixed_kill_rate": round(avg_fixed, 4),
                "avg_queen_kill_rate": round(avg_queen, 4),
                "improvement_pct": round(improvement, 1),
                "success_gate": improvement >= 10,
            }
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
