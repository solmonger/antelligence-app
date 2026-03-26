#!/usr/bin/env python3
"""Evaluate Queen-guided vs fixed-policy nanobot simulation.

Runs the simulation under both conditions across multiple seeds and
patients (tumor configurations), comparing kill rates.

Success gate: queen policy must improve kill rate ≥10% vs fixed baseline
across ≥5 seeds and ≥3 patient configurations.

Usage:
    python3 scripts/evaluate_queen.py [--seeds 5] [--patients 3] [--steps 50]
    python3 scripts/evaluate_queen.py --json
"""

import argparse
import json
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from nanobot_simulation import TumorNanobotModel, QueenNanobot
from biofvm import create_trail_pheromone, create_alarm_pheromone, create_recruitment_pheromone


# Patient configurations (tumor params)
PATIENT_CONFIGS = [
    {"tumor_radius": 80.0, "label": "small_tumor"},
    {"tumor_radius": 150.0, "label": "medium_tumor"},
    {"tumor_radius": 200.0, "label": "large_tumor"},
]


def run_trial(n_steps, n_bots, tumor_radius, seed, with_queen, episode_length=10):
    """Run a single simulation trial.

    Args:
        n_steps: Simulation steps
        n_bots: Number of nanobots
        tumor_radius: Tumor radius
        seed: Random seed
        with_queen: Whether to enable Queen episodic planner
        episode_length: Steps between Queen parameter adjustments

    Returns:
        Result dict
    """
    np.random.seed(seed)
    import random
    random.seed(seed)

    model = TumorNanobotModel(
        domain_size=max(400.0, tumor_radius * 3),
        voxel_size=20.0,
        n_nanobots=n_bots,
        tumor_radius=tumor_radius,
        agent_type="heuristic",
        with_queen=with_queen,
        use_llm_queen=False,  # Heuristic queen for reproducible benchmarks
    )

    # Add pheromone fields
    create_trail_pheromone(model.microenv)
    create_alarm_pheromone(model.microenv)
    create_recruitment_pheromone(model.microenv)

    queen = None
    if with_queen:
        queen = QueenNanobot(model=model, use_llm=False, episode_length=episode_length)

    total_cells = len(model.geometry.tumor_cells)
    start = time.time()

    for step in range(n_steps):
        model.step()
        if queen:
            queen.step()

    runtime = time.time() - start
    living = len(model.geometry.get_living_cells())
    kills = total_cells - living
    kill_rate = (kills / total_cells * 100) if total_cells > 0 else 0
    deliveries = sum(bot.deliveries_made for bot in model.nanobots)

    result = {
        "kill_rate": round(kill_rate, 2),
        "kills": kills,
        "total_cells": total_cells,
        "deliveries": deliveries,
        "runtime_s": round(runtime, 2),
    }

    if queen:
        result["queen_episodes"] = queen.episode_counter
        result["final_params"] = queen.worker_params.copy()

    return result


def run_evaluation(n_seeds, patients, n_steps, n_bots, episode_length, verbose=True):
    """Run full evaluation across seeds and patients.

    Returns:
        Evaluation result dict with per-patient and aggregate stats
    """
    results = {"fixed": {}, "queen": {}}

    for patient in patients:
        label = patient["label"]
        radius = patient["tumor_radius"]
        results["fixed"][label] = []
        results["queen"][label] = []

        for seed in range(n_seeds):
            for policy, with_queen in [("fixed", False), ("queen", True)]:
                if verbose:
                    print(f"  {label} seed={seed} policy={policy}...", end=" ", flush=True)
                trial = run_trial(
                    n_steps=n_steps,
                    n_bots=n_bots,
                    tumor_radius=radius,
                    seed=seed,
                    with_queen=with_queen,
                    episode_length=episode_length,
                )
                trial["seed"] = seed
                results[policy][label].append(trial)
                if verbose:
                    print(f"kill={trial['kill_rate']}%")

    # Aggregate stats
    summary = {}
    for policy in ["fixed", "queen"]:
        all_kills = []
        for label in results[policy]:
            kills = [r["kill_rate"] for r in results[policy][label]]
            all_kills.extend(kills)
            summary[f"{policy}_{label}_avg"] = round(np.mean(kills), 2) if kills else 0
        summary[f"{policy}_overall_avg"] = round(np.mean(all_kills), 2) if all_kills else 0

    improvement = summary["queen_overall_avg"] - summary["fixed_overall_avg"]
    summary["improvement_pct"] = round(improvement, 2)
    summary["success_gate_passed"] = improvement >= 10.0

    return {
        "ok": True,
        "summary": summary,
        "runs": results,
        "config": {
            "n_seeds": n_seeds,
            "n_patients": len(patients),
            "n_steps": n_steps,
            "n_bots": n_bots,
            "episode_length": episode_length,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Queen vs fixed policy")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--patients", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--bots", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    patients = PATIENT_CONFIGS[:args.patients]

    if not args.json:
        print(f"Evaluating Queen vs Fixed ({args.seeds} seeds, {len(patients)} patients, {args.steps} steps)")
        print("=" * 60)

    result = run_evaluation(
        n_seeds=args.seeds,
        patients=patients,
        n_steps=args.steps,
        n_bots=args.bots,
        episode_length=args.episode_length,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        s = result["summary"]
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Fixed policy avg:  {s['fixed_overall_avg']}%")
        print(f"Queen policy avg:  {s['queen_overall_avg']}%")
        print(f"Improvement:       {s['improvement_pct']:+.2f}%")
        print(f"Success gate (≥10%): {'PASSED' if s['success_gate_passed'] else 'NOT PASSED'}")

        for patient in patients:
            label = patient["label"]
            print(f"\n  {label}:")
            print(f"    Fixed: {s.get(f'fixed_{label}_avg', 0)}%")
            print(f"    Queen: {s.get(f'queen_{label}_avg', 0)}%")


if __name__ == "__main__":
    main()
