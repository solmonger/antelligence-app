#!/usr/bin/env python3
"""Batch experiment runner — YAML-configured parameter sweeps.

Runs simulation configurations in batch, collects metrics, and
optionally submits results on-chain.

Usage:
    python3 scripts/batch_runner.py --config experiments/sweep.yaml [--submit] [--output-dir results/]

Config format (YAML):
    name: "tumor-size-sweep"
    seeds: [42, 43, 44, 45, 46]
    defaults:
      steps: 300
      nanobots: 10
      domain_size: 400.0
    sweep:
      tumor_radius: [100, 150, 200]
"""

import sys
import os
import types
import argparse
import json
import time
import itertools
from datetime import datetime, timezone
from pathlib import Path

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
import yaml
from backend.nanobot_simulation import TumorNanobotModel
from backend.tumor_environment import CellPhase


def run_single(config):
    """Run one simulation configuration and return metrics."""
    np.random.seed(config["seed"])
    start = time.time()

    model = TumorNanobotModel(
        domain_size=config.get("domain_size", 400.0),
        voxel_size=config.get("voxel_size", 10.0),
        n_nanobots=config.get("nanobots", 10),
        tumor_radius=config.get("tumor_radius", 150.0),
        agent_type="Rule-Based",
        with_queen=config.get("with_queen", False),
    )

    initial_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    n_steps = config.get("steps", 300)

    for _ in range(n_steps):
        model.step()

    elapsed = time.time() - start
    final_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    kills = initial_alive - final_alive
    kill_rate = kills / initial_alive if initial_alive > 0 else 0

    apoptotic = sum(1 for c in model.geometry.tumor_cells if c.phase == CellPhase.APOPTOTIC)
    necrotic = sum(1 for c in model.geometry.tumor_cells if c.phase == CellPhase.NECROTIC)
    deliveries = sum(b.deliveries_made for b in model.nanobots)
    drug = sum(b.total_drug_delivered for b in model.nanobots)

    return {
        "seed": config["seed"],
        "domain_size": config.get("domain_size", 400.0),
        "n_nanobots": config.get("nanobots", 10),
        "tumor_radius": config.get("tumor_radius", 150.0),
        "steps": n_steps,
        "initial_cells": initial_alive,
        "final_alive": final_alive,
        "kills": kills,
        "kill_rate": round(kill_rate, 4),
        "apoptotic": apoptotic,
        "necrotic": necrotic,
        "deliveries": deliveries,
        "total_drug_ug": round(drug, 2),
        "elapsed_sec": round(elapsed, 2),
        "strategy_type": "rule-based",
        "model_used": "none",
    }


def load_config(path):
    """Load YAML experiment config."""
    with open(path) as f:
        return yaml.safe_load(f)


def expand_sweep(config):
    """Expand sweep parameters into individual run configs."""
    defaults = config.get("defaults", {})
    seeds = config.get("seeds", [42])
    sweep = config.get("sweep", {})

    if not sweep:
        return [{"seed": s, **defaults} for s in seeds]

    # Cartesian product of sweep values
    keys = list(sweep.keys())
    values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        for seed in seeds:
            cfg = dict(defaults)
            cfg["seed"] = seed
            for k, v in zip(keys, combo):
                cfg[k] = v
            configs.append(cfg)

    return configs


def generate_report(experiment_name, results, output_dir):
    """Generate markdown report with results."""
    report_path = output_dir / f"{experiment_name}-report.md"

    # Group by sweep parameter
    lines = [
        f"# Experiment: {experiment_name}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Configuration",
        f"- Runs: {len(results)}",
        f"- Seeds: {sorted(set(r['seed'] for r in results))}",
        "",
        "## Results",
        "",
        "| Seed | Tumor R | Bots | Kill% | Kills | Deliveries | Drug(µg) | Time(s) |",
        "|------|---------|------|-------|-------|------------|----------|---------|",
    ]

    for r in sorted(results, key=lambda x: (-x["kill_rate"], x["seed"])):
        lines.append(
            f"| {r['seed']} | {r['tumor_radius']:.0f} | {r['n_nanobots']} | "
            f"{r['kill_rate']*100:.1f}% | {r['kills']}/{r['initial_cells']} | "
            f"{r['deliveries']} | {r['total_drug_ug']} | {r['elapsed_sec']} |"
        )

    # Summary stats
    kill_rates = [r["kill_rate"] for r in results]
    lines.extend([
        "",
        "## Summary",
        f"- Mean kill rate: {np.mean(kill_rates)*100:.1f}%",
        f"- Std kill rate: {np.std(kill_rates)*100:.1f}%",
        f"- Min: {min(kill_rates)*100:.1f}%, Max: {max(kill_rates)*100:.1f}%",
        f"- Total runs: {len(results)}",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--submit", action="store_true", help="Submit results on-chain")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config.get("name", "unnamed")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = expand_sweep(config)
    print(f"Experiment: {experiment_name}")
    print(f"Runs: {len(runs)}")
    print()

    results = []
    for i, run_config in enumerate(runs):
        print(f"  [{i+1}/{len(runs)}] seed={run_config['seed']} tumor_r={run_config.get('tumor_radius', '?')} nanobots={run_config.get('nanobots', '?')}... ", end="", flush=True)
        result = run_single(run_config)
        results.append(result)
        print(f"kill_rate={result['kill_rate']*100:.1f}% ({result['elapsed_sec']}s)")

    # Save raw results
    results_path = output_dir / f"{experiment_name}-results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved: {results_path}")

    # Generate report
    report_path = generate_report(experiment_name, results, output_dir)
    print(f"Report saved: {report_path}")

    # Submit to chain if requested
    if args.submit:
        print("\nSubmitting to chain...")
        for result in results:
            metrics_path = output_dir / f"metrics-seed{result['seed']}-r{result['tumor_radius']:.0f}.json"
            metrics_path.write_text(json.dumps(result, indent=2))
            print(f"  Saved {metrics_path}")
        print("Run 'python3 scripts/antelligence_cli.py submit --metrics FILE' for each.")


if __name__ == "__main__":
    main()
