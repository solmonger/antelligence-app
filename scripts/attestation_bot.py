#!/usr/bin/env python3
"""Attestation bot — re-runs random submissions for reproducibility spot-checks.

Picks k% of recent simulation submissions, re-runs them with the same config,
and verifies metrics are within tolerance. Reports discrepancies.

Usage:
    python3 scripts/attestation_bot.py results.json [--sample-pct 20] [--tolerance 5]
    python3 scripts/attestation_bot.py --from-dir ./results/ --sample-pct 10
"""

import argparse
import json
import os
import sys
import random
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.verify import verify_artifact_integrity, verify_metrics_tolerance
from chain.ipfs import create_simulation_artifact


def spot_check_run(original_result, steps, n_bots, with_queen, episode_length):
    """Re-run a simulation and compare metrics."""
    from nanobot_simulation import TumorNanobotModel, QueenNanobot
    from biofvm import create_trail_pheromone, create_alarm_pheromone, create_recruitment_pheromone

    seed = original_result.get("seed", 0)
    radius = original_result.get("tumor_radius", 150)

    np.random.seed(seed)
    random.seed(seed)

    domain = max(400.0, radius * 3)
    model = TumorNanobotModel(
        domain_size=domain, voxel_size=20.0, n_nanobots=n_bots,
        tumor_radius=radius, agent_type="heuristic",
        with_queen=with_queen, use_llm_queen=False,
    )
    create_trail_pheromone(model.microenv)
    create_alarm_pheromone(model.microenv)
    create_recruitment_pheromone(model.microenv)

    queen = None
    if with_queen:
        queen = QueenNanobot(model=model, use_llm=False, episode_length=episode_length)

    total_cells = len(model.geometry.tumor_cells)
    for _ in range(steps):
        model.step()
        if queen:
            queen.step()

    living = len(model.geometry.get_living_cells())
    kills = total_cells - living
    kill_rate = (kills / total_cells * 100) if total_cells > 0 else 0

    return {
        "kill_rate_pct": round(kill_rate, 2),
        "kills": kills,
        "deliveries": sum(bot.deliveries_made for bot in model.nanobots),
    }


def run_spot_checks(sweep_data, sample_pct=20, tolerance_pct=5.0, verbose=True):
    """Run spot checks on a sample of sweep results."""
    config = sweep_data.get("config", {})
    results = sweep_data.get("results", [])

    if not results:
        return {"ok": True, "checked": 0, "passed": 0, "message": "No results to check"}

    # Sample k%
    n_sample = max(1, int(len(results) * sample_pct / 100))
    sample = random.sample(results, min(n_sample, len(results)))

    checks = []
    for original in sample:
        if verbose:
            print(f"  Spot-checking seed={original.get('seed')} patient={original.get('patient')}...", end=" ", flush=True)

        recomputed = spot_check_run(
            original,
            steps=config.get("steps", 50),
            n_bots=config.get("n_bots", 5),
            with_queen=config.get("with_queen", True),
            episode_length=config.get("episode_length", 10),
        )

        tolerance_result = verify_metrics_tolerance(
            {"kill_rate": original.get("kill_rate_pct", 0), "deliveries": original.get("deliveries", 0)},
            {"kill_rate": recomputed["kill_rate_pct"], "deliveries": recomputed["deliveries"]},
            tolerance_pct=tolerance_pct,
        )

        check = {
            "seed": original.get("seed"),
            "patient": original.get("patient"),
            "original_kill_rate": original.get("kill_rate_pct"),
            "recomputed_kill_rate": recomputed["kill_rate_pct"],
            "passed": tolerance_result["ok"],
            "details": tolerance_result["checks"],
        }
        checks.append(check)

        if verbose:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"{status} (orig={check['original_kill_rate']}% recomp={check['recomputed_kill_rate']}%)")

    passed = sum(1 for c in checks if c["passed"])
    return {
        "ok": passed == len(checks),
        "checked": len(checks),
        "passed": passed,
        "failed": len(checks) - passed,
        "sample_pct": sample_pct,
        "tolerance_pct": tolerance_pct,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Attestation spot-check bot")
    parser.add_argument("input", help="JSON results from batch_runner.py")
    parser.add_argument("--sample-pct", type=float, default=20)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    text = Path(args.input).read_text()
    start = text.find("{")
    data = json.loads(text[start:])

    if not args.json:
        print(f"Attestation Bot: spot-checking {args.sample_pct}% of runs (tolerance: {args.tolerance}%)")
        print("=" * 50)

    result = run_spot_checks(data, args.sample_pct, args.tolerance, verbose=not args.json)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nResults: {result['passed']}/{result['checked']} passed")
        if result["failed"] > 0:
            print(f"ALERT: {result['failed']} checks failed!")
            for c in result["checks"]:
                if not c["passed"]:
                    print(f"  FAIL: seed={c['seed']} patient={c['patient']} "
                          f"orig={c['original_kill_rate']}% vs recomp={c['recomputed_kill_rate']}%")


if __name__ == "__main__":
    main()
