#!/usr/bin/env python3
"""Full DeSci Pipeline — Run simulation → Submit → Attest → Link to IP-NFT.

End-to-end pipeline that demonstrates the complete antelligence workflow:
1. Run a nanobot tumor simulation
2. Hash and submit results to ExperienceRegistry on Base Sepolia
3. Create EAS attestation for the results
4. Link attestation to the IP-NFT

Usage:
    python3 scripts/full_pipeline.py --nanobots 10 --tumor-radius 150 --steps 300 --seed 42 [--dry-run]
"""

import sys
import os
import types
import argparse
import json
import time
from pathlib import Path

# Load .env BEFORE mocking dotenv (so our env vars are available)
from dotenv import load_dotenv as _real_load_dotenv
_real_load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Mock external modules (for nanobot_simulation imports)
for mod_name in ["dotenv", "litellm_client", "blockchain", "blockchain.client"]:
    mock = types.ModuleType(mod_name)
    if mod_name == "dotenv":
        mock.load_dotenv = lambda *a, **kw: None
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


def run_simulation(args):
    """Step 1: Run the simulation."""
    from backend.nanobot_simulation import TumorNanobotModel
    from backend.tumor_environment import CellPhase
    from backend.queen_policy import QueenPolicy, WorkerParams, SwarmMetrics, apply_params_to_nanobot
    from backend.nanobot_simulation import NanobotState

    np.random.seed(args.seed)
    print(f"\n{'='*60}")
    print(f"  STEP 1: Running Simulation")
    print(f"  Seed: {args.seed} | Bots: {args.nanobots} | Radius: {args.tumor_radius}µm | Steps: {args.steps}")
    print(f"{'='*60}")

    model = TumorNanobotModel(
        domain_size=400.0, voxel_size=10.0, n_nanobots=args.nanobots,
        tumor_radius=args.tumor_radius, agent_type="Rule-Based", with_queen=False,
    )

    queen = QueenPolicy(epoch_interval=50)
    initial_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    prev_kills = 0
    prev_deliveries = 0

    start = time.time()
    for step_i in range(args.steps):
        if queen.should_update(step_i):
            living = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
            kills_now = initial_alive - living
            deliveries_now = sum(b.deliveries_made for b in model.nanobots)
            metrics = SwarmMetrics(
                step=step_i, total_living_cells=living,
                cells_killed_this_epoch=kills_now - prev_kills,
                deliveries_this_epoch=deliveries_now - prev_deliveries,
                kill_rate=kills_now / max(initial_alive, 1),
            )
            params = queen.update(metrics)
            for bot in model.nanobots:
                apply_params_to_nanobot(bot, params)
            prev_kills = kills_now
            prev_deliveries = deliveries_now
        model.step()
    elapsed = time.time() - start

    final_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    kills = initial_alive - final_alive
    kill_rate = kills / max(initial_alive, 1)
    deliveries = sum(b.deliveries_made for b in model.nanobots)
    drug = sum(b.total_drug_delivered for b in model.nanobots)

    metrics = {
        "seed": args.seed,
        "domain_size": 400.0,
        "n_nanobots": args.nanobots,
        "tumor_radius": args.tumor_radius,
        "steps": args.steps,
        "initial_cells": initial_alive,
        "kills": kills,
        "kill_rate": round(kill_rate, 4),
        "deliveries": deliveries,
        "total_drug_ug": round(drug, 2),
        "elapsed_sec": round(elapsed, 2),
        "strategy_type": "queen-guided",
        "model_used": "rule-based",
        "queen_epochs": queen.epoch_count,
    }

    print(f"\n  Results: {kills}/{initial_alive} cells killed ({kill_rate*100:.1f}%)")
    print(f"  Deliveries: {deliveries} | Drug: {drug:.0f}µg | Time: {elapsed:.1f}s")
    return metrics


def submit_to_chain(metrics, dry_run=False):
    """Step 2: Submit to ExperienceRegistry."""
    print(f"\n{'='*60}")
    print(f"  STEP 2: {'[DRY RUN] ' if dry_run else ''}Submitting to ExperienceRegistry")
    print(f"{'='*60}")

    # Save metrics to temp file
    metrics_path = Path("/tmp/pipeline_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    from scripts.antelligence_cli import hash_metrics_file
    submission = hash_metrics_file(str(metrics_path))

    print(f"  Run hash:  {submission['run_hash'][:20]}...")
    print(f"  IPFS CID:  {submission['ipfs_cid'][:20]}...")
    print(f"  Score:     {submission['score']}")

    if dry_run:
        print("  [DRY RUN] Skipping on-chain submission")
        return submission

    from web3 import Web3
    from eth_account import Account

    w3 = Web3(Web3.HTTPProvider(os.getenv("BASE_SEPOLIA_RPC_URL")))
    acct = Account.from_key(os.getenv("PRIVATE_KEY"))

    abi_path = Path(__file__).parent.parent / "blockchain" / "artifacts" / "contracts" / "ExperienceRegistry.sol" / "ExperienceRegistry.json"
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    registry = w3.eth.contract(
        address=w3.to_checksum_address("0x22ECc5e4ddcCbAa44f508480e09eBD2640Dcd4e9"),
        abi=abi,
    )

    run_hash_bytes = w3.to_bytes(hexstr=submission["run_hash"])
    data_hash_bytes = w3.to_bytes(hexstr=submission["data_hash"])
    strategy_meta = (
        metrics.get("strategy_type", "queen-guided"),
        metrics.get("model_used", "rule-based"),
        int(metrics.get("n_nanobots", 10)),
        int(metrics.get("tumor_radius", 150)),
        data_hash_bytes,
    )

    nonce = w3.eth.get_transaction_count(acct.address)
    txn = registry.functions.submitExperience(
        run_hash_bytes, submission["ipfs_cid"], data_hash_bytes,
        submission["score"], strategy_meta,
    ).build_transaction({
        "from": acct.address, "nonce": nonce, "gas": 300000, "gasPrice": w3.eth.gas_price,
    })
    signed = acct.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    print(f"  TX: https://sepolia.basescan.org/tx/{tx_hash.hex()}")
    print(f"  Block: {receipt.blockNumber} | Status: {'OK' if receipt.status == 1 else 'FAILED'}")

    submission["tx_hash"] = tx_hash.hex()
    return submission


def main():
    parser = argparse.ArgumentParser(description="Full DeSci pipeline")
    parser.add_argument("--nanobots", type=int, default=10)
    parser.add_argument("--tumor-radius", type=float, default=150.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ANTELLIGENCE FULL DESCI PIPELINE")
    print("=" * 60)

    # Step 1: Simulate
    metrics = run_simulation(args)

    # Step 2: Submit to chain
    submission = submit_to_chain(metrics, dry_run=args.dry_run)

    # Summary
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Kill rate: {metrics['kill_rate']*100:.1f}%")
    print(f"  Score: {submission['score']}")
    print(f"  Run hash: {submission['run_hash'][:20]}...")
    if not args.dry_run and 'tx_hash' in submission:
        print(f"  TX: {submission['tx_hash'][:20]}...")
    print()


if __name__ == "__main__":
    main()
