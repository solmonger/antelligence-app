#!/usr/bin/env python3
"""Attestation Bot — Reproducibility verification for simulation results.

Re-runs a percentage of submitted simulations, compares metrics against
the on-chain record, and attests the result on ExperienceRegistry.

Usage:
    python3 scripts/attestation_bot.py verify --metrics FILE [--tolerance 0.05]
    python3 scripts/attestation_bot.py attest --run-hash HASH --quality SCORE
    python3 scripts/attestation_bot.py spot-check --sample-pct 10

The bot:
1. Loads a submitted metrics file
2. Re-runs the simulation with the same parameters + seed
3. Compares kill_rate within tolerance
4. If reproducible, submits an attestation on-chain
"""

import sys
import os
import types
import argparse
import json
import time
from pathlib import Path

# Mock external modules
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
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


def reproduce_simulation(metrics: dict) -> dict:
    """Re-run a simulation with the same parameters and return metrics."""
    from backend.nanobot_simulation import TumorNanobotModel
    from backend.tumor_environment import CellPhase

    seed = metrics.get("seed", 42)
    np.random.seed(seed)

    model = TumorNanobotModel(
        domain_size=metrics.get("domain_size", 400.0),
        voxel_size=metrics.get("voxel_size", 10.0),
        n_nanobots=metrics.get("n_nanobots", 10),
        tumor_radius=metrics.get("tumor_radius", 150.0),
        agent_type="Rule-Based",
        with_queen=False,
    )

    initial_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    n_steps = metrics.get("steps", 300)

    for _ in range(n_steps):
        model.step()

    final_alive = sum(1 for c in model.geometry.tumor_cells if c.is_alive)
    kills = initial_alive - final_alive
    kill_rate = kills / initial_alive if initial_alive > 0 else 0
    deliveries = sum(b.deliveries_made for b in model.nanobots)
    drug = sum(b.total_drug_delivered for b in model.nanobots)

    return {
        "kill_rate": round(kill_rate, 4),
        "kills": kills,
        "initial_cells": initial_alive,
        "deliveries": deliveries,
        "total_drug_ug": round(drug, 2),
    }


def check_reproducibility(original: dict, reproduced: dict, tolerance: float = 0.05) -> dict:
    """Compare original and reproduced metrics within tolerance."""
    checks = {}

    for key in ["kill_rate", "kills", "deliveries"]:
        orig_val = original.get(key, 0)
        repr_val = reproduced.get(key, 0)

        if orig_val == 0 and repr_val == 0:
            checks[key] = {"match": True, "original": orig_val, "reproduced": repr_val, "delta": 0}
        elif orig_val == 0:
            checks[key] = {"match": False, "original": orig_val, "reproduced": repr_val, "delta": repr_val}
        else:
            delta = abs(repr_val - orig_val) / abs(orig_val)
            checks[key] = {
                "match": delta <= tolerance,
                "original": orig_val,
                "reproduced": repr_val,
                "delta": round(delta, 4),
            }

    all_match = all(c["match"] for c in checks.values())
    quality = 100 if all_match else max(0, int(100 * (1 - max(c["delta"] for c in checks.values()))))

    return {
        "reproducible": all_match,
        "quality_score": quality,
        "checks": checks,
    }


def cmd_verify(args):
    """Verify reproducibility of a metrics file."""
    path = Path(args.metrics)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    original = json.loads(path.read_text())
    print(f"Original: kill_rate={original.get('kill_rate', '?')}, seed={original.get('seed', '?')}")
    print(f"Re-running simulation...")

    start = time.time()
    reproduced = reproduce_simulation(original)
    elapsed = time.time() - start
    print(f"Reproduced in {elapsed:.1f}s: kill_rate={reproduced['kill_rate']}")

    result = check_reproducibility(original, reproduced, args.tolerance)

    print(f"\nReproducibility: {'PASS' if result['reproducible'] else 'FAIL'}")
    print(f"Quality score: {result['quality_score']}/100")
    for key, check in result["checks"].items():
        status = "MATCH" if check["match"] else "MISMATCH"
        print(f"  {key}: {check['original']} → {check['reproduced']} ({status}, delta={check['delta']})")

    if args.json:
        print(json.dumps(result, indent=2))


def cmd_attest(args):
    """Submit attestation on-chain for a verified run."""
    from web3 import Web3
    from eth_account import Account

    rpc = os.getenv("BASE_SEPOLIA_RPC_URL")
    pk = os.getenv("PRIVATE_KEY")
    if not rpc or not pk:
        print("ERROR: BASE_SEPOLIA_RPC_URL and PRIVATE_KEY required", file=sys.stderr)
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(rpc))
    acct = Account.from_key(pk)

    abi_path = Path(__file__).parent.parent / "blockchain" / "artifacts" / "contracts" / "ExperienceRegistry.sol" / "ExperienceRegistry.json"
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    registry = w3.eth.contract(
        address=w3.to_checksum_address("0x22ECc5e4ddcCbAa44f508480e09eBD2640Dcd4e9"),
        abi=abi,
    )

    run_hash_bytes = w3.to_bytes(hexstr=args.run_hash)
    quality = min(100, max(0, args.quality))

    print(f"Attesting run {args.run_hash[:20]}... quality={quality}")

    nonce = w3.eth.get_transaction_count(acct.address)
    txn = registry.functions.attestExperience(
        run_hash_bytes,
        quality,
        args.notes or "",
    ).build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "gas": 200000,
        "gasPrice": w3.eth.gas_price,
    })

    signed = acct.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"TX: https://sepolia.basescan.org/tx/{tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status == 1:
        print(f"Attestation submitted! Block: {receipt.blockNumber}")
    else:
        print("ERROR: Transaction reverted", file=sys.stderr)
        sys.exit(1)


def cmd_spot_check(args):
    """Spot-check random sample of submissions from leaderboard DB."""
    import sqlite3

    db_path = Path(__file__).parent.parent / "data" / "leaderboard.db"
    if not db_path.exists():
        print("No leaderboard DB found. Run leaderboard.py record first.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM submissions ORDER BY RANDOM() LIMIT ?",
                        (max(1, args.sample_pct),)).fetchall()
    conn.close()

    if not rows:
        print("No submissions to check.")
        return

    print(f"Spot-checking {len(rows)} submission(s)...\n")
    for row in rows:
        original = {
            "seed": row["seed"],
            "domain_size": 400.0,
            "n_nanobots": row["n_nanobots"],
            "tumor_radius": row["tumor_radius"],
            "steps": row["steps"],
            "kill_rate": row["kill_rate"],
            "deliveries": row["deliveries"] if "deliveries" in row.keys() else 0,
        }

        print(f"Run: {row['run_hash'][:20]}...")
        reproduced = reproduce_simulation(original)
        result = check_reproducibility(original, reproduced)
        status = "REPRODUCIBLE" if result["reproducible"] else "NOT REPRODUCIBLE"
        print(f"  {status} (quality={result['quality_score']}/100)")


def main():
    parser = argparse.ArgumentParser(description="Attestation bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="Verify reproducibility")
    p_verify.add_argument("--metrics", required=True)
    p_verify.add_argument("--tolerance", type=float, default=0.05)
    p_verify.add_argument("--json", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_attest = sub.add_parser("attest", help="Submit on-chain attestation")
    p_attest.add_argument("--run-hash", required=True)
    p_attest.add_argument("--quality", type=int, required=True, help="0-100")
    p_attest.add_argument("--notes", default="")
    p_attest.set_defaults(func=cmd_attest)

    p_spot = sub.add_parser("spot-check", help="Spot-check random submissions")
    p_spot.add_argument("--sample-pct", type=int, default=10)
    p_spot.set_defaults(func=cmd_spot_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
