"""Submission CLI for posting simulation attestations on-chain.

Posts simulation results (config hash, kill rate, metrics) to the
ExperienceRegistry contract on Base Sepolia via the deployed TumorIntel.

Usage:
    python3 -m chain.submit --config config.json --metrics metrics.json
    python3 -m chain.submit --dry-run --config config.json

Requires: web3.py or cast CLI for transaction submission.
Falls back to cast CLI if web3 not installed.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chain.ipfs import pin_simulation, compute_artifact_hash


# Contract addresses on Base Sepolia
TUMOR_INTEL_ADDRESS = os.getenv("TUMOR_INTEL_ADDR", "0x1118c23879bC0319981bd12d0497E1496310f4CE")
BASE_SEPOLIA_CHAIN_ID = 84532


def submit_via_cast(
    config_hash: str,
    kill_rate: int,
    nanobot_count: int,
    tumor_radius: int,
    steps: int,
    rpc_url: str,
    private_key: str,
    dry_run: bool = True,
) -> Dict:
    """Submit simulation attestation using Foundry's cast CLI.

    Args:
        config_hash: Hex config hash (32 bytes)
        kill_rate: Kill rate scaled by 10000
        nanobot_count: Number of nanobots
        tumor_radius: Tumor radius in µm
        steps: Simulation steps
        rpc_url: Base Sepolia RPC URL
        private_key: Deployer private key
        dry_run: If True, only estimate gas

    Returns:
        Result dict with tx_hash or estimate
    """
    # For now, we can't call verifySimulation without a real ZK proof.
    # Instead, store the attestation data for when proofs are ready.
    # The isVerified() check can be called once proofs are submitted.

    if dry_run:
        # Estimate gas for the verification call
        try:
            result = subprocess.run(
                [
                    "cast", "estimate",
                    TUMOR_INTEL_ADDRESS,
                    "isVerified(bytes32)(bool)",
                    f"0x{config_hash}",
                    "--rpc-url", rpc_url,
                ],
                capture_output=True, text=True, timeout=15,
            )
            return {
                "ok": True,
                "dry_run": True,
                "contract": TUMOR_INTEL_ADDRESS,
                "config_hash": config_hash,
                "gas_estimate": result.stdout.strip(),
                "message": "Dry run complete. Submit ZK proof via verifySimulation() to attest on-chain.",
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # Live submission would require a ZK proof — placeholder for now
    return {
        "ok": False,
        "error": "Live submission requires ZK proof generation (SP1). Use --dry-run for gas estimates.",
        "contract": TUMOR_INTEL_ADDRESS,
        "config_hash": config_hash,
    }


def create_attestation_bundle(
    config: dict,
    metrics: dict,
    run_id: Optional[str] = None,
) -> Dict:
    """Create a complete attestation bundle: IPFS artifact + on-chain data.

    Args:
        config: Simulation config
        metrics: Simulation results
        run_id: Optional run ID

    Returns:
        Bundle with IPFS pin result + on-chain submission data
    """
    # Pin to IPFS
    ipfs_result = pin_simulation(config, metrics, run_id=run_id, backend="dry-run")

    # Prepare on-chain data
    config_hash = ipfs_result["config_hash"]
    kill_rate = int(metrics.get("kill_rate", 0) * 100)  # Scale to basis points
    nanobot_count = config.get("nanobot_count", config.get("n_nanobots", 0))
    tumor_radius = config.get("tumor_radius", 0)
    steps = config.get("steps", config.get("n_steps", 0))

    return {
        "ok": True,
        "ipfs": ipfs_result,
        "onchain": {
            "contract": TUMOR_INTEL_ADDRESS,
            "chain_id": BASE_SEPOLIA_CHAIN_ID,
            "config_hash": config_hash,
            "kill_rate_bps": kill_rate,
            "nanobot_count": nanobot_count,
            "tumor_radius": tumor_radius,
            "steps": steps,
        },
        "status": "ready_for_proof",
        "next_step": "Generate ZK proof with SP1, then call verifySimulation(publicValues, proofBytes)",
    }


def main():
    parser = argparse.ArgumentParser(description="Submit simulation attestation")
    parser.add_argument("--config", required=True, help="Config JSON file or inline JSON")
    parser.add_argument("--metrics", required=True, help="Metrics JSON file or inline JSON")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = json.loads(args.config)

    # Load metrics
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = json.loads(args.metrics)

    bundle = create_attestation_bundle(config, metrics)
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
