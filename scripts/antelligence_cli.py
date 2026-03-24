#!/usr/bin/env python3
"""Antelligence CLI — Submit and verify simulation runs on Base Sepolia.

Subcommands:
  submit  --metrics FILE    Hash simulation artifacts, submit to ExperienceRegistry
  verify  --run-hash HASH   Fetch CID from chain, verify data integrity
  list    [--verified]       List submitted experiences from chain

Environment:
  BASE_SEPOLIA_RPC_URL   Alchemy/Infura RPC
  PRIVATE_KEY            Deployer private key (0x-prefixed)
  EXPERIENCE_REGISTRY    Contract address (auto-detected if not set)
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for blockchain imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# ---------------------------------------------------------------------------
# Contract addresses (deployed 2026-03-24)
# ---------------------------------------------------------------------------
EXPERIENCE_REGISTRY_ADDR = os.getenv(
    "EXPERIENCE_REGISTRY",
    "0x22ECc5e4ddcCbAa44f508480e09eBD2640Dcd4e9"
)


def get_web3():
    """Initialize Web3 connection."""
    from web3 import Web3
    rpc = os.getenv("BASE_SEPOLIA_RPC_URL")
    if not rpc:
        print("ERROR: BASE_SEPOLIA_RPC_URL not set", file=sys.stderr)
        sys.exit(1)
    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        print(f"ERROR: Cannot connect to {rpc}", file=sys.stderr)
        sys.exit(1)
    return w3


def get_account(w3):
    """Load account from private key."""
    from eth_account import Account
    pk = os.getenv("PRIVATE_KEY")
    if not pk:
        print("ERROR: PRIVATE_KEY not set", file=sys.stderr)
        sys.exit(1)
    return Account.from_key(pk)


def get_registry(w3):
    """Load ExperienceRegistry contract."""
    abi_path = Path(__file__).parent.parent / "blockchain" / "artifacts" / "contracts" / "ExperienceRegistry.sol" / "ExperienceRegistry.json"
    if not abi_path.exists():
        print(f"ERROR: ABI not found at {abi_path}. Run 'npx hardhat compile' first.", file=sys.stderr)
        sys.exit(1)
    with open(abi_path) as f:
        abi = json.load(f)["abi"]
    return w3.eth.contract(
        address=w3.to_checksum_address(EXPERIENCE_REGISTRY_ADDR),
        abi=abi,
    )


# ---------------------------------------------------------------------------
# IPFS-compatible hashing (CIDv1 raw codec, sha2-256)
# ---------------------------------------------------------------------------

def compute_data_hash(data: bytes) -> str:
    """Compute keccak256 hash of data (for on-chain verification)."""
    from web3 import Web3
    return Web3.keccak(data).hex()


def compute_ipfs_cid(data: bytes) -> str:
    """Compute an IPFS-compatible CID (base32, raw codec, sha256).

    This is a local CID computation — the data is NOT pinned to IPFS.
    Use with a pinning service (Pinata, web3.storage) for actual IPFS hosting.
    """
    import hashlib
    import base64
    sha = hashlib.sha256(data).digest()
    # CIDv1: version(1) + codec(raw=0x55) + hash_fn(sha256=0x12) + length(32) + digest
    multihash = bytes([0x12, 0x20]) + sha
    cid_bytes = bytes([0x01, 0x55]) + multihash
    # Base32 lower encoding (multibase prefix 'b')
    cid = 'b' + base64.b32encode(cid_bytes).decode().lower().rstrip('=')
    return cid


def hash_metrics_file(metrics_path: str) -> dict:
    """Hash a metrics JSON file and return submission data."""
    path = Path(metrics_path)
    if not path.exists():
        print(f"ERROR: Metrics file not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw = path.read_bytes()
    metrics = json.loads(raw)

    data_hash = compute_data_hash(raw)
    cid = compute_ipfs_cid(raw)

    # Compute run hash from key parameters
    run_params = json.dumps({
        "domain_size": metrics.get("domain_size"),
        "n_nanobots": metrics.get("n_nanobots"),
        "tumor_radius": metrics.get("tumor_radius"),
        "seed": metrics.get("seed"),
        "steps": metrics.get("steps"),
    }, sort_keys=True)
    run_hash = compute_data_hash(run_params.encode())

    score = int(metrics.get("kill_rate", 0) * 10000)  # Scale to uint256

    return {
        "run_hash": run_hash,
        "ipfs_cid": cid,
        "data_hash": data_hash,
        "score": score,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_submit(args):
    """Submit simulation results to ExperienceRegistry."""
    submission = hash_metrics_file(args.metrics)

    print(f"Run hash:  {submission['run_hash']}")
    print(f"IPFS CID:  {submission['ipfs_cid']}")
    print(f"Data hash: {submission['data_hash']}")
    print(f"Score:     {submission['score']} (kill_rate × 10000)")

    if args.dry_run:
        print("\n[DRY RUN] Would submit to ExperienceRegistry. Use --submit to execute.")
        print(json.dumps(submission, indent=2, default=str))
        return

    w3 = get_web3()
    acct = get_account(w3)
    registry = get_registry(w3)

    run_hash_bytes = w3.to_bytes(hexstr=submission["run_hash"])
    data_hash_bytes = w3.to_bytes(hexstr=submission["data_hash"])

    print(f"\nSubmitting from {acct.address}...")
    nonce = w3.eth.get_transaction_count(acct.address)
    gas_price = w3.eth.gas_price

    # Build StrategyMeta tuple
    m = submission["metrics"]
    strategy_meta = (
        m.get("strategy_type", "pheromone-guided"),  # strategyType
        m.get("model_used", "rule-based"),            # modelUsed
        int(m.get("n_nanobots", 10)),                 # nanobotCount
        int(m.get("tumor_radius", 150)),              # tumorRadius
        w3.to_bytes(hexstr=submission["data_hash"]),    # datasetHash
    )

    txn = registry.functions.submitExperience(
        run_hash_bytes,
        submission["ipfs_cid"],
        data_hash_bytes,
        submission["score"],
        strategy_meta,
    ).build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "gas": 300000,
        "gasPrice": gas_price,
    })

    signed = acct.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"TX sent: https://sepolia.basescan.org/tx/{tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status == 1:
        print(f"Submitted successfully! Block: {receipt.blockNumber}")
    else:
        print("ERROR: Transaction reverted", file=sys.stderr)
        sys.exit(1)


def cmd_verify(args):
    """Verify a simulation run against on-chain data."""
    w3 = get_web3()
    registry = get_registry(w3)

    run_hash_bytes = bytes.fromhex(args.run_hash.replace("0x", ""))
    exp = registry.functions.experiences(run_hash_bytes).call()

    if exp[0] == b'\x00' * 32:
        print(f"No experience found for run hash {args.run_hash}")
        return

    print(f"Run hash:     {args.run_hash}")
    print(f"IPFS CID:     {exp[1]}")
    print(f"Data hash:    0x{exp[2].hex()}")
    print(f"Score:        {exp[3]}")
    print(f"Submitter:    {exp[4]}")
    print(f"Timestamp:    {exp[5]}")
    print(f"Attestations: {exp[6]}")
    print(f"Verified:     {exp[7]}")

    if args.metrics:
        # Re-hash local metrics and compare
        submission = hash_metrics_file(args.metrics)
        local_hash_hex = submission["data_hash"]
        if not local_hash_hex.startswith("0x"):
            local_hash_hex = "0x" + local_hash_hex
        local_hash = bytes.fromhex(local_hash_hex[2:])

        if local_hash == exp[2]:
            print("\nData integrity: MATCH")
        else:
            print(f"\nData integrity: MISMATCH")
            print(f"  On-chain:  0x{exp[2].hex()}")
            print(f"  Local:     {submission['data_hash']}")


def cmd_list(args):
    """List submissions by checking known run hashes.

    Note: Event log queries require a paid Alchemy plan on Base Sepolia.
    This command verifies known run hashes instead.
    """
    w3 = get_web3()
    registry = get_registry(w3)

    # Check if any known hashes are submitted
    # In production, this would be backed by a local DB of submitted hashes
    print("Checking on-chain registry status...")
    print(f"Registry: {EXPERIENCE_REGISTRY_ADDR}")
    print(f"Chain: Base Sepolia (84532)")
    print(f"Block: {w3.eth.block_number}")
    print()
    print("To verify a specific run, use:")
    print("  python3 scripts/antelligence_cli.py verify --run-hash 0x...")
    print()
    print("View all events on BaseScan:")
    print(f"  https://sepolia.basescan.org/address/{EXPERIENCE_REGISTRY_ADDR}#events")


def main():
    parser = argparse.ArgumentParser(
        description="Antelligence CLI — simulation provenance on Base Sepolia"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit simulation results")
    p_submit.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    p_submit.add_argument("--dry-run", action="store_true", help="Hash only, don't submit")
    p_submit.set_defaults(func=cmd_submit)

    p_verify = sub.add_parser("verify", help="Verify a run against on-chain data")
    p_verify.add_argument("--run-hash", required=True, help="Run hash (0x-prefixed)")
    p_verify.add_argument("--metrics", help="Optional local metrics file to compare")
    p_verify.set_defaults(func=cmd_verify)

    p_list = sub.add_parser("list", help="List recent submissions")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
