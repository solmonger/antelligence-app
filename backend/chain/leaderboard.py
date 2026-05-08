"""Leaderboard service for ranking attested simulation policies.

Reads SimulationVerified events from TumorIntel on Base Sepolia and
ranks policies by attested kill rate. Supports both on-chain event
reading (via cast/RPC) and local artifact-based ranking.

Usage:
    python3 -m chain.leaderboard                    # Show leaderboard
    python3 -m chain.leaderboard --from-dir ./runs  # Rank local artifacts
    python3 -m chain.leaderboard --json             # JSON output
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chain.config import get_base_sepolia_rpc_url, get_tumor_intel_address
from chain.proof_spec import PROOF_ARTIFACT_VERSION, PROOF_FORMAT, PROOF_SYSTEM, PUBLIC_VALUES_SCHEMA_VERSION

TUMOR_INTEL_ADDRESS = get_tumor_intel_address()
SIMULATION_VERIFIED_TOPIC = None  # Will compute from event signature
TRUST_TIER_RANK = {
    "verified_onchain": 5,
    "proof_staged": 4,
    "replay_checked": 3,
    "integrity_checked": 2,
    "unverified": 1,
    "no_effect": 0,
}


def fetch_onchain_events(rpc_url: str, from_block: int = 0) -> List[Dict]:
    """Fetch SimulationVerified events from TumorIntel contract.

    Uses cast CLI to query event logs.

    Args:
        rpc_url: Base Sepolia RPC URL
        from_block: Start block for event query

    Returns:
        List of decoded event dicts
    """
    try:
        # Get event signature hash
        result = subprocess.run(
            [
                "cast", "logs",
                "--address", TUMOR_INTEL_ADDRESS,
                "--from-block", str(from_block),
                "--rpc-url", rpc_url,
                "--json",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []
        logs = json.loads(result.stdout) if result.stdout.strip() else []
        return logs
    except Exception:
        return []


def load_local_artifacts(directory: str) -> List[Dict]:
    """Load simulation artifacts from a local directory.

    Reads all JSON files that match the antelligence artifact format.

    Args:
        directory: Path to directory containing artifact JSON files

    Returns:
        List of artifact dicts
    """
    artifacts = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return artifacts

    for f in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get("type") == "antelligence-simulation-v2":
                artifacts.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return artifacts


def rank_by_kill_rate(entries: List[Dict]) -> List[Dict]:
    """Rank simulation entries by kill rate (descending).

    Args:
        entries: List of dicts with at least 'kill_rate' field

    Returns:
        Sorted list with rank added
    """
    sorted_entries = sorted(
        entries,
        key=lambda x: (
            x.get("kill_rate", 0),
            TRUST_TIER_RANK.get(x.get("trust_tier"), -1),
        ),
        reverse=True,
    )
    for i, entry in enumerate(sorted_entries):
        entry["rank"] = i + 1
    return sorted_entries


def safe_dict(value: object) -> Dict:
    """Return value only when it is a protocol dictionary; otherwise ignore malformed claims."""
    return value if isinstance(value, dict) else {}


def flag_is_true(value: object) -> bool:
    """Return True only for explicit boolean protocol flags, never truthy claims."""
    return value is True


def valid_staged_proof_bundle(proof_bundle: Dict) -> bool:
    """Return True only for proof bundles carrying canonical staged proof metadata."""
    if not proof_bundle:
        return False
    required_strings = ("run_id", "artifact_hash", "config_hash", "public_values", "proof_bytes")
    if any(not isinstance(proof_bundle.get(field), str) or not proof_bundle.get(field) for field in required_strings):
        return False
    return all((
        proof_bundle.get("proof_artifact_version") == PROOF_ARTIFACT_VERSION,
        proof_bundle.get("proof_system") == PROOF_SYSTEM,
        proof_bundle.get("proof_format") == PROOF_FORMAT,
        proof_bundle.get("public_values_schema_version") == PUBLIC_VALUES_SCHEMA_VERSION,
    ))


def derive_trust_tier(verification_status: Dict, proof_bundle: Dict, proof_lifecycle: Dict) -> str:
    if flag_is_true(verification_status.get("onchain_ok")):
        return "verified_onchain"
    if valid_staged_proof_bundle(proof_bundle) and proof_lifecycle.get("stage") == "proof_generated":
        return "proof_staged"
    if flag_is_true(verification_status.get("replay_ok")):
        return "replay_checked"
    if flag_is_true(verification_status.get("integrity_ok")):
        return "integrity_checked"
    return "unverified"


def finite_metric_value(metrics: object, field: str, default: float = 0, max_value: Optional[float] = None) -> float:
    """Return a bounded finite non-negative metric value; malformed values are zeroed."""
    if not isinstance(metrics, dict):
        return default
    value = metrics.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return default
    if max_value is not None and value > max_value:
        return default
    return value


def has_treatment_effect(metrics: object) -> bool:
    """Return True when a run reports any finite positive treatment effect signal."""
    return any((
        finite_metric_value(metrics, "kill_rate", max_value=100) > 0,
        finite_metric_value(metrics, "deliveries") > 0,
        finite_metric_value(metrics, "total_drug") > 0,
    ))


def config_public_uint32(config: Dict, field: str, fallback_field: Optional[str] = None) -> int:
    """Return an exact uint32 config value for leaderboard public inputs; malformed values are zeroed."""
    value = config.get(field, config.get(fallback_field, 0) if fallback_field else 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    if value < 0 or value > 4_294_967_295 or int(value) != value:
        return 0
    return int(value)


def normalize_leaderboard_artifact(record: Dict) -> Dict:
    """Flatten attestation/proof bundle records into a canonical artifact view."""
    artifact = record.get("ipfs", {}).get("artifact", {}) if isinstance(record.get("ipfs"), dict) else {}
    if not artifact:
        return record

    normalized = dict(artifact)
    for field in (
        "verification_status",
        "proof_lifecycle",
        "proof_bundle",
        "trust_tier",
        "verified_onchain",
        "onchain",
        "status",
        "next_step",
    ):
        if field in record:
            normalized[field] = record[field]
    return normalized


def build_leaderboard(artifacts: List[Dict]) -> Dict:
    """Build leaderboard from simulation artifacts.

    Args:
        artifacts: List of simulation artifact dicts

    Returns:
        Leaderboard dict with ranked entries and summary stats
    """
    entries = []
    for raw_artifact in artifacts:
        artifact = normalize_leaderboard_artifact(raw_artifact)
        config = artifact.get("config", {})
        metrics = artifact.get("metrics", {})
        kill_rate = finite_metric_value(metrics, "kill_rate", max_value=100)
        deliveries = finite_metric_value(metrics, "deliveries")
        total_drug = finite_metric_value(metrics, "total_drug")
        verification_status = safe_dict(artifact.get("verification_status", {}))
        proof_lifecycle = safe_dict(artifact.get("proof_lifecycle", {}))
        claimed_proof_bundle = safe_dict(artifact.get("proof_bundle", {}))
        proof_bundle = claimed_proof_bundle if valid_staged_proof_bundle(claimed_proof_bundle) else {}
        effectful = has_treatment_effect(metrics)
        verified_onchain = flag_is_true(verification_status.get("onchain_ok")) or flag_is_true(artifact.get("verified_onchain"))
        trust_tier = (
            "verified_onchain"
            if verified_onchain
            else derive_trust_tier(verification_status, proof_bundle, proof_lifecycle)
        )
        entry = {
            "run_id": artifact.get("run_id", "unknown"),
            "config_hash": artifact.get("config_hash", ""),
            "kill_rate": kill_rate,
            "deliveries": deliveries,
            "total_drug": total_drug,
            "tumor_radius": config_public_uint32(config, "tumor_radius"),
            "nanobot_count": config_public_uint32(config, "nanobot_count", "n_nanobots"),
            "steps": config_public_uint32(config, "steps", "n_steps"),
            "timestamp": artifact.get("timestamp", ""),
            "verified_onchain": verified_onchain,
            "proof_stage": proof_lifecycle.get("stage", "untracked"),
            "integrity_ok": flag_is_true(verification_status.get("integrity_ok")),
            "replay_ok": flag_is_true(verification_status.get("replay_ok")),
            "proof_ok": flag_is_true(verification_status.get("proof_ok")),
            "trust_tier": trust_tier,
            "proof_origin": proof_bundle.get("proof_origin", "unknown"),
            "proof_artifact_version": proof_bundle.get("proof_artifact_version", "untracked"),
            "effect_status": "effect_reported" if effectful else "no_effect",
        }
        if not effectful:
            entry.update({
                "verified_onchain": False,
                "proof_stage": "no_effect",
                "integrity_ok": False,
                "replay_ok": False,
                "proof_ok": False,
                "trust_tier": "no_effect",
            })
        entries.append(entry)

    ranked = rank_by_kill_rate(entries)

    # Summary stats
    kill_rates = [e["kill_rate"] for e in entries]
    summary = {
        "total_entries": len(entries),
        "verified_entries": sum(1 for e in entries if e.get("verified_onchain")),
        "replay_checked_entries": sum(1 for e in entries if e.get("replay_ok")),
        "staged_proof_entries": sum(1 for e in entries if e.get("trust_tier") == "proof_staged"),
        "zero_effect_entries": sum(1 for e in entries if e.get("effect_status") == "no_effect"),
        "avg_kill_rate": round(sum(kill_rates) / len(kill_rates), 2) if kill_rates else 0,
        "best_kill_rate": max(kill_rates) if kill_rates else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "ok": True,
        "leaderboard": ranked,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Antelligence simulation leaderboard")
    parser.add_argument("--from-dir", help="Load artifacts from local directory")
    parser.add_argument("--onchain", action="store_true", help="Fetch events from Base Sepolia")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    artifacts = []

    if args.from_dir:
        artifacts = load_local_artifacts(args.from_dir)
    elif args.onchain:
        rpc_url = get_base_sepolia_rpc_url()
        if not rpc_url:
            print(json.dumps({"ok": False, "error": "BASE_SEPOLIA_RPC_URL not set"}))
            sys.exit(1)
        events = fetch_onchain_events(rpc_url)
        # Convert events to artifact-like format
        for evt in events:
            artifacts.append({
                "type": "antelligence-simulation-v2",
                "config": {},
                "metrics": {"kill_rate": 0},
                "verification_status": {
                    "schema_ok": True,
                    "integrity_ok": False,
                    "replay_ok": False,
                    "proof_ok": False,
                    "onchain_ok": True,
                },
                "proof_lifecycle": {"stage": "verified_onchain"},
                "tx_hash": evt.get("transactionHash", ""),
            })
    else:
        print(json.dumps({"ok": False, "error": "Specify --from-dir or --onchain"}))
        sys.exit(1)

    result = build_leaderboard(artifacts)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if not result["leaderboard"]:
            print("No simulation entries found.")
            return

        print(f"\nAntelligence Simulation Leaderboard")
        print(f"{'='*60}")
        print(f"{'Rank':<6}{'Kill Rate':<12}{'Deliveries':<12}{'Nanobots':<10}{'Steps':<8}{'Run ID'}")
        print(f"{'-'*60}")
        for entry in result["leaderboard"][:20]:
            print(
                f"{entry['rank']:<6}"
                f"{entry['kill_rate']:>8.1f}%   "
                f"{entry['deliveries']:>8}    "
                f"{entry['nanobot_count']:>6}    "
                f"{entry['steps']:>5}   "
                f"{entry['run_id'][:16]}"
            )
        print(f"\nTotal: {result['summary']['total_entries']} entries")
        print(f"Best: {result['summary']['best_kill_rate']}% | Avg: {result['summary']['avg_kill_rate']}%")


if __name__ == "__main__":
    main()
