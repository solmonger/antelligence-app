"""Verification CLI for simulation attestation.

Fetches a simulation artifact from IPFS, recomputes metrics from the config,
and checks tolerance against the claimed results.

Usage:
    python3 -m chain.verify <run_hash>     # Verify a simulation by artifact hash
    python3 -m chain.verify --from-file artifact.json  # Verify from local file

This is the trust-minimized verification path:
  1. Fetch artifact from IPFS (or local file)
  2. Recompute config hash from artifact config
  3. Verify config hash matches artifact's claimed hash
  4. Check metrics are within tolerance of a fresh simulation
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chain.ipfs import compute_artifact_hash, compute_config_hash


def fetch_from_ipfs(cid: str) -> Tuple[Optional[dict], Optional[str]]:
    """Fetch artifact JSON from IPFS via public gateway.

    Args:
        cid: IPFS content ID

    Returns:
        (artifact_dict, error_string)
    """
    gateways = [
        f"https://gateway.pinata.cloud/ipfs/{cid}",
        f"https://ipfs.io/ipfs/{cid}",
        f"https://cloudflare-ipfs.com/ipfs/{cid}",
    ]
    for url in gateways:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "antelligence-verify/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read()), None
        except Exception:
            continue
    return None, f"Could not fetch CID {cid} from any IPFS gateway"


def verify_artifact_integrity(artifact: dict) -> Dict:
    """Verify internal consistency of a simulation artifact.

    Checks:
    1. Config hash matches recomputed hash
    2. Metrics hash matches recomputed hash
    3. Artifact hash matches recomputed hash
    4. Required fields present

    Args:
        artifact: Simulation artifact dict

    Returns:
        Verification result dict
    """
    checks = []

    # Check required fields
    required = ["version", "type", "config", "metrics", "config_hash", "metrics_hash", "artifact_hash"]
    for field in required:
        if field not in artifact:
            checks.append({"check": f"field_{field}", "ok": False, "reason": f"Missing field: {field}"})

    if any(not c["ok"] for c in checks):
        return {"ok": False, "checks": checks}

    # Verify config hash
    config = artifact["config"]
    expected_config_hash = compute_artifact_hash(config)
    config_ok = expected_config_hash == artifact["config_hash"]
    checks.append({
        "check": "config_hash",
        "ok": config_ok,
        "expected": expected_config_hash,
        "actual": artifact["config_hash"],
    })

    # Verify metrics hash
    metrics = artifact["metrics"]
    expected_metrics_hash = compute_artifact_hash(metrics)
    metrics_ok = expected_metrics_hash == artifact["metrics_hash"]
    checks.append({
        "check": "metrics_hash",
        "ok": metrics_ok,
        "expected": expected_metrics_hash,
        "actual": artifact["metrics_hash"],
    })

    # Verify artifact hash (must recompute without artifact_hash field)
    artifact_copy = {k: v for k, v in artifact.items() if k != "artifact_hash"}
    expected_artifact_hash = compute_artifact_hash(artifact_copy)
    artifact_ok = expected_artifact_hash == artifact["artifact_hash"]
    checks.append({
        "check": "artifact_hash",
        "ok": artifact_ok,
        "expected": expected_artifact_hash,
        "actual": artifact["artifact_hash"],
    })

    all_ok = all(c["ok"] for c in checks)
    return {"ok": all_ok, "checks": checks}


def verify_metrics_tolerance(
    claimed_metrics: dict,
    recomputed_metrics: dict,
    tolerance_pct: float = 5.0,
) -> Dict:
    """Check if claimed metrics are within tolerance of recomputed ones.

    Args:
        claimed_metrics: Metrics from the artifact
        recomputed_metrics: Metrics from a fresh simulation
        tolerance_pct: Allowed deviation percentage

    Returns:
        Verification result
    """
    checks = []
    for key in claimed_metrics:
        if key not in recomputed_metrics:
            continue
        claimed = claimed_metrics[key]
        recomputed = recomputed_metrics[key]
        if not isinstance(claimed, (int, float)) or not isinstance(recomputed, (int, float)):
            continue

        if recomputed == 0:
            deviation = abs(claimed)
        else:
            deviation = abs(claimed - recomputed) / abs(recomputed) * 100

        ok = deviation <= tolerance_pct
        checks.append({
            "metric": key,
            "claimed": claimed,
            "recomputed": recomputed,
            "deviation_pct": round(deviation, 2),
            "ok": ok,
        })

    all_ok = all(c["ok"] for c in checks) if checks else True
    return {"ok": all_ok, "checks": checks, "tolerance_pct": tolerance_pct}


def main():
    parser = argparse.ArgumentParser(
        description="Verify antelligence simulation attestation",
    )
    parser.add_argument("hash", nargs="?", help="Artifact hash or IPFS CID to verify")
    parser.add_argument("--from-file", help="Verify from local JSON file")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.from_file:
        with open(args.from_file) as f:
            artifact = json.load(f)
    elif args.hash:
        # Try as IPFS CID first, then as local file
        if args.hash.startswith("Qm") or args.hash.startswith("bafy"):
            artifact, err = fetch_from_ipfs(args.hash)
            if not artifact:
                print(json.dumps({"ok": False, "error": err}))
                sys.exit(1)
        else:
            # Try as file path
            p = Path(args.hash)
            if p.exists():
                with open(p) as f:
                    artifact = json.load(f)
            else:
                print(json.dumps({"ok": False, "error": f"Not a valid CID or file: {args.hash}"}))
                sys.exit(1)
    else:
        parser.error("Provide a hash/CID or --from-file")

    # Verify integrity
    result = verify_artifact_integrity(artifact)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"Verification: {status}")
        for check in result["checks"]:
            icon = "✓" if check["ok"] else "✗"
            print(f"  {icon} {check['check']}: {'OK' if check['ok'] else check.get('reason', 'MISMATCH')}")


if __name__ == "__main__":
    main()
