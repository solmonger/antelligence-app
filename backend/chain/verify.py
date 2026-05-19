"""Verification CLI for simulation attestation.

Fetches a simulation artifact from IPFS or disk, verifies artifact integrity,
and optionally replays the simulation from captured config to compare claimed vs
recomputed metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chain.config import get_base_sepolia_rpc_url, get_tumor_intel_address
from chain.ipfs import compute_artifact_hash
from chain.proof_spec import (
    PROGRAM_VERSION,
    PROOF_ARTIFACT_VERSION,
    PROOF_BOUNDARY_VERSION,
    PROOF_FORMAT,
    PROOF_SYSTEM,
    PROOF_TRANSPORT_ORIGIN_STATUSES,
    PROOF_TRANSPORT_PROTOCOL_VERSION,
    PROOF_TRANSPORT_STATUS_ACTIVE,
    PUBLIC_VALUES_ABI_TYPES,
    PUBLIC_VALUES_FIELDS,
    PUBLIC_VALUES_SCHEMA_VERSION,
    TRANSPORT_METADATA_REQUIRED_KEYS,
    build_proof_transport_metadata,
    build_transport_commitment_inputs,
    compute_transport_commitment,
    decode_public_values_payload,
    encode_public_values_payload,
    validate_proof_transport_mock_flag,
    validate_proof_transport_origin_status,
    validate_public_values_payload,
)
from simulation_replay import replay_artifact_metrics


def fetch_from_ipfs(cid: str) -> Tuple[Optional[dict], Optional[str]]:
    """Fetch artifact JSON from IPFS via public gateway."""
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


def canonical_artifact_view(artifact: dict) -> dict:
    """Return the canonical artifact fields used for integrity hashing.

    This intentionally excludes auxiliary runtime/proof/onchain metadata added by
    later pipeline stages so integrity remains anchored to the original artifact.
    """
    canonical_keys = [
        "version",
        "type",
        "run_id",
        "timestamp",
        "config",
        "metrics",
        "config_hash",
        "metrics_hash",
    ]
    return {k: artifact[k] for k in canonical_keys if k in artifact}


def verify_artifact_integrity(artifact: dict) -> Dict:
    """Verify internal consistency of a simulation artifact."""
    checks = []

    required = ["version", "type", "config", "metrics", "config_hash", "metrics_hash", "artifact_hash"]
    for field in required:
        if field not in artifact:
            checks.append({"check": f"field_{field}", "ok": False, "reason": f"Missing field: {field}"})

    if any(not c["ok"] for c in checks):
        return {"ok": False, "checks": checks}

    config = artifact["config"]
    expected_config_hash = compute_artifact_hash(config)
    config_ok = expected_config_hash == artifact["config_hash"]
    checks.append({
        "check": "config_hash",
        "ok": config_ok,
        "expected": expected_config_hash,
        "actual": artifact["config_hash"],
    })

    metrics = artifact["metrics"]
    expected_metrics_hash = compute_artifact_hash(metrics)
    metrics_ok = expected_metrics_hash == artifact["metrics_hash"]
    checks.append({
        "check": "metrics_hash",
        "ok": metrics_ok,
        "expected": expected_metrics_hash,
        "actual": artifact["metrics_hash"],
    })

    expected_artifact_hash = compute_artifact_hash(canonical_artifact_view(artifact))
    artifact_ok = expected_artifact_hash == artifact["artifact_hash"]
    checks.append({
        "check": "artifact_hash",
        "ok": artifact_ok,
        "expected": expected_artifact_hash,
        "actual": artifact["artifact_hash"],
    })

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def _is_finite_metric_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def verify_metrics_tolerance(
    claimed_metrics: dict,
    recomputed_metrics: dict,
    tolerance_pct: float = 5.0,
) -> Dict:
    """Check if claimed metrics are within tolerance of recomputed ones."""
    if not (isinstance(tolerance_pct, (int, float)) and not isinstance(tolerance_pct, bool)
            and math.isfinite(tolerance_pct) and tolerance_pct >= 0):
        return {
            "ok": False,
            "checks": [{
                "check": "tolerance_pct_valid",
                "ok": False,
                "reason": f"tolerance_pct must be a non-negative finite number, got {tolerance_pct!r}",
            }],
            "tolerance_pct": tolerance_pct,
        }
    checks = []
    for key in claimed_metrics:
        claimed = claimed_metrics[key]
        claimed_numeric = _is_finite_metric_number(claimed)
        if key not in recomputed_metrics:
            checks.append({
                "metric": key,
                "claimed": claimed,
                "recomputed": None,
                "deviation_pct": None,
                "ok": False,
                "reason": "missing_recomputed_metric" if claimed_numeric else "non_numeric_claimed_metric_value",
            })
            continue
        recomputed = recomputed_metrics[key]
        recomputed_numeric = _is_finite_metric_number(recomputed)
        if not claimed_numeric or not recomputed_numeric:
            checks.append({
                "metric": key,
                "claimed": claimed,
                "recomputed": recomputed,
                "deviation_pct": None,
                "ok": False,
                "reason": "non_numeric_computed_or_claimed_metric_value",
            })
            continue

        if recomputed == 0:
            if claimed == 0:
                deviation = 0.0
            else:
                # Non-zero claimed vs zero recomputed: relative deviation is
                # undefined/infinite.  A metric that the recomputer says is
                # zero but the claim says is non-zero must always fail,
                # regardless of tolerance — this is a metric-accounting
                # invariant.
                checks.append({
                    "metric": key,
                    "claimed": claimed,
                    "recomputed": recomputed,
                    "deviation_pct": None,
                    "ok": False,
                    "reason": "nonzero_claimed_vs_zero_recomputed_deviation_undefined",
                })
                continue
        else:
            deviation = abs(claimed - recomputed) / abs(recomputed) * 100

        checks.append({
            "metric": key,
            "claimed": claimed,
            "recomputed": recomputed,
            "deviation_pct": round(deviation, 2),
            "ok": deviation <= tolerance_pct,
        })

    # Flag phantom metrics: keys in recomputed that were never claimed.
    # A phantom metric could mask that the recomputation produced different
    # outputs than the original claim, which is a metric-accounting invariant.
    phantom_keys = [k for k in recomputed_metrics if k not in claimed_metrics]
    for key in phantom_keys:
        checks.append({
            "metric": key,
            "claimed": None,
            "recomputed": recomputed_metrics[key],
            "deviation_pct": None,
            "ok": False,
            "reason": "phantom_recomputed_metric_not_in_claimed",
        })

    all_ok = all(c["ok"] for c in checks) if checks else True
    return {"ok": all_ok, "checks": checks, "tolerance_pct": tolerance_pct}


def verify_public_values_schema(artifact: dict) -> Dict:
    onchain = artifact.get("onchain", {})
    metadata = onchain.get("public_values_metadata", {})
    payload = onchain.get("public_values_payload", {})
    checks = []
    if onchain:
        public_values = onchain.get("public_values")
        checks.append({
            "check": "public_values_present",
            "ok": bool(public_values),
        })
        checks.append({
            "check": "public_values_schema_version",
            "ok": metadata.get("schema_version") == onchain.get("public_values_schema_version"),
            "expected": onchain.get("public_values_schema_version"),
            "actual": metadata.get("schema_version"),
        })
        checks.append({
            "check": "public_values_schema_version_canonical",
            "ok": onchain.get("public_values_schema_version") == PUBLIC_VALUES_SCHEMA_VERSION,
            "expected": PUBLIC_VALUES_SCHEMA_VERSION,
            "actual": onchain.get("public_values_schema_version"),
        })
        checks.append({
            "check": "public_values_program_version",
            "ok": metadata.get("program_version") == onchain.get("program_version"),
            "expected": onchain.get("program_version"),
            "actual": metadata.get("program_version"),
        })
        checks.append({
            "check": "public_values_program_version_canonical",
            "ok": onchain.get("program_version") == PROGRAM_VERSION,
            "expected": PROGRAM_VERSION,
            "actual": onchain.get("program_version"),
        })
        checks.append({
            "check": "public_values_field_list",
            "ok": metadata.get("fields") == list(PUBLIC_VALUES_FIELDS),
            "expected": list(PUBLIC_VALUES_FIELDS),
            "actual": metadata.get("fields"),
        })
        checks.append({
            "check": "public_values_abi_types",
            "ok": metadata.get("abi_types") == list(PUBLIC_VALUES_ABI_TYPES),
            "expected": list(PUBLIC_VALUES_ABI_TYPES),
            "actual": metadata.get("abi_types"),
        })
        payload_present = all(field in payload for field in PUBLIC_VALUES_FIELDS)
        checks.append({
            "check": "public_values_payload_present",
            "ok": payload_present,
        })
        decoded_payload = None
        if public_values:
            try:
                decoded_payload = decode_public_values_payload(public_values)
                checks.append({
                    "check": "public_values_decode",
                    "ok": True,
                    "decoded": decoded_payload,
                })
            except Exception as exc:
                checks.append({
                    "check": "public_values_decode",
                    "ok": False,
                    "reason": str(exc),
                })
        if payload_present:
            checks.extend(validate_public_values_payload(payload))
            try:
                expected_encoding = encode_public_values_payload(payload)
                checks.append({
                    "check": "public_values_payload_encoding",
                    "ok": expected_encoding == public_values,
                    "expected": public_values,
                    "actual": expected_encoding,
                })
            except Exception as exc:
                checks.append({
                    "check": "public_values_payload_encoding",
                    "ok": False,
                    "expected": public_values,
                    "actual": None,
                    "reason": str(exc),
                })
        if payload_present and decoded_payload is not None:
            checks.append({
                "check": "decoded_payload_matches_declared_payload",
                "ok": decoded_payload == payload,
                "expected": payload,
                "actual": decoded_payload,
            })
            artifact_config_hash = artifact.get("config_hash")
            if artifact_config_hash is not None:
                checks.append({
                    "check": "artifact_config_hash_matches_payload",
                    "ok": artifact_config_hash == decoded_payload.get("config_hash"),
                    "expected": decoded_payload.get("config_hash"),
                    "actual": artifact_config_hash,
                })
            checks.append({
                "check": "onchain_config_hash_matches_payload",
                "ok": onchain.get("config_hash") == decoded_payload.get("config_hash"),
                "expected": decoded_payload.get("config_hash"),
                "actual": onchain.get("config_hash"),
            })
            for field in ("kill_rate_bps", "nanobot_count", "tumor_radius", "steps"):
                checks.append({
                    "check": f"onchain_{field}_matches_payload",
                    "ok": onchain.get(field) == decoded_payload.get(field),
                    "expected": decoded_payload.get(field),
                    "actual": onchain.get(field),
                })
    return {"ok": all(c["ok"] for c in checks) if checks else True, "checks": checks}


def verify_artifact_replay(artifact: dict, tolerance_pct: float = 5.0) -> Dict:
    """Replay the simulation from artifact config and compare metrics.

    Passes the full claimed and recomputed numeric metric dictionaries to
    verify_metrics_tolerance instead of cherry-picking two keys, so that
    phantom-metric detection (keys present in recomputed but absent from
    claimed) and missing-metric detection fire correctly in the replay path.
    Structural metadata keys (normalized_config, replay_metadata, etc.) are
    excluded from tolerance comparison because they are not numeric metrics.
    """
    recomputed = replay_artifact_metrics(artifact)
    claimed = artifact.get("metrics", {})

    # Normalize claimed key names to match recomputed output convention
    normalized_claimed: Dict[str, object] = {}
    for key, value in claimed.items():
        normalized_key = "deliveries" if key == "total_deliveries" else key
        normalized_claimed[normalized_key] = value

    # Separate numeric metrics from structural metadata in recomputed output
    _STRUCTURAL_KEYS = {"normalized_config", "replay_metadata", "swarm_provenance"}
    numeric_recomputed = {k: v for k, v in recomputed.items() if k not in _STRUCTURAL_KEYS}

    tolerance_result = verify_metrics_tolerance(
        normalized_claimed,
        numeric_recomputed,
        tolerance_pct=tolerance_pct,
    )
    return {
        "ok": tolerance_result["ok"],
        "recomputed_metrics": recomputed,
        "tolerance": tolerance_result,
    }


def verify_proof_bundle_schema(record: dict) -> Dict:
    """Verify the staged SP1/Groth16 proof bundle boundary.

    The current bundle is intentionally non-cryptographic, but its transport shape,
    schema versions, and commitments must still be internally consistent.
    """
    proof_bundle = record.get("proof_bundle") if isinstance(record.get("proof_bundle"), dict) else record
    if not isinstance(proof_bundle, dict) or not proof_bundle:
        return {"ok": False, "checks": [{"check": "proof_bundle_present", "ok": False, "reason": "Missing proof_bundle"}], "decoded_public_values": None}

    checks = [{"check": "proof_bundle_present", "ok": True}]
    checks.append(validate_proof_transport_origin_status(
        str(proof_bundle.get("proof_origin", "")),
        str(proof_bundle.get("prover_status", "")),
    ))
    checks.append(validate_proof_transport_mock_flag(
        str(proof_bundle.get("proof_origin", "")),
        bool(proof_bundle.get("is_mock", False)),
    ))
    decoded_public_values = None

    checks.extend([
        {
            "check": "proof_artifact_version",
            "ok": proof_bundle.get("proof_artifact_version") == PROOF_ARTIFACT_VERSION,
            "expected": PROOF_ARTIFACT_VERSION,
            "actual": proof_bundle.get("proof_artifact_version"),
        },
        {
            "check": "proof_system",
            "ok": proof_bundle.get("proof_system") == PROOF_SYSTEM,
            "expected": PROOF_SYSTEM,
            "actual": proof_bundle.get("proof_system"),
        },
        {
            "check": "proof_format",
            "ok": proof_bundle.get("proof_format") == PROOF_FORMAT,
            "expected": PROOF_FORMAT,
            "actual": proof_bundle.get("proof_format"),
        },
        {
            "check": "public_values_schema_version",
            "ok": proof_bundle.get("public_values_schema_version") == PUBLIC_VALUES_SCHEMA_VERSION,
            "expected": PUBLIC_VALUES_SCHEMA_VERSION,
            "actual": proof_bundle.get("public_values_schema_version"),
        },
        {
            "check": "program_version",
            "ok": proof_bundle.get("program_version") == PROGRAM_VERSION,
            "expected": PROGRAM_VERSION,
            "actual": proof_bundle.get("program_version"),
        },
        {
            "check": "proof_boundary_version",
            "ok": proof_bundle.get("proof_boundary_version") == PROOF_BOUNDARY_VERSION,
            "expected": PROOF_BOUNDARY_VERSION,
            "actual": proof_bundle.get("proof_boundary_version"),
        },
    ])

    public_values = proof_bundle.get("public_values")
    try:
        decoded_public_values = decode_public_values_payload(public_values)
        checks.append({"check": "public_values_decode", "ok": True, "decoded": decoded_public_values})
    except Exception as exc:
        checks.append({"check": "public_values_decode", "ok": False, "reason": str(exc)})

    if decoded_public_values is not None:
        payload_checks = validate_public_values_payload(decoded_public_values)
        checks.extend(payload_checks)
        checks.append({
            "check": "proof_bundle_config_hash_matches_payload",
            "ok": proof_bundle.get("config_hash") == decoded_public_values.get("config_hash"),
            "expected": decoded_public_values.get("config_hash"),
            "actual": proof_bundle.get("config_hash"),
        })
        onchain = record.get("onchain", {}) if isinstance(record.get("onchain"), dict) else {}
        if onchain:
            checks.append({
                "check": "onchain_public_values_matches_proof_bundle",
                "ok": onchain.get("public_values") == public_values,
                "expected": public_values,
                "actual": onchain.get("public_values"),
            })
            for field in PUBLIC_VALUES_FIELDS:
                # Only check fields that are present as flat onchain keys.
                # Commitment/protocol/status fields live inside public_values_payload
                # and the encoded public_values string — they are not separate
                # onchain columns, so a missing flat key is not a mismatch.
                if field not in onchain:
                    continue
                checks.append({
                    "check": f"onchain_{field}_matches_proof_bundle_payload",
                    "ok": onchain.get(field) == decoded_public_values.get(field),
                    "expected": decoded_public_values.get(field),
                    "actual": onchain.get(field),
                })

    artifact = record.get("ipfs", {}).get("artifact", {}) if isinstance(record.get("ipfs"), dict) else {}
    if artifact:
        if decoded_public_values is not None:
            checks.append({
                "check": "proof_payload_config_hash_matches_artifact",
                "ok": decoded_public_values.get("config_hash") == artifact.get("config_hash"),
                "expected": artifact.get("config_hash"),
                "actual": decoded_public_values.get("config_hash"),
            })
        checks.extend([
            {
                "check": "run_id_matches_artifact",
                "ok": proof_bundle.get("run_id") == artifact.get("run_id"),
                "expected": artifact.get("run_id"),
                "actual": proof_bundle.get("run_id"),
            },
            {
                "check": "artifact_hash_matches_artifact",
                "ok": proof_bundle.get("artifact_hash") == artifact.get("artifact_hash"),
                "expected": artifact.get("artifact_hash"),
                "actual": proof_bundle.get("artifact_hash"),
            },
            {
                "check": "witness_commitment_matches_config_hash",
                "ok": proof_bundle.get("witness_commitment") == artifact.get("config_hash"),
                "expected": artifact.get("config_hash"),
                "actual": proof_bundle.get("witness_commitment"),
            },
            {
                "check": "trace_commitment_matches_artifact_hash",
                "ok": proof_bundle.get("trace_commitment") == artifact.get("artifact_hash"),
                "expected": artifact.get("artifact_hash"),
                "actual": proof_bundle.get("trace_commitment"),
            },
        ])

    proof_bytes = proof_bundle.get("proof_bytes", "")
    try:
        normalized_proof_bytes = proof_bytes[2:] if proof_bytes.startswith("0x") else proof_bytes
        proof_bytes_len = len(bytes.fromhex(normalized_proof_bytes))
        checks.append({"check": "proof_bytes_hex", "ok": True, "bytes": proof_bytes_len})
        checks.append({"check": "proof_bytes_non_empty", "ok": proof_bytes_len > 0, "actual": proof_bytes_len})
    except Exception as exc:
        normalized_proof_bytes = None
        checks.append({"check": "proof_bytes_hex", "ok": False, "reason": str(exc)})

    transport_metadata = proof_bundle.get("transport_metadata") or {}
    checks.append({
        "check": "transport_metadata_present",
        "ok": bool(transport_metadata),
    })
    if transport_metadata and decoded_public_values is not None and normalized_proof_bytes is not None:
        normalized_public_values = public_values if public_values.startswith("0x") else f"0x{public_values}"
        public_values_raw = bytes.fromhex(normalized_public_values[2:])
        proof_bytes_raw = bytes.fromhex(normalized_proof_bytes)
        expected_public_values_commitment = hashlib.sha256(public_values_raw).hexdigest()
        expected_proof_bytes_commitment = hashlib.sha256(proof_bytes_raw).hexdigest()
        expected_transport_commitment = compute_transport_commitment(*build_transport_commitment_inputs(
            public_values=normalized_public_values,
            proof_bytes=proof_bytes if proof_bytes.startswith("0x") else f"0x{normalized_proof_bytes}",
            proof_origin=str(transport_metadata.get("proof_origin", "")),
            prover_status=str(transport_metadata.get("prover_status", "")),
            is_mock=bool(transport_metadata.get("is_mock", False)),
            artifact_version=str(transport_metadata.get("artifact_version", "")),
            proof_system=str(transport_metadata.get("proof_system", "")),
            proof_format=str(transport_metadata.get("proof_format", "")),
            program_version=str(transport_metadata.get("program_version", "")),
            public_values_schema_version=str(transport_metadata.get("public_values_schema_version", "")),
            proof_boundary_version=str(transport_metadata.get("proof_boundary_version", "")),
            protocol_version=str(transport_metadata.get("protocol_version", "")),
            status=str(transport_metadata.get("status", "")),
        ))
        checks.extend([
            {
                "check": "transport_metadata_keys_exact",
                "ok": tuple(transport_metadata.keys()) == TRANSPORT_METADATA_REQUIRED_KEYS,
                "expected": list(TRANSPORT_METADATA_REQUIRED_KEYS),
                "actual": list(transport_metadata.keys()),
            },
            {
                "check": "transport_artifact_version",
                "ok": transport_metadata.get("artifact_version") == proof_bundle.get("proof_artifact_version"),
                "expected": proof_bundle.get("proof_artifact_version"),
                "actual": transport_metadata.get("artifact_version"),
            },
            {
                "check": "transport_proof_system",
                "ok": transport_metadata.get("proof_system") == PROOF_SYSTEM,
                "expected": PROOF_SYSTEM,
                "actual": transport_metadata.get("proof_system"),
            },
            {
                "check": "transport_proof_format",
                "ok": transport_metadata.get("proof_format") == PROOF_FORMAT,
                "expected": PROOF_FORMAT,
                "actual": transport_metadata.get("proof_format"),
            },
            {
                "check": "transport_public_values_schema_version",
                "ok": transport_metadata.get("public_values_schema_version") == proof_bundle.get("public_values_schema_version"),
                "expected": proof_bundle.get("public_values_schema_version"),
                "actual": transport_metadata.get("public_values_schema_version"),
            },
            {
                "check": "transport_program_version",
                "ok": transport_metadata.get("program_version") == proof_bundle.get("program_version"),
                "expected": proof_bundle.get("program_version"),
                "actual": transport_metadata.get("program_version"),
            },
            {
                "check": "transport_proof_boundary_version",
                "ok": transport_metadata.get("proof_boundary_version") == proof_bundle.get("proof_boundary_version"),
                "expected": proof_bundle.get("proof_boundary_version"),
                "actual": transport_metadata.get("proof_boundary_version"),
            },
            {
                "check": "transport_protocol_version",
                "ok": transport_metadata.get("protocol_version") == PROOF_TRANSPORT_PROTOCOL_VERSION,
                "expected": PROOF_TRANSPORT_PROTOCOL_VERSION,
                "actual": transport_metadata.get("protocol_version"),
            },
            {
                "check": "transport_status",
                "ok": transport_metadata.get("status") == PROOF_TRANSPORT_STATUS_ACTIVE,
                "expected": PROOF_TRANSPORT_STATUS_ACTIVE,
                "actual": transport_metadata.get("status"),
            },
            {
                "check": "transport_proof_origin_matches_bundle",
                "ok": transport_metadata.get("proof_origin") == proof_bundle.get("proof_origin"),
                "expected": proof_bundle.get("proof_origin"),
                "actual": transport_metadata.get("proof_origin"),
            },
            {
                "check": "transport_prover_status_matches_bundle",
                "ok": transport_metadata.get("prover_status") == proof_bundle.get("prover_status"),
                "expected": proof_bundle.get("prover_status"),
                "actual": transport_metadata.get("prover_status"),
            },
            {
                "check": "transport_commitment",
                "ok": transport_metadata.get("transport_commitment") == expected_transport_commitment,
                "expected": expected_transport_commitment,
                "actual": transport_metadata.get("transport_commitment"),
            },
            {
                "check": "transport_public_values_commitment",
                "ok": transport_metadata.get("public_values_commitment") == expected_public_values_commitment,
                "expected": expected_public_values_commitment,
                "actual": transport_metadata.get("public_values_commitment"),
            },
            {
                "check": "transport_proof_bytes_commitment",
                "ok": transport_metadata.get("proof_bytes_commitment") == expected_proof_bytes_commitment,
                "expected": expected_proof_bytes_commitment,
                "actual": transport_metadata.get("proof_bytes_commitment"),
            },
            {
                "check": "transport_public_values_bytes",
                "ok": transport_metadata.get("public_values_bytes") == len(public_values_raw),
                "expected": len(public_values_raw),
                "actual": transport_metadata.get("public_values_bytes"),
            },
            {
                "check": "transport_proof_bytes_length",
                "ok": transport_metadata.get("proof_bytes_length") == len(proof_bytes_raw),
                "expected": len(proof_bytes_raw),
                "actual": transport_metadata.get("proof_bytes_length"),
            },
            {
                "check": "transport_is_mock_matches_bundle",
                "ok": transport_metadata.get("is_mock") == proof_bundle.get("is_mock"),
                "expected": proof_bundle.get("is_mock"),
                "actual": transport_metadata.get("is_mock"),
            },
            validate_proof_transport_origin_status(
                str(transport_metadata.get("proof_origin", "")),
                str(transport_metadata.get("prover_status", "")),
            ),
            validate_proof_transport_mock_flag(
                str(transport_metadata.get("proof_origin", "")),
                bool(transport_metadata.get("is_mock", False)),
            ),
        ])

    return {"ok": all(check["ok"] for check in checks), "checks": checks, "decoded_public_values": decoded_public_values}


def check_onchain_verification(config_hash: str) -> Dict:
    """Check whether the current TumorIntel contract marks a config hash as verified."""
    rpc_url = get_base_sepolia_rpc_url()
    contract = get_tumor_intel_address()
    if not rpc_url or not contract:
        return {"ok": False, "reason": "missing chain config"}
    try:
        import subprocess
        result = subprocess.run(
            [
                "cast",
                "call",
                contract,
                "isVerified(bytes32)(bool)",
                f"0x{config_hash}",
                "--rpc-url",
                rpc_url,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
        return {"ok": True, "verified": result.lower() in {"true", "0x1", "1"}, "raw": result}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _flag_is_true(value: object) -> bool:
    """Return True only for explicit boolean True, never truthy claims."""
    return value is True


STAGED_PROOF_LIFECYCLE_STAGES = {"proof_generated", "submitted_onchain"}


def _valid_staged_proof_bundle(proof_bundle: Dict) -> bool:
    """Return True only for proof bundles carrying canonical staged proof metadata.

    Mirrors the leaderboard's valid_staged_proof_bundle gate so that the
    verify-path trust-tier derivation cannot promote a fabricated or
    schema-invalid proof bundle to proof_staged — the same origin/status
    contract enforced by verify_proof_bundle_schema must hold here too.
    """
    if not proof_bundle:
        return False
    required_strings = ("run_id", "artifact_hash", "config_hash", "public_values", "proof_bytes")
    if any(not isinstance(proof_bundle.get(field), str) or not proof_bundle.get(field) for field in required_strings):
        return False
    proof_origin = proof_bundle.get("proof_origin")
    if not isinstance(proof_origin, str) or proof_origin not in PROOF_TRANSPORT_ORIGIN_STATUSES:
        return False
    return all((
        proof_bundle.get("proof_artifact_version") == PROOF_ARTIFACT_VERSION,
        proof_bundle.get("proof_system") == PROOF_SYSTEM,
        proof_bundle.get("proof_format") == PROOF_FORMAT,
        proof_bundle.get("public_values_schema_version") == PUBLIC_VALUES_SCHEMA_VERSION,
        proof_bundle.get("program_version") == PROGRAM_VERSION,
        proof_bundle.get("proof_boundary_version") == PROOF_BOUNDARY_VERSION,
    ))


def _derive_trust_tier(verification_status: Dict, proof_bundle: Dict | None, proof_lifecycle: Dict | None) -> str:
    if _flag_is_true(verification_status.get("onchain_ok")):
        return "verified_onchain"
    if _valid_staged_proof_bundle(proof_bundle or {}) and (proof_lifecycle or {}).get("stage") in STAGED_PROOF_LIFECYCLE_STAGES:
        return "proof_staged"
    if _flag_is_true(verification_status.get("replay_ok")):
        return "replay_checked"
    if _flag_is_true(verification_status.get("integrity_ok")):
        return "integrity_checked"
    return "unverified"


def _artifact_is_mock(artifact: Dict, proof_bundle: Dict | None) -> bool:
    # Trust-tier prefixing is reserved for artifacts explicitly marked as mock.
    # Staged proof bundles currently carry mock transport metadata while they
    # are awaiting a real verifier; that state is already represented by
    # proof_ok=False/proof_staged and should not rewrite historical tiers.
    return artifact.get("is_mock") is True


def verify_artifact(artifact: dict, tolerance_pct: float = 5.0, replay: bool = True) -> Dict:
    integrity = verify_artifact_integrity(artifact)
    public_values = verify_public_values_schema(artifact) if integrity["ok"] else None
    proof_bundle_result = verify_proof_bundle_schema(artifact) if integrity["ok"] and isinstance(artifact.get("proof_bundle"), dict) else None

    replay_result = None
    if replay and integrity["ok"]:
        replay_result = verify_artifact_replay(artifact, tolerance_pct=tolerance_pct)

    onchain = check_onchain_verification(artifact.get("config_hash", "")) if integrity["ok"] else None
    onchain_verified = isinstance(onchain, dict) and _flag_is_true(onchain.get("ok")) and _flag_is_true(onchain.get("verified"))

    prior_status = artifact.get("verification_status", {}) if isinstance(artifact.get("verification_status"), dict) else {}
    verification_status = {
        "schema_ok": _flag_is_true(prior_status.get("schema_ok", True)) and (public_values["ok"] if public_values is not None else True) and (proof_bundle_result["ok"] if proof_bundle_result is not None else True),
        "integrity_ok": integrity["ok"],
        "replay_ok": replay_result["ok"] if replay_result is not None else _flag_is_true(prior_status.get("replay_ok", False)),
        "proof_ok": (proof_bundle_result["ok"] if proof_bundle_result is not None else _flag_is_true(prior_status.get("proof_ok", False))) and onchain_verified,
        "onchain_ok": onchain_verified,
    }

    prior_lifecycle = artifact.get("proof_lifecycle", {}) if isinstance(artifact.get("proof_lifecycle"), dict) else {}
    proof_lifecycle = dict(prior_lifecycle)
    if verification_status["onchain_ok"]:
        proof_lifecycle.update({
            "stage": "verified_onchain",
            "is_final": True,
            "note": "Proof accepted by verifier-capable TumorIntel contract.",
        })
    elif not proof_lifecycle and proof_bundle_result is not None:
        proof_lifecycle = {
            "stage": "proof_generated",
            "proof_system": PROOF_SYSTEM,
            "is_final": False,
            "note": "Proof bundle validated locally; awaiting on-chain verifier acceptance.",
        }
    valid_proof_bundle = artifact.get("proof_bundle") if proof_bundle_result is not None and proof_bundle_result["ok"] else None
    trust_tier = _derive_trust_tier(verification_status, valid_proof_bundle, proof_lifecycle)
    proof_bundle = artifact.get("proof_bundle") if isinstance(artifact.get("proof_bundle"), dict) else None
    is_mock = _artifact_is_mock(artifact, proof_bundle)
    if is_mock and trust_tier != "unverified" and not trust_tier.startswith("mock_"):
        trust_tier = f"mock_{trust_tier}"

    # Explicitly differentiate between local/simulated and cryptographically verified
    trust_metadata = {
        "is_cryptographic": verification_status["onchain_ok"],
        "is_mock": is_mock,
        "trust_tier": trust_tier
    }

    overall_ok = (
        integrity["ok"]
        and (public_values["ok"] if public_values is not None else True)
        and (proof_bundle_result["ok"] if proof_bundle_result is not None else True)
        and (replay_result["ok"] if replay_result is not None else True)
    )
    return {
        "ok": overall_ok,
        "integrity": integrity,
        "public_values": public_values,
        "proof_bundle": proof_bundle_result,
        "replay": replay_result,
        "onchain": onchain,
        "verification_status": verification_status,
        "proof_lifecycle": proof_lifecycle,
        "trust_tier": trust_tier,
        "trust_metadata": trust_metadata,
        "is_mock": is_mock,
    }


def _load_artifact(args: argparse.Namespace) -> dict:
    if args.from_file:
        with open(args.from_file) as f:
            return json.load(f)
    if args.hash:
        if args.hash.startswith("Qm") or args.hash.startswith("bafy"):
            artifact, err = fetch_from_ipfs(args.hash)
            if not artifact:
                print(json.dumps({"ok": False, "error": err}))
                sys.exit(1)
            return artifact

        p = Path(args.hash)
        if p.exists():
            with open(p) as f:
                return json.load(f)

        print(json.dumps({"ok": False, "error": f"Not a valid CID or file: {args.hash}"}))
        sys.exit(1)

    raise SystemExit("Provide a hash/CID or --from-file")


def main():
    parser = argparse.ArgumentParser(description="Verify antelligence simulation attestation")
    parser.add_argument("hash", nargs="?", help="Artifact hash or IPFS CID to verify")
    parser.add_argument("--from-file", help="Verify from local JSON file")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Metric deviation tolerance percentage")
    parser.add_argument("--no-replay", action="store_true", help="Skip simulation replay and only verify integrity")
    args = parser.parse_args()

    artifact = _load_artifact(args)
    result = verify_artifact(artifact, tolerance_pct=args.tolerance, replay=not args.no_replay)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    status = "PASS" if result["ok"] else "FAIL"
    print(f"Verification: {status}")
    print(f"  Integrity: {'PASS' if result['integrity']['ok'] else 'FAIL'}")
    for check in result["integrity"]["checks"]:
        icon = "✓" if check["ok"] else "✗"
        print(f"    {icon} {check['check']}: {'OK' if check['ok'] else check.get('reason', 'MISMATCH')}")

    if result["replay"] is None:
        print("  Replay: SKIPPED")
    else:
        print(f"  Replay: {'PASS' if result['replay']['ok'] else 'FAIL'}")
        for check in result["replay"]["tolerance"]["checks"]:
            icon = "✓" if check["ok"] else "✗"
            print(
                f"    {icon} {check['metric']}: claimed={check['claimed']} "
                f"recomputed={check['recomputed']} deviation={check['deviation_pct']}%"
            )


if __name__ == "__main__":
    main()
