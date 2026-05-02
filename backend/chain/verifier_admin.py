"""Admin helpers for configuring verifier contracts and submitting proof verification."""

from __future__ import annotations

import json
import subprocess
from typing import Dict

from .config import get_base_sepolia_rpc_url, get_private_key, get_tumor_intel_address
from .proof_lifecycle import build_lifecycle, build_verification_status


def set_verifier_address(verifier_address: str) -> Dict:
    rpc_url = get_base_sepolia_rpc_url()
    private_key = get_private_key()
    contract = get_tumor_intel_address()
    
    # Validate verifier address is not empty and is valid
    if not verifier_address or not verifier_address.startswith("0x"):
        raise ValueError("Invalid verifier address")

    result = subprocess.run(
        [
            "cast",
            "send",
            contract,
            "setVerifier(address)",
            verifier_address,
            "--rpc-url",
            rpc_url,
            "--private-key",
            private_key,
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return {"ok": True, "tx": json.loads(result.stdout), "verifier": verifier_address}


def submit_proof_verification(public_values: str, proof_bytes: str) -> Dict:
    rpc_url = get_base_sepolia_rpc_url()
    private_key = get_private_key()
    contract = get_tumor_intel_address()
    result = subprocess.run(
        [
            "cast",
            "send",
            contract,
            "verifySimulation(bytes,bytes)",
            public_values,
            proof_bytes,
            "--rpc-url",
            rpc_url,
            "--private-key",
            private_key,
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    return {
        "ok": True,
        "tx": json.loads(result.stdout),
        "proof_lifecycle": build_lifecycle(
            "verified_onchain",
            note="Proof accepted by verifier-capable TumorIntel contract.",
        ),
        "verification_status": build_verification_status(
            schema_ok=True,
            integrity_ok=True,
            replay_ok=True,
            proof_ok=True,
            onchain_ok=True,
        ),
    }
