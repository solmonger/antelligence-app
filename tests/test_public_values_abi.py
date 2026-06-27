
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from chain.proof_spec import build_public_values_payload, encode_public_values_payload, decode_public_values_payload

def test_public_values_abi_invariants():
    # Verify that payload fields are strictly enforced and encoding is symmetric
    payload = build_public_values_payload(
        config_hash="0x" + ("1" * 64),
        kill_rate_bps=5000,
        nanobot_count=100,
        tumor_radius=50,
        steps=1000,
    )

    encoded = encode_public_values_payload(payload)
    decoded = decode_public_values_payload(encoded)

    assert decoded["config_hash"] == payload["config_hash"]
    assert decoded["kill_rate_bps"] == payload["kill_rate_bps"]
    assert decoded["nanobot_count"] == payload["nanobot_count"]
    assert decoded["tumor_radius"] == payload["tumor_radius"]
    assert decoded["steps"] == payload["steps"]

    # ABI length is 5 fields * 32 bytes = 160 bytes; this matches TumorIntel.sol.
    assert len(bytes.fromhex(encoded[2:])) == 160
