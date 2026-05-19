

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from chain.proof_spec import (
    build_proof_transport_metadata,
    build_public_values_payload,
    encode_public_values_payload,
    TRANSPORT_METADATA_REQUIRED_KEYS,
)

def test_transport_metadata_drift_protection():
    # Provide a valid public_values payload (160 bytes ABI-encoded) so that
    # decode_public_values_payload inside build_proof_transport_metadata succeeds.
    payload = build_public_values_payload(
        config_hash="0x" + "ab" * 32,
        kill_rate_bps=1250,
        nanobot_count=10,
        tumor_radius=100,
        steps=20,
    )
    public_values = encode_public_values_payload(payload)
    # proof_bytes can be arbitrary hex of even length
    proof_bytes = "0x" + "00" * 64

    metadata = build_proof_transport_metadata(
        public_values=public_values,
        proof_bytes=proof_bytes,
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True
    )

    assert "transport_commitment" in metadata

def test_transport_metadata_invalid_keys():
    # Anchor the expected number of transport metadata keys.
    assert len(TRANSPORT_METADATA_REQUIRED_KEYS) == 16
