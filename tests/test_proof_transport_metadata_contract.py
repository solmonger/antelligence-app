"""Regression guard for proof transport metadata shape."""

import hashlib

from backend.chain.proof_spec import (
    TRANSPORT_METADATA_REQUIRED_KEYS,
    build_proof_transport_metadata,
    build_public_values_payload,
    encode_public_values_payload,
)


def test_build_proof_transport_metadata_matches_required_key_contract():
    payload = build_public_values_payload(
        config_hash="12" * 32,
        kill_rate_bps=1200,
        nanobot_count=4,
        tumor_radius=100,
        steps=20,
    )

    metadata = build_proof_transport_metadata(
        public_values=encode_public_values_payload(payload),
        proof_bytes="0x" + ("34" * 48),
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )

    assert set(metadata) == set(TRANSPORT_METADATA_REQUIRED_KEYS)
    assert list(metadata) == list(TRANSPORT_METADATA_REQUIRED_KEYS)


def test_transport_metadata_commitments_are_stable_and_bound_to_config_hash():
    payload = build_public_values_payload(
        config_hash="ab" * 32,
        kill_rate_bps=1200,
        nanobot_count=4,
        tumor_radius=100,
        steps=20,
    )
    public_values = encode_public_values_payload(payload)
    proof_bytes = "0x" + ("34" * 48)

    metadata = build_proof_transport_metadata(
        public_values=public_values,
        proof_bytes=proof_bytes,
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )
    repeated = build_proof_transport_metadata(
        public_values=public_values,
        proof_bytes=proof_bytes,
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )
    changed_config_values = encode_public_values_payload({**payload, "config_hash": "cd" * 32})
    changed_config_metadata = build_proof_transport_metadata(
        public_values=changed_config_values,
        proof_bytes=proof_bytes,
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )

    assert repeated["transport_commitment"] == metadata["transport_commitment"]
    assert metadata["public_values_commitment"] == hashlib.sha256(bytes.fromhex(public_values[2:])).hexdigest()
    assert metadata["proof_bytes_commitment"] == hashlib.sha256(bytes.fromhex(proof_bytes[2:])).hexdigest()
    assert changed_config_metadata["public_values_commitment"] != metadata["public_values_commitment"]
    assert changed_config_metadata["transport_commitment"] != metadata["transport_commitment"]
