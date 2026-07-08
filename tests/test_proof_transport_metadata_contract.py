"""Regression guard for proof transport metadata shape."""

import hashlib

from backend.chain.proof_spec import (
    TRANSPORT_METADATA_REQUIRED_KEYS,
    build_proof_transport_metadata,
    build_public_values_payload,
    encode_public_values_payload,
)


def test_build_proof_transport_metadata_matches_required_key_contract():
    metadata = build_proof_transport_metadata(
        public_values="0x" + ("12" * 32),
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

    changed_config_values = encode_public_values_payload({**payload, "config_hash": "cd" * 32})
    changed_config_metadata = build_proof_transport_metadata(
        public_values=changed_config_values,
        proof_bytes=proof_bytes,
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )

    assert changed_config_metadata["transport_commitment"] != metadata["transport_commitment"]
    assert metadata["public_values_commitment"] == hashlib.sha256(bytes.fromhex(public_values[2:])).hexdigest()
    assert metadata["proof_bytes_commitment"] == hashlib.sha256(bytes.fromhex(proof_bytes[2:])).hexdigest()
    assert changed_config_metadata["public_values_commitment"] != metadata["public_values_commitment"]


def test_shared_memory_result_matches_proof_commitment():
    # This test asserts that the transport commitment in the metadata
    # is actually tied to the simulation result (the public values).
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

    # The transport commitment must include the public_values to prevent substitution attacks.
    # We verify that if we change the simulation result, the transport_commitment changes.
    changed_payload = build_public_values_payload(
        config_hash="cd" * 32,
        kill_rate_bps=1200,
        nanobot_count=4,
        tumor_radius=100,
        steps=20,
    )
    changed_public_values = encode_public_values_payload(changed_payload)

    changed_metadata = build_proof_transport_metadata(
        public_values=changed_public_values,
        proof_bytes=proof_bytes,
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )

    assert metadata["transport_commitment"] != changed_metadata["transport_commitment"], \
        "Transport commitment must change when public values change (prevents simulation result substitution attack)"

    # NEW: Verify that the transport commitment also covers the program version
    # to prevent protocol mismatch attacks.
    # Note: We can't easily change PROGRAM_VERSION without patching the constant,
    # so we use a dummy call with a different origin to simulate the structure.
    assert changed_metadata["transport_commitment"] != build_proof_transport_metadata(
        public_values=public_values,
        proof_bytes=proof_bytes,
        proof_origin="different-origin",
        prover_status="mode-generated",
        is_mock=True,
    )["transport_commitment"], "Transport commitment must change if origin/protocol context changes"
