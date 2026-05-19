import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from chain import proof_spec
from chain.proof_spec import (
    build_proof_transport_metadata,
    build_public_values_payload,
    encode_public_values_payload,
)


def _encoded_public_values() -> str:
    payload = build_public_values_payload(
        config_hash="0x" + ("ab" * 32),
        kill_rate_bps=1250,
        nanobot_count=4,
        tumor_radius=100,
        steps=20,
    )
    return encode_public_values_payload(payload)


def test_transport_commitment_changes_when_transport_only_metadata_changes(monkeypatch):
    public_values = _encoded_public_values()

    baseline = build_proof_transport_metadata(
        public_values=public_values,
        proof_bytes="0xabcd",
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )

    monkeypatch.setattr(proof_spec, "PROOF_FORMAT", "groth16-v2")
    changed = build_proof_transport_metadata(
        public_values=public_values,
        proof_bytes="0xabcd",
        proof_origin="mock",
        prover_status="mock-generated",
        is_mock=True,
    )

    assert changed["proof_format"] == "groth16-v2"
    assert changed["transport_commitment"] != baseline["transport_commitment"]
