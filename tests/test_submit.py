"""Unit tests for submission CLI."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.submit import create_attestation_bundle, encode_public_values, TUMOR_INTEL_ADDRESS


class TestAttestationBundle:
    def test_create_bundle(self):
        config = {"tumor_radius": 150, "nanobot_count": 10, "steps": 300}
        metrics = {"kill_rate": 45.5, "deliveries": 30}
        bundle = create_attestation_bundle(config, metrics)

        assert bundle["ok"] is True
        assert bundle["onchain"]["contract"] == TUMOR_INTEL_ADDRESS
        assert bundle["onchain"]["chain_id"] == 84532
        assert bundle["onchain"]["kill_rate_bps"] == 4550
        assert bundle["onchain"]["nanobot_count"] == 10
        assert bundle["onchain"]["tumor_radius"] == 150
        assert bundle["onchain"]["steps"] == 300
        assert len(bundle["onchain"]["config_hash"]) == 64

    def test_bundle_has_ipfs(self):
        bundle = create_attestation_bundle(
            config={"x": 1},
            metrics={"kill_rate": 10.0},
        )
        assert "ipfs" in bundle
        assert "artifact_hash" in bundle["ipfs"]

    def test_bundle_status(self):
        bundle = create_attestation_bundle(
            config={"tumor_radius": 100},
            metrics={"kill_rate": 30.0},
        )
        assert bundle["status"] == "ready_for_submission"
        assert bundle["proof_lifecycle"]["stage"] == "bundle_created"
        assert bundle["proof_lifecycle"]["proof_system"] == "sp1+groth16"
        assert bundle["verification_status"]["integrity_ok"] is True
        assert "verifySimulation" in bundle["next_step"]

    def test_bundle_contains_public_values(self):
        bundle = create_attestation_bundle(
            config={"tumor_radius": 100, "nanobot_count": 4, "steps": 20},
            metrics={"kill_rate": 12.0},
        )
        assert bundle["onchain"]["public_values"].startswith("0x")
        assert len(bundle["onchain"]["public_values"]) > 10
        assert bundle["onchain"]["public_values_schema_version"] == "public-values-v1"
        assert bundle["onchain"]["public_values_metadata"]["program_version"] == "tumor-intel-proof-v1"

    def test_encode_public_values(self):
        encoded = encode_public_values("00" * 32, 1234, 5, 150, 40)
        assert encoded.startswith("0x")
