"""Unit tests for submission CLI."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.submit import create_attestation_bundle, TUMOR_INTEL_ADDRESS


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
        assert bundle["status"] == "ready_for_proof"
        assert "verifySimulation" in bundle["next_step"]
