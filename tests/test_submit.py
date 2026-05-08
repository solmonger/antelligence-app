"""Unit tests for submission CLI."""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain import proof_spec
from chain.submit import (
    TUMOR_INTEL_ADDRESS,
    create_attestation_bundle,
    encode_public_values,
    submit_via_cast,
)


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

    def test_bundle_rejects_kill_rate_above_full_clearance(self):
        try:
            create_attestation_bundle(
                config={"tumor_radius": 100, "nanobot_count": 4, "steps": 20},
                metrics={"kill_rate": 100.01},
            )
        except ValueError as exc:
            assert "kill_rate must be between 0 and 100" in str(exc)
        else:
            raise AssertionError("out-of-range kill_rate should not enter public values")

    def test_bundle_rejects_non_finite_kill_rate_public_value(self):
        try:
            create_attestation_bundle(
                config={"tumor_radius": 100, "nanobot_count": 4, "steps": 20},
                metrics={"kill_rate": "NaN"},
            )
        except ValueError as exc:
            assert "kill_rate must be finite" in str(exc)
        else:
            raise AssertionError("non-finite kill_rate should not enter public values")

    def test_bundle_rejects_kill_rate_that_would_truncate_public_values(self):
        try:
            create_attestation_bundle(
                config={"tumor_radius": 100, "nanobot_count": 4, "steps": 20},
                metrics={"kill_rate": 12.345},
            )
        except ValueError as exc:
            assert "kill_rate must encode exactly to basis points" in str(exc)
        else:
            raise AssertionError("fractional-bps kill_rate should not be silently truncated")

    def test_bundle_rejects_config_values_that_would_truncate_public_values(self):
        try:
            create_attestation_bundle(
                config={"tumor_radius": 100, "nanobot_count": 4.5, "steps": 20},
                metrics={"kill_rate": 12.0},
            )
        except ValueError as exc:
            assert "nanobot_count must encode exactly to uint32" in str(exc)
        else:
            raise AssertionError("fractional uint32 config value should not be silently truncated")

    def test_bundle_rejects_negative_public_value_config_fields(self):
        try:
            create_attestation_bundle(
                config={"tumor_radius": -1, "nanobot_count": 4, "steps": 20},
                metrics={"kill_rate": 12.0},
            )
        except ValueError as exc:
            assert "tumor_radius must be between 0 and 4294967295" in str(exc)
        else:
            raise AssertionError("negative uint32 config value should not enter public values")

    def test_bundle_rejects_public_values_encoder_drift(self):
        def encode_with_drift(payload):
            drifted = dict(payload)
            drifted["kill_rate_bps"] = payload["kill_rate_bps"] + 1
            return proof_spec.encode_public_values_payload(drifted)

        with patch("chain.submit.encode_public_values_payload", side_effect=encode_with_drift):
            try:
                create_attestation_bundle(
                    config={"tumor_radius": 100, "nanobot_count": 4, "steps": 20},
                    metrics={"kill_rate": 12.0},
                )
            except ValueError as exc:
                assert "encoded public values drifted from payload" in str(exc)
            else:
                raise AssertionError("drifted public values should not enter an attestation bundle")

    def test_bundle_onchain_metadata_preserves_simulation_commitments(self):
        bundle = create_attestation_bundle(
            config={"tumor_radius": 100, "nanobot_count": 4, "steps": 20},
            metrics={"kill_rate": 12.0, "deliveries": 7},
            run_id="commitment-contract",
        )

        artifact = bundle["ipfs"]["artifact"]
        commitments = bundle["onchain"]["simulation_commitments"]
        assert commitments == {
            "config_hash": artifact["config_hash"],
            "metrics_hash": artifact["metrics_hash"],
            "artifact_hash": artifact["artifact_hash"],
        }
        assert bundle["onchain"]["config_hash"] == commitments["config_hash"]
        assert bundle["onchain"]["public_values_payload"]["config_hash"] == commitments["config_hash"]

    def test_encode_public_values(self):
        encoded = encode_public_values("00" * 32, 1234, 5, 150, 40)
        assert encoded.startswith("0x")

    @patch("chain.submit.subprocess.run")
    def test_submit_via_cast_preserves_existing_0x_prefix(self, mock_run):
        mock_run.return_value.stdout = "21000\n"
        result = submit_via_cast(
            config_hash="0x" + ("ab" * 32),
            kill_rate=1234,
            nanobot_count=5,
            tumor_radius=150,
            steps=40,
            rpc_url="http://rpc.test",
            private_key="0xprivate",
            dry_run=True,
        )

        assert result["ok"] is True
        cast_args = mock_run.call_args.args[0]
        assert cast_args[4] == "0x" + ("ab" * 32)

    @patch("chain.submit.subprocess.run")
    def test_submit_via_cast_rejects_out_of_range_kill_rate_before_cast(self, mock_run):
        try:
            submit_via_cast(
                config_hash="0x" + ("ab" * 32),
                kill_rate=10001,
                nanobot_count=5,
                tumor_radius=150,
                steps=40,
                rpc_url="http://rpc.test",
                private_key="0xprivate",
                dry_run=True,
            )
        except ValueError as exc:
            assert "kill_rate must be between 0 and 10000" in str(exc)
        else:
            raise AssertionError("out-of-range kill_rate should not reach cast submission")

        mock_run.assert_not_called()
