"""Unit tests for verification CLI."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.ipfs import create_simulation_artifact, compute_artifact_hash
from chain.proof_adapter import create_proof_bundle
from chain.proof_spec import build_proof_transport_metadata, build_public_values_payload, compute_transport_commitment, encode_public_values_payload
from chain.verify import (
    check_onchain_verification,
    verify_artifact,
    verify_artifact_integrity,
    verify_artifact_replay,
    verify_metrics_tolerance,
    verify_proof_bundle_schema,
    verify_public_values_schema,
)


class TestVerifyIntegrity:
    def test_valid_artifact(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 150, "nanobots": 10},
            metrics={"kill_rate": 45.5, "deliveries": 30},
        )
        result = verify_artifact_integrity(artifact)
        assert result["ok"] is True
        assert all(c["ok"] for c in result["checks"])

    def test_tampered_config_hash(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 150},
            metrics={"kill_rate": 45.5},
        )
        artifact["config_hash"] = "deadbeef" * 8
        result = verify_artifact_integrity(artifact)
        assert result["ok"] is False
        failed = [c for c in result["checks"] if not c["ok"]]
        assert any(c["check"] == "config_hash" for c in failed)

    def test_tampered_metrics(self):
        artifact = create_simulation_artifact(
            config={"x": 1},
            metrics={"kill_rate": 45.5},
        )
        # Tamper with metrics after creation
        artifact["metrics"]["kill_rate"] = 99.9
        result = verify_artifact_integrity(artifact)
        assert result["ok"] is False

    def test_missing_field(self):
        artifact = {"config": {}, "metrics": {}}
        result = verify_artifact_integrity(artifact)
        assert result["ok"] is False


class TestMetricsTolerance:
    def test_within_tolerance(self):
        claimed = {"kill_rate": 45.5, "deliveries": 30}
        recomputed = {"kill_rate": 44.0, "deliveries": 29}
        result = verify_metrics_tolerance(claimed, recomputed, tolerance_pct=5.0)
        assert result["ok"] is True

    def test_outside_tolerance(self):
        claimed = {"kill_rate": 45.5}
        recomputed = {"kill_rate": 20.0}
        result = verify_metrics_tolerance(claimed, recomputed, tolerance_pct=5.0)
        assert result["ok"] is False

    def test_exact_match(self):
        claimed = {"kill_rate": 45.5}
        recomputed = {"kill_rate": 45.5}
        result = verify_metrics_tolerance(claimed, recomputed)
        assert result["ok"] is True
        assert result["checks"][0]["deviation_pct"] == 0.0

    def test_empty_metrics(self):
        result = verify_metrics_tolerance({}, {})
        assert result["ok"] is True

    def test_missing_recomputed_numeric_metric_fails(self):
        result = verify_metrics_tolerance({"kill_rate": 45.5}, {})
        assert result["ok"] is False
        assert result["checks"] == [
            {
                "metric": "kill_rate",
                "claimed": 45.5,
                "recomputed": None,
                "deviation_pct": None,
                "ok": False,
                "reason": "missing_recomputed_metric",
            }
        ]

    def test_boolean_metrics_are_not_accepted_as_numeric_treatment_accounting(self):
        result = verify_metrics_tolerance({"kill_rate": True}, {"kill_rate": 1})

        assert result["ok"] is False
        assert result["checks"] == [
            {
                "metric": "kill_rate",
                "claimed": True,
                "recomputed": 1,
                "deviation_pct": None,
                "ok": False,
                "reason": "non_numeric_metric_value",
            }
        ]

    def test_non_numeric_claimed_metric_fails_even_without_recomputed_counterpart(self):
        result = verify_metrics_tolerance({"kill_rate": "NaN"}, {})

        assert result["ok"] is False
        assert result["checks"] == [
            {
                "metric": "kill_rate",
                "claimed": "NaN",
                "recomputed": None,
                "deviation_pct": None,
                "ok": False,
                "reason": "non_numeric_claimed_metric_value",
            }
        ]

    def test_non_numeric_claimed_and_recomputed_metrics_fail_closed(self):
        result = verify_metrics_tolerance({"kill_rate": "0"}, {"kill_rate": "0"})

        assert result["ok"] is False
        assert result["checks"] == [
            {
                "metric": "kill_rate",
                "claimed": "0",
                "recomputed": "0",
                "deviation_pct": None,
                "ok": False,
                "reason": "non_numeric_metric_value",
            }
        ]


class TestPublicValuesBoundary:
    def test_build_public_values_payload_rejects_fractional_uint32_inputs(self):
        try:
            build_public_values_payload(
                config_hash="0x" + ("ab" * 32),
                kill_rate_bps=1250,
                nanobot_count=4.5,
                tumor_radius=100,
                steps=20,
            )
        except ValueError as exc:
            assert "nanobot_count must encode exactly to uint32" in str(exc)
        else:
            raise AssertionError("fractional proof-boundary inputs should not be silently truncated")

    def test_build_public_values_payload_rejects_uint32_overflow(self):
        try:
            build_public_values_payload(
                config_hash="0x" + ("ab" * 32),
                kill_rate_bps=1250,
                nanobot_count=2**32,
                tumor_radius=100,
                steps=20,
            )
        except ValueError as exc:
            assert "nanobot_count must be between 0 and 4294967295" in str(exc)
        else:
            raise AssertionError("overflowing proof-boundary inputs should not enter uint32 ABI payloads")

    def test_build_public_values_payload_rejects_kill_rate_bps_above_100_percent(self):
        try:
            build_public_values_payload(
                config_hash="0x" + ("ab" * 32),
                kill_rate_bps=10001,
                nanobot_count=4,
                tumor_radius=100,
                steps=20,
            )
        except ValueError as exc:
            assert "kill_rate_bps must be between 0 and 10000" in str(exc)
        else:
            raise AssertionError("kill_rate_bps above 100 percent should not enter proof-boundary payloads")

    def test_encode_public_values_payload_rejects_fractional_payload_drift(self):
        payload = build_public_values_payload(
            config_hash="0x" + ("ab" * 32),
            kill_rate_bps=1250,
            nanobot_count=4,
            tumor_radius=100,
            steps=20,
        )
        payload["steps"] = 20.5

        try:
            encode_public_values_payload(payload)
        except ValueError as exc:
            assert "steps must encode exactly to uint32" in str(exc)
        else:
            raise AssertionError("direct public-value encoding should not truncate fractional payload drift")

    def test_encode_public_values_payload_rejects_extra_fields(self):
        payload = build_public_values_payload(
            config_hash="0x" + ("ab" * 32),
            kill_rate_bps=1250,
            nanobot_count=4,
            tumor_radius=100,
            steps=20,
        )
        payload["uncommitted_shadow_metric"] = 999999

        try:
            encode_public_values_payload(payload)
        except ValueError as exc:
            assert "public-values payload fields must exactly match" in str(exc)
        else:
            raise AssertionError("extra public-value fields should not be ignored before proof-boundary encoding")

    def test_decode_public_values_payload_rejects_trailing_bytes(self):
        from chain.proof_spec import decode_public_values_payload

        payload = build_public_values_payload(
            config_hash="0x" + ("ab" * 32),
            kill_rate_bps=1250,
            nanobot_count=4,
            tumor_radius=100,
            steps=20,
        )
        encoded = encode_public_values_payload(payload)

        try:
            decode_public_values_payload(encoded + "00")
        except ValueError as exc:
            assert "public_values must be exactly 160 bytes" in str(exc)
        else:
            raise AssertionError("trailing bytes must not be accepted at the proof boundary")

    def test_transport_metadata_rejects_noncanonical_public_values_length(self):
        from chain.proof_spec import build_proof_transport_metadata

        try:
            build_proof_transport_metadata(
                public_values="0x00",
                proof_bytes="0xabcd",
                proof_origin="mock",
                prover_status="mock-generated",
                is_mock=True,
            )
        except ValueError as exc:
            assert "public_values must be exactly 160 bytes" in str(exc)
        else:
            raise AssertionError("transport metadata should not bless malformed public values")

    def test_transport_metadata_rejects_origin_status_mock_drift(self):
        payload = build_public_values_payload(
            config_hash="0x" + ("ab" * 32),
            kill_rate_bps=1250,
            nanobot_count=4,
            tumor_radius=100,
            steps=20,
        )
        public_values = encode_public_values_payload(payload)

        try:
            build_proof_transport_metadata(
                public_values=public_values,
                proof_bytes="0xabcd",
                proof_origin="sp1-groth16-adapter",
                prover_status="mock-generated",
                is_mock=True,
            )
        except ValueError as exc:
            assert "proof transport origin/status mismatch" in str(exc)
        else:
            raise AssertionError("adapter-origin transport metadata must not reuse mock proof status")

    def test_transport_metadata_rejects_mock_origin_with_nonmock_flag(self):
        payload = build_public_values_payload(
            config_hash="0x" + ("ab" * 32),
            kill_rate_bps=1250,
            nanobot_count=4,
            tumor_radius=100,
            steps=20,
        )
        public_values = encode_public_values_payload(payload)

        try:
            build_proof_transport_metadata(
                public_values=public_values,
                proof_bytes="0xabcd",
                proof_origin="mock",
                prover_status="mock-generated",
                is_mock=False,
            )
        except ValueError as exc:
            assert "proof transport mock flag mismatch" in str(exc)
        else:
            raise AssertionError("mock-origin transport metadata must carry is_mock=True")

    def test_transport_commitment_binds_schema_and_boundary_versions(self, monkeypatch):
        from chain import proof_spec

        payload = build_public_values_payload(
            config_hash="0x" + ("ab" * 32),
            kill_rate_bps=1250,
            nanobot_count=4,
            tumor_radius=100,
            steps=20,
        )
        public_values = encode_public_values_payload(payload)
        baseline = build_proof_transport_metadata(
            public_values=public_values,
            proof_bytes="0xabcd",
            proof_origin="mock",
            prover_status="mock-generated",
            is_mock=True,
        )

        monkeypatch.setattr(proof_spec, "PUBLIC_VALUES_SCHEMA_VERSION", "public-values-v2")
        monkeypatch.setattr(proof_spec, "PROOF_BOUNDARY_VERSION", "sp1-groth16-adapter-v2")
        upgraded = build_proof_transport_metadata(
            public_values=public_values,
            proof_bytes="0xabcd",
            proof_origin="mock",
            prover_status="mock-generated",
            is_mock=True,
        )

        assert upgraded["public_values_schema_version"] == "public-values-v2"
        assert upgraded["proof_boundary_version"] == "sp1-groth16-adapter-v2"
        assert upgraded["transport_commitment"] != baseline["transport_commitment"]


class TestReplayVerification:
    def test_replay_returns_metrics(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2, "seed": 7, "domain_size": 40, "voxel_size": 20},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        result = verify_artifact_replay(artifact, tolerance_pct=100.0)
        assert "recomputed_metrics" in result
        assert "kill_rate" in result["recomputed_metrics"]
        assert "deliveries" in result["recomputed_metrics"]

    def test_full_verification_includes_replay(self):
        from chain.submit import create_attestation_bundle
        artifact = create_attestation_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2, "seed": 7, "domain_size": 40, "voxel_size": 20},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        artifact = {**artifact["ipfs"]["artifact"], "onchain": artifact["onchain"]}
        result = verify_artifact(artifact, tolerance_pct=100.0, replay=True)
        assert "integrity" in result
        assert "public_values" in result
        assert "replay" in result
        assert "onchain" in result
        assert result["integrity"]["ok"] is True
        assert result["public_values"]["ok"] is True
        assert result["replay"] is not None

    def test_public_values_schema_validation(self):
        from chain.submit import create_attestation_bundle
        artifact = create_attestation_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        result = verify_public_values_schema({"config_hash": artifact["ipfs"]["artifact"]["config_hash"], "onchain": artifact["onchain"]})
        assert result["ok"] is True
        assert len(result["checks"]) >= 6

    def test_public_values_schema_rejects_artifact_config_hash_mismatch(self):
        from chain.submit import create_attestation_bundle
        artifact = create_attestation_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        result = verify_public_values_schema({"config_hash": "ff" * 32, "onchain": artifact["onchain"]})
        assert result["ok"] is False
        assert any(check["check"] == "artifact_config_hash_matches_payload" and check["ok"] is False for check in result["checks"])

    def test_public_values_schema_rejects_onchain_config_hash_mismatch(self):
        from chain.submit import create_attestation_bundle
        artifact = create_attestation_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        artifact["onchain"]["config_hash"] = "aa" * 32
        result = verify_public_values_schema({"config_hash": artifact["ipfs"]["artifact"]["config_hash"], "onchain": artifact["onchain"]})
        assert result["ok"] is False
        assert any(check["check"] == "onchain_config_hash_matches_payload" and check["ok"] is False for check in result["checks"])

    def test_public_values_schema_rejects_colluding_noncanonical_versions(self):
        from chain.submit import create_attestation_bundle
        artifact = create_attestation_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        onchain = artifact["onchain"]
        onchain["public_values_schema_version"] = "public-values-v999"
        onchain["program_version"] = "tumor-intel-proof-v999"
        onchain["public_values_metadata"]["schema_version"] = "public-values-v999"
        onchain["public_values_metadata"]["program_version"] = "tumor-intel-proof-v999"

        result = verify_public_values_schema({"config_hash": artifact["ipfs"]["artifact"]["config_hash"], "onchain": onchain})

        assert result["ok"] is False
        assert any(
            check["check"] == "public_values_schema_version_canonical" and check["ok"] is False
            for check in result["checks"]
        )
        assert any(
            check["check"] == "public_values_program_version_canonical" and check["ok"] is False
            for check in result["checks"]
        )

    def test_public_values_schema_rejects_malformed_declared_payload_without_crashing(self):
        from chain.submit import create_attestation_bundle
        artifact = create_attestation_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        artifact["onchain"]["public_values_payload"]["config_hash"] = "not-hex"

        result = verify_public_values_schema({"config_hash": artifact["ipfs"]["artifact"]["config_hash"], "onchain": artifact["onchain"]})

        assert result["ok"] is False
        assert any(
            check["check"] == "public_values_payload_encoding" and check["ok"] is False and "reason" in check
            for check in result["checks"]
        )

    def test_public_values_schema_rejects_declared_kill_rate_above_full_clearance(self):
        from chain.submit import create_attestation_bundle
        artifact = create_attestation_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
        )
        onchain = artifact["onchain"]
        onchain["kill_rate_bps"] = 10001
        onchain["public_values_payload"]["kill_rate_bps"] = 10001

        result = verify_public_values_schema({"config_hash": artifact["ipfs"]["artifact"]["config_hash"], "onchain": onchain})

        assert result["ok"] is False
        assert any(
            check["check"] == "kill_rate_bps_uint32_boundary" and check["ok"] is False
            for check in result["checks"]
        )

    def test_proof_bundle_schema_validation(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-1",
        )
        result = verify_proof_bundle_schema(bundle)
        assert result["ok"] is True
        assert any(check["check"] == "proof_artifact_version" and check["ok"] is True for check in result["checks"])
        assert any(check["check"] == "transport_commitment" and check["ok"] is True for check in result["checks"])
        assert result["decoded_public_values"]["config_hash"] == bundle["proof_bundle"]["config_hash"]

    def test_proof_bundle_schema_rejects_transport_commitment_mismatch(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-2",
        )
        bundle["proof_bundle"]["transport_metadata"]["transport_commitment"] = "deadbeef"
        result = verify_proof_bundle_schema(bundle)
        assert result["ok"] is False
        assert any(check["check"] == "transport_commitment" and check["ok"] is False for check in result["checks"])

    def test_proof_bundle_schema_rejects_public_values_for_different_artifact_config(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-config-mismatch",
        )
        wrong_config_hash = "aa" * 32
        payload = dict(bundle["onchain"]["public_values_payload"])
        payload["config_hash"] = wrong_config_hash
        public_values = encode_public_values_payload(payload)
        proof_bundle = bundle["proof_bundle"]
        proof_bundle["config_hash"] = wrong_config_hash
        proof_bundle["public_values"] = public_values
        proof_bundle["transport_metadata"] = build_proof_transport_metadata(
            public_values=public_values,
            proof_bytes=proof_bundle["proof_bytes"],
            proof_origin=proof_bundle["proof_origin"],
            prover_status=proof_bundle["prover_status"],
            is_mock=proof_bundle["is_mock"],
        )
        bundle["onchain"]["config_hash"] = wrong_config_hash
        bundle["onchain"]["public_values"] = public_values
        bundle["onchain"]["public_values_payload"] = payload

        result = verify_proof_bundle_schema(bundle)

        assert result["ok"] is False
        assert any(
            check["check"] == "proof_payload_config_hash_matches_artifact" and check["ok"] is False
            for check in result["checks"]
        )

    def test_proof_bundle_schema_rejects_transport_origin_mismatch(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-origin-mismatch",
        )
        bundle["proof_bundle"]["transport_metadata"]["proof_origin"] = "different-prover"

        result = verify_proof_bundle_schema(bundle)

        assert result["ok"] is False
        assert any(
            check["check"] == "transport_proof_origin_matches_bundle" and check["ok"] is False
            for check in result["checks"]
        )

    def test_proof_bundle_schema_rejects_colluding_origin_status_drift(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-colluding-status-drift",
        )
        proof_bundle = bundle["proof_bundle"]
        proof_bundle["prover_status"] = "mock-generated"
        proof_bundle["transport_metadata"]["prover_status"] = "mock-generated"
        proof_bundle["transport_metadata"]["transport_commitment"] = compute_transport_commitment(
            proof_bundle["public_values"],
            proof_bundle["proof_bytes"],
            proof_bundle["proof_origin"],
            proof_bundle["prover_status"],
            "tumor-intel-proof-v1",
        )

        result = verify_proof_bundle_schema(bundle)

        assert result["ok"] is False
        assert any(
            check["check"] == "proof_transport_origin_status" and check["ok"] is False
            for check in result["checks"]
        )

    def test_proof_bundle_schema_rejects_mock_origin_with_nonmock_flag(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-mock-flag-drift",
        )
        proof_bundle = bundle["proof_bundle"]
        proof_bundle["proof_origin"] = "mock"
        proof_bundle["prover_status"] = "mock-generated"
        proof_bundle["is_mock"] = False
        proof_bundle["transport_metadata"]["proof_origin"] = "mock"
        proof_bundle["transport_metadata"]["prover_status"] = "mock-generated"
        proof_bundle["transport_metadata"]["is_mock"] = False
        proof_bundle["transport_metadata"]["transport_commitment"] = compute_transport_commitment(
            proof_bundle["public_values"],
            proof_bundle["proof_bytes"],
            proof_bundle["proof_origin"],
            proof_bundle["prover_status"],
            "tumor-intel-proof-v1",
        )

        result = verify_proof_bundle_schema(bundle)

        assert result["ok"] is False
        assert any(
            check["check"] == "proof_transport_mock_flag" and check["ok"] is False
            for check in result["checks"]
        )

    def test_proof_bundle_schema_rejects_tampered_transport_value_commitments(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-transport-commitment-mismatch",
        )
        bundle["proof_bundle"]["transport_metadata"]["public_values_commitment"] = "00" * 32
        bundle["proof_bundle"]["transport_metadata"]["proof_bytes_commitment"] = "11" * 32

        result = verify_proof_bundle_schema(bundle)

        assert result["ok"] is False
        assert any(
            check["check"] == "transport_public_values_commitment" and check["ok"] is False
            for check in result["checks"]
        )
        assert any(
            check["check"] == "transport_proof_bytes_commitment" and check["ok"] is False
            for check in result["checks"]
        )

    def test_verify_artifact_includes_proof_bundle_validation(self):
        bundle = create_proof_bundle(
            config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2, "seed": 7, "domain_size": 40, "voxel_size": 20},
            metrics={"kill_rate": 0.0, "deliveries": 0},
            run_id="proof-verify-3",
        )
        artifact = {
            **bundle["ipfs"]["artifact"],
            "onchain": bundle["onchain"],
            "proof_bundle": bundle["proof_bundle"],
            "proof_lifecycle": bundle["proof_lifecycle"],
            "verification_status": bundle["verification_status"],
            "trust_tier": bundle["trust_tier"],
        }
        result = verify_artifact(artifact, tolerance_pct=100.0, replay=False)
        assert "proof_bundle" in result
        assert result["proof_bundle"]["ok"] is True
        assert result["verification_status"]["schema_ok"] is True
        assert result["verification_status"]["integrity_ok"] is True
        assert result["verification_status"]["proof_ok"] is False
        assert result["verification_status"]["onchain_ok"] is False
        assert result["proof_lifecycle"]["stage"] == "proof_generated"
        assert result["trust_tier"] == "proof_staged"

    def test_verify_artifact_marks_verified_onchain_only_when_chain_accepts(self, monkeypatch):
        bundle = create_proof_bundle(
            config={"tumor_radius": 55, "nanobot_count": 3, "steps": 4},
            metrics={"kill_rate": 12.5, "deliveries": 1},
            run_id="proof-verify-4",
        )
        artifact = {
            **bundle["ipfs"]["artifact"],
            "onchain": bundle["onchain"],
            "proof_bundle": bundle["proof_bundle"],
            "proof_lifecycle": bundle["proof_lifecycle"],
            "verification_status": bundle["verification_status"],
        }

        monkeypatch.setattr("chain.verify.check_onchain_verification", lambda config_hash: {"ok": True, "verified": True, "raw": "true"})
        result = verify_artifact(artifact, tolerance_pct=100.0, replay=False)

        assert result["proof_bundle"]["ok"] is True
        assert result["verification_status"]["proof_ok"] is True
        assert result["verification_status"]["onchain_ok"] is True
        assert result["proof_lifecycle"]["stage"] == "verified_onchain"
        assert result["trust_tier"] == "verified_onchain"

    def test_onchain_check_returns_shape(self):
        result = check_onchain_verification("00" * 32)
        assert "ok" in result
