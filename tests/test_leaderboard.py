"""Unit tests for leaderboard service."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.ipfs import create_simulation_artifact
from chain.leaderboard import rank_by_kill_rate, build_leaderboard, normalize_leaderboard_artifact
from chain.proof_adapter import create_proof_bundle
from chain.proof_spec import build_proof_transport_metadata


def mark_bundle_as_real_adapter_proof(artifact):
    proof_bundle = artifact["proof_bundle"]
    proof_bundle["is_mock"] = False
    proof_bundle["transport_metadata"] = build_proof_transport_metadata(
        public_values=proof_bundle["public_values"],
        proof_bytes=proof_bundle["proof_bytes"],
        proof_origin=proof_bundle["proof_origin"],
        prover_status=proof_bundle["prover_status"],
        is_mock=False,
    )


class TestRanking:
    def test_rank_by_kill_rate(self):
        entries = [
            {"kill_rate": 30.0, "run_id": "a"},
            {"kill_rate": 50.0, "run_id": "b"},
            {"kill_rate": 10.0, "run_id": "c"},
        ]
        ranked = rank_by_kill_rate(entries)
        assert ranked[0]["run_id"] == "b"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["run_id"] == "a"
        assert ranked[2]["run_id"] == "c"

    def test_empty_list(self):
        ranked = rank_by_kill_rate([])
        assert ranked == []

    def test_single_entry(self):
        ranked = rank_by_kill_rate([{"kill_rate": 45.5}])
        assert ranked[0]["rank"] == 1

    def test_equal_kill_rate_prefers_stronger_trust_tier(self):
        entries = [
            {"kill_rate": 50.0, "trust_tier": "unverified", "run_id": "unverified"},
            {"kill_rate": 50.0, "trust_tier": "proof_staged", "run_id": "proof"},
            {"kill_rate": 50.0, "trust_tier": "replay_checked", "run_id": "replay"},
        ]

        ranked = rank_by_kill_rate(entries)

        assert [entry["run_id"] for entry in ranked] == ["proof", "replay", "unverified"]
        assert [entry["rank"] for entry in ranked] == [1, 2, 3]


class TestBuildLeaderboard:
    def test_from_artifacts(self):
        artifacts = [
            create_simulation_artifact(
                config={"tumor_radius": 150, "nanobot_count": 10, "steps": 300},
                metrics={"kill_rate": 45.5, "deliveries": 30},
                run_id="run-001",
            ),
            create_simulation_artifact(
                config={"tumor_radius": 150, "nanobot_count": 5, "steps": 300},
                metrics={"kill_rate": 22.0, "deliveries": 10},
                run_id="run-002",
            ),
        ]
        result = build_leaderboard(artifacts)
        assert result["ok"] is True
        assert len(result["leaderboard"]) == 2
        assert result["leaderboard"][0]["run_id"] == "run-001"
        assert result["leaderboard"][0]["rank"] == 1
        assert result["summary"]["best_kill_rate"] == 45.5
        assert result["summary"]["total_entries"] == 2

    def test_empty_artifacts(self):
        result = build_leaderboard([])
        assert result["ok"] is True
        assert len(result["leaderboard"]) == 0
        assert result["summary"]["total_entries"] == 0

    def test_summary_stats(self):
        artifacts = [
            create_simulation_artifact(
                config={"tumor_radius": 100},
                metrics={"kill_rate": 40.0},
            ),
            create_simulation_artifact(
                config={"tumor_radius": 100},
                metrics={"kill_rate": 60.0},
            ),
        ]
        result = build_leaderboard(artifacts)
        assert result["summary"]["avg_kill_rate"] == 50.0
        assert result["summary"]["best_kill_rate"] == 60.0

    def test_summary_average_includes_zero_kill_runs(self):
        artifacts = [
            create_simulation_artifact(
                config={"tumor_radius": 100},
                metrics={"kill_rate": 0.0},
            ),
            create_simulation_artifact(
                config={"tumor_radius": 100},
                metrics={"kill_rate": 60.0},
            ),
        ]
        result = build_leaderboard(artifacts)
        assert result["summary"]["avg_kill_rate"] == 30.0

    def test_malformed_metrics_payload_cannot_crash_leaderboard_accounting(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 30.0, "deliveries": 2, "total_drug": 1.0},
            run_id="malformed-metrics-payload",
        )
        artifact["metrics"] = "kill_rate=30"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["kill_rate"] == 0
        assert entry["deliveries"] == 0
        assert entry["total_drug"] == 0
        assert entry["effect_status"] == "no_effect"
        assert entry["trust_tier"] == "no_effect"
        assert result["summary"]["zero_effect_entries"] == 1

    def test_non_finite_metrics_cannot_poison_summary_or_effect_status(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": float("nan"), "deliveries": float("inf"), "total_drug": 0},
            run_id="non-finite-metrics",
        )

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["kill_rate"] == 0
        assert entry["deliveries"] == 0
        assert entry["effect_status"] == "no_effect"
        assert entry["trust_tier"] == "no_effect"
        assert result["summary"]["avg_kill_rate"] == 0
        assert result["summary"]["best_kill_rate"] == 0
        assert result["summary"]["zero_effect_entries"] == 1

    def test_negative_metrics_cannot_create_negative_leaderboard_accounting(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": -12.0, "deliveries": -1, "total_drug": -0.5},
            run_id="negative-metrics",
        )

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["kill_rate"] == 0
        assert entry["deliveries"] == 0
        assert entry["total_drug"] == 0
        assert entry["effect_status"] == "no_effect"
        assert entry["trust_tier"] == "no_effect"
        assert result["summary"]["avg_kill_rate"] == 0
        assert result["summary"]["best_kill_rate"] == 0
        assert result["summary"]["zero_effect_entries"] == 1

    def test_kill_rate_above_100_cannot_poison_leaderboard_accounting(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 125.0, "deliveries": 3, "total_drug": 1.0},
            run_id="impossible-kill-rate",
        )

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["kill_rate"] == 0
        assert entry["deliveries"] == 3
        assert entry["total_drug"] == 1.0
        assert entry["effect_status"] == "effect_reported"
        assert result["summary"]["avg_kill_rate"] == 0
        assert result["summary"]["best_kill_rate"] == 0

    def test_capped_metric_fields_flags_kill_rate_domain_violation(self):
        """When kill_rate exceeds 100, it is domain-capped and the field is auditable."""
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 125.0, "deliveries": 3},
            run_id="capped-kill-rate-audit",
        )

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["kill_rate"] == 0
        assert entry["capped_metric_fields"] == ["kill_rate"]

    def test_capped_metric_fields_empty_when_all_metrics_within_domain(self):
        """When no metrics exceed domain boundaries, capped_metric_fields is empty."""
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 50.0, "deliveries": 3},
            run_id="no-capped-fields",
        )

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["kill_rate"] == 50.0
        assert entry["capped_metric_fields"] == []

    def test_capped_metric_fields_distinguishes_missing_from_capped(self):
        """A missing kill_rate and a domain-capped kill_rate must be distinguishable.

        Missing: capped_metric_fields does not list kill_rate (default 0, no capping occurred).
        Capped: capped_metric_fields lists kill_rate (value was >100, zeroed by cap).
        This is a core metric-accounting invariant.
        """
        missing_artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"deliveries": 3},
            run_id="missing-kill-rate",
        )
        capped_artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 200.0, "deliveries": 3},
            run_id="capped-kill-rate",
        )

        result = build_leaderboard([missing_artifact, capped_artifact])
        missing_entry = next(e for e in result["leaderboard"] if e["run_id"] == "missing-kill-rate")
        capped_entry = next(e for e in result["leaderboard"] if e["run_id"] == "capped-kill-rate")

        # Both have kill_rate=0 but they are distinguishable
        assert missing_entry["kill_rate"] == 0
        assert capped_entry["kill_rate"] == 0
        assert missing_entry["capped_metric_fields"] == []
        assert capped_entry["capped_metric_fields"] == ["kill_rate"]

    def test_malformed_config_counts_cannot_leak_into_leaderboard_public_inputs(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": -100, "nanobot_count": True, "steps": 1.5},
            metrics={"kill_rate": 10.0, "deliveries": 1},
            run_id="malformed-config-counts",
        )

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["tumor_radius"] == 0
        assert entry["nanobot_count"] == 0
        assert entry["steps"] == 0
        assert entry["effect_status"] == "effect_reported"

    def test_malformed_config_payload_cannot_crash_leaderboard_public_inputs(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 10.0, "deliveries": 1},
            run_id="malformed-config-payload",
        )
        artifact["config"] = "tumor_radius=100"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["tumor_radius"] == 0
        assert entry["nanobot_count"] == 0
        assert entry["steps"] == 0
        assert entry["kill_rate"] == 10.0
        assert entry["effect_status"] == "effect_reported"

    def test_zero_effect_proof_runs_remain_in_metrics_without_inflated_trust(self):
        failed_effect = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 0.0, "deliveries": 0, "total_drug": 0},
            run_id="zero-effect-proof",
        )
        successful_effect = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 40.0, "deliveries": 2, "total_drug": 1.5},
            run_id="effect-proof",
        )

        result = build_leaderboard([failed_effect, successful_effect])
        zero_entry = next(e for e in result["leaderboard"] if e["run_id"] == "zero-effect-proof")

        assert result["summary"]["total_entries"] == 2
        assert result["summary"]["avg_kill_rate"] == 20.0
        assert result["summary"]["zero_effect_entries"] == 1
        assert zero_entry["rank"] == 2
        assert zero_entry["effect_status"] == "no_effect"
        assert zero_entry["trust_tier"] == "no_effect"
        assert zero_entry["verified_onchain"] is False
        assert zero_entry["integrity_ok"] is False
        assert zero_entry["replay_ok"] is False
        assert zero_entry["proof_ok"] is False

        # The successful_effect entry is mock (is_mock=True), so it gets mock_proof_staged.
        effect_entry = next(e for e in result["leaderboard"] if e["run_id"] == "effect-proof")
        assert effect_entry["trust_tier"] == "mock_proof_staged"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_legacy_onchain_flag_promotes_verified_trust_tier(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="legacy-onchain",
        )
        artifact["verified_onchain"] = True

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["verified_onchain"] is True
        assert entry["trust_tier"] == "verified_onchain"
        assert result["summary"]["verified_entries"] == 1

    def test_non_boolean_legacy_onchain_flag_cannot_self_promote(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="string-onchain-claim",
        )
        artifact["verified_onchain"] = "true"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["verified_onchain"] is False
        assert entry["trust_tier"] == "unverified"
        assert result["summary"]["verified_entries"] == 0

    def test_artifact_trust_tier_cannot_self_promote_without_verification_evidence(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="self-claimed-onchain",
        )
        artifact["trust_tier"] = "verified_onchain"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["verified_onchain"] is False
        assert entry["trust_tier"] == "unverified"
        assert result["summary"]["verified_entries"] == 0

    def test_non_boolean_replay_and_integrity_flags_cannot_promote_trust_tier(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="string-verification-flags",
        )
        artifact["verification_status"] = {"replay_ok": "true", "integrity_ok": "true"}

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] == "unverified"
        assert entry["replay_ok"] is False
        assert entry["integrity_ok"] is False

    def test_malformed_proof_metadata_cannot_crash_or_promote_trust(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="malformed-proof-metadata",
        )
        artifact["verification_status"] = "onchain_ok=true"
        artifact["proof_lifecycle"] = "proof_generated"
        artifact["proof_bundle"] = "proof_origin=mock"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] == "unverified"
        assert entry["verified_onchain"] is False
        assert entry["proof_stage"] == "untracked"
        assert entry["proof_origin"] == "unknown"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_incomplete_proof_bundle_cannot_self_promote_to_staged_trust(self):
        artifact = create_simulation_artifact(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="incomplete-proof-claim",
        )
        artifact["proof_bundle"] = {"proof_origin": "mock", "proof_bytes": "0xabcd"}

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] == "unverified"
        assert entry["proof_origin"] == "unknown"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_staged_proof_entries_are_labeled_honestly(self):
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="proof-stage-1",
        )
        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]
        assert entry["run_id"] == "proof-stage-1"
        assert entry["kill_rate"] == 22.0
        assert entry["deliveries"] == 5
        assert entry["nanobot_count"] == 3
        assert entry["steps"] == 10
        # Mock proof bundles (is_mock=True) must be prefixed, not claim proof_staged.
        assert entry["trust_tier"] == "mock_proof_staged"
        assert entry["proof_origin"] == "sp1-groth16-adapter"
        assert entry["proof_ok"] is False
        # Mock proofs must not inflate the staged-proof count.
        assert result["summary"]["staged_proof_entries"] == 0

    def test_submitted_onchain_entries_remain_staged_until_chain_accepts(self):
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="proof-submitted",
        )
        artifact["proof_lifecycle"]["stage"] = "submitted_onchain"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["verified_onchain"] is False
        assert entry["proof_stage"] == "submitted_onchain"
        # Mock proof gets mock_ prefix
        assert entry["trust_tier"] == "mock_proof_staged"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_normalize_leaderboard_artifact_flattens_attestation_bundle(self):
        artifact = create_proof_bundle(
            config={"tumor_radius": 90, "nanobot_count": 7, "steps": 12},
            metrics={"kill_rate": 31.5, "deliveries": 2},
            run_id="proof-stage-2",
        )
        normalized = normalize_leaderboard_artifact(artifact)
        assert normalized["run_id"] == "proof-stage-2"
        assert normalized["metrics"]["kill_rate"] == 31.5
        assert normalized["config"]["nanobot_count"] == 7
        assert normalized["proof_bundle"]["proof_origin"] == "sp1-groth16-adapter"
        assert normalized["verification_status"]["integrity_ok"] is True

    def test_unknown_proof_origin_cannot_achieve_staged_trust(self):
        """A proof bundle with a fabricated proof_origin must not qualify for proof_staged."""
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="fabricated-origin",
        )
        artifact["proof_bundle"]["proof_origin"] = "fabricated-origin"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] != "proof_staged"
        assert entry["proof_origin"] == "unknown"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_mismatched_program_version_cannot_achieve_staged_trust(self):
        """A proof bundle with a non-canonical program_version must not qualify for proof_staged."""
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="stale-program-version",
        )
        artifact["proof_bundle"]["program_version"] = "stale-v0"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] != "proof_staged"
        assert entry["proof_origin"] == "unknown"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_transport_metadata_protocol_drift_cannot_achieve_staged_trust(self):
        """Leaderboard uses the same schema validator as verify.py for transport metadata."""
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="stale-transport-protocol",
        )
        artifact["proof_bundle"]["transport_metadata"]["protocol_version"] = "stale-protocol-v0"

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] != "proof_staged"
        assert entry["proof_origin"] == "unknown"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_transport_mock_flag_mismatch_cannot_achieve_staged_trust(self):
        """Top-level and transport_metadata mock flags must agree before leaderboard promotion."""
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="mock-flag-mismatch",
        )
        artifact["proof_bundle"]["is_mock"] = False

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] != "proof_staged"
        assert entry["proof_origin"] == "unknown"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_malformed_transport_metadata_cannot_achieve_staged_trust(self):
        """Missing transport metadata is not sufficient for proof_staged leaderboard ranking."""
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="missing-transport-metadata",
        )
        artifact["proof_bundle"].pop("transport_metadata")

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] != "proof_staged"
        assert entry["proof_origin"] == "unknown"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_mock_proof_bundle_gets_mock_prefix_not_proof_staged(self):
        """Mock proof bundles (is_mock=True) must get mock_proof_staged, not proof_staged."""
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="mock-proof-tier",
        )
        # create_proof_bundle always sets is_mock=True
        assert artifact["proof_bundle"]["is_mock"] is True

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] == "mock_proof_staged"
        assert entry["trust_tier"] != "proof_staged"
        assert result["summary"]["staged_proof_entries"] == 0

    def test_non_mock_proof_bundle_gets_genuine_proof_staged(self):
        """A proof bundle with is_mock=False should get proof_staged, not mock_proof_staged."""
        artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 22.0, "deliveries": 5},
            run_id="real-proof-tier",
        )
        mark_bundle_as_real_adapter_proof(artifact)

        result = build_leaderboard([artifact])
        entry = result["leaderboard"][0]

        assert entry["trust_tier"] == "proof_staged"
        assert result["summary"]["staged_proof_entries"] == 1

    def test_mock_proof_ranks_below_real_proof_at_same_kill_rate(self):
        """At equal kill_rate, a mock_proof_staged entry must rank below proof_staged."""
        mock_artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 50.0, "deliveries": 5},
            run_id="mock-at-50",
        )
        real_artifact = create_proof_bundle(
            config={"tumor_radius": 100, "nanobot_count": 3, "steps": 10},
            metrics={"kill_rate": 50.0, "deliveries": 5},
            run_id="real-at-50",
        )
        mark_bundle_as_real_adapter_proof(real_artifact)

        result = build_leaderboard([mock_artifact, real_artifact])
        ranked = result["leaderboard"]

        assert ranked[0]["run_id"] == "real-at-50"
        assert ranked[0]["trust_tier"] == "proof_staged"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["run_id"] == "mock-at-50"
        assert ranked[1]["trust_tier"] == "mock_proof_staged"
        assert ranked[1]["rank"] == 2

    def test_all_mock_trust_tiers_rank_above_no_effect(self):
        """Every mock_ prefixed trust tier must rank above no_effect at equal kill_rate.

        Before the TRUST_TIER_RANK dict was extended, mock_ tiers not present
        (mock_replay_checked, mock_integrity_checked, mock_unverified) would
        silently default to -1, placing them BELOW no_effect (rank 0) — a
        metric-accounting invariant violation. This test locks the fix.
        """
        from chain.leaderboard import TRUST_TIER_RANK
        no_effect_rank = TRUST_TIER_RANK.get("no_effect", 0)
        mock_tiers = {k: v for k, v in TRUST_TIER_RANK.items() if k.startswith("mock_")}
        assert mock_tiers, "at least one mock_ tier must exist in TRUST_TIER_RANK"
        for tier, rank in mock_tiers.items():
            assert rank >= no_effect_rank, (
                f"mock_ tier {tier!r} has rank {rank} < no_effect rank {no_effect_rank}; "
                "mock proofs must not silently fall below no_effect in ranking"
            )

    def test_mock_proof_staged_ranks_above_unverified_at_equal_kill_rate(self):
        """A mock_proof_staged entry must rank above unverified when kill_rates match."""
        from chain.leaderboard import rank_by_kill_rate
        entries = [
            {"kill_rate": 50.0, "trust_tier": "unverified", "run_id": "unverified-50"},
            {"kill_rate": 50.0, "trust_tier": "mock_proof_staged", "run_id": "mock-proof-50"},
        ]
        ranked = rank_by_kill_rate(entries)
        assert ranked[0]["run_id"] == "mock-proof-50"
        assert ranked[1]["run_id"] == "unverified-50"
