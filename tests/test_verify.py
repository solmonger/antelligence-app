"""Unit tests for verification CLI."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.ipfs import create_simulation_artifact, compute_artifact_hash
from chain.verify import verify_artifact_integrity, verify_metrics_tolerance


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
