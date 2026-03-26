"""Unit tests for leaderboard service."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.ipfs import create_simulation_artifact
from chain.leaderboard import rank_by_kill_rate, build_leaderboard


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
