import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.leaderboard import rank_by_kill_rate

class TestLeaderboardStability:
    def test_leaderboard_rank_stability_on_empty_metrics(self):
        # Regression: leaderboard ranking should be stable even when metrics are missing/malformed.
        entries = [
            {"run_id": "a", "kill_rate": 0.0, "trust_tier": "unverified"},
            {"run_id": "b", "kill_rate": 0.0, "trust_tier": "integrity_checked"},
        ]
        ranked = rank_by_kill_rate(entries)
        # Higher trust tier should be ranked higher
        assert ranked[0]["run_id"] == "b"
        assert ranked[1]["run_id"] == "a"
