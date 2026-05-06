
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.ipfs import create_simulation_artifact
from chain.leaderboard import build_leaderboard

def test_onchain_verification_propagation():
    """Verify that onchain_ok status propagates to verified_onchain and trust_perm."""
    artifact = create_simulation_artifact(
        config={"tumor_radius": 100},
        metrics={"kill_rate": 50.0},
        run_id="onchain-test"
    )
    # Inject verification_status
    artifact["verification_status"] = {"onchain_ok": True}
    
    result = build_leaderboard([artifact])
    entry = result["leaderboard"][0]
    
    assert entry["verified_onchain"] is True
    assert entry["trust_tier"] == "verified_onchain"

def test_replay_verification_propagation():
    """Verify that replay_ok status propagates to trust_tier."""
    artifact = create_simulation_artifact(
        config={"tumor_radius": 100},
        metrics={"kill_rate": 50.0},
        run_id="replay-test"
    )
    # Inject verification_status
    artifact["verification_status"] = {"replay_ok": True}
    
    result = build_leaderboard([artifact])
    entry = result["leaderboard"][0]
    
    assert entry["replay_ok"] is True
    assert entry["trust_tier"] == "replay_checked"
