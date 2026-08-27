from scripts.generate_queen_dpo_pairs import generate_pairs


def _entry(run_id: str, score: float, trust_tier: str) -> dict:
    return {
        "run_id": run_id,
        "kill_rate": score,
        "deliveries": 1,
        "nanobot_count": 10,
        "steps": 100,
        "trust_tier": trust_tier,
        "pheromone_params": {
            "trail_decay": score / 100,
            "recruitment_diffusion": 1e-6,
        },
    }


def test_generate_pairs_excludes_unverified_runs_from_both_sides():
    pairs = generate_pairs(
        [
            _entry("unverified-high", 99.0, "unverified"),
            _entry("replay-good", 70.0, "replay_checked"),
            _entry("proof-good", 40.0, "proof_staged"),
        ]
    )

    assert len(pairs) == 1
    assert pairs[0]["meta"]["chosen_run_id"] == "replay-good"
    assert pairs[0]["meta"]["rejected_run_id"] == "proof-good"
    assert pairs[0]["meta"]["chosen_trust_tier"] == "replay_checked"
    assert pairs[0]["meta"]["rejected_trust_tier"] == "proof_staged"


def test_generate_pairs_requires_two_training_eligible_runs():
    pairs = generate_pairs(
        [
            _entry("integrity-only", 80.0, "integrity_checked"),
            _entry("unverified", 30.0, "unverified"),
        ]
    )

    assert pairs == []
