
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add backend to sys.path so we can import the bot's dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from scripts.attestation_bot import run_spot_checks


def test_attestation_bot_dry_run(tmp_path: Path):
    """
    Test that the attestation bot can process a dummy JSON input
    and report results without needing real blockchain/IPFS side effects.
    """
    # 1. Create a dummy input file
    dummy_input_path = tmp_path / "dummy_results.json"
    dummy_data = {
        "config": {
            "steps": 50,
            "n_bots": 5,
            "with_queen": True,
            "episode_length": 10
        },
        "results": [
            {
                "seed": 42,
                "patient": "test_patient_1",
                "kill_rate_pct": 10.0,
                "deliveries": 5,
                "tumor_radius": 150
            },
            {
                "seed": 123,
                "patient": "test_patient_2",
                "kill_rate_pct": 20.0,
                "deliveries": 10,
                "tumor_radius": 150
            }
        ]
    }
    
    dummy_input_path.write_text(json.dumps(dummy_data))

    try:
        # 2. Run the core logic without replaying live simulations or touching
        # blockchain/IPFS side effects.
        def fake_spot_check(original, steps, n_bots, with_queen, episode_length):
            return {
                "kill_rate_pct": original["kill_rate_pct"],
                "kills": 0,
                "deliveries": original["deliveries"],
            }

        with patch("scripts.attestation_bot.spot_check_run", side_effect=fake_spot_check):
            result = run_spot_checks(dummy_data, sample_pct=100, tolerance_pct=5.0, verbose=False)

        # 3. Assertions
        assert result["ok"] is True, f"Dry run failed: {result}"
        assert result["checked"] == 2
        assert result["passed"] == 2
        
        print("Dry run test passed successfully!")

    finally:
        # Cleanup
        if dummy_input_path.exists():
            dummy_input_path.unlink()

if __name__ == "__main__":
    test_attestation_bot_dry_run()
