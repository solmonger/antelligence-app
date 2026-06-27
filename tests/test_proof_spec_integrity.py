
import pytest
from typing import Any
from backend.chain.proof_spec import build_public_values_payload, PUBLIC_VALUES_FIELDS

def test_public_values_payload_robustness():
    # Verify that payload construction rejects non-finite or negative inputs explicitly
    config_hash = "0x" + "a" * 64

    # Test valid
    payload = build_public_values_payload(config_hash, 100, 1000, 50, 200)
    assert payload["kill_rate_bps"] == 100
    assert payload["nanobot_count"] == 1000
    assert payload["tumor_radius"] == 50
    assert payload["steps"] == 200

    # Test invalid values (negative or too large)
    with pytest.raises(ValueError, match="must be between 0 and 10000"):
        build_public_values_payload(config_hash, -1, 1000, 50, 200)

    with pytest.raises(ValueError, match="must be between 0 and 4294967295"):
        build_public_values_payload(config_hash, 100, -1, 50, 200)

    # Test non-finite value
    with pytest.raises(ValueError, match="must be finite"):
        build_public_values_payload(config_hash, 100, float('inf'), 50, 200) # type: ignore
