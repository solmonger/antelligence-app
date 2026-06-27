
import pytest
from backend.chain.proof_spec import build_public_values_payload

def test_public_values_kill_rate_bps_boundary():
    config_hash = "0x" + "a" * 64
    # KILL_RATE_BPS_MAX is 10000. 10001 should fail.
    with pytest.raises(ValueError, match="must be between 0 and 10000"):
        build_public_values_payload(
            config_hash=config_hash,
            kill_rate_bps=10001,
            nanobot_count=1000,
            tumor_radius=50,
            steps=100
        )

def test_public_values_nanobot_count_boundary():
    config_hash = "0x" + "a" * 64
    # Test UINT32_MAX boundary
    boundary = 4294967295
    # Should succeed at boundary
    payload = build_public_values_payload(
        config_hash=config_hash,
        kill_rate_bps=500,
        nanobot_count=boundary,
        tumor_radius=50,
        steps=100
    )
    assert payload["nanobot_count"] == boundary

    # Should fail at boundary + 1
    with pytest.raises(ValueError, match="must be between 0 and 4294967295"):
        build_public_values_payload(
            config_hash=config_hash,
            kill_rate_bps=500,
            nanobot_count=boundary + 1,
            tumor_radius=50,
            steps=100
        )
