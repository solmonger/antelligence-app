
import pytest
from backend.chain.proof_spec import build_public_values_payload, encode_public_values_payload

def test_public_values_strict_negative_nanobot_count():
    config_hash = "0x" + "a" * 64
    # The current validation logic should block negative integers
    with pytest.raises(ValueError):
        build_public_values_payload(
            config_hash=config_hash,
            kill_rate_bps=500,
            nanobot_count=-1,
            tumor_radius=50,
            steps=100
        )

def test_public_values_too_large_tumor_radius():
    config_hash = "0x" + "a" * 64
    # Ensure tumor_radius boundary is checked against UINT32_MAX
    too_large = 4294967296
    with pytest.raises(ValueError, match="must be between 0 and 4294967295"):
        build_public_values_payload(
            config_hash=config_hash,
            kill_rate_bps=500,
            nanobot_count=1000,
            tumor_radius=too_large,
            steps=100
        )
