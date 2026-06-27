
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from chain.proof_spec import build_public_values_payload

def test_protocol_invariants_nanobot_floor():
    # nanobot_count=0 is valid per the current spec (min_value=0 in _uint32_public_value).
    # This test anchors the current protocol behaviour.
    payload = build_public_values_payload(
        config_hash="0x" + ("ab" * 32),
        kill_rate_bps=1250,
        nanobot_count=0,
        tumor_radius=100,
        steps=20,
    )
    assert payload["nanobot_count"] == 0
