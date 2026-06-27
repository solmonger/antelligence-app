"""Verify proof bundle protocol invariant: proof bytes integrity."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.proof_adapter import create_proof_bundle
from chain.verify import verify_proof_bundle_schema

def test_proof_bundle_rejects_empty_proof_bytes():
    bundle = create_proof_bundle(
        config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
        metrics={"kill_rate": 0.0, "deliveries": 0},
        run_id="proof-verify-empty-bytes",
    )
    # Tamper: set proof bytes to empty
    bundle["proof_bundle"]["proof_bytes"] = "0x"

    result = verify_proof_bundle_schema(bundle)

    # Assert verification actually catches this
    assert result["ok"] is False
    assert any(check["check"] == "proof_bytes_non_empty" and check["ok"] is False for check in result["checks"])

if __name__ == "__main__":
    bundle = create_proof_bundle(
        config={"tumor_radius": 40, "nanobot_count": 2, "steps": 2},
        metrics={"kill_rate": 0.0, "deliveries": 0},
        run_id="proof-verify-empty-bytes-main",
    )
    bundle["proof_bundle"]["proof_bytes"] = "0x"
    result = verify_proof_bundle_schema(bundle)
    print(f"Test result: {result['ok']}, Checks failed: {[c['check'] for c in result['checks'] if not c['ok']]}")
