

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from chain.verify import _derive_trust_tier

def test_trust_tier_derivation():
    # 1. verified_onchain
    assert _derive_trust_tier({"onchain_ok": True}, None, None) == "verified_onchain"
    
    # 2. proof_staged requires a _valid_staged_proof_bundle (canonical schema) AND
    #    a lifecycle stage in STAGED_PROOF_LIFECYCLE_STAGES. A bare dict like
    #    {"some": "bundle"} does not pass _valid_staged_proof_bundle, so it falls
    #    through to "unverified".
    assert _derive_trust_tier({"onchain_ok": False}, {"some": "bundle"}, None) == "unverified"
    
    # 3. proof_staged (lifecycle stage is proof_generated, but no valid bundle)
    #    Falls through without a valid proof_bundle.
    assert _derive_trust_tier({"onchain_ok": False}, None, {"stage": "proof_generated"}) == "unverified"
    
    # 4. replay_checked
    assert _derive_trust_tier({"onchain_ok": False, "replay_ok": True}, None, None) == "replay_checked"
    
    # 5. integrity_checked
    assert _derive_trust_tier({"onchain_ok": False, "replay_ok": False, "integrity_ok": True}, None, None) == "integrity_checked"
    
    # 6. unverified
    assert _derive_trust_tier({"onchain_ok": False, "replay_ok": False, "integrity_ok": False}, None, None) == "unverified"
    
    print("Trust tier derivation tests passed!")

if __name__ == "__main__":
    test_trust_tier_derivation()
