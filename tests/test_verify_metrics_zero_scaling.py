
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from chain.verify import verify_metrics_tolerance

def test_metrics_tolerance_with_zero_recomputed_metric_value_scaling():
    # Non-zero claimed vs zero recomputed: deviation is undefined (invariant violation),
    # so deviation_pct is None and the check fails with the appropriate reason.
    claimed = {"kill_rate": 0.5}
    recomputed = {"kill_rate": 0.0}
    result = verify_metrics_tolerance(claimed, recomputed, tolerance_pct=5.0)
    assert result["ok"] is False
    assert result["checks"][0]["deviation_pct"] is None
    assert result["checks"][0]["reason"] == "nonzero_claimed_vs_zero_recomputed_deviation_undefined"
