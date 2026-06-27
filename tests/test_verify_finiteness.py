
import pytest
from backend.chain.verify import verify_metrics_tolerance

def test_verify_metrics_tolerance_finiteness_check():
    # Verify that non-finite metrics correctly return failure
    claimed = {"kill_rate": float('nan')}
    recomputed = {"kill_rate": 0.5}

    result = verify_metrics_tolerance(claimed, recomputed)

    assert result["ok"] is False
    failed_checks = [c for c in result["checks"] if not c["ok"]]
    assert len(failed_checks) == 1
    assert failed_checks[0]["reason"] == "non_numeric_computed_or_claimed_metric_value"

    claimed = {"kill_rate": float('inf')}
    result = verify_metrics_tolerance(claimed, recomputed)
    assert result["ok"] is False
    assert result["checks"][0]["reason"] == "non_numeric_computed_or_claimed_metric_value"
