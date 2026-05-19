
import pytest
from backend.chain.verify import verify_metrics_tolerance

def test_verify_metrics_tolerance_missing_metric_behavior():
    # Verify that a missing metric is treated as failure with missing_recomputed_metric reason
    claimed = {"kill_rate": 0.5, "extra": 100}
    recomputed = {"kill_rate": 0.5}

    result = verify_metrics_tolerance(claimed, recomputed)

    assert result["ok"] is False
    # Check that extra is the failed check
    failed_checks = [c for c in result["checks"] if not c["ok"]]
    assert len(failed_checks) == 1
    assert failed_checks[0]["metric"] == "extra"
    assert failed_checks[0]["reason"] == "missing_recomputed_metric"
