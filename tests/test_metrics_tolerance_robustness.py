
import pytest
from backend.chain.verify import verify_metrics_tolerance

def test_metrics_tolerance_nan_handling():
    """Ensure tolerance checker handles NaNs robustly without crashing."""
    claimed = {"value": float("nan")}
    recomputed = {"value": 100.0}

    result = verify_metrics_tolerance(claimed, recomputed)

    assert not result["ok"]
    assert len(result["checks"]) == 1
    assert result["checks"][0]["ok"] is False
    assert result["checks"][0]["reason"] == "non_numeric_computed_or_claimed_metric_value"

def test_metrics_tolerance_infinity_handling():
    """Ensure tolerance checker handles infinity robustly."""
    claimed = {"value": float("inf")}
    recomputed = {"value": 100.0}

    result = verify_metrics_tolerance(claimed, recomputed)

    assert not result["ok"]
    assert result["checks"][0]["reason"] == "non_numeric_computed_or_claimed_metric_value"
