
import pytest
from backend.chain.verify import verify_metrics_tolerance

def test_verify_metrics_tolerance_exact_match():
    # Verify exact match cases with zero deviation
    claimed = {"kill_rate": 0.5}
    recomputed = {"kill_rate": 0.5}

    result = verify_metrics_tolerance(claimed, recomputed)

    assert result["ok"] is True
    assert result["checks"][0]["deviation_pct"] == 0.0

def test_verify_metrics_tolerance_small_deviation():
    # Verify small deviation within tolerance
    claimed = {"kill_rate": 0.52}
    recomputed = {"kill_rate": 0.5} # 4% deviation (0.02 / 0.5)

    # default tolerance is 5.0%
    result = verify_metrics_tolerance(claimed, recomputed)
    assert result["ok"] is True
    assert result["checks"][0]["deviation_pct"] == 4.0

def test_verify_metrics_tolerance_large_deviation():
    # Verify large deviation exceeding tolerance
    claimed = {"kill_rate": 0.6}
    recomputed = {"kill_rate": 0.5} # 20% deviation

    result = verify_metrics_tolerance(claimed, recomputed)
    assert result["ok"] is False
    assert result["checks"][0]["deviation_pct"] == 20.0
