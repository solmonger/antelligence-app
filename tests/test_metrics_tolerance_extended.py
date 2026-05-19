import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from chain.verify import verify_metrics_tolerance

class TestMetricsToleranceExtended:
    def test_missing_recomputed_metric_value_is_not_silently_accepted(self):
        # Current behavior ensures missing recomputed metrics fail.
        # This test ensures we don't accidentally regress to 'ok: True'
        # if the code logic for non-existent dictionary keys changes.
        claimed = {"kill_rate": 45.5}
        recomputed = {}
        result = verify_metrics_tolerance(claimed, recomputed)

        assert result["ok"] is False
        assert any(c["metric"] == "kill_rate" and c["reason"] == "missing_recomputed_metric" for c in result["checks"])
