import pytest
from backend.chain.verify import _is_finite_metric_number

def test_is_finite_metric_number_edge_cases():
    assert _is_finite_metric_number(0) is True
    assert _is_finite_metric_number(1.5) is True
    assert _is_finite_metric_number(float('inf')) is False
    assert _is_finite_metric_number(float('nan')) is False
    assert _is_finite_metric_number(True) is False
    assert _is_finite_metric_number("1.0") is False
    assert _is_finite_metric_number(None) is False

def test_is_finite_metric_number_integers():
    assert _is_finite_metric_number(100) is True
    assert _is_finite_metric_number(-100) is True
