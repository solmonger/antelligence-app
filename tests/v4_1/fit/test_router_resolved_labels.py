"""Brief S15b: router training labels use S'-resolved matching; the override helper delegates to v41_resolve."""
from __future__ import annotations

import pytest

from antelligence.experiments import v41_fit, v41_override, v41_resolve
from antelligence.experiments.v41_router import FEATURE_NAMES, ROUTER_MODELS

CHEAP, MID = ROUTER_MODELS[0], ROUTER_MODELS[1]
NF = len(FEATURE_NAMES)
SP = {"target": "B", "release": "yes", "amount": "3"}
SPR = {"target": "r20"}


def _d(target, release=1, amount=3):
    return {"target": target, "release": release, "amount": amount}


def _s(decisions, tmap, sp=SP, spr=SPR, stale=False, src="exact"):
    decisions = {m: decisions.get(m, _d("A", 0, 1)) for m in ROUTER_MODELS}
    return v41_fit.Sample("SW-B", "c", {"target": {}, "release": {}, "amount": {}}, {}, tuple(float(i) for i in range(NF)), decisions, sp,
                          tmap, spr, src, stale)


ON_B = {"A": "r10", "B": "r20", "S": "r20", "X": "explore"}    # current target is S''s region
OFF_B = {"A": "r10", "B": "r20", "S": "r10", "X": "explore"}


def test_s_answer_that_resolves_to_sprimes_region_is_credited():
    s = _s({MID: _d("S")}, ON_B)
    assert v41_fit.resolved_matches(s, MID) == {"release": True, "amount": True, "target": True}
    assert v41_fit.resolved_matches(_s({MID: _d("S")}, OFF_B), MID)["target"] is False


def test_router_label_is_the_cheapest_best_resolved_match():
    # raw letters say neither model matches B; resolved, only MID's S does
    s = _s({MID: _d("S")}, ON_B)
    _, rows = v41_fit.router_rows([s])
    assert [r.label for r in rows] == [MID]
    # a tie goes to the cheaper model
    t = _s({CHEAP: _d("B"), MID: _d("S")}, ON_B)
    assert v41_fit.router_rows([t])[1][0].label == CHEAP


def test_lower_and_upper_bounds_differ_only_on_undecidable_derived_rows():
    derived = _s({MID: _d("S", 0, 1)}, {"S": "e30:D", "B": "e60:B"}, spr={"target": "e60:B"}, stale=True, src="derived")
    assert v41_fit.resolved_matches(derived, MID, "lower")["target"] is False
    assert v41_fit.resolved_matches(derived, MID, "upper")["target"] is True
    assert v41_fit.router_rows([derived])[1][0].label == ROUTER_MODELS[0]   # lower: nobody matches -> cheapest
    assert v41_fit.router_rows([derived], bound="upper")[1][0].label == MID
    exact = _s({MID: _d("S")}, ON_B)
    assert (v41_fit.router_rows([exact])[1][0].label == v41_fit.router_rows([exact], bound="upper")[1][0].label)
    with pytest.raises(v41_fit.FitError):
        v41_fit.router_rows([exact], bound="middle")


def test_fit_report_records_the_bound_and_basis():
    samples = [_s({MID: _d("S")}, ON_B) for _ in range(20)] + [_s({CHEAP: _d("B")}, OFF_B) for _ in range(20)]
    _, rep = v41_fit.fit_router(samples, k=3)
    assert rep["bound"] == "lower" and rep["label_basis"] == "sprime_resolved"
    assert rep["resolution"] == {"exact": 40}
    assert rep["sprime_match_share_per_question"][MID]["target"] == pytest.approx(0.5)
    assert rep["sprime_match_share_per_question_raw"][MID]["target"] == 0.0
    _, up = v41_fit.fit_router(samples, k=3, bound="upper")
    assert up["bound"] == "upper"


# --- override helper delegates to the canonical resolver ---------------------------------------------------------------

CANDS = [{"label": "A", "region_id": 7}, {"label": "B", "region_id": 9}]


def test_override_resolver_delegates_to_v41_resolve(monkeypatch):
    calls = []
    real = v41_resolve.resolve_label
    monkeypatch.setattr(v41_override, "resolve_label", lambda *a: calls.append(a) or real(*a))
    assert v41_override.resolve_target_region("S", "B", CANDS) == 9
    assert calls == [("S", {"A": "r7", "B": "r9"}, "r9")]


@pytest.mark.parametrize("label,current,expected", [
    ("A", None, 7), ("B", "A", 9), ("S", "B", 9), ("S", None, None), ("S", "S", None), ("S", "X", None),
    ("X", "A", "X"), ("Z", None, None)])
def test_override_resolver_results_unchanged(label, current, expected):
    assert v41_override.resolve_target_region(label, current, CANDS) == expected
    assert v41_override.is_override(label, current, CANDS, 9) is (expected != 9)
