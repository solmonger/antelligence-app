"""Brief S16: worker override rate, its P2 reporting rule, the P1 gap secondary, S9 restriction, and the gap gate
being gone. Fake backends only; no network."""

from __future__ import annotations

import asyncio

import pytest

from antelligence.experiments import v41_analysis as A
from antelligence.experiments import v41_cells, v41_override as O
from antelligence.kernel.decision import Answer
from antelligence.providers.decision_backends import FakeBackend, hashed_answers
from antelligence.experiments.v41_harness import BONSAI, D1, FW, Q4, Q9, Worker
from tests.v4_1.analysis.conftest import load_script, make_cell, synth
from tests.v4_1.harness.conftest import CASE_DEV, build_rig, d1_responder

CANDS = [{"label": "A", "region_id": 7}, {"label": "B", "region_id": 9}]


# --- resolver ----------------------------------------------------------------


def test_resolve_target_region_rules():
    r = O.resolve_target_region
    assert r("A", None, CANDS) == 7 and r("B", "A", CANDS) == 9
    assert r("S", "B", CANDS) == 9            # stay = the current target's region
    assert r("S", None, CANDS) is None and r("S", "S", CANDS) is None   # stay with no target: no region
    assert r("X", "A", CANDS) == "X"
    assert r("Z", None, CANDS) is None        # unknown letter


def test_is_override_semantics():
    f = O.is_override
    assert f("A", None, CANDS, 7) is False    # picked the assigned region
    assert f("B", None, CANDS, 7) is True
    assert f("S", "A", CANDS, 7) is False     # kept the queen's target
    assert f("S", "B", CANDS, 7) is True      # kept a different target
    assert f("S", None, CANDS, 7) is True     # stayed with nothing: not following the assignment
    assert f("X", "A", CANDS, 7) is True
    assert f("A", None, CANDS, None) is None  # no assignment: not comparable
    assert O.queen_region({"region_ids": [9, 7]}) == 9 and O.queen_region(None) is None and O.queen_region({"region_ids": []}) is None
    assert O.override_rate(1, 4) == 0.25 and O.override_rate(0, 0) is None


# --- cells -------------------------------------------------------------------


def test_cell_schema_has_override_fields_and_checks_consistency():
    ok = make_cell("SW-B", "c", 0, 0.5)
    assert v41_cells.validate_cell(ok) == []
    bad = make_cell("SW-B", "c", 0, 0.5, override_count=41, override_decisions=40)
    assert any("override_count" in e for e in v41_cells.validate_cell(bad))
    assert "override_decisions" in v41_cells.FIELD_NAMES and "override_count" in v41_cells.FIELD_NAMES


# --- analysis ----------------------------------------------------------------


def _cells(sw_b_rate, fw_rate, n=10):
    cells = []
    for i in range(n):
        cells.append(make_cell("SW-B", f"c{i}", 0, 0.6, override_decisions=100, override_count=int(sw_b_rate * 100)))
        cells.append(make_cell("FW", f"c{i}", 0, 0.6, override_decisions=100, override_count=int(fw_rate * 100)))
        cells.append(make_cell("S", f"c{i}", 0, 0.3, override_decisions=0, override_count=0))
    return cells


def test_worker_override_is_pooled_per_arm():
    w = A.worker_override(_cells(0.05, 0.20))["arms"]
    assert w["SW-B"] == {"overrides": 50, "decisions": 1000, "rate": 0.05}
    assert w["FW"]["rate"] == 0.2
    assert w["S"] == {"overrides": 0, "decisions": 0, "rate": None}   # no assignments: no rate, not 0


@pytest.mark.parametrize("b,f,followed", [(0.05, 0.05, True), (0.05, 0.20, False), (0.20, 0.05, False), (0.20, 0.30, False),
                                          (0.09, 0.09, True), (0.10, 0.05, False)])  # strictly below 10%
def test_p2_reading_rule(b, f, followed):
    res = A.p2_reading(_cells(b, f))
    assert res["followed_the_queen"] is followed
    assert res["reading"] == (A.P2_FOLLOWED_LABEL if followed else A.P2_FRONTIER_LABEL)
    assert A.P2_FOLLOWED_LABEL == "workers followed the queen" and A.P2_FRONTIER_LABEL == "small matches frontier"


def test_p2_reading_without_data_gives_no_label():
    cells = [make_cell("SW-B", "c", 0, 0.5, override_decisions=0, override_count=0), make_cell("FW", "c", 0, 0.5)]
    res = A.p2_reading(cells)
    assert res["followed_the_queen"] is None and res["reading"] is None and res["reading_note"]


def test_p2_and_analyze_carry_the_reading():
    cells = synth(n_cases=12, effect=0.05)
    p2 = A.p2(cells, A.p1(cells))
    assert p2["reading"] == A.P2_FOLLOWED_LABEL     # make_cell default: 2 of 40 = 5% in both arms
    res = A.analyze(cells)
    assert res["p2"]["reading"] == A.P2_FOLLOWED_LABEL
    assert res["worker_override"]["arms"]["SW-B"]["rate"] == pytest.approx(0.05)
    assert res["thresholds"]["override_followed_max"] == 0.10


# --- gap secondary and S9 ----------------------------------------------------


def _gap_cells():
    """12 cases; the first 5 have S' - S = 0.30, the rest 0.05. SW-B beats SW-noBoard by 0.1 only on the gap cases."""
    cells = []
    for i in range(12):
        gap = i < 5
        base = 0.4
        for seed in (0, 1):
            cells += [make_cell("S", f"c{i}", seed, base), make_cell("S-prime", f"c{i}", seed, base + (0.30 if gap else 0.05)),
                      make_cell("SW-noBoard", f"c{i}", seed, base + 0.05 + 0.01 * i),
                      make_cell("SW-B", f"c{i}", seed, base + 0.05 + 0.01 * i + (0.10 if gap else 0.0))]
    return cells


def test_gap_secondary_restricts_p1_to_gap_cases_and_reports_the_share():
    cells = _gap_cells()
    g = A.gap_cases(cells)
    assert g["n_cases"] == 12 and g["n_cases_gap"] == 5 and g["share"] == pytest.approx(5 / 12)
    sec = A.p1_gap_secondary(cells)
    assert sec["secondary"] is True and sec["n_cases"] == 5 and sec["n_cases_gap"] == 5 and sec["gap_share"] == pytest.approx(5 / 12)
    assert sec["mean_diff"] == pytest.approx(0.10)
    assert A.p1(cells)["mean_diff"] == pytest.approx(0.10 * 5 / 12)   # the primary still uses all cases


def test_gap_threshold_is_inclusive_and_case_level():
    cells = [make_cell("S", "a", 0, 0.5), make_cell("S-prime", "a", 0, 0.65),     # 0.15: in
             make_cell("S", "b", 0, 0.5), make_cell("S-prime", "b", 0, 0.64)]     # 0.14: out
    assert A.gap_cases(cells)["cases"] == ["a"]


def test_gap_secondary_with_no_gap_cases_is_empty_not_an_error():
    cells = [make_cell("S", "a", 0, 0.5), make_cell("S-prime", "a", 0, 0.52), make_cell("SW-B", "a", 0, 0.6),
             make_cell("SW-noBoard", "a", 0, 0.5)]
    sec = A.p1_gap_secondary(cells)
    assert sec["n_cases"] == 0 and sec["passed"] is False and sec["gap_share"] == 0.0


def test_s9_stays_restricted_to_gap_cases():
    cells = _gap_cells()
    s9 = A.secondary(cells)["S9"]
    assert s9["min_gap"] == 0.15 and s9["n_cases_gap"] == 5
    assert s9["arms"]["SW-B"]["n_cases"] == 5
    # SW-B - S on the gap cases: mean(0.05 + 0.01 i + 0.10), i in 0..4 = 0.17; S' - S = 0.30
    assert s9["arms"]["SW-B"]["ratio"] == pytest.approx(0.17 / 0.30)


def test_analyze_reports_gap_share_and_secondary():
    res = A.analyze(_gap_cells())
    assert res["gap_share"]["share"] == pytest.approx(5 / 12)
    assert res["p1_gap_secondary"]["n_cases_gap"] == 5
    # the secondary never feeds P2's gate
    assert res["p2"]["p1_passed"] == res["p1"]["passed"]


# --- prereg text and report --------------------------------------------------


def test_prereg_template_carries_the_rules():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[3] / "docs" / "v4.1" / "PREREGISTRATION.template.md").read_text()
    assert '"workers followed the queen", not as "small matches frontier"' in text
    assert "below 10% (strictly) in both SW-B and FW" in text
    assert "P1 gap secondary" in text and "restricted to the evaluation cases with S' - S >= 0.15" in text
    assert "not a gate" in text
    assert "80% of cases" in text and "not part of this preregistration" in text


def test_report_shows_reading_and_gap_secondary():
    rep = load_script("v41_report")
    res = A.analyze(_gap_cells())
    md = rep.render(res)
    assert "Worker override rate" in md and "workers followed the queen" in md
    assert "P1 gap secondary" in md and "results.json#p1_gap_secondary.mean_diff" in md


# --- harness: the diagnostic is computed from real decisions ------------------


def _forced_target_workers(label):
    def respond(req):
        out = hashed_answers(req)
        q = req.questions["target"]
        pick = label if label in q.options else q.options[0]
        probs = {o: (0.9 if o == pick else 0.1 / (len(q.options) - 1)) for o in q.options}
        out["target"] = Answer("choice", pick, probs, 0.9)
        return out

    ws = {}
    for name, model in ((D1, "d1-fake"), (Q4, "q4-fake"), (Q9, "q9-fake"), (BONSAI, "bonsai-fake"), (FW, "fw-fake")):
        inner = d1_responder(0.05, respond) if name == D1 else respond
        ws[name] = Worker(FakeBackend(inner, model=model, name=name))
    return ws


def _run(tmp_path, label):
    rig = build_rig(tmp_path, workers=_forced_target_workers(label))
    return asyncio.run(rig.harness.run("SW-B", CASE_DEV, 1))


def test_harness_explore_always_overrides(tmp_path):
    res = _run(tmp_path, "X")
    cell, log = res.cell, [r for r in res.decision_log if r.get("kind") == "decision"]
    assert cell["override_decisions"] > 0 and cell["override_count"] == cell["override_decisions"]
    assert all(r["override"] is True and r["target_region"] == "X" for r in log if r["override"] is not None)
    assert v41_cells.validate_cell(cell) == []


def test_harness_staying_follows_the_queen(tmp_path):
    res = _run(tmp_path, "S")
    cell = res.cell
    log = [r for r in res.decision_log if r.get("kind") == "decision" and r["override"] is not None]
    assert cell["override_decisions"] == len(log) > 0
    assert cell["override_count"] == sum(1 for r in log if r["override"])
    # first round: each squad adopts the queen's first region, and S keeps it
    first = [r for r in log if r["round"] == 0]
    assert first and all(r["override"] is False and r["target_region"] == r["queen_region"] for r in first)


# --- the gap gate is gone ----------------------------------------------------


def test_tuning_no_longer_gates_on_the_gap_share():
    from antelligence.worlds.tumor_mri import tuning as T
    assert "gap_frac" not in T.STRICT and "gap_frac" not in T.WIDENED and T.STRICT["gap"] == 0.15
    low_gap = {"median_sprime": 0.72, "median_s": 0.40, "gap_frac": 0.0, "n_cases": 10}
    assert T.in_band(low_gap, T.STRICT)                       # a 0% gap share no longer fails the band
    assert T.penalty(low_gap) == pytest.approx(0.03 + 0.0)    # nor adds a penalty
    assert T.penalty({**low_gap, "gap_frac": 1.0}) == T.penalty(low_gap)
    assert T.summarise([0.4, 0.4], [0.6, 0.5])["gap_frac"] == 0.5   # the share is still reported
