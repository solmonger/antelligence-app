"""A1 diagnostics helper: pure-function checks on synthetic data (no run directory, no MRI, no truth)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import v42_diagnose as D  # noqa: E402
from antelligence.kernel.colony.quorum import ZoneSpec  # noqa: E402
from antelligence.worlds.tumor_mri.colony_policies import NEED_NORM  # noqa: E402

HALF = {"dosed": 30, "dose_ok": 40, "slow_uptake": 60, "washout": 60, "tumour_here": 40, "not_tumour": 80, "need_more": 15}


def _decision(tick, ant, target="A", release=0, amount=3, tvox=None, job="dose", reason="ok"):
    w = {"target": target, "release": release, "amount": amount}
    if tvox is not None:
        w["target_vox"] = list(tvox)
    return {"tick": tick, "ant": ant, "kind": "decision", "action": "decide",
            "params": {"world": w, "meta": {"job": job, "reason": reason, "stimuli": {}}, "emit": [], "ask": None}}


def test_decisions_and_signature():
    col = {"action_log": [_decision(0, 0), _decision(5, 1, tvox=(1, 2, 3)), {"kind": "answer", "tick": 5, "ant": 2}]}
    d = D.decisions(col)
    assert set(d) == {(0, 0), (5, 1)}
    assert D.sig(d[(5, 1)]) == ("A", 0, 3, (1, 2, 3))
    assert D.sig(d[(0, 0)])[3] is None


def test_positions_and_looks_read_the_logged_turns():
    col = {"reading_logs": {"0": [{"kind": "dose", "tick": 0, "pos_vox": [1, 1, 1], "value": 0.0},
                                  {"kind": "look", "tick": 0, "pos_vox": [1, 1, 1], "value": 0.7}]}}
    assert D.positions(col) == {(0, 0): (1, 1, 1)}
    assert D.looks(col) == {(0, 0): 0.7}


def test_latest_need_picks_the_latest_queen_step_not_after_tick():
    by = {0: {1: "high"}, 30: {1: "low"}, 60: {1: "none"}}
    assert D.latest_need(by, 29) == {1: "high"}
    assert D.latest_need(by, 30) == {1: "low"}
    assert D.latest_need({}, 5) == {}


def test_predicted_top_skips_closed_zones_and_uses_need_over_distance():
    near = SimpleNamespace(region_id=1, p_tumour=0.9, centroid_vox=(0, 0, 10))
    far = SimpleNamespace(region_id=2, p_tumour=0.9, centroid_vox=(0, 0, 40))
    case = SimpleNamespace(cands={"A": near, "B": far})
    pos = (0, 0, 0)
    assert D.predicted_top(case, pos, {}, []) == "A"
    assert D.predicted_top(case, pos, {}, [1]) == "B"
    assert D.predicted_top(case, pos, {1: "none", 2: "high"}, []) == "B"
    assert NEED_NORM["none"] == 0.0 and D.predicted_top(case, pos, {1: "none", 2: "none"}, []) is None


def test_replay_quorum_closes_a_zone_after_two_authors_and_checks_final_status():
    zone = ZoneSpec(7, (10.0, 10.0, 10.0), lambda p: 2.0 * ((p[0] - 10) ** 2 + (p[1] - 10) ** 2 + (p[2] - 10) ** 2) ** 0.5 <= 12.0)
    case = SimpleNamespace(zones=[zone])
    pos_log = {str(a): [{"kind": "dose", "tick": t, "pos_vox": [10, 10, 10], "value": 0.0} for t in range(0, 20, 5)] for a in range(8)}
    marks = [{"ant": 0, "mark_kind": "dose_ok", "pos_vox": [10, 10, 10], "tick": 0, "amount_milli": 1000, "reading_ppm": 0},
             {"ant": 1, "mark_kind": "dose_ok", "pos_vox": [10, 10, 11], "tick": 5, "amount_milli": 1000, "reading_ppm": 0}]
    col = {"arm": "S-marks", "params": {"read_radius_mm": 6.0, "confirm_radius_mm": 4.0, "half_life": HALF, "quorum_k": 2},
           "marks": marks, "reading_logs": pos_log, "region_status": {"7": "done"}}
    rp = D.replay(col, case)
    assert rp["events"][0][1:] == (7, "done") and rp["events"][0][0] == 6
    assert rp["status"][10][7] == "done" and rp["status"][5][7] == "open"
    assert rp["status_ok"] and rp["marks_ok"]
    # a foreign mark is visible to ant 2 at tick 10 (mark by ant 0 at tick 0 is within 6 mm of (10, 10, 10))
    assert D.foreign(rp["rows"][(10, 2)], 2)
    assert not D.foreign(rp["rows"][(10, 0)], 0) or all(set(r["authors"]) != {0} for r in D.foreign(rp["rows"][(10, 0)], 0))


def test_main_refuses_an_output_path_outside_the_research_folder(tmp_path):
    with pytest.raises(SystemExit):
        D.main(["--run-dir", str(tmp_path), "--cases-root", str(tmp_path), "--out", str(tmp_path / "x.json")])
