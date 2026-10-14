"""The scripted queen: allotment, delivered share, need bands, alarms, fixed note, and what she reads."""

import pytest

from antelligence.experiments.v42_harness import make_board
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen, need_band
from tests.v4_2.q1.helpers import REGIONS


def queen(**kw):
    return ScriptedQueen(REGIONS, total_drug=1000.0, n_squads=8, voxel_mm=2.0, **kw)


def dosed(tick, pos, milli=1000, ant=0):
    return {"subtype": "mark", "mark_kind": "dosed", "ant": ant, "pos_vox": list(pos), "amount_milli": milli, "reading_ppm": 0, "tick": tick}


def mark(kind, tick, pos, ant=0):
    return {"subtype": "mark", "mark_kind": kind, "ant": ant, "pos_vox": list(pos), "amount_milli": 1000, "reading_ppm": 0, "tick": tick}


def call(q, tick, marks=(), events=()):
    return q(tick, {"tick": tick, "regions": [{"region_id": r.region_id, "centroid_vox": list(r.centroid_vox)} for r in REGIONS],
                    "marks": list(marks), "quorum_events": list(events)})


def test_bands():
    assert [need_band(x) for x in (0.0, 0.0499, 0.05, 0.2499, 0.25, 0.5999, 0.6, 1.0)] == \
        ["none", "none", "low", "low", "medium", "medium", "high", "high"]


def test_tick_zero_need_is_scan_mass_normalised_by_its_maximum():
    out = call(queen(), 0)
    rows = {r["region_id"]: r for r in out["need_rows"]}
    assert rows[0]["need_milli"] == 1000 and rows[1]["need_milli"] == round(1000 * (0.7 * 3000) / (0.8 * 4000))
    assert rows[0]["delivered_milli"] == 0 and out["zone_need"] == {0: "high", 1: "high"}
    assert set(out) == {"zone_need", "colony_note", "need_rows"}                  # never options, routes or gates


def test_allotment_and_delivered_share():
    q = queen()
    mass = {0: 0.8 * 4000, 1: 0.7 * 3000}
    assert q.allotment(0) == pytest.approx(1000 * mass[0] / sum(mass.values()))
    share = q.delivered_share([dosed(0, (10, 10, 10), 1000)])                      # one full squad share = 125 drug
    assert share[0] == pytest.approx(125.0 / q.allotment(0)) and share[1] == 0.0
    full = q.delivered_share([dosed(0, (10, 10, 10), 1000, ant=a) for a in range(8)])
    assert full[0] == 1.0                                                          # capped at 1


def test_delivered_zone_drops_to_none_and_other_zone_becomes_the_top_need():
    q = queen()
    marks = [dosed(5, (10, 10, 10), 1000, ant=a) for a in range(5)]               # 625 drug > the 603 allotment
    out = call(q, 30, marks)
    rows = {r["region_id"]: r for r in out["need_rows"]}
    assert rows[0]["need"] == "none" and rows[0]["delivered_milli"] == 1000
    assert rows[1]["need"] == "high" and rows[1]["need_milli"] == 1000             # renormalised by the new maximum


def test_marks_from_the_same_tick_or_later_are_not_read():
    out = call(queen(), 30, [dosed(30, (10, 10, 10), 1000)])
    assert {r["region_id"]: r["delivered_milli"] for r in out["need_rows"]} == {0: 0, 1: 0}


def test_stall_alarm_after_two_unchanged_steps_and_board_entry():
    board = make_board("S-marks", "phantom", 1, "synthetic", {}, {})
    q = queen(board=board)
    call(q, 0), call(q, 30)
    assert q.alarms == []
    call(q, 60)
    assert [a["note"] for a in q.alarms] == ["stall"] and q.alarms[0]["kind"] == "other"
    [e] = board.by_type("alarm")
    assert e.payload["note"] == "stall" and e.payload["kind"] == "other"
    q2 = queen()
    call(q2, 0), call(q2, 30, [dosed(5, (10, 10, 10), 300)]), call(q2, 60, [dosed(5, (10, 10, 10), 300)]), call(q2, 90, [dosed(5, (10, 10, 10), 300)])
    assert [a["tick"] for a in q2.alarms] == [90]                                  # progress at 30 resets the count


def test_dispute_alarm_fires_once_per_dispute():
    q = queen()
    pair = [mark("tumour_here", 5, (10, 10, 10), ant=1), mark("not_tumour", 8, (11, 10, 10), ant=2)]
    for t in (0, 30, 60, 90):
        call(q, t, [m for m in pair if m["tick"] < t])
    disputes = [a for a in q.alarms if a["note"].startswith("dispute")]
    assert [(a["note"], a["tick"]) for a in disputes] == [("dispute:0", 60)]
    far = [mark("tumour_here", 5, (10, 10, 10), ant=1), mark("not_tumour", 8, (30, 30, 30), ant=2)]
    q = queen()
    for t in (30, 60, 90):
        call(q, t, far)
    assert not [a for a in q.alarms if a["note"].startswith("dispute")]            # 4 mm apart is a dispute; far apart is not


def test_note_is_a_fixed_template_and_deterministic():
    a, b = call(queen(), 0), call(queen(), 0)
    assert a == b and a["colony_note"].startswith("Colony note, tick 0: 0% of the drug allotment delivered.")
    assert len(a["colony_note"]) <= 200
    ev = [{"status": "done"}, {"status": "not_tumour"}, {"status": "done"}]
    assert "Zones done 2, not tumour 1." in call(queen(), 30, events=ev)["colony_note"]


def test_queen_reads_only_inputs_and_records_the_keys_she_was_given():
    q = queen()
    q(0, {"tick": 0, "regions": []})
    assert q.reads == [(0, ("regions", "tick"))]
    assert not hasattr(q, "world") and not hasattr(q, "scorer")
