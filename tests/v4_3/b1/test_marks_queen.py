"""B1 sections 4 and 5: the good_uptake kind in the board / marks / colony config, and the queen's uptake scaling (B3)."""

import json

import pytest

from antelligence.kernel.board import MARK_KINDS, ValidationError, validate_payload
from antelligence.kernel.colony.episode import QUEEN_MARK_KINDS, ColonyConfig, QUEEN_INPUT_KEYS
from antelligence.kernel.colony.marks import DEFAULT_HALF_LIFE, MarkError, MarkField, MarkParams, complete_half_life
from antelligence.worlds.tumor_mri.colony_queen import ScriptedQueen
from antelligence.worlds.tumor_mri.colony_state import KIND_ORDER, KIND_WORDS, marks_sentence
from tests.v4_2.q1.helpers import REGIONS

SEVEN = {"dosed": 30, "dose_ok": 40, "slow_uptake": 60, "washout": 60, "tumour_here": 40, "not_tumour": 80, "need_more": 15}


# --- the mark kind --------------------------------------------------------------------------------------------------------

def test_good_uptake_validates_on_the_board():
    p = {"subtype": "mark", "mark_kind": "good_uptake", "ant": 1, "pos_vox": [1, 2, 3], "amount_milli": 1000, "reading_ppm": 800000}
    validate_payload("evidence", p)
    with pytest.raises(ValidationError):
        validate_payload("evidence", {**p, "mark_kind": "great_uptake"})


def test_marks_params_loads_a_seven_kind_half_life():
    mp = MarkParams(half_life=SEVEN)
    assert mp.half_life["good_uptake"] == 60 and set(mp.half_life) == set(MARK_KINDS)
    assert mp.ttl("good_uptake") == 480
    assert MarkParams(half_life={**SEVEN, "good_uptake": 25}).half_life["good_uptake"] == 25
    with pytest.raises(MarkError):
        MarkParams(half_life={"dosed": 1})                                   # still needs every old kind
    with pytest.raises(MarkError):
        MarkParams(half_life={**SEVEN, "bogus": 3})
    assert complete_half_life(SEVEN) == DEFAULT_HALF_LIFE


def test_good_uptake_decays_with_half_life_60_and_aggregates_like_any_mark():
    f = MarkField("S-marks", MarkParams())
    assert f.deposit(2, "good_uptake", (5, 5, 5), 0)["mark_kind"] == "good_uptake"
    s = f.aggregate("good_uptake", (5, 5, 5), 60)
    assert s.strength == pytest.approx(0.5) and s.distinct_authors == 1
    assert f.aggregate("good_uptake", (5, 5, 5), 480 + 1).strength == 0.0 or not f.near((5, 5, 5), 482, 6.0)
    rows = f.marks_near((5, 5, 5), 1)
    assert [r["kind"] for r in rows] == ["good_uptake"]


def test_colony_config_accepts_seven_kinds_and_omits_the_default_in_to_dict():
    c = ColonyConfig.from_dict({"half_life": SEVEN})
    assert c.half_life["good_uptake"] == 60
    assert c.to_dict() == ColonyConfig().to_dict() and "good_uptake" not in c.to_dict()["half_life"]
    assert json.dumps(c.to_dict(), sort_keys=True) == json.dumps(ColonyConfig(half_life=SEVEN).to_dict(), sort_keys=True)
    assert list(c.to_dict()["half_life"]) == list(SEVEN)                    # same keys, same order as v4.2
    c2 = ColonyConfig(half_life={**SEVEN, "good_uptake": 25})
    assert c2.to_dict()["half_life"]["good_uptake"] == 25 and c2.to_dict() != c.to_dict()
    assert ColonyConfig.from_dict(c2.to_dict()) == c2


def test_queen_and_state_know_the_kind():
    assert "good_uptake" in QUEEN_MARK_KINDS and "slow_uptake" in QUEEN_MARK_KINDS
    assert set(QUEEN_MARK_KINDS) >= {"dosed", "dose_ok", "tumour_here", "not_tumour"} and QUEEN_INPUT_KEYS == ("tick", "regions")
    assert KIND_ORDER[-1] == "good_uptake" and KIND_WORDS["good_uptake"] == "good uptake" and len(KIND_ORDER) == len(MARK_KINDS)
    row = {"kind": "good_uptake", "pos_vox": [1, 1, 1], "strength": 1.0, "distinct_authors": 2, "strong": True, "disputed": False, "authors": [1, 2]}
    assert "good uptake by 2 ants" in marks_sentence({"marks_near": [row]})


# --- the queen ------------------------------------------------------------------------------------------------------------

def queen(**kw):
    return ScriptedQueen(REGIONS, total_drug=1000.0, n_squads=8, voxel_mm=2.0, **kw)


def mark(kind, tick, pos, ant=0, milli=1000):
    return {"subtype": "mark", "mark_kind": kind, "ant": ant, "pos_vox": list(pos), "amount_milli": milli, "reading_ppm": 0, "tick": tick}


def call(q, tick, marks=()):
    return q(tick, {"tick": tick, "regions": [{"region_id": r.region_id, "centroid_vox": list(r.centroid_vox)} for r in REGIONS],
                    "marks": list(marks), "quorum_events": []})


Z0, Z1 = (10, 10, 10), (30, 10, 10)          # inside zone 0 and zone 1
MASS = {0: 0.8 * 4000, 1: 0.7 * 3000}


def rows(out):
    return {r["region_id"]: r for r in out["need_rows"]}


def test_scaling_off_ignores_uptake_marks_and_adds_nothing():
    marks = [mark("good_uptake", 5, Z0), mark("slow_uptake", 5, Z1)]
    assert call(queen(), 30, marks) == call(queen(), 30, [])
    r = rows(call(queen(), 30, marks))
    assert all("uptake_factor_milli" not in x for x in r.values())


def test_factor_1_25_when_more_than_half_is_good():
    out = call(queen(uptake_scaling=True), 30, [mark("good_uptake", 25, Z1, ant=0), mark("good_uptake", 25, Z1, ant=1), mark("slow_uptake", 25, Z1, ant=2)])
    r = rows(out)
    assert r[1]["uptake_factor_milli"] == 1250 and r[0]["uptake_factor_milli"] == 1000
    raw0, raw1 = MASS[0], MASS[1] * 1.25
    top = max(raw0, raw1)
    assert r[0]["need_milli"] == round(1000 * raw0 / top) and r[1]["need_milli"] == round(1000 * raw1 / top)


def test_factor_0_5_when_more_than_half_is_slow_and_scaling_precedes_normalisation():
    out = call(queen(uptake_scaling=True), 30, [mark("slow_uptake", 25, Z0), mark("slow_uptake", 26, Z0, ant=3), mark("good_uptake", 25, Z0, ant=4)])
    r = rows(out)
    assert r[0]["uptake_factor_milli"] == 500 and r[1]["uptake_factor_milli"] == 1000
    # raw need is scaled BEFORE the maximum is taken: zone 1 (unscaled) is now the top
    assert r[1]["need_milli"] == 1000 and r[0]["need_milli"] == round(1000 * (MASS[0] * 0.5) / MASS[1])


def test_exactly_half_and_no_marks_are_factor_one():
    q = queen(uptake_scaling=True)
    r = rows(call(q, 30, [mark("good_uptake", 25, Z1, ant=0), mark("slow_uptake", 25, Z1, ant=1)]))
    assert r[1]["uptake_factor_milli"] == 1000                                   # equal strength: not "more than half"
    assert {x["uptake_factor_milli"] for x in rows(call(q, 30, [])).values()} == {1000}
    assert rows(call(q, 0))[0]["uptake_factor_milli"] == 1000                    # tick-0 call (the S-solo queen) has no marks


def test_strength_decays_by_age_and_half_life_per_kind():
    q = queen(uptake_scaling=True)
    # a good mark deposited at tick 0 (60-tick half-life) is worth 0.25 at tick 120; a fresh slow mark (tick 119) is worth ~1
    r = rows(call(q, 120, [mark("good_uptake", 0, Z1), mark("slow_uptake", 119, Z1, ant=1)]))
    assert r[1]["uptake_factor_milli"] == 500
    # the other way round: an old slow mark loses to a fresh good one
    r = rows(call(q, 120, [mark("slow_uptake", 0, Z1), mark("good_uptake", 119, Z1, ant=1)]))
    assert r[1]["uptake_factor_milli"] == 1250
    # a custom half-life shifts the balance: slow decays fast (5 ticks) so after 20 ticks a good mark of the same age wins
    hl = {**DEFAULT_HALF_LIFE, "slow_uptake": 5}
    r = rows(call(queen(uptake_scaling=True, half_life=hl), 60, [mark("slow_uptake", 40, Z1), mark("good_uptake", 40, Z1, ant=1)]))
    assert r[1]["uptake_factor_milli"] == 1250
    assert rows(call(q, 60, [mark("slow_uptake", 40, Z1), mark("good_uptake", 40, Z1, ant=1)]))[1]["uptake_factor_milli"] == 1000
    # amounts weigh in
    r = rows(call(q, 30, [mark("good_uptake", 25, Z1, milli=1000), mark("slow_uptake", 25, Z1, ant=1, milli=400)]))
    assert r[1]["uptake_factor_milli"] == 1250


def test_expired_future_and_out_of_zone_marks_do_not_count():
    q = queen(uptake_scaling=True)
    ttl = 8 * 60
    r = rows(call(q, ttl + 1, [mark("slow_uptake", 0, Z0)]))                      # older than 8 half-lives: not live
    assert r[0]["uptake_factor_milli"] == 1000
    assert rows(call(q, 30, [mark("slow_uptake", 30, Z0), mark("slow_uptake", 31, Z0)]))[0]["uptake_factor_milli"] == 1000   # tick >= now
    assert rows(call(q, 30, [mark("slow_uptake", 5, (80, 80, 80))]))[0]["uptake_factor_milli"] == 1000                        # in no zone


def test_other_mark_kinds_never_count_as_uptake():
    r = rows(call(queen(uptake_scaling=True), 30, [mark("tumour_here", 20, Z1), mark("washout", 20, Z1), mark("dose_ok", 20, Z1)]))
    assert r[1]["uptake_factor_milli"] == 1000


def test_scaling_does_not_touch_delivered_share_or_alarms():
    q_on, q_off = queen(uptake_scaling=True), queen()
    dosed = [dict(mark("dosed", 5, Z0), amount_milli=1000, ant=a) for a in range(3)]
    marks = dosed + [mark("slow_uptake", 5, Z0), mark("good_uptake", 5, Z1)]
    a, b = call(q_on, 30, marks), call(q_off, 30, dosed)
    assert a["colony_note"].split(" Need:")[0] == b["colony_note"].split(" Need:")[0]   # the delivered % is not scaled; the need bands are
    assert q_on.alarms == q_off.alarms
    assert [x["delivered_milli"] for x in a["need_rows"]] == [x["delivered_milli"] for x in b["need_rows"]]
