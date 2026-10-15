"""A2r1: scripted ``decide_turn`` tests of the dose move-on (spot choice, persistent goal, arrival, timeout, skip, switch off)."""

from dataclasses import replace

import numpy as np

from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_policies import ScanInfo, decide_turn, known_dosed_spots
from antelligence.worlds.tumor_mri.questions import STAY
from tests.v4_2.q1.helpers import REGIONS, inputs, log_look, mark_row
from tests.v4_3.a2.helpers import R43, state

VM = 2.0
ON, OFF = R43, replace(R43, dose_move_on=False)


def pmap(only_centre=False):
    p = np.full((45, 25, 25), 0.1)
    if only_centre:
        p[10, 10, 10] = 0.9
    else:
        p[4:17, 4:17, 4:17] = 0.9          # zone A (centroid (10, 10, 10), zone radius 7.9 voxels)
        p[24:37, 4:17, 4:17] = 0.9         # zone B
    return p


def scan(p=None, regions=REGIONS):
    return ScanInfo(regions, VM, p if p is not None else pmap())


def dose_log(pos, value=1.2, tick=0):
    return {"tick": tick, "pos_vox": list(pos), "kind": "dose", "value": value, "band": 0, "word": None}


def turn(st, sc, tick, pos, dose_rel, **kw):
    return decide_turn(inputs(tick=tick, round=tick // 5, pos=pos, dose_rel=dose_rel, **kw), st, sc, 0)


def test_dose_ok_with_payload_left_moves_on_to_the_nearest_new_spot():
    st, sc = state(ON), scan()
    d = turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    assert d.job == "dose" and d.info["reason"] == "dose_ok" and not d.release
    assert d.target_vox == (7, 10, 10) and d.target_label == STAY       # 3 voxels = 6 mm away, lexicographic tie
    assert st.dose_goal == {"spot": (7, 10, 10), "zone": 0, "age": 0}
    assert st.counters["moved_on_dose_ok"] == 1


def test_the_goal_persists_while_the_release_rule_still_applies_on_the_way():
    st, sc = state(ON), scan()
    turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    d = turn(st, sc, 5, (9, 10, 10), 0.0)                                # on the way: dose below D_t, c >= 0.6
    assert d.release and d.target_vox == (7, 10, 10) and d.target_label == STAY
    assert st.dose_goal["age"] == 1 and st.counters["moved_on_dose_ok"] == 1
    d = turn(st, sc, 10, (9, 9, 10), 1.5, reading_log=[dose_log((9, 9, 10), 1.5)])   # dose OK again on the way: goal kept
    assert not d.release and d.target_vox == (7, 10, 10) and st.counters["moved_on_dose_ok"] == 1


def test_the_goal_clears_on_arrival_and_the_zone_centroid_pulls_again():
    st, sc = state(ON), scan()
    turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    d = turn(st, sc, 5, (7, 10, 10), 0.0)                                # arrived, dose below D_t: releases here
    assert st.dose_goal is None and st.counters["dose_goal_reached"] == 1
    assert d.release and d.target_vox is None and d.target_label == "A"


def test_arrival_at_a_dosed_spot_moves_on_again_and_the_arrival_voxel_is_known():
    st, sc = state(ON), scan()
    turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    log = [dose_log((10, 10, 10)), dose_log((7, 10, 10), 1.1, tick=5)]
    d = turn(st, sc, 5, (7, 10, 10), 1.1, reading_log=log)
    assert st.counters["dose_goal_reached"] == 1 and st.counters["moved_on_dose_ok"] == 2
    assert d.target_vox is not None and d.target_vox != (7, 10, 10)
    for k in ((10, 10, 10), (7, 10, 10)):                                # >= 6 mm from both known dosed spots
        assert VM * np.linalg.norm(np.subtract(d.target_vox, k)) >= 6.0 - 1e-9
    assert st.dose_goal["spot"] == d.target_vox and st.dose_goal["age"] == 0


def test_the_goal_lapses_after_six_decision_turns():
    st, sc = state(ON), scan()
    turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    for k in range(1, 6):                                                # turns 1..5 still head for the spot
        d = turn(st, sc, 5 * k, (13, 10, 10), 0.0)
        assert d.target_vox == (7, 10, 10), k
    d = turn(st, sc, 30, (13, 10, 10), 0.0)                              # the sixth turn since it was set: timeout
    assert st.dose_goal is None and st.counters["dose_goal_timeout"] == 1
    assert d.target_vox is None and d.target_label == "A"


def test_the_goal_clears_when_its_zone_is_skipped():
    st, sc = state(ON), scan()
    turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    st.skip_zones.add(0)
    d = turn(st, sc, 5, (9, 10, 10), 0.0)
    assert st.dose_goal is None and st.counters["dose_goal_zone_dropped"] == 1
    assert d.target_label == "B"                                         # zone A is gone; the next-best target


def test_no_spot_left_skips_the_zone_and_takes_the_next_best_target():
    st, sc = state(ON), scan(pmap(only_centre=True))
    d = turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    assert 0 in st.skip_zones and st.counters["zone_skipped_no_spot"] == 1 and st.dose_goal is None
    assert d.job == "dose" and d.target_label == "B" and not d.release


def test_known_spots_are_own_releases_own_dose_readings_and_the_current_voxel():
    st = state(ON)
    st.released_at.extend([(5, 5, 5), (6, 6, 6)])
    inp = inputs(pos=(10, 10, 10), reading_log=[dose_log((8, 8, 8), 1.0), dose_log((9, 9, 9), 0.99), log_look((4, 4, 4), 0.9)])
    assert known_dosed_spots(inp, st) == [(5, 5, 5), (6, 6, 6), (8, 8, 8), (10, 10, 10)]   # 0.99 and looks do not count


def test_release_voxels_of_earlier_decisions_are_remembered_and_excluded():
    st, sc = state(ON), scan()
    d = turn(st, sc, 0, (10, 10, 10), 0.0)                               # releases at (10, 10, 10)
    assert d.release and st.released_at == [(10, 10, 10)]
    d = turn(st, sc, 5, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    assert d.target_vox is not None and VM * np.linalg.norm(np.subtract(d.target_vox, (10, 10, 10))) >= 6.0 - 1e-9


def test_sharing_arms_add_visible_dosed_and_dose_ok_marks_and_nothing_else():
    st = state(ON)
    marks = [mark_row("dosed", (7, 10, 10)), mark_row("dose_ok", (10, 7, 10)), mark_row("slow_uptake", (10, 10, 13)),
             mark_row("tumour_here", (10, 10, 7))]
    inp = inputs(pos=(10, 10, 10), marks=marks)
    assert known_dosed_spots(inp, st) == [(7, 10, 10), (10, 7, 10), (10, 10, 10)]
    sc = scan()
    d = decide_turn(inputs(pos=(10, 10, 10), dose_rel=1.2, marks=[mark_row("dosed", (7, 10, 10))],
                           reading_log=[dose_log((10, 10, 10))]), state(ON), sc, 0)
    solo = decide_turn(inputs(pos=(10, 10, 10), dose_rel=1.2, reading_log=[dose_log((10, 10, 10))]), state(ON), sc, 0)
    assert solo.target_vox == (7, 10, 10)                                # without the mark the ant would go there
    assert d.target_vox == (9, 8, 8)                                     # with it, the next spot >= 6 mm from (7, 10, 10)
    assert VM * np.linalg.norm(np.subtract(d.target_vox, (7, 10, 10))) >= 6.0


def test_solo_input_has_no_marks_and_the_set_is_own_only():
    inp = inputs(pos=(10, 10, 10))
    assert inp.marks is None                                             # S-solo: the marks input is absent
    st = state(ON)
    st.released_at.append((6, 6, 6))
    assert known_dosed_spots(inp, st) == [(6, 6, 6), (10, 10, 10)]


def test_switch_off_changes_nothing():
    st, sc = state(OFF), scan()
    d = turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    assert d.target_vox is None and d.target_label == "A" and st.dose_goal is None
    assert not any(k in st.counters for k in ("moved_on_dose_ok", "zone_skipped_no_spot", "dose_goal_reached", "dose_goal_timeout"))


def test_without_the_scan_map_the_switch_has_no_effect():
    st, sc = state(ON), ScanInfo(REGIONS, VM)
    d = turn(st, sc, 0, (10, 10, 10), 1.2, reading_log=[dose_log((10, 10, 10))])
    assert d.target_vox is None and d.target_label == "A" and st.dose_goal is None
    assert st.counters["move_on_no_scan_map"] == 1 and "moved_on_dose_ok" not in st.counters


def test_a_low_estimate_at_a_dosed_spot_is_not_a_move_on():
    st, sc = state(ON), scan()
    d = turn(st, sc, 0, (10, 10, 10), 1.2, look=0.4, reading_log=[dose_log((10, 10, 10))])     # c < 0.6: reason look_low
    assert d.info["reason"] == "look_low" and d.target_vox is None and st.dose_goal is None


def test_v42_has_the_switch_off_and_v43_on_and_both_are_in_the_hash():
    assert AntRules.v42().dose_move_on is False and AntRules.v43().dose_move_on is True and AntRules.v43().move_on_mm == 6.0
    assert AntRules.v43().to_dict()["dose_move_on"] is True and "move_on_mm" in AntRules.v42().to_dict()
    assert AntRules.v43().sha() != replace(AntRules.v43(), dose_move_on=False).sha()
    assert AntRules.v43().sha() != replace(AntRules.v43(), move_on_mm=8.0).sha()
    assert AntRules.from_dict({"version": "v43"}).dose_move_on is True
