"""B2: scripted ``decide_turn`` tests of the probe-and-hold foraging rule, branch by branch (dose and explore jobs)."""

from dataclasses import replace

import numpy as np
import pytest

from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_policies import (_dose_targets, known_uptake_spots, uptake_class, uptake_info)
from antelligence.worlds.tumor_mri.questions import EXPLORE, STAY
from tests.v4_2.q1.helpers import inputs, mark_row
from tests.v4_3.a2.helpers import R43, state
from tests.v4_3.b2.helpers import POS, RF, VM, fresh, pmap, scan, turn, uptake


def mm(a, b):
    return VM * float(np.linalg.norm(np.subtract(a, b)))


# --- the switch and the rules ------------------------------------------------------------------------------------------------

def test_the_rules_and_their_defaults():
    assert (RF.foraging, RF.probe_level, RF.info_mm, RF.good_u, RF.poor_u, RF.good_level, RF.middling_level, RF.hold_max_turns,
            RF.zone_halve) == (True, 1, 6.0, 0.7, 0.4, 3, 2, 2, 0.5)
    assert not AntRules.v42().foraging and not AntRules.v43().foraging
    assert RF == replace(AntRules.v43(), foraging=True) and AntRules.from_dict({"version": "v43", "foraging": True}) == RF
    d = RF.to_dict()
    assert {"foraging", "probe_level", "info_mm", "good_u", "poor_u", "good_level", "middling_level", "hold_max_turns",
            "zone_halve"} <= set(d)
    assert RF.sha() not in (AntRules.v42().sha(), AntRules.v43().sha()) and replace(RF, zone_halve=0.4).sha() != RF.sha()


def test_the_switch_off_leaves_the_rules_dict_and_hash_as_before():
    for r in (AntRules.v42(), AntRules.v43()):
        assert not any(k in r.to_dict() for k in ("foraging", "probe_level", "info_mm", "good_u", "zone_halve"))
    assert AntRules.v42().sha() == "9b9cbcadaa1af354e2b282f818493c72763eefdf5429cbba7ce85017619cefc2"
    assert AntRules.v43().sha() == "ddf0cd1fe7229f97db4f7747f221d7892fdf40b69af5d1b6a7677578ef617855"


def test_foraging_needs_the_a2r1_goal():
    with pytest.raises(ValueError):
        replace(AntRules.v43(), dose_move_on=False, foraging=True)
    with pytest.raises(ValueError):
        AntRules.from_dict({"foraging": True})                    # the v4.2 preset has no dose_move_on


def test_sensor_words_match_the_world():
    from antelligence.worlds.tumor_mri.foraging import uptake_word
    for v in (0.0, 0.39, 0.4, 0.69, 0.7, 1.2):
        assert uptake_class(v, RF) == uptake_word(v)


# --- no information: probe and hold ------------------------------------------------------------------------------------------

def test_no_information_probes_at_ten_percent_and_holds():
    st, sc = fresh()
    d = turn(st, sc, 0)
    assert d.release and d.level == 1 and d.info["reason"] == "probe" and d.job == "dose"
    assert d.target_label == STAY and d.target_vox == POS
    assert st.hold == {"pos": POS, "age": 0, "tick": 0} and st.stay is None
    assert st.counters["probes"] == 1 and st.counters["probes_saved_by_marks"] == 0


def test_the_explore_job_probes_too_and_stays_on_its_voxel():
    st, sc = fresh()
    d = turn(st, sc, 0, candidates=[], visited_regions=frozenset(), zone_need={})       # nothing to dose: the job is explore
    assert d.job == "explore" and d.release and d.level == 1 and d.info["reason"] == "probe"
    assert d.target_label == STAY and d.target_vox == POS and st.hold is not None


def test_only_a_tumour_spot_is_probed():
    for kw, why in (({"look": 0.4}, "look_low"), ({"payload_fraction": 0.0}, "payload_empty"), ({"dose_rel": 1.3}, "dose_ok")):
        st, sc = fresh()
        d = turn(st, sc, 0, **kw)
        assert not d.release and d.info["reason"] == why and st.hold is None and st.counters["probes"] == 0, why
    st, sc = fresh()
    d = turn(st, sc, 0, candidates=[], visited_regions=frozenset(), zone_need={}, scan_p=0.3, look=0.9)   # explore needs scan p >= 0.5
    assert d.job == "explore" and not d.release and d.info["reason"] == "scan_low" and st.hold is None


def test_information_beyond_six_mm_does_not_count():
    st, sc = fresh()
    d = turn(st, sc, 0, reading_log=[uptake((10, 10, 14), 0.9, tick=0)])                # 8 mm away
    assert d.info["reason"] == "probe"
    st, sc = fresh()
    d = turn(st, sc, 0, reading_log=[uptake((10, 10, 13), 0.9, tick=0)])                # exactly 6 mm: within
    assert d.info["reason"] == "ok_good"


# --- the reading: good, middling, poor ---------------------------------------------------------------------------------------

def probed(rules=RF, p=None):
    st, sc = fresh(rules, p)
    turn(st, sc, 0)
    return st, sc


@pytest.mark.parametrize("value,level,reason,cls", [(0.9, 3, "ok_good", "good"), (0.7, 3, "ok_good", "good"),
                                                     (0.55, 2, "ok_middling", "middling"), (0.4, 2, "ok_middling", "middling")])
def test_the_reading_sets_the_amount_and_the_stay(value, level, reason, cls):
    st, sc = probed()
    d = turn(st, sc, 5, reading_log=[uptake(POS, value)])
    assert d.job == "dose" and d.release and d.level == level and d.info["reason"] == reason
    assert d.target_label == STAY and d.target_vox == POS
    assert st.hold is None and st.stay == {"pos": POS, "cls": cls} and st.counters["holds_resolved"] == 1


def test_the_reading_forces_the_dose_job_even_when_the_ant_would_explore():
    st, sc = probed()
    d = turn(st, sc, 5, reading_log=[uptake(POS, 0.9)], candidates=[], visited_regions=frozenset(), zone_need={})
    assert d.job == "dose" and d.release and d.level == 3 and d.target_vox == POS


def test_a_poor_reading_releases_nothing_halves_the_zone_and_moves_to_the_nearest_unprobed_voxel():
    st, sc = probed()
    d = turn(st, sc, 5, reading_log=[uptake(POS, 0.2)])
    assert d.job == "dose" and not d.release and d.info["reason"] == "poor_uptake"
    assert st.hold is None and st.stay is None and st.counters["no_release_poor"] == 1
    assert st.zone_mult == {0: 0.5} and 0 not in st.skip_zones and st.counters["zones_halved"] == 1
    spot = d.target_vox
    assert spot is not None and d.target_label == STAY and st.dose_goal == {"spot": spot, "zone": 0, "age": 0}
    assert sc.zone_at(spot) == 0 and sc.p_tumour[spot] >= 0.5 and mm(spot, POS) >= 6.0
    # nearest: no zone voxel with p >= 0.5 and >= 6 mm from the reading is closer to the ant (ties: lexicographic)
    best = min(((mm(v, POS), tuple(v)) for v in sc.zone_voxels(0) if mm(v, POS) > 6.0 + 1e-6))
    assert (mm(spot, POS), spot) == best


def test_poor_goal_persists_like_the_a2r1_goal_and_clears_on_arrival():
    st, sc = probed()
    d = turn(st, sc, 5, reading_log=[uptake(POS, 0.2)])
    spot = d.target_vox
    d = turn(st, sc, 10, pos=(spot[0], spot[1], spot[2] + 2), reading_log=[uptake(POS, 0.2)], look=0.2)   # on the way
    assert d.target_vox == spot and st.dose_goal["age"] == 1
    d = turn(st, sc, 15, pos=spot)                                                                           # arrived: nothing known here
    assert st.dose_goal is None and st.counters["dose_goal_reached"] == 1
    assert d.info["reason"] == "probe" and d.release and d.level == 1 and st.hold["pos"] == spot            # an unprobed voxel is probed


def test_a_poor_zone_is_halved_once_and_never_dropped():
    st, sc = probed()
    turn(st, sc, 5, reading_log=[uptake(POS, 0.2)])
    spot = st.dose_goal["spot"]
    st.hold = {"pos": spot, "age": 0, "tick": 10}                                  # a second probe in the same zone ...
    turn(st, sc, 15, pos=spot, reading_log=[uptake(POS, 0.2), uptake(spot, 0.1, tick=15)])
    assert st.zone_mult == {0: 0.5} and st.counters["zones_halved"] == 1 and st.counters["no_release_poor"] == 2
    assert 0 not in st.skip_zones
    plain = {t.label: t.value for t in _dose_targets(inputs(), state(R43), sc, 0)}
    halved = {t.label: t.value for t in _dose_targets(inputs(), st, sc, 0)}
    assert halved["A"] == pytest.approx(0.5 * plain["A"]) and halved["B"] == plain["B"]


def test_the_halving_reaches_spot_targets_in_the_zone():
    st, sc = fresh()
    st.zone_mult[0] = 0.5
    rows = [mark_row("tumour_here", (12, 10, 10), strength=0.8, authors=(3,))]
    base = [t for t in _dose_targets(inputs(marks=rows), state(R43), sc, 0) if t.kind == "spot"][0]
    half = [t for t in _dose_targets(inputs(marks=rows), st, sc, 0) if t.kind == "spot"][0]
    assert half.value == pytest.approx(0.5 * base.value)


def test_an_exhausted_zone_is_skipped_not_dropped_for_uptake():
    st, sc = fresh(p=pmap(only_centre=True))
    turn(st, sc, 0)
    d = turn(st, sc, 5, reading_log=[uptake(POS, 0.2)])
    assert 0 in st.skip_zones and st.counters["zone_exhausted"] == 1 and st.dose_goal is None
    assert st.zone_mult == {0: 0.5}
    assert d.job == "dose" and not d.release and d.info["reason"] == "poor_uptake" and d.target_label == "B"


def test_poor_without_a_scan_map_stays_put_and_does_not_skip():
    from antelligence.worlds.tumor_mri.colony_policies import ScanInfo
    from tests.v4_2.q1.helpers import REGIONS
    st, sc = state(RF), ScanInfo(REGIONS, VM)
    turn(st, sc, 0)
    d = turn(st, sc, 5, reading_log=[uptake(POS, 0.2)])
    assert not d.release and st.counters["forage_no_scan_map"] == 1 and not st.skip_zones and st.dose_goal is None


def test_a_poor_explorer_goes_on_as_a_dose_ant_towards_the_new_spot():
    st, sc = fresh()
    d = turn(st, sc, 0, reading_log=[uptake(POS, 0.2, tick=0)], candidates=[], visited_regions=frozenset(), zone_need={})
    assert d.job == "dose" and not d.release and st.counters["poor_explore_to_dose"] == 1      # the explore job met known poor tissue
    assert d.target_vox == st.dose_goal["spot"] and mm(d.target_vox, POS) >= 6.0


# --- hold age and timeout ----------------------------------------------------------------------------------------------------

def test_the_hold_waits_ages_and_times_out():
    st, sc = probed()
    for k, tick in enumerate((5, 10), start=1):
        d = turn(st, sc, tick)                                                    # no reading yet
        assert not d.release and d.info["reason"] == "hold" and d.job == "dose" and d.target_label == STAY
        assert d.target_vox == POS and st.hold["age"] == k and st.counters["hold_turns"] == k
    d = turn(st, sc, 15)                                                          # hold_max_turns = 2 reached
    assert st.counters["hold_timeouts"] == 1 and d.info["reason"] == "probe" and st.hold == {"pos": POS, "age": 0, "tick": 15}
    assert st.counters["probes"] == 2


def test_a_reading_about_another_voxel_or_an_older_release_does_not_end_the_hold():
    st, sc = probed()
    d = turn(st, sc, 5, reading_log=[uptake((10, 10, 14), 0.9, tick=5)])        # about a voxel 4 voxels away
    assert d.info["reason"] == "hold"
    st, sc = probed()
    st.hold["tick"] = 7
    d = turn(st, sc, 10, reading_log=[uptake(POS, 0.9, tick=5)])                 # read before the probe was released
    assert d.info["reason"] == "hold"


def test_an_empty_payload_ends_the_hold_without_forcing_the_job():
    st, sc = probed()
    d = turn(st, sc, 5, reading_log=[uptake(POS, 0.9)], payload_fraction=0.0)
    assert st.hold is None and st.stay is None and st.counters["hold_dropped"] == 1 and not d.release


# --- stay until the dose is OK -----------------------------------------------------------------------------------------------

def test_stay_releases_per_class_each_turn_until_the_dose_is_ok():
    st, sc = probed()
    turn(st, sc, 5, reading_log=[uptake(POS, 0.55)])                             # middling: stay with 25%
    log = [uptake(POS, 0.55)]
    for tick, dose in ((10, 0.4), (15, 0.8)):
        d = turn(st, sc, tick, reading_log=log, dose_rel=dose, candidates=[], visited_regions=frozenset(), zone_need={})
        assert d.job == "dose" and d.release and d.level == 2 and d.info["reason"] == "ok_middling" and d.target_vox == POS
        assert st.stay == {"pos": POS, "cls": "middling"}
    d = turn(st, sc, 20, reading_log=log, dose_rel=1.2)                          # the dose reached D_t: the stay ends
    assert st.stay is None and st.counters["stay_ended_dose_ok"] == 1
    assert d.job == "dose" and not d.release and d.info["reason"] == "dose_ok"
    assert st.counters["moved_on_dose_ok"] == 1 and d.target_vox == st.dose_goal["spot"]           # A2r1 takes over


def test_stay_ends_on_empty_payload_or_a_falling_estimate_or_a_move():
    for kw, key in (({"payload_fraction": 0.0}, "stay_ended_payload_empty"), ({"look": 0.3}, "stay_ended_look_low"),
                    ({"pos": (10, 10, 11)}, "stay_moved")):
        st, sc = probed()
        turn(st, sc, 5, reading_log=[uptake(POS, 0.9)])
        turn(st, sc, 10, reading_log=[uptake(POS, 0.9)])
        d = turn(st, sc, 15, reading_log=[uptake(POS, 0.9)], **kw)
        assert st.stay is None and st.counters[key] == 1, key
        assert not (d.release and d.info["reason"] == "ok_good" and d.target_vox == POS), key


# --- uptake information: marks and the information barrier -------------------------------------------------------------------

def test_a_foreign_good_mark_saves_the_probe():
    st, sc = fresh()
    d = turn(st, sc, 0, marks=[mark_row("good_uptake", (12, 10, 10), authors=(3,))])
    assert d.release and d.level == 3 and d.info["reason"] == "ok_good" and st.hold is None and st.stay is None
    assert st.counters["probes"] == 0 and st.counters["probes_saved_by_marks"] == 1


def test_a_foreign_slow_mark_is_poor_tissue():
    st, sc = fresh()
    d = turn(st, sc, 0, marks=[mark_row("slow_uptake", (12, 10, 10), authors=(3,))])
    assert not d.release and d.info["reason"] == "poor_uptake" and st.counters["probes"] == 0
    assert st.zone_mult == {0: 0.5} and st.dose_goal is not None and mm(st.dose_goal["spot"], (12, 10, 10)) >= 6.0
    assert st.counters["probes_saved_by_marks"] == 1


def test_in_s_solo_the_marks_input_is_none_and_no_foreign_mark_counts():
    st, sc = fresh()
    d = turn(st, sc, 0, marks=None)
    assert d.info["reason"] == "probe" and st.counters["probes_saved_by_marks"] == 0
    # the helper itself: the same board is visible only through ``marks``
    row = mark_row("good_uptake", (12, 10, 10), authors=(3,))
    assert uptake_info(inputs(marks=[row]), POS, 0, RF)[0] == {"cls": "good", "src": "foreign"}
    assert uptake_info(inputs(marks=None), POS, 0, RF) == (None, False)
    assert known_uptake_spots(inputs(marks=[row])) == [(12, 10, 10)] and known_uptake_spots(inputs(marks=None)) == []


def test_the_ants_own_mark_repeats_its_reading_and_is_not_foreign():
    st, sc = fresh()
    d = turn(st, sc, 0, marks=[mark_row("good_uptake", (12, 10, 10), authors=(0,))])
    assert d.info["reason"] == "probe"
    st, sc = fresh()
    d = turn(st, sc, 0, marks=[mark_row("good_uptake", (12, 10, 10), authors=(0, 3))])        # merged with another ant's: foreign
    assert d.info["reason"] == "ok_good"


def test_other_mark_kinds_carry_no_uptake_information():
    st, sc = fresh()
    d = turn(st, sc, 0, marks=[mark_row(k, (11, 10, 10), authors=(3,)) for k in ("dosed", "dose_ok", "washout", "tumour_here")])
    assert d.info["reason"] == "probe"


def test_the_nearest_information_wins_with_the_stated_ties():
    own_poor, own_good = uptake((10, 10, 12), 0.2, tick=1), uptake((10, 10, 8), 0.9, tick=3)      # both 4 mm away
    near_mark = mark_row("good_uptake", (10, 10, 11), authors=(3,))                             # 2 mm away
    cls = lambda **kw: uptake_info(inputs(**kw), POS, 0, RF)  # noqa: E731
    assert cls(reading_log=[own_poor, own_good])[0] == {"cls": "good", "src": "own"}              # tie: the later own reading
    assert cls(reading_log=[own_poor], marks=[mark_row("good_uptake", (10, 10, 12), authors=(3,))])[0]["src"] == "own"   # own first
    assert cls(reading_log=[own_poor], marks=[near_mark]) == ({"cls": "good", "src": "foreign"}, True)   # nearer foreign mark wins
    d_own_far = cls(reading_log=[uptake((10, 10, 8), 0.2)], marks=[near_mark])
    assert d_own_far[0]["cls"] == "good" and d_own_far[1] is True


def test_a_probe_is_not_saved_when_the_ant_has_its_own_information():
    st, sc = fresh()
    d = turn(st, sc, 0, reading_log=[uptake((10, 10, 8), 0.9, tick=0)], marks=[mark_row("good_uptake", (10, 10, 11), authors=(3,))])
    assert d.info["reason"] == "ok_good" and st.counters["probes_saved_by_marks"] == 0


def test_known_information_gives_the_amounts_without_a_stay():
    for value, level, reason in ((0.9, 3, "ok_good"), (0.5, 2, "ok_middling")):
        st, sc = fresh()
        d = turn(st, sc, 0, reading_log=[uptake((11, 10, 10), value, tick=0)])
        assert d.release and d.level == level and d.info["reason"] == reason and st.hold is None and st.stay is None
    st, sc = fresh()
    d = turn(st, sc, 0, reading_log=[uptake((11, 10, 10), 0.1, tick=0)])
    assert not d.release and d.info["reason"] == "poor_uptake" and st.counters["no_release_poor"] == 1


def test_the_new_amounts_replace_the_old_fifty_and_seventy_five():
    slow = [{"tick": 0, "pos_vox": [11, 10, 10], "kind": "uptake", "value": 0.2, "band": 0, "word": "slow"}]
    st, sc = state(R43), scan()
    assert turn(st, sc, 0, reading_log=slow).level == 4                                    # switch off: 75% if slow, as before
    st, sc = fresh()
    d = turn(st, sc, 0, reading_log=[uptake((11, 10, 10), 0.9, tick=0)], marks=[mark_row("slow_uptake", (12, 10, 10), authors=(3,))])
    assert d.level == 3                                                                       # switch on: the nearest information rules


def test_the_switch_off_never_probes():
    st, sc = state(R43), scan()
    d = turn(st, sc, 0)
    assert d.release and d.level == 3 and d.info["reason"] == "ok" and st.hold is None and not st.counters
