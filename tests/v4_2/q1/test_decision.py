"""The one decision function: release rule, judgement marks, S-marks adjustments, S-talk ask / answer, thresholds."""

import pytest

from antelligence.worlds.tumor_mri.colony_policies import (AntState, EMPTY_FRACTION, ColonyScript, automatic_marks, decide_turn,
                                                           judgement_marks, release_rule)
from antelligence.kernel.types import LocalView
from tests.v4_2.q1.helpers import SCAN, cand, inputs, log_look, mark_row


def decide(inp, st=None, ant=0):
    st = st or AntState()
    return decide_turn(inp, st, SCAN, ant), st


def no_verify():
    """Thresholds that keep the verify job out of the way, so the S-marks action in these tests is a zone target."""
    st = AntState()
    st.thresholds.theta["verify"] = 1.0
    return st


# --- release rule -------------------------------------------------------------------------------------------------------

def test_release_needs_look_half_and_dose_below_target():
    assert release_rule(inputs(look=0.5), "dose", 0.5, (10, 10, 10))[0]
    assert not release_rule(inputs(look=0.49), "dose", 0.49, (10, 10, 10))[0]
    assert not release_rule(inputs(dose_rel=1.0), "dose", 0.8, (10, 10, 10))[0]
    assert release_rule(inputs(dose_rel=0.99), "dose", 0.8, (10, 10, 10))[0]


def test_release_amount_is_half_and_three_quarters_after_own_slow_uptake():
    assert release_rule(inputs(), "dose", 0.8, (10, 10, 10))[1] == 3
    slow = {"tick": 0, "pos_vox": [10, 11, 10], "kind": "uptake", "value": 0.4, "band": None, "word": "slow"}
    assert release_rule(inputs(reading_log=[slow]), "dose", 0.8, (10, 10, 10))[1] == 4
    ok = dict(slow, word="as expected", value=0.9)
    assert release_rule(inputs(reading_log=[ok]), "dose", 0.8, (10, 10, 10))[1] == 3
    far = dict(slow, pos_vox=[40, 10, 10])
    assert release_rule(inputs(reading_log=[far]), "dose", 0.8, (10, 10, 10))[1] == 3     # not near this zone


def test_fast_washout_skips_release_unless_look_is_065():
    fast = {"tick": 0, "pos_vox": [10, 10, 10], "kind": "washout", "value": 0.05, "band": None, "word": "fast"}
    assert not release_rule(inputs(reading_log=[fast], look=0.64), "dose", 0.64, (10, 10, 10))[0]
    assert release_rule(inputs(reading_log=[fast], look=0.65), "dose", 0.65, (10, 10, 10))[0]


def test_explore_releases_only_at_065_and_verify_never():
    assert not release_rule(inputs(look=0.6), "explore", 0.6, (10, 10, 10))[0]
    assert release_rule(inputs(look=0.65), "explore", 0.65, (10, 10, 10))[0]
    assert not release_rule(inputs(look=0.99), "verify", 0.99, (10, 10, 10))[0]
    assert not release_rule(inputs(payload_fraction=EMPTY_FRACTION / 2), "dose", 0.9, (10, 10, 10))[0]


# --- judgement marks ----------------------------------------------------------------------------------------------------

def test_judgement_marks():
    assert judgement_marks(inputs(look=0.65, scan_p=0.49), 0.65) == ["tumour_here"]
    assert judgement_marks(inputs(look=0.64, scan_p=0.49), 0.64) == []
    assert judgement_marks(inputs(look=0.8, scan_p=0.9), 0.8) == ["tumour_here"]
    assert judgement_marks(inputs(look=0.79, scan_p=0.9), 0.79) == []
    assert judgement_marks(inputs(look=0.3, scan_p=0.5), 0.3) == ["not_tumour"]
    assert judgement_marks(inputs(look=0.31, scan_p=0.5), 0.31) == []
    assert judgement_marks(inputs(look=0.3, scan_p=0.49), 0.3) == []
    assert judgement_marks(inputs(look=0.5, scan_p=0.4, payload_fraction=0.0), 0.5) == ["need_more"]
    assert judgement_marks(inputs(look=0.5, scan_p=0.4, payload_fraction=0.5), 0.5) == []
    assert judgement_marks(inputs(look=0.5, scan_p=0.4, payload_fraction=0.0, dose_rel=1.0), 0.5) == []


# --- the three arms are one function -----------------------------------------------------------------------------------

def test_absent_shared_inputs_equal_empty_shared_inputs():
    """S-solo is the same function with the shared inputs missing: None and 'nothing shared' decide identically."""
    a, _ = decide(inputs(marks=None, region_status=None, messages=None))
    b, _ = decide(inputs(marks=[], region_status={}, messages=None))
    assert (a.job, a.target_label, a.target_vox, a.release, a.level, a.marks, a.ask) == \
           (b.job, b.target_label, b.target_vox, b.release, b.level, b.marks, b.ask)


def test_no_arm_name_reaches_the_decision_function():
    import inspect
    from antelligence.worlds.tumor_mri import colony_policies as cp
    src = inspect.getsource(cp)
    for name in ("S-solo", "S-marks", "S-talk"):
        assert f'"{name}"' not in src and f"'{name}'" not in src.replace('"S-marks"', '')


def test_strong_done_zone_is_skipped_and_open_zone_chosen():
    d, _ = decide(inputs(look=0.2, scan_p=0.3, pos=(20, 10, 10), region_status={0: "done", 1: "open"}, marks=[],
                         candidates=[cand("A", 0, (10, 10, 10), 0.8, 20.0), cand("B", 1, (30, 10, 10), 0.7, 40.0)]))
    assert d.job == "dose" and d.target_label == "B"
    d2, _ = decide(inputs(look=0.2, scan_p=0.3, pos=(20, 10, 10), region_status={0: "open", 1: "open"}, marks=[],
                          candidates=[cand("A", 0, (10, 10, 10), 0.8, 20.0), cand("B", 1, (30, 10, 10), 0.7, 40.0)]))
    assert d2.target_label == "A"


def test_strong_dosed_mark_means_dose_already_high_here():
    row = mark_row("dosed", (10, 10, 10), strong=True, authors=(3, 4), strength=1.2)
    d, _ = decide(inputs(marks=[row]))
    assert not d.release
    d, _ = decide(inputs(marks=[mark_row("dosed", (10, 10, 10), strong=False)]))
    assert d.release                                                       # one weak dosed mark does not stop it
    d, _ = decide(inputs(marks=None))
    assert d.release


def test_others_slow_uptake_and_washout_marks_count_like_own_readings():
    d, _ = decide(inputs(marks=[mark_row("slow_uptake", (10, 11, 10))]))
    assert d.release and d.level == 4
    d, _ = decide(inputs(look=0.6, marks=[mark_row("washout", (10, 10, 10))]))
    assert not d.release
    d, _ = decide(inputs(look=0.6, marks=[mark_row("washout", (40, 10, 10))]))
    assert d.release


def test_tumour_here_and_need_more_spots_of_other_ants_are_extra_targets():
    cands = [cand("A", 0, (10, 10, 10), 0.8, 60.0)]
    spot = mark_row("tumour_here", (12, 12, 10), strength=1.0, authors=(5,))
    d, _ = decide(inputs(look=0.2, scan_p=0.3, pos=(30, 30, 30), candidates=cands, zone_need={0: "low"}, marks=[spot]))
    assert d.target_vox == (12, 12, 10) and d.job == "dose"
    own = mark_row("tumour_here", (12, 12, 10), authors=(0,))
    d, _ = decide(inputs(look=0.2, scan_p=0.3, pos=(30, 30, 30), candidates=cands, zone_need={0: "low"}, marks=[own]), ant=0)
    assert d.target_vox is None                                            # its own mark is not a lead
    disputed = mark_row("tumour_here", (12, 12, 10), authors=(5,), disputed=True)
    d, _ = decide(inputs(look=0.2, scan_p=0.3, pos=(30, 30, 30), candidates=cands, zone_need={0: "low"}, marks=[disputed]))
    assert d.target_vox != (12, 12, 10)


def test_marks_make_a_region_count_as_visited_for_explore():
    base = dict(look=0.2, scan_p=0.3, payload_fraction=0.0, zone_need={}, visited_regions=frozenset())
    d, _ = decide(inputs(**base, marks=None))
    assert d.job == "explore" and d.target_vox is None and d.target_label == "X"      # the world's own Explore
    d, _ = decide(inputs(**base, marks=[mark_row("dose_ok", (30, 10, 10))], region_status={}))
    assert d.job == "explore" and d.target_vox is not None and abs(d.target_vox[0] - 10) <= 1   # region 1 is taken; go to 0


# --- thresholds and jobs ------------------------------------------------------------------------------------------------

def test_dose_job_success_lowers_theta_and_failure_raises_it():
    st = AntState()
    d1, _ = decide(inputs(), st)
    assert d1.release and d1.job == "dose"
    d2, _ = decide(inputs(tick=5, round=1), st)                            # the release of turn 1 had a look >= 0.5, no slow reading
    assert st.thresholds.theta["dose"] == pytest.approx(0.4)
    st2 = AntState()
    decide(inputs(look=0.4, scan_p=0.45), st2)
    decide(inputs(tick=5, round=1, look=0.4, scan_p=0.45), st2)
    assert st2.thresholds.theta["dose"] == pytest.approx(0.55)             # no release: a turn without success
    slow = {"tick": 5, "pos_vox": [10, 10, 10], "kind": "uptake", "value": 0.3, "band": None, "word": "slow"}
    st3 = AntState()
    decide(inputs(), st3)
    decide(inputs(tick=5, round=1, uptake={"value": 0.3, "word": "slow", "about_vox": [10, 10, 10]}, reading_log=[slow]), st3)
    assert st3.thresholds.theta["dose"] == pytest.approx(0.55)             # slow uptake about that release: not a success


def test_explore_success_and_verify_success():
    st = AntState()
    st.prev = {"job": "explore", "released": False, "look": 0.2, "pos": None, "spot": None}
    decide(inputs(look=0.7, scan_p=0.3), st)                               # found tumour where the scan said no
    assert st.thresholds.theta["explore"] == pytest.approx(0.4)
    st.prev = {"job": "verify", "released": False, "look": 0.5, "pos": None, "spot": (10, 10, 10)}
    log = [log_look((10, 10, 10), 0.5), log_look((10, 11, 10), 0.9)]       # mean 0.7 leaves [0.35, 0.65]
    decide(inputs(reading_log=log), st)
    assert st.thresholds.theta["verify"] == pytest.approx(0.4)
    st.prev = {"job": "verify", "released": False, "look": 0.5, "pos": None, "spot": (10, 10, 10)}
    decide(inputs(reading_log=[log_look((10, 10, 10), 0.5), log_look((10, 11, 10), 0.55)]), st)
    assert st.thresholds.theta["verify"] == pytest.approx(0.45)


def test_job_choice_follows_p_and_tie_break():
    # all stimuli equal 1 and equal thresholds: dose > verify > explore
    st = AntState()
    d, _ = decide(inputs(zone_need={}, scan_p=1.0, look=0.2, candidates=[cand("A", 0, (10, 10, 10), 1.0, 5.0)],
                         reading_log=[log_look((14, 10, 10), 0.5)]), st)
    assert d.job == "dose"
    d, _ = decide(inputs(payload_fraction=0.0, look=0.2, scan_p=0.3, reading_log=[log_look((14, 10, 10), 0.5)]), AntState())
    assert d.job == "verify"                                               # dose stimulus is 0 with an empty payload; verify 1.0 = explore 1.0


def test_verify_target_is_an_unlooked_voxel_within_4mm_of_a_grey_look():
    d, _ = decide(inputs(payload_fraction=0.0, look=0.2, scan_p=0.3, reading_log=[log_look((14, 10, 10), 0.5)]))
    assert d.job == "verify" and d.target_vox is not None and d.target_vox != (14, 10, 10)
    assert 2.0 * sum((a - b) ** 2 for a, b in zip(d.target_vox, (14, 10, 10))) ** 0.5 <= 4.0 + 1e-9
    assert not d.release


# --- S-talk ---------------------------------------------------------------------------------------------------------------

def test_unsure_talker_takes_the_smarks_action_and_asks():
    unsure = dict(look=0.5, scan_p=0.6, reading_log=[log_look((10, 10, 10), 0.5)])
    a, _ = decide(inputs(**unsure, marks=[], messages=None), no_verify())
    b, st = decide(inputs(**unsure, marks=[], messages=[]), no_verify())
    assert a.ask is None and b.ask == {"question": "is_tumour", "region_id": 0}
    assert (a.job, a.target_label, a.target_vox, a.release, a.level) == (b.job, b.target_label, b.target_vox, b.release, b.level)
    assert st.pending["estimate"] == pytest.approx(0.5)


def test_confident_talker_does_not_ask():
    sure = dict(look=0.9, scan_p=0.9, reading_log=[log_look((10, 10, 10), 0.9)], marks=[], messages=[])
    assert decide(inputs(**sure))[0].ask is None


def test_talker_with_no_own_look_near_the_target_estimates_from_the_current_look_and_still_asks():
    far = dict(look=0.5, scan_p=0.6, reading_log=[log_look((30, 30, 30), 0.1)], marks=[], messages=[])   # nothing logged near the target
    d, st = decide(inputs(**far), no_verify())
    assert d.ask == {"question": "is_tumour", "region_id": 0}
    assert st.pending["estimate"] == pytest.approx(0.5)               # the current look, not the far-away 0.1
    assert st.counters["asks"] == 1 and st.counters["asks_on_current_look"] == 1
    assert "ask_skipped_no_reading_to_quote" not in st.counters
    sure = dict(far, look=0.9)                                          # a confident current look is not unsure: no ask
    assert decide(inputs(**sure), no_verify())[0].ask is None


def test_disputed_target_is_unsure_even_with_a_confident_estimate():
    row = mark_row("tumour_here", (10, 10, 10), disputed=True)
    d, _ = decide(inputs(look=0.9, scan_p=0.9, reading_log=[log_look((10, 10, 10), 0.9)], marks=[row], messages=[]))
    assert d.ask is not None


def test_ask_budget_one_per_two_rounds():
    unsure = dict(look=0.5, scan_p=0.6, reading_log=[log_look((10, 10, 10), 0.5)], marks=[], messages=[])
    st = AntState()
    assert decide(inputs(**unsure, round=0), st)[0].ask is not None
    st.pending = None
    assert decide(inputs(**unsure, round=1), st)[0].ask is None
    assert decide(inputs(**unsure, round=2), st)[0].ask is not None


def answer(ans, region=0, pos=None):
    return {"type": "answer", "question": "is_tumour", "answer": ans, "about": {"region_id": region, "pos_vox": pos},
            "msg_id": 2, "from_ant": 3, "to_ant": 0, "tick": 1, "note": ""}


def test_answer_folds_into_the_estimate_and_replaces_the_look_for_that_target_that_turn():
    unsure = dict(look=0.5, scan_p=0.6, reading_log=[log_look((10, 10, 10), 0.5)], marks=[])
    st = no_verify()
    decide(inputs(**unsure, messages=[]), st)
    assert st.pending is not None
    d, _ = decide(inputs(**unsure, round=1, tick=5, messages=[answer("no")]), st)
    assert st.pending is None and st.counters["answers_used"] == 1 and d.target_label != "A"   # (0.5 + 0.2) / 2 = 0.35 < 0.5: ineligible now
    assert not st.skip_zones and not st.skip_spots                                              # ... but nothing is dropped for good
    assert not d.release                                                                        # raw look 0.5 would have released
    st = no_verify()
    decide(inputs(**unsure, messages=[]), st)
    d, _ = decide(inputs(**unsure, round=1, tick=5, messages=[answer("yes")]), st)
    assert d.target_label == "A" and d.release                                                  # (0.5 + 0.8) / 2 = 0.65 keeps it
    st = no_verify()
    decide(inputs(**unsure, messages=[]), st)
    decide(inputs(**unsure, round=1, tick=5, messages=[answer("unknown")]), st)
    assert not st.skip_zones and st.counters["answers_used"] == 0                               # unknown is ignored
    st = no_verify()
    decide(inputs(**unsure, messages=[]), st)
    decide(inputs(**unsure, round=1, tick=5, messages=[]), st)
    assert st.pending is None                                                                   # unanswered: settled at the next turn anyway


# --- the wrapper ----------------------------------------------------------------------------------------------------------

def test_colony_script_builds_a_valid_scripted_intent():
    obs = dict(squad=0, tick=0, round=0, position_vox=[10, 10, 10], payload=10.0, payload_fraction=1.0, local_p_tumour=0.7,
               local_dose_rel=0.0, candidates=[cand("A", 0, (10, 10, 10), 0.8, 0.0)], local_look=0.8, look=0.8,
               reading_log=[], uptake_reading=None, zone_need={0: "high"}, visited_self={"regions": [0, 1], "walked_count": 1})
    it = ColonyScript(SCAN, 0).decide(LocalView(agent_id="squad-0", tick=0, revision=0, seed=1, observation=obs))
    assert it.action == "decide" and it.meta["scripted"] is True
    assert it.params["release"] == 1 and it.params["amount"] == 3 and it.params["target"] == "A"
    assert [d.kind for d in it.emit] == ["tumour_here"]


def test_automatic_marks_rule():
    ctx = {"ant": 1, "tick": 5, "pos_vox": [3, 3, 3], "release_fraction": 0.5, "zone": 0, "readings": [
        {"kind": "look", "value": 0.9}, {"kind": "dose", "value": 1.2},
        {"kind": "uptake", "value": 0.3, "word": "slow", "pos_vox": [2, 2, 2]},
        {"kind": "washout", "value": 0.05, "word": "fast", "pos_vox": [3, 3, 3]}]}
    out = automatic_marks(ctx)
    assert [(k, tuple(p), a) for k, p, a, _ in out] == [("dosed", (3, 3, 3), 0.5), ("dose_ok", (3, 3, 3), 1.0),
                                                        ("slow_uptake", (2, 2, 2), 1.0), ("washout", (3, 3, 3), 1.0)]
    assert automatic_marks({**ctx, "zone": None, "release_fraction": 0.0, "readings": ctx["readings"][1:2]}) == []
    ok = {**ctx, "release_fraction": 0.0, "readings": [{"kind": "dose", "value": 0.99}, {"kind": "uptake", "value": 0.9, "word": "as expected"},
                                                          {"kind": "washout", "value": 0.0, "word": "normal"}]}
    assert automatic_marks(ok) == []


# --- move on (goal 7: only own fast washout with look < 0.65; never a single-voxel dose_ok) -------------------------------

def washout_row(pos, tick=0):
    return {"tick": tick, "pos_vox": list(pos), "kind": "washout", "value": 0.1, "band": 0, "word": "fast"}


def test_single_voxel_dose_ok_never_abandons_a_zone():
    st = no_verify()
    d, _ = decide(inputs(dose_rel=1.0, look=0.8), st)
    assert not d.release and d.info["reason"] == "dose_ok"
    assert d.target_label == "A" and not st.skip_zones and not st.skip_spots
    assert not [k for k in st.counters if k.startswith("moved_on_")]
    for kw in (dict(look=0.4), dict(look=0.8, marks=[mark_row("dose_ok", (10, 10, 10), strong=True)], dose_rel=0.0)):
        st = no_verify()
        d, _ = decide(inputs(**kw), st)
        assert not st.skip_zones and not [k for k in st.counters if k.startswith("moved_on_")]


def test_fast_washout_with_low_look_moves_on_in_every_arm():
    kw = dict(look=0.55, reading_log=[washout_row((10, 10, 10))])
    for shared in (dict(), dict(marks=[], region_status={}), dict(marks=[], region_status={}, messages=[])):   # S-solo / S-marks / S-talk
        st = no_verify()
        d, _ = decide(inputs(**kw, **shared), st)
        assert not d.release and d.info["reason"] == "washout"
        assert st.skip_zones == {0} and st.counters["moved_on_washout"] == 1 and d.target_label != "A"
    st = no_verify()
    decide(inputs(look=0.7, reading_log=[washout_row((10, 10, 10))]), st)             # look >= 0.65: releases, no skip
    assert not st.skip_zones
    st = no_verify()                                                                  # sharing arms: another ant's washout mark counts
    decide(inputs(look=0.55, marks=[mark_row("washout", (10, 10, 10))], region_status={}), st)
    assert st.skip_zones == {0}
