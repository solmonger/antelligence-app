"""Unit tests for each v4.3 rule (design readings 3 to 10), plus AntRules and CombinedEstimate themselves."""

import json
from dataclasses import replace

import pytest

from antelligence.kernel.colony.cards import Card
from antelligence.kernel.colony.episode import ColonyConfig
from antelligence.kernel.colony.messages import answer_question
from antelligence.kernel.colony.thresholds import Thresholds
from antelligence.worlds.tumor_mri import WorldConfig
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.colony_policies import (AntState, ColonyScript, _settle_previous, decide_turn, judgement_marks,
                                                           release_rule)
from antelligence.worlds.tumor_mri.combined import (DEFAULT_COEFFICIENTS_PATH, CombinedEstimate, CombinedEstimateError)
from tests.v4_2.q1.helpers import SCAN, inputs, log_look
from tests.v4_2.q1.test_decision import answer
from tests.v4_3.a2.helpers import COEF, R42, R43, estimate, log_est, state

POS = (10, 10, 10)


# --- AntRules and CombinedEstimate -------------------------------------------------------------------------------------

def test_v42_preset_is_the_default_and_v43_differs_where_the_design_says():
    assert AntRules.v42() == AntRules() and R42.version == "v42" and not R42.use_combined
    assert (R42.release_c, R42.explore_release_c, R42.explore_scan_min, R42.fold_drop_c, R42.fold_can_permit) == (0.5, 0.65, 0.0, 0.5, True)
    assert (R42.theta_step_up, R42.theta_step_down, R42.theta_floor, R42.theta_ceil, R42.resolved_only) == (0.05, 0.1, 0.1, 1.0, False)
    assert (R42.verify_per_spot_max, R42.verify_share_max, R42.card_min_resolved, R42.answer_rule) == (None, None, 1, "v42")
    assert (R43.release_c, R43.explore_release_c, R43.explore_scan_min, R43.washout_release_c) == (0.6, 0.6, 0.5, 0.65)
    assert (R43.tumour_here_c, R43.tumour_here_scan_max, R43.tumour_here_any_c, R43.not_tumour_c, R43.not_tumour_scan_min) == \
           (0.8, 0.5, None, 0.2, 0.5)
    assert (R43.grey, R43.verify_per_spot_max, R43.verify_share_max) == ((0.35, 0.65), 2, 0.25)
    assert (R43.theta_step_up, R43.theta_step_down, R43.theta_floor, R43.theta_ceil, R43.resolved_only) == (0.05, 0.05, 0.1, 1.0, True)
    assert (R43.fold_drop_c, R43.fold_can_permit, R43.answer_rule, R43.card_min_resolved, R43.use_combined) == (0.6, False, "v43", 3, True)


def test_ant_rules_dict_roundtrip_sha_and_errors():
    assert AntRules.from_dict(R43.to_dict()) == R43 and AntRules.from_dict(R42.to_dict()) == R42
    assert AntRules.from_dict({"version": "v43"}) == R43 and AntRules.from_dict({}) == R42
    assert AntRules.from_dict({"version": "v43", "release_c": 0.7}).release_c == 0.7
    json.dumps(R43.to_dict())
    assert R42.sha() != R43.sha() and R43.sha() == AntRules.v43().sha() and len(R43.sha()) == 64
    with pytest.raises(ValueError):
        AntRules.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        AntRules(version="v44")


def test_combined_estimate_arithmetic_schema_and_json(tmp_path):
    ident = CombinedEstimate([0.0, 0.0, 1.0, 0.0])                    # c = sigmoid(logit(look)) = look
    assert ident(0.3, 0.7) == pytest.approx(0.7) and ident(0.9, 0.001) == pytest.approx(0.001)
    assert ident(0.5, 0.0) == pytest.approx(0.001) and ident(0.5, 1.0) == pytest.approx(0.999)       # clipped
    est = estimate()
    import math
    ls, ll = math.log(0.6 / 0.4), math.log(0.8 / 0.2)
    assert est(0.6, 0.8) == pytest.approx(1 / (1 + math.exp(-(-0.2 + 0.8 * ls + 1.2 * ll + 0.1 * ls * ll))))
    assert est.to_dict()["coef"] == COEF["coef"] and est.to_dict()["schema"] == "combined_v4_3"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(COEF))
    assert CombinedEstimate.from_json(p)(0.4, 0.5) == est(0.4, 0.5) and CombinedEstimate.from_json(p).sha() == est.sha()
    with pytest.raises(CombinedEstimateError):
        CombinedEstimate.from_dict({**COEF, "schema": "other"})
    with pytest.raises(CombinedEstimateError):
        CombinedEstimate.from_dict({**COEF, "coef": [1, 2, 3]})
    with pytest.raises(CombinedEstimateError):
        CombinedEstimate.from_dict({**COEF, "features": ["1"]})
    assert DEFAULT_COEFFICIENTS_PATH.parts[-4:] == ("v4.3-20261015", "combined", "coefficients.json") or \
        DEFAULT_COEFFICIENTS_PATH.parts[-3:] == ("v4.3-20261015", "combined", "coefficients.json")
    assert "docs" in DEFAULT_COEFFICIENTS_PATH.parts


# --- release rule ------------------------------------------------------------------------------------------------------

def test_release_at_c_059_vs_061():
    assert not release_rule(inputs(), "dose", 0.59, POS, R43)[0]
    assert release_rule(inputs(), "dose", 0.61, POS, R43)[0]
    assert release_rule(inputs(), "dose", 0.6, POS, R43)[0]
    assert release_rule(inputs(), "dose", 0.55, POS, R42)[0]                     # v4.2 threshold was 0.5


def test_release_uses_the_estimate_not_the_raw_look_in_decide_turn():
    """Raw look 0.95 but c = 0.59: no release. c = 0.61 and a low raw look: release."""
    low = decide_turn(inputs(look=0.95, est=0.59, scan_p=0.7), state(no_verify=True), SCAN, 0)
    assert not low.release and low.info["reason"] == "look_low"
    high = decide_turn(inputs(look=0.2, est=0.61, scan_p=0.7), state(no_verify=True), SCAN, 0)
    assert high.release and high.info["reason"] == "ok"
    # the same estimate found in the reading log at this tick (what the driver provides) is read too
    logged = decide_turn(inputs(look=0.95, tick=5, reading_log=[log_est(POS, 0.61, tick=5)]), state(no_verify=True), SCAN, 0)
    assert logged.release
    # no est anywhere: falls back to the raw look
    raw = decide_turn(inputs(look=0.62, reading_log=[log_look(POS, 0.62)]), state(no_verify=True), SCAN, 0)
    assert raw.release


def test_explore_job_needs_scan_p_at_least_half():
    assert not release_rule(inputs(scan_p=0.49), "explore", 0.9, POS, R43)[0]
    assert release_rule(inputs(scan_p=0.49), "explore", 0.9, POS, R43)[2] == "scan_low"
    assert release_rule(inputs(scan_p=0.5), "explore", 0.9, POS, R43)[0]
    assert not release_rule(inputs(scan_p=0.9), "explore", 0.59, POS, R43)[0]
    assert release_rule(inputs(scan_p=0.1), "explore", 0.65, POS, R42)[0]        # v4.2: no scan condition
    assert not release_rule(inputs(scan_p=0.9), "verify", 0.99, POS, R43)[0]    # verify never releases


def test_washout_skip_below_065_is_unchanged():
    fast = {"tick": 0, "pos_vox": list(POS), "kind": "washout", "value": 0.05, "band": None, "word": "fast"}
    assert not release_rule(inputs(reading_log=[fast]), "dose", 0.64, POS, R43)[0]
    assert release_rule(inputs(reading_log=[fast]), "dose", 0.65, POS, R43)[0]


# --- fold-in (S-talk) --------------------------------------------------------------------------------------------------

def _asked(rules, logged):
    """A talker that asked about zone 0 with estimate = the logged value (est under v4.3, the raw look under v4.2)."""
    st = state(rules, no_verify=True)
    if rules.use_combined:
        base = dict(look=0.95, est=logged, scan_p=0.6, reading_log=[log_est(POS, logged)], marks=[])
    else:
        base = dict(look=logged, scan_p=0.6, reading_log=[log_look(POS, logged)], marks=[])
    decide_turn(inputs(**base, messages=[]), st, SCAN, 0)
    assert st.pending is not None and st.pending["estimate"] == pytest.approx(logged)
    return st


def test_fold_in_never_permits_a_release_in_v43():
    """The asked estimate is 0.5; a yes answer folds to (0.5 + 0.8) / 2 = 0.65. With the current look below the release
    threshold (v4.2: 0.45 < 0.5, v4.3: c = 0.55 < 0.6) v4.2 lets the folded value permit a release, v4.3 does not."""
    st = _asked(R42, 0.5)
    d = decide_turn(inputs(look=0.45, scan_p=0.6, round=1, tick=5, marks=[], reading_log=[log_look(POS, 0.5), log_look(POS, 0.45, 5)],
                           messages=[answer("yes")]), st, SCAN, 0)
    assert d.release and st.counters["answers_used"] == 1
    st = _asked(R43, 0.5)
    d = decide_turn(inputs(look=0.95, est=0.55, scan_p=0.6, round=1, tick=5, marks=[],
                           reading_log=[log_est(POS, 0.5), log_est(POS, 0.55, tick=5)], messages=[answer("yes")]), st, SCAN, 0)
    assert not d.release and st.counters["answers_used"] == 1
    assert st.counters["release_changed_by_answer"] == 0              # the folded value changed nothing for the release test


def test_fold_in_can_lower_and_drops_the_target_for_one_turn_only():
    st = _asked(R43, 0.6)
    d = decide_turn(inputs(look=0.95, est=0.9, scan_p=0.6, round=1, tick=5, marks=[],
                           reading_log=[log_est(POS, 0.6), log_est(POS, 0.9, tick=5)], messages=[answer("no")]), st, SCAN, 0)
    assert not d.release                                         # min(0.9, (0.6 + 0.2) / 2 = 0.4) = 0.4 < 0.6
    assert st.counters["release_changed_by_answer"] == 1
    assert d.target_label != "A" and not st.skip_zones and not st.skip_spots     # ineligible this turn, never skipped for good
    d2 = decide_turn(inputs(look=0.95, est=0.9, scan_p=0.6, round=2, tick=9, marks=[], messages=[]),
                     state(R43, no_verify=True), SCAN, 0)
    assert d2.release and d2.target_label == "A"


def test_fold_in_between_threshold_and_c_uses_the_lower_value():
    """c = 0.9 and a folded 0.65 (>= 0.6): the release stands and the zone stays eligible (fold_drop_c is 0.6)."""
    st = _asked(R43, 0.5)
    d = decide_turn(inputs(look=0.95, est=0.9, scan_p=0.6, round=1, tick=5, marks=[],
                           reading_log=[log_est(POS, 0.5), log_est(POS, 0.9, tick=5)], messages=[answer("yes")]), st, SCAN, 0)
    assert d.release and d.target_label == "A"                    # min(0.9, 0.65) = 0.65 >= 0.6


def test_marks_from_others_never_allow_a_release_the_rule_forbids():
    spot = {"kind": "tumour_here", "pos_vox": [10, 10, 10], "strength": 1.0, "distinct_authors": 2, "strong": True,
            "disputed": False, "authors": [3]}
    d = decide_turn(inputs(look=0.95, est=0.5, scan_p=0.7, marks=[spot]), state(no_verify=True), SCAN, 0)
    assert not d.release                                          # the release test still uses the ant's own c


# --- judgement marks ---------------------------------------------------------------------------------------------------

def test_tumour_here_band():
    assert judgement_marks(inputs(scan_p=0.49), 0.8, R43) == ["tumour_here"]
    assert judgement_marks(inputs(scan_p=0.49), 0.79, R43) == []
    assert judgement_marks(inputs(scan_p=0.5), 0.9, R43) == []                 # scan p must be < 0.5
    assert judgement_marks(inputs(scan_p=0.95), 0.99, R43) == []               # no "look >= 0.8 anywhere" branch
    assert judgement_marks(inputs(scan_p=0.49), 0.65, R43) == []               # v4.2's 0.65 branch is gone
    assert judgement_marks(inputs(scan_p=0.95), 0.8, R42) == ["tumour_here"]   # v4.2 untouched
    assert judgement_marks(inputs(scan_p=0.49), 0.65, R42) == ["tumour_here"]


def test_not_tumour_band():
    assert judgement_marks(inputs(scan_p=0.5), 0.2, R43) == ["not_tumour"]
    assert judgement_marks(inputs(scan_p=0.5), 0.21, R43) == []
    assert judgement_marks(inputs(scan_p=0.49), 0.1, R43) == []
    assert judgement_marks(inputs(scan_p=0.5), 0.3, R42) == ["not_tumour"]


def test_need_more():
    empty = dict(payload_fraction=0.0, dose_rel=0.5, scan_p=0.7)
    assert judgement_marks(inputs(**empty), 0.6, R43) == ["need_more"]
    assert judgement_marks(inputs(**empty), 0.59, R43) == []
    assert judgement_marks(inputs(**{**empty, "payload_fraction": 0.5}), 0.9, R43) == []
    assert judgement_marks(inputs(**{**empty, "dose_rel": 1.0}), 0.9, R43) == []
    assert judgement_marks(inputs(**empty), 0.55, R42) == ["need_more"]


def test_decide_turn_marks_come_from_c_not_the_raw_look():
    d = decide_turn(inputs(look=0.99, est=0.5, scan_p=0.3), state(no_verify=True), SCAN, 0)
    assert d.marks == ()
    d = decide_turn(inputs(look=0.1, est=0.85, scan_p=0.3), state(no_verify=True), SCAN, 0)
    assert [k for k, _ in d.marks] == ["tumour_here"]


# --- answers -----------------------------------------------------------------------------------------------------------

def test_v43_answers_0_6_and_0_4_with_unknown_between():
    kw = dict(has_own_need_more=False, voxel_mm=1.0, answer_rule="v43")
    a = (10, 10, 10)

    def look(est, raw=0.95, pos=(11, 10, 10)):
        return log_est(pos, est, raw=raw)

    assert answer_question("is_tumour", a, [look(0.6)], **kw) == ("yes", pytest.approx(0.6))
    assert answer_question("is_tumour", a, [look(0.4)], **kw) == ("no", pytest.approx(0.4))
    assert answer_question("is_tumour", a, [look(0.5)], **kw)[0] == "unknown"
    assert answer_question("is_tumour", a, [look(0.59)], **kw)[0] == "unknown"
    assert answer_question("is_tumour", a, [look(0.41)], **kw)[0] == "unknown"
    assert answer_question("is_tumour", a, [look(0.8), look(0.4, pos=(10, 12, 10))], **kw) == ("yes", pytest.approx(0.6))
    assert answer_question("is_tumour", a, [look(0.9, pos=(30, 10, 10))], **kw) == ("unknown", None)     # none within 6 mm
    assert answer_question("is_tumour", a, [], **kw) == ("unknown", None)
    # the estimate is read, not the raw value (raw 0.95 would say yes)
    assert answer_question("is_tumour", a, [look(0.3, raw=0.95)], **kw)[0] == "no"
    # a log without est falls back to the raw value
    assert answer_question("is_tumour", a, [log_look((11, 10, 10), 0.7)], **kw)[0] == "yes"
    assert answer_question("should_i_come", a, [look(0.6)], **kw)[0] == "yes"
    assert answer_question("should_i_come", a, [look(0.59)], **kw)[0] == "no"
    assert answer_question("should_i_come", a, [look(0.1)], **{**kw, "has_own_need_more": True})[0] == "yes"
    assert answer_question("dosed_enough", a, [], **kw) == ("unknown", None)


def test_v42_answers_are_untouched_even_when_est_is_logged():
    kw = dict(has_own_need_more=False, voxel_mm=1.0)
    a = (10, 10, 10)
    assert answer_question("is_tumour", a, [log_est((11, 10, 10), 0.3, raw=0.55)], **kw)[0] == "yes"       # raw 0.55 >= 0.5
    assert answer_question("should_i_come", a, [log_est((11, 10, 10), 0.3, raw=0.55)], **kw)[0] == "yes"
    assert answer_question("is_tumour", a, [log_est((11, 10, 10), 0.3, raw=0.45)], **kw)[0] == "no"


def test_colony_config_answer_rule_keys_and_to_dict():
    assert "answer_rule" not in ColonyConfig().to_dict()                       # v4.2 headers and hashes unchanged
    assert ColonyConfig.from_dict({"answer_rule": "v43"}).to_dict()["answer_rule"] == "v43"
    ColonyConfig.from_dict({"ant_rules": {"version": "v43"}, "enabled": False, "arm": "S"})   # ignored like enabled / arm
    with pytest.raises(ValueError):
        ColonyConfig.from_dict({"answer_rule": "v44"})
    with pytest.raises(ValueError):
        ColonyConfig.from_dict({"bogus": 1})


# --- verify ------------------------------------------------------------------------------------------------------------

def _verify_world(**kw):
    """One grey spot (by est; the raw value 0.95 is not grey), nothing else to do: only the verify job has a stimulus."""
    base = dict(look=0.95, est=0.5, scan_p=0.6, candidates=[], visited_regions=frozenset({0, 1}), zone_need={},
                reading_log=[log_est(POS, 0.5)])
    base.update(kw)
    return inputs(**base)


def test_verify_spots_come_from_est_and_the_local_mean_must_be_grey():
    st = state(R43)
    st.decision_turns = 100                                           # keep the share cap out of this test
    d = decide_turn(_verify_world(), st, SCAN, 0)
    assert d.job == "verify" and st.verify_by_spot[POS] == 1
    # a grey look next to a very high one: the 4 mm mean is not grey, so it is not a spot
    st2 = state(R43)
    st2.decision_turns = 100
    log = [log_est(POS, 0.5), log_est((11, 10, 10), 0.95, tick=0)]     # mean 0.725
    d = decide_turn(_verify_world(reading_log=log), st2, SCAN, 0)
    assert d.job != "verify"


def test_verify_per_spot_cap_of_two():
    st = state(R43)
    st.decision_turns = 100
    jobs = []
    for k in range(4):
        d = decide_turn(_verify_world(tick=5 * k, round=k), st, SCAN, 0)
        jobs.append(d.job)
    assert jobs[:2] == ["verify", "verify"] and "verify" not in jobs[2:]
    assert st.verify_by_spot[POS] == 2 and st.verify_turns == 2
    assert d.info["stimuli"]["verify"] == 0.0                         # blocked: stimulus 0
    # v4.2 rules have no cap
    st42 = state(R42)
    raw_grey = dict(reading_log=[log_look(POS, 0.5)], look=0.5)
    assert [decide_turn(_verify_world(tick=5 * k, round=k, **raw_grey), st42, SCAN, 0).job for k in range(6)].count("verify") >= 3


def test_verify_share_cap_of_25_percent_arithmetic():
    """Verify is allowed only if (verify_turns + 1) <= 0.25 * (decision_turns + 1), counts over previous turns.
    d = 0, 1, 2: 1 <= 0.25, 0.5, 0.75 is false -> no verify. d = 3: 1 <= 1.0 -> verify (v becomes 1).
    Then 2 <= 0.25 (d + 1) needs d >= 7: d = 4, 5, 6 blocked, d = 7 verify (v = 2); next needs d >= 11. So d = 3, 7, 11, ..."""
    rules = replace(R43, verify_per_spot_max=None)                    # isolate the share cap
    st = state(rules)
    verify_at = []
    for d in range(40):
        v_before, d_before = st.verify_turns, st.decision_turns
        assert d_before == d
        out = decide_turn(_verify_world(tick=5 * d, round=d), st, SCAN, 0)
        allowed = (v_before + 1) <= 0.25 * (d_before + 1)
        assert (out.job == "verify") == allowed
        if out.job == "verify":
            verify_at.append(d)
    assert verify_at == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    assert st.verify_turns == 10 and st.decision_turns == 40
    assert st.verify_turns <= 0.25 * st.decision_turns


def test_verify_turn_is_counted_only_with_a_target_spot():
    st = state(R43)
    st.decision_turns = 100
    decide_turn(_verify_world(reading_log=[]), st, SCAN, 0)           # nothing grey: no spot, nothing counted
    assert st.verify_turns == 0 and not st.verify_by_spot


# --- thresholds --------------------------------------------------------------------------------------------------------

def _prev(**kw):
    p = {"job": "dose", "released": True, "look": 0.9, "pos": POS, "spot": None}
    p.update(kw)
    return p


def _theta(st, job):
    return round(st.thresholds.theta[job], 10)


@pytest.mark.parametrize("word,expect", [("good", 0.45), ("middling", 0.45), ("poor", 0.55), ("slow", 0.55), ("as expected", 0.5)])
def test_dose_theta_moves_only_on_a_resolved_uptake_reading(word, expect):
    st = state(R43)
    st.prev = _prev()
    _settle_previous(inputs(uptake={"word": word, "about_vox": [10, 11, 10], "value": 0.5}), st, 0.9)
    assert _theta(st, "dose") == expect


def test_dose_unresolved_without_reading_far_reading_or_release():
    for kw, uptake in ((dict(), None),
                       (dict(), {"word": "good", "about_vox": [40, 10, 10], "value": 1.0}),       # about a voxel > 6 mm away
                       (dict(released=False, pos=None), {"word": "good", "about_vox": [10, 10, 10], "value": 1.0})):
        st = state(R43)
        st.prev = _prev(**kw)
        _settle_previous(inputs(uptake=uptake), st, 0.9)
        assert _theta(st, "dose") == 0.5, (kw, uptake)
        assert st.counters["theta_unresolved"] == 1


def test_v42_mode_still_updates_every_turn():
    st = state(R42)
    st.prev = _prev()
    _settle_previous(inputs(uptake=None), st, 0.9)                    # release on a look >= 0.5, no slow reading: success
    assert _theta(st, "dose") == 0.4                                  # v4.2: step down 0.1
    st.prev = _prev(released=False)
    _settle_previous(inputs(uptake=None), st, 0.9)
    assert _theta(st, "dose") == 0.45                                 # a turn without success: up 0.05


@pytest.mark.parametrize("c,scan,expect", [(0.6, 0.5, 0.45), (0.9, 0.9, 0.45), (0.34, 0.9, 0.55), (0.1, 0.1, 0.55),
                                           (0.5, 0.9, 0.5),             # between 0.35 and 0.6: unresolved
                                           (0.9, 0.49, 0.5)])           # high c but scan p < 0.5: not a success, not a failure
def test_explore_theta_resolution_on_an_uncovered_spot(c, scan, expect):
    st = state(R43)
    st.prev = _prev(job="explore", released=False, pos=None)
    log = [log_est(POS, c, tick=5, scan_p=scan)]
    _settle_previous(inputs(tick=5, scan_p=scan, reading_log=log), st, c)
    assert _theta(st, "explore") == expect


def test_explore_covered_spot_is_unresolved():
    st = state(R43)
    st.prev = _prev(job="explore", released=False, pos=None)
    log = [log_est((11, 10, 10), 0.5, tick=0), log_est(POS, 0.9, tick=5)]           # an own look 2 mm away at an earlier tick
    _settle_previous(inputs(tick=5, scan_p=0.9, reading_log=log), st, 0.9)
    assert _theta(st, "explore") == 0.5
    st = state(R43)
    st.prev = _prev(job="explore", released=False, pos=None)
    log = [log_est((30, 10, 10), 0.5, tick=0), log_est(POS, 0.9, tick=5)]           # an earlier look too far away: not covered
    _settle_previous(inputs(tick=5, scan_p=0.9, reading_log=log), st, 0.9)
    assert _theta(st, "explore") == 0.45


def test_verify_theta_success_failure_and_unresolved():
    def run(est_values, count):
        st = state(R43)
        st.verify_by_spot[POS] = count
        st.prev = _prev(job="verify", released=False, pos=None, spot=POS)
        log = [log_est(POS, v, tick=i) for i, v in enumerate(est_values)]
        _settle_previous(inputs(tick=9, reading_log=log), st, 0.5)
        return _theta(st, "verify")

    assert run([0.5, 0.9], 1) == 0.45        # mean 0.7: out of the grey band -> success
    assert run([0.5, 0.1], 1) == 0.45        # mean 0.3: out of the grey band -> success
    assert run([0.5, 0.6], 2) == 0.55        # mean 0.55 still grey and the spot has had 2 verify turns -> failure
    assert run([0.5, 0.6], 1) == 0.5         # still grey but only 1 verify turn so far -> unresolved
    assert run([], 1) == 0.5                 # no own look near the spot -> unresolved


def test_theta_steps_are_0_05_both_ways_and_clamped():
    st = state(R43)
    st.thresholds.theta["dose"] = 0.1
    st.prev = _prev()
    _settle_previous(inputs(uptake={"word": "good", "about_vox": list(POS), "value": 1.0}), st, 0.9)
    assert _theta(st, "dose") == 0.1                                  # floor
    st.thresholds.theta["dose"] = 1.0
    st.prev = _prev()
    _settle_previous(inputs(uptake={"word": "poor", "about_vox": list(POS), "value": 0.1}), st, 0.9)
    assert _theta(st, "dose") == 1.0                                  # ceiling
    st.thresholds.theta["dose"] = 0.5
    for word, want in (("poor", 0.55), ("poor", 0.6), ("good", 0.55), ("good", 0.5)):
        st.prev = _prev()
        _settle_previous(inputs(uptake={"word": word, "about_vox": list(POS), "value": 0.5}), st, 0.9)
        assert _theta(st, "dose") == want
    assert (Thresholds().step_up, Thresholds().step_down, Thresholds().floor, Thresholds().ceil) == (0.05, 0.1, 0.1, 1.0)
    custom = Thresholds(step_up=0.2, step_down=0.3, floor=0.2, ceil=0.8)
    custom.update("dose", True)
    assert custom.theta["dose"] == pytest.approx(0.2)
    for _ in range(4):
        custom.update("dose", False)
    assert custom.theta["dose"] == pytest.approx(0.8)


def test_colony_script_takes_rules_and_none_means_v42():
    assert ColonyScript(SCAN, 0).rules == R42 and ColonyScript(SCAN, 0).state.rules == R42
    s = ColonyScript(SCAN, 1, R43)
    assert s.state.rules == R43 and s.state.thresholds.step_down == 0.05 and s.state.thresholds.theta["dose"] == 0.5
    assert AntState().rules == R42 and AntState().thresholds.step_down == 0.1


# --- cards -------------------------------------------------------------------------------------------------------------

def _card(min_resolved, confirmed, contradicted=0, claims=6):
    c = Card("ant-00", "S-marks", min_resolved=min_resolved)
    c.in_run = {"claims": claims, "confirmed": confirmed, "contradicted": contradicted}
    return c


def test_card_band_two_vs_three_resolved_claims():
    assert _card(3, 2).band() == "mixed"                              # 2 resolved, both confirmed: too few
    assert _card(3, 3).band() == "usually right"                      # 3 resolved, 100% confirmed
    assert _card(3, 3, 1).band() == "usually right"                   # 4 resolved, 75% confirmed
    assert _card(3, 2, 1).band() == "mixed"                           # 3 resolved but 67% < 70%
    assert _card(3, 3, claims=4).band() == "new"                      # fewer than 5 claims is still "new"
    assert _card(1, 1).band() == "usually right"                      # legacy: one resolved claim is enough
    assert _card(1, 0, 0).band() == "mixed"                           # legacy: nothing resolved -> mixed


def test_card_serialisation_omits_the_legacy_min_resolved():
    assert "min_resolved" not in Card("ant-00", "S-marks").to_dict()
    d = Card("ant-00", "S-marks", min_resolved=3).to_dict()
    assert d["min_resolved"] == 3 and Card.from_dict(d).min_resolved == 3
    assert Card.from_dict(Card("ant-00", "S-marks").to_dict()).min_resolved == 1


# --- configs -----------------------------------------------------------------------------------------------------------

def test_v43_configs_load():
    import yaml
    from tests.v4_3.a2.test_legacy_golden import REPO
    frozen = WorldConfig.from_yaml(REPO / "config" / "v4_2" / "world.yaml")
    for name, f2, f3 in (("world.yaml", True, True), ("world_gateA.yaml", False, False)):
        path = REPO / "config" / "v4_3" / name
        w = WorldConfig.from_yaml(path)
        doc = yaml.safe_load(path.read_text())
        c = ColonyConfig.from_dict({**doc["colony"], "quiet_rule": False})
        assert c.answer_rule == "v42"                                 # the harness sets it from the ant rules
        assert AntRules.from_dict(doc["colony"]["ant_rules"]) == (AntRules.v43_foraging() if f2 else R43)   # B2 is on in the Step B world only
        assert (w.facts.f1_simlook, w.facts.f2, w.facts.f3) == (True, f2, f3)
        assert w.drug_scale_k == frozen.drug_scale_k and w.dose_threshold == frozen.dose_threshold
        assert w.with_(facts=frozen.facts) == frozen                  # nothing else differs from the frozen v4.2 world
