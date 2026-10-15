"""S-talk answer consumer (decision 2026-10-14): the combined estimate replaces the look in the release test and dose-target
eligibility for the pending target on that turn. The 0.35 drop rule is gone. S-solo / S-marks never see answers."""

import hashlib
import json

import pytest

from antelligence.worlds.tumor_mri.colony_policies import ColonyScript, decide_turn
from tests.v4_2.q1.helpers import SCAN, harness, inputs, log_look, phantom
from tests.v4_2.q1.test_decision import answer, no_verify


def _asked(look_logged, **kw):
    """A talker that asked about zone 0 with estimate = the logged look, then (next turn) gets an answer."""
    st = no_verify()
    base = dict(look=look_logged, scan_p=0.6, reading_log=[log_look((10, 10, 10), look_logged)], marks=[])
    decide_turn(inputs(**base, messages=[]), st, SCAN, 0)
    assert st.pending is not None
    return st, base


def test_yes_answer_at_estimate_half_lets_release_happen_where_the_raw_look_would_not():
    st, base = _asked(0.5)
    raw = dict(base, look=0.45)                                   # current raw look 0.45 < 0.5: no release on its own
    d0 = decide_turn(inputs(**raw, round=1, tick=5, messages=[]), no_verify(), SCAN, 0)
    assert not d0.release and d0.info["reason"] == "look_low"
    d1 = decide_turn(inputs(**raw, round=1, tick=5, messages=[answer("yes")]), st, SCAN, 0)
    assert d1.release and d1.info["reason"] == "ok"               # (0.5 + 0.8) / 2 = 0.65 >= 0.5
    assert st.counters["answers_used"] == 1 and not st.skip_zones


def test_no_answer_at_055_blocks_release_where_the_raw_look_would_release():
    st, base = _asked(0.55)
    raw = dict(base, look=0.55)
    d0 = decide_turn(inputs(**raw, round=1, tick=5, messages=[]), no_verify(), SCAN, 0)
    assert d0.release
    d1 = decide_turn(inputs(**raw, round=1, tick=5, messages=[answer("no")]), st, SCAN, 0)
    assert not d1.release                                         # (0.55 + 0.2) / 2 = 0.375 < 0.5
    assert st.counters["answers_used"] == 1


def test_low_combined_estimate_is_not_dropped_for_good_only_ineligible_on_that_turn():
    st, base = _asked(0.5)
    d = decide_turn(inputs(**base, round=1, tick=5, messages=[answer("no")]), st, SCAN, 0)    # estimate 0.35: old rule dropped it
    assert d.target_label != "A"                                  # not eligible this turn
    assert not st.skip_zones and not st.skip_spots                # but never added to a permanent skip set
    d2 = decide_turn(inputs(**base, round=2, tick=9, messages=[]), no_verify_state(st), SCAN, 0)
    assert d2.target_label == "A"                                 # next turn the raw look decides again


def no_verify_state(st):
    st.thresholds.theta["verify"] = 1.0
    return st


def test_judgement_marks_still_use_the_raw_look():
    st, base = _asked(0.5)
    raw = dict(base, look=0.9, scan_p=0.4)                        # raw 0.9 >= 0.8: tumour_here whatever the answer says
    d = decide_turn(inputs(**raw, round=1, tick=5, messages=[answer("no")]), st, SCAN, 0)
    assert [k for k, _ in d.marks] == ["tumour_here"]


# --- S-solo / S-marks are unchanged -------------------------------------------------------------------------------------

# Digest of every decision (action, params, emit, meta) of a 120-tick phantom episode, seed 3, recorded on the code BEFORE
# this change (integration cfcb464). No answer ever reaches S-solo / S-marks, so the shared function must not move.
GOLDEN = {"S-solo": "97e5c9f6b6ab218559bba5d96cd3a03414387d45e9f60d112b57228c7b972f37",
          "S-marks": "7747535e611e794bcbd4e7ff76e5b79f959d8775b71685c175b7e548b8b6846e"}


def _digest(arm, monkeypatch):
    rows = []
    orig = ColonyScript.decide

    def decide(this, view):
        it = orig(this, view)
        rows.append([this.ant, view.tick, it.action, it.params, [(d.kind, d.pos, d.ttl) for d in it.emit], it.meta])
        return it

    monkeypatch.setattr(ColonyScript, "decide", decide)
    harness(120, noisy=True).run_episode(arm, phantom(), 3)
    assert rows
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize("arm", ("S-solo", "S-marks"))
def test_solo_and_marks_decisions_are_byte_identical_to_before(arm, monkeypatch):
    assert _digest(arm, monkeypatch) == GOLDEN[arm]
