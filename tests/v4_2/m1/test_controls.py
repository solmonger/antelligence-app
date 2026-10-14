import dataclasses

import numpy as np
import pytest

from antelligence.experiments.v42_controls import DisplaceMarks, PerfectMarks, quiet_pair, shuffled_other
from antelligence.kernel.colony.episode import default_auto_marks, run_colony_episode
from antelligence.kernel.colony.quorum import ZoneSpec
from antelligence.kernel.signal import SignalDraft
from tests.v4_2.colony.fake_world import ARMS, BASE_KEYS, FakeWorld, Scripted


def sphere(c, r):
    return lambda p: sum((a - b) ** 2 for a, b in zip(p, c)) ** 0.5 <= r


BRAIN = np.array([(x, y, 0) for x in range(0, 60) for y in range(-10, 11)])
Z1 = ZoneSpec(1, (10, 0, 0), sphere((10, 0, 0), 5))
Z2 = ZoneSpec(2, (40, 0, 0), sphere((40, 0, 0), 5))
Z3 = ZoneSpec(3, (13, 0, 0), sphere((13, 0, 0), 5))      # overlaps Z1


def disp(zones, seed=1):
    return DisplaceMarks(zones, BRAIN, 1.0, "caseA", seed)


def test_displaced_stays_in_same_zone_and_is_seeded():
    d = disp([Z1, Z2])
    for i in range(30):
        new = d(i, 0, "tumour_here", (10, 1, 0))
        assert Z1.contains(new) and tuple(new) in {tuple(v) for v in BRAIN}
    assert d(5, 0, "dosed", (40, 0, 0)) == disp([Z1, Z2])(5, 0, "dosed", (40, 0, 0))     # same stream -> same voxel
    moved = {tuple(d(i, 0, "x", (10, 0, 0))) for i in range(50)}
    assert len(moved) > 5                                                                # actually random


def test_displaced_outside_every_zone_within_6mm_of_spot():
    d = disp([Z1, Z2])
    spot = (25, 3, 0)
    assert d.zone_of(spot) is None
    for i in range(40):
        new = d(i, 1, "washout", spot)
        assert np.linalg.norm(np.array(new) - np.array(spot)) <= 6.0 + 1e-9


def test_displaced_overlap_uses_nearest_regions_zone():
    d = disp([Z1, Z3])
    assert d.zone_of((11, 0, 0)) == 1                  # in both; closer to centroid (10,0,0)
    assert d.zone_of((12, 0, 0)) == 3                  # closer to (13,0,0)
    assert d.zone_of((11.5, 0, 0)) == 1                # tie 1.5 vs 1.5 -> lower region id
    for i in range(30):
        assert Z3.contains(d(i, 0, "x", (12, 0, 0)))
    assert any(not Z1.contains(d(i, 0, "x", (12, 0, 0))) for i in range(60))   # not confined to the overlap with Z1


def test_displaced_no_brain_voxel_keeps_spot():
    d = DisplaceMarks([Z1], np.zeros((0, 3), int), 1.0, "c", 1)
    assert d(0, 0, "x", (30, 0, 0)) == (30, 0, 0)


def test_displaced_keeps_kind_amount_time_in_a_real_episode():
    def pol(view):
        i = int(view.agent_id.split("-")[1])
        if i == 0 and view.tick == 0:
            return {"release": 1}, [SignalDraft(kind="tumour_here", pos=(5.0, 0.0, 0.0), ttl=1)], None
        return {"release": 0}, [], None
    kw = dict(arm=ARMS["S-marks"], base_keys=BASE_KEYS)
    w = FakeWorld(looks={(0, 0, 0): 0.9})
    base = run_colony_episode(w, lambda a: Scripted(pol, []), **kw).colony["marks"]
    d = DisplaceMarks([ZoneSpec(1, (10, 0, 0), sphere((10, 0, 0), 8))], BRAIN, 1.0, "fake", 0)
    w2 = FakeWorld(looks={(0, 0, 0): 0.9})
    ctrl = run_colony_episode(w2, lambda a: Scripted(pol, []), mark_pos_fn=d, **kw).colony["marks"]
    assert len(base) == len(ctrl) and len(d.log) == len(ctrl)
    for b, c in zip(base, ctrl):
        assert {k: b[k] for k in ("mark_kind", "ant", "amount_milli", "tick")} == {k: c[k] for k in ("mark_kind", "ant", "amount_milli", "tick")}
    assert any(b["pos_vox"] != c["pos_vox"] for b, c in zip(base, ctrl))


def test_shuffle_never_picks_the_asker_and_is_seeded():
    for asker in range(8):
        picks = {shuffled_other("c", 3, m, asker, 8) for m in range(1, 200)}
        assert asker not in picks and picks <= set(range(8)) and len(picks) > 3
    assert shuffled_other("c", 3, 7, 2, 8) == shuffled_other("c", 3, 7, 2, 8)
    assert shuffled_other("c", 3, 7, 1, 2) == 0
    with pytest.raises(ValueError):
        shuffled_other("c", 3, 7, 0, 1)


class Truth:
    def label(self, pos):
        return pos[0] >= 8

    def u(self, pos):
        return 0.4 if pos[0] >= 8 else 1.0

    def delta(self, pos):
        return 0.08 if pos[0] >= 8 else 0.0


def pol(view):
    i = int(view.agent_id.split("-")[1])
    if i == 0 and view.tick in (0, 5):
        return {"release": 0}, [SignalDraft(kind="tumour_here", pos=(5.0, 0.0, 0.0), ttl=1)], None
    return {"release": 0}, [], None


def test_perfect_hook_rewrites_what_others_read_but_not_the_author_inputs():
    kw = dict(arm=ARMS["S-marks"], base_keys=BASE_KEYS)
    looks = {(0, 0, 0): 0.9, (10, 0, 0): 0.7}
    seen_a, seen_b = [], []
    plain = run_colony_episode(FakeWorld(looks=looks), lambda a: Scripted(pol, seen_a), **kw)
    pm = PerfectMarks(Truth())
    perfect = run_colony_episode(FakeWorld(looks=looks), pm.wrap_policy_for(lambda a: Scripted(pol, seen_b)),
                                 auto_mark_rule=pm.wrap_auto_rule(default_auto_marks), **kw)
    assert {r["mark_kind"] for r in plain.colony["marks"] if r["ant"] == 0} == {"tumour_here"}
    assert {r["mark_kind"] for r in perfect.colony["marks"] if r["ant"] == 0} == {"not_tumour"}     # truth at (5,0,0): not tumour
    assert pm.rewritten == 2
    own_keys = ("look", "reading_log", "uptake_reading", "washout_reading", "visited_self", "zone_need", "colony_note")
    a0 = [(t, {k: o[k] for k in own_keys}) for ag, t, o in seen_a if ag == "squad-0"]
    b0 = [(t, {k: o[k] for k in own_keys}) for ag, t, o in seen_b if ag == "squad-0"]
    assert a0 == b0 and a0
    # another ant at tick 5 reads the corrected mark
    other = {k: v for k, v in [(ag, o) for ag, t, o in seen_b if t == 5]}
    kinds = {r["kind"] for r in other["squad-1"]["marks_near"]}
    assert "not_tumour" in kinds and "tumour_here" not in kinds


def test_perfect_auto_marks_keep_only_what_truth_confirms():
    pm = PerfectMarks(Truth(), fast_delta=0.02)
    rule = pm.wrap_auto_rule(lambda ctx: [("slow_uptake", (10, 0, 0), 1.0, 600000), ("slow_uptake", (2, 0, 0), 1.0, 600000),
                                          ("washout", (10, 0, 0), 1.0, 1), ("washout", (2, 0, 0), 1.0, 1),
                                          ("dosed", (2, 0, 0), 0.3, 0)])
    out = rule({})
    assert [(k, p) for k, p, _, _ in out] == [("slow_uptake", (10, 0, 0)), ("washout", (10, 0, 0)), ("dosed", (2, 0, 0))]
    assert out[0][3] == 400000 and out[1][3] == 80000 and out[2][3] == 0
    assert pm.dropped == 2


def test_quiet_pair_differs_only_in_quiet_rule():
    p = quiet_pair()
    assert not p["S-marks"].quiet_rule and p["S-marks-quiet"].quiet_rule
    assert dataclasses.replace(p["S-marks-quiet"], quiet_rule=False) == p["S-marks"]
