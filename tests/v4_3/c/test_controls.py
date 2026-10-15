"""C: the v4.3 controls. Uptake marks displaced to a random zone, perfect uptake marks from a hand-made field, registry robustness."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pytest

from antelligence.experiments import v42_harness, v43_arms
from antelligence.experiments.v42_controls import DisplaceMarks
from antelligence.experiments.v43_controls import (DisplaceMarksV43, PerfectMarksV43, UPTAKE_KINDS, true_uptake_class)
from antelligence.experiments.v43_harness import V43Harness
from antelligence.kernel.colony.quorum import ZoneSpec
from antelligence.worlds.tumor_mri import OBS_KEYS
from antelligence.worlds.tumor_mri.colony_obs import expected_keys
from antelligence.worlds.tumor_mri.colony_policies import ColonyScript
from antelligence.worlds.tumor_mri.foraging import foraging_uptake
from tests.v4_2.q1.helpers import noisy_look_factory, phantom_cfg
from tests.v4_3.a2.helpers import R43, estimate
from tests.v4_3.b1.helpers import load_yaml_configs, phantom_with_channels


# --- a small hand-made world: a 3 x 1 x 1 row of brain blocks, one zone round each centre ---------------------------------------

CENTRES = [(5, 5, 5), (25, 5, 5), (45, 5, 5)]
VMM = 1.0


def zones() -> List[ZoneSpec]:
    return [ZoneSpec(i + 1, c, (lambda c: lambda p: VMM * math.dist(p, c) <= 4.0 + 1e-9)(c)) for i, c in enumerate(CENTRES)]


def brain() -> np.ndarray:
    pts = [(x, y, z) for c in CENTRES for x in range(c[0] - 4, c[0] + 5) for y in range(c[1] - 4, c[1] + 5) for z in range(c[2] - 4, c[2] + 5)
           if VMM * math.dist((x, y, z), c) <= 4.0 + 1e-9]
    return np.array(pts)


def zone_of(pos: Sequence[int]) -> int:
    return next(z.region_id for z in zones() if z.contains(pos))


def disp(case: str = "c1", seed: int = 5) -> DisplaceMarksV43:
    return DisplaceMarksV43(zones(), brain(), VMM, case, seed)


def test_uptake_marks_land_in_a_random_zone_of_any_zone():
    d = disp()
    src = (5, 5, 5)                                              # the mark's own spot is in zone 1
    dest = [d(i, 0, "good_uptake" if i % 2 else "slow_uptake", src) for i in range(120)]
    assert all(d_.__class__ is tuple for d_ in dest)
    zs = [zone_of(p) for p in dest]
    assert set(zs) == {1, 2, 3}                                  # all three zones are reachable, not just the mark's own
    assert min(zs.count(z) for z in (1, 2, 3)) >= 20             # roughly uniform over zones (120 draws, expect 40 each)
    brain_set = {tuple(p) for p in brain()}
    assert all(p in brain_set for p in dest)
    assert {r["to_zone"] for r in d.log} == {1, 2, 3}


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_uptake_displacement_is_deterministic_per_seed_and_differs_across_seeds(seed):
    a = [disp(seed=seed)(i, 0, "good_uptake", (5, 5, 5)) for i in range(30)]
    b = [disp(seed=seed)(i, 3, "good_uptake", (25, 5, 5)) for i in range(30)]    # author and origin do not matter, only (case, seed, index)
    assert a == b
    other = [disp(seed=seed + 100)(i, 0, "good_uptake", (5, 5, 5)) for i in range(30)]
    assert a != other
    assert a != [disp(case="c2", seed=seed)(i, 0, "good_uptake", (5, 5, 5)) for i in range(30)]


def test_other_marks_move_exactly_as_in_v42():
    v42 = DisplaceMarks(zones(), brain(), VMM, "c1", 5)
    v43 = disp()
    for i, kind in enumerate(["dosed", "dose_ok", "tumour_here", "not_tumour", "need_more", "washout"] * 5):
        pos = CENTRES[i % 3]
        assert v43(i, 1, kind, pos) == v42(i, 1, kind, pos)
        assert zone_of(v43(i, 1, kind, pos)) == zone_of(pos)           # same zone, as in v4.2


def test_uptake_marks_are_never_returned_to_a_voxel_outside_the_brain_even_when_a_zone_is_empty():
    zs = zones() + [ZoneSpec(9, (90, 90, 90), lambda p: False)]          # a zone that holds no brain voxel
    d = DisplaceMarksV43(zs, brain(), VMM, "c1", 5)
    brain_set = {tuple(p) for p in brain()}
    assert all(d(i, 0, "slow_uptake", (5, 5, 5)) in brain_set for i in range(60))
    assert 9 not in {r["to_zone"] for r in d.log}


# --- perfect marks ---------------------------------------------------------------------------------------------------------------

class FieldTruth:
    def __init__(self, u: Dict[tuple, float], labels: Dict[tuple, bool] | None = None) -> None:
        self._u, self._labels = u, labels or {}

    def label(self, pos) -> bool:
        return self._labels.get(tuple(pos), False)

    def u(self, pos) -> float:
        return self._u.get(tuple(pos), 1.0)

    def delta(self, pos) -> float:
        return 0.0


def test_true_uptake_class_thresholds():
    assert true_uptake_class(1.0) == true_uptake_class(0.7) == "good_uptake"
    assert true_uptake_class(0.6999) is None and true_uptake_class(0.4) is None and true_uptake_class(0.5) is None
    assert true_uptake_class(0.3999) == true_uptake_class(0.05) == "slow_uptake"


def test_perfect_uptake_marks_take_the_true_class_and_true_u_from_a_hand_made_field():
    field = {(1, 1, 1): 0.9, (2, 2, 2): 0.2, (3, 3, 3): 0.55, (4, 4, 4): 0.7, (5, 5, 5): 0.39}
    pm = PerfectMarksV43(FieldTruth(field), fast_delta=0.02)

    def rule(ctx):                                               # the author's own marks: wrong or right, at five voxels
        return [("slow_uptake", (1, 1, 1), 1.0, 123), ("good_uptake", (2, 2, 2), 1.0, 456), ("good_uptake", (3, 3, 3), 1.0, 700000),
                ("slow_uptake", (4, 4, 4), 1.0, 100000), ("good_uptake", (5, 5, 5), 1.0, 800000), ("dosed", (9, 9, 9), 0.25, 0)]

    out = pm.wrap_auto_rule(rule)({"readings": []})
    by_pos = {tuple(p): (k, ppm, a) for k, p, a, ppm in out}
    assert by_pos[(1, 1, 1)] == ("good_uptake", 900000, 1.0)        # u 0.9: good, reading is the true u
    assert by_pos[(2, 2, 2)] == ("slow_uptake", 200000, 1.0)
    assert (3, 3, 3) not in by_pos                                  # middling: dropped
    assert by_pos[(4, 4, 4)] == ("good_uptake", 700000, 1.0)
    assert by_pos[(5, 5, 5)] == ("slow_uptake", 390000, 1.0)
    assert by_pos[(9, 9, 9)] == ("dosed", 0, 0.25)                  # world facts pass unchanged
    assert pm.dropped == 1 and pm.rewritten == 4


def test_perfect_judgement_marks_still_come_from_the_true_label():
    pm = PerfectMarksV43(FieldTruth({}, {(1, 1, 1): True, (2, 2, 2): False}), fast_delta=0.02)
    assert pm._draft({"kind": "not_tumour", "pos_vox": (1, 1, 1)})["kind"] == "tumour_here"
    assert pm._draft({"kind": "tumour_here", "pos_vox": (2, 2, 2)})["kind"] == "not_tumour"
    assert pm._draft({"kind": "need_more", "pos_vox": (2, 2, 2)})["kind"] == "need_more"


def test_perfect_arm_on_the_phantom_marks_every_uptake_mark_with_the_true_class():
    case = phantom_with_channels()
    w, c, _ = load_yaml_configs("config/v4_3/world.yaml", 120)
    h = V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory)
    rec = h.run_episode("S-marks-perfect", case, 1)
    ch = case.scorer_channels
    u = foraging_uptake(case.crop.truth, ch["tc"], ch["et"], case.crop.case_id, "mid")
    ups = [m for m in rec["colony"]["marks"] if m["mark_kind"] in UPTAKE_KINDS]
    assert ups
    for m in ups:
        val = float(u[tuple(m["pos_vox"])])
        assert m["mark_kind"] == true_uptake_class(val)
        assert m["reading_ppm"] == int(round(val * 1_000_000))
    assert rec["control"]["rewritten"] + rec["control"]["dropped"] >= 0 and "fast_delta" in rec["control"]


def test_displaced_arm_on_the_phantom_moves_uptake_marks_and_reports_it():
    case = phantom_with_channels()
    w, c, _ = load_yaml_configs("config/v4_3/world.yaml", 120)
    h = V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory)
    base, mov = h.run_episode("S-marks", case, 1), h.run_episode("S-marks-displaced", case, 1)
    assert mov["control"]["uptake_to_random_zone"] > 0 and mov["control"]["displaced"] >= mov["control"]["uptake_to_random_zone"]
    t0 = min(m["tick"] for m in base["colony"]["marks"] if m["mark_kind"] == "good_uptake")   # the runs agree up to the first deposit
    a = sorted(tuple(m["pos_vox"]) for m in base["colony"]["marks"] if m["mark_kind"] == "good_uptake" and m["tick"] == t0)
    b = sorted(tuple(m["pos_vox"]) for m in mov["colony"]["marks"] if m["mark_kind"] == "good_uptake" and m["tick"] == t0)
    assert len(a) == len(b) and a != b
    again = h.run_episode("S-marks-displaced", case, 1)
    assert again["colony"]["marks"] == mov["colony"]["marks"]          # deterministic


# --- the information barrier for the controls --------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["S-marks-displaced", "S-marks-perfect", "S-talk-shuffled"])
def test_barrier_holds_for_control_arms_under_v43(arm, monkeypatch):
    seen = []
    orig = ColonyScript.decide

    def spy(self, view):
        seen.append(set(view.observation))
        return orig(self, view)

    monkeypatch.setattr(ColonyScript, "decide", spy)
    w, c, _ = load_yaml_configs("config/v4_3/world.yaml", 60)
    V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory).run_episode(arm, phantom_with_channels(), 1)
    assert seen and all(k == set(expected_keys(arm)) for k in seen)
    assert set(OBS_KEYS) <= seen[0]


def test_perfect_marks_leave_the_authors_first_observation_alone(monkeypatch):
    """The wrapper acts on the policy output, so at the first tick the author's observation is the same in S-marks and
    S-marks-perfect (cards carry the arm name, so they are left out of the comparison)."""
    firsts = {}
    orig = ColonyScript.decide
    current = ["S-marks"]

    def spy(self, view):
        if view.tick == 0:
            firsts[(current[0], view.agent_id)] = {k: repr(v) for k, v in view.observation.items() if k not in ("card_self", "cards_public")}
        return orig(self, view)

    monkeypatch.setattr(ColonyScript, "decide", spy)
    w, c, _ = load_yaml_configs("config/v4_3/world.yaml", 30)
    for arm in ("S-marks", "S-marks-perfect"):
        current[0] = arm
        V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory).run_episode(arm, phantom_with_channels(), 1)
    ids = sorted(a for (arm, a) in firsts if arm == "S-marks")
    assert ids and all(firsts[("S-marks", a)] == firsts[("S-marks-perfect", a)] for a in ids)


# --- registry robustness -----------------------------------------------------------------------------------------------------------

@pytest.fixture
def clean_registry():
    """Empty the shared v4.2 registries (another test module may have edited them) and restore them afterwards."""
    arms, controls = dict(v42_harness.ARMS), dict(v42_harness.ARM_CONTROLS)
    v42_harness.ARMS.clear()
    v42_harness.ARM_CONTROLS.clear()
    yield
    v42_harness.ARMS.clear()
    v42_harness.ARMS.update(arms)
    v42_harness.ARM_CONTROLS.clear()
    v42_harness.ARM_CONTROLS.update(controls)


def test_register_all_restores_missing_arms_and_is_idempotent(clean_registry):
    assert "S-marks-displaced" not in v42_harness.ARMS
    names = v43_arms.register_all()
    assert set(names) <= set(v42_harness.ARMS) and v42_harness.resolve_arm("S-marks-perfect").verdict is False
    before = dict(v42_harness.ARMS)
    v43_arms.register_all()
    assert v42_harness.ARMS == before
    assert not v42_harness.ARM_CONTROLS                                   # v4.3 never writes the global control table


def test_every_arm_runs_after_the_registries_were_cleared_and_v42_controls_cannot_leak_in(clean_registry):
    def poisoned(*a, **k):
        raise AssertionError("a v4.2 control hook was used by the v4.3 harness")

    v42_harness.ARM_CONTROLS["S-marks-displaced"] = poisoned
    v42_harness.ARM_CONTROLS["S-marks-perfect"] = poisoned
    v43_arms.register_all()
    w, c, _ = load_yaml_configs("config/v4_3/world.yaml", 40)
    case = phantom_with_channels()
    for arm in ("S", "S-solo", "S-marks", "S-talk", "S-marks-quiet", "S'", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect"):
        h = V43Harness(w, c, R43, estimate(), look_fn_factory=noisy_look_factory)
        rec = h.run_episode(arm, case, 1)
        assert rec["arm"] == arm and rec["metrics"]["coverage"] >= 0.0
    # and the v4.2 harness is untouched by all of this: it still looks the poisoned hook up in the global table
    v42 = v42_harness.V42Harness(phantom_cfg(20), look_fn_factory=noisy_look_factory)
    assert v42.arm_controls("S-marks-displaced") is poisoned and v42.arm_def("S-solo").name == "S-solo"


def test_v43_arms_exempt_set_and_freeze_need():
    for a in ("S", "S-solo", "S'"):
        assert not v43_arms.needs_freeze(a)
    for a in ("S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect", "S-marks-quiet"):
        assert v43_arms.needs_freeze(a)
    with pytest.raises(KeyError):
        v43_arms.needs_freeze("S-nope")
