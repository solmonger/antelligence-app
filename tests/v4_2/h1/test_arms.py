"""H1 control arms on the synthetic phantom with a fake close look (no data, no network)."""

import numpy as np
import pytest

from antelligence.experiments import v42_arms  # noqa: F401  (registers)
from antelligence.experiments.v42_arms import CONTROL_ARMS
from antelligence.experiments.v42_harness import ARMS, resolve_arm
from antelligence.worlds.tumor_mri import OBS_KEYS
from antelligence.worlds.tumor_mri.colony_obs import expected_keys
from antelligence.worlds.tumor_mri.colony_policies import ColonyScript
from tests.v4_2.q1.helpers import harness, noisy_look_factory, phantom


@pytest.fixture(scope="module")
def case():
    return phantom()


def run(arm, case, seed=1, ticks=60, **facts):
    return harness(ticks, noisy=True, **facts).run_episode(arm, case, seed)


def test_arms_registered():
    for a in CONTROL_ARMS:
        assert resolve_arm(a).kind == "ant"
    assert ARMS["S-marks-perfect"].verdict is False
    assert ARMS["S-marks-displaced"].driver_arm == "S-marks-displaced" and ARMS["S-talk-shuffled"].driver_arm == "S-talk-shuffled"


def test_displaced_differs_only_in_mark_positions(case):
    base, disp = run("S-marks", case), run("S-marks-displaced", case)
    bm, dm = base["colony"]["marks"], disp["colony"]["marks"]
    assert dm and disp["control"]["displaced"] > 0
    # before any ant could read a displaced mark the two runs are the same run: the first tick's marks agree on author, kind, tick and amount
    t0 = min(m["tick"] for m in bm)
    first_b = sorted((m["ant"], m["mark_kind"], m["tick"], m["amount_milli"]) for m in bm if m["tick"] == t0)
    first_d = sorted((m["ant"], m["mark_kind"], m["tick"], m["amount_milli"]) for m in dm if m["tick"] == t0)
    assert first_b == first_d
    # positions: every displaced deposit sits where the control put it, inside the brain
    brain = case.crop.brain
    assert all(brain[tuple(m["pos_vox"])] for m in dm)
    moved = [m for m in dm if m["tick"] == t0]
    assert sorted(map(tuple, (m["pos_vox"] for m in moved))) != sorted(map(tuple, (m["pos_vox"] for m in bm if m["tick"] == t0))) or len(moved) == 0


def test_shuffled_never_asks_the_asker(case):
    rec = run("S-talk-shuffled", case)
    asks = [m for m in rec["colony"]["messages"] if m["type"] == "ask"]
    assert asks, "the phantom run must contain asks for this test to mean anything"
    assert all(m["from_ant"] != m["to_ant"] for m in rec["colony"]["messages"])


def test_perfect_rewrites_from_truth(case):
    rec = run("S-marks-perfect", case, f2=True, f2_source="B", f2_share=0.2, u_core=0.5, f3=True, f3_decay=0.04, f3_share=0.1)
    c = rec["control"]
    assert c["fast_delta"] == pytest.approx(0.5 * 0.04)               # goal 5.3
    truth = case.crop.truth
    for m in rec["colony"]["marks"]:
        if m["mark_kind"] in ("tumour_here", "not_tumour"):
            assert (m["mark_kind"] == "tumour_here") == bool(truth[tuple(m["pos_vox"])])


@pytest.mark.parametrize("arm", CONTROL_ARMS)
def test_barrier_holds_for_control_arms(arm, case, monkeypatch):
    seen = []
    orig = ColonyScript.decide

    def spy(self, view):
        seen.append(set(view.observation))
        return orig(self, view)

    monkeypatch.setattr(ColonyScript, "decide", spy)
    run(arm, case)
    assert seen and all(k == set(expected_keys(arm)) for k in seen)
    assert set(OBS_KEYS) <= seen[0]


def test_perfect_leaves_author_observation_alone(case, monkeypatch):
    """The wrapper acts on the policy output; the observation an ant gets in S-marks-perfect is the S-marks key set and, at the
    first tick, the very same values as in S-marks."""
    firsts = {}
    orig = ColonyScript.decide

    def spy(self, view):
        firsts.setdefault((current[0], self.ant, view.tick), {k: repr(v) for k, v in view.observation.items() if k not in ("card_self", "cards_public")})  # cards carry the arm name
        return orig(self, view)

    current = ["S-marks"]
    monkeypatch.setattr(ColonyScript, "decide", spy)
    run("S-marks", case)
    current[0] = "S-marks-perfect"
    run("S-marks-perfect", case)
    t0 = min(t for (_, _, t) in firsts)
    same = [firsts[("S-marks", a, t0)] == firsts[("S-marks-perfect", a, t0)] for a in range(8) if ("S-marks", a, t0) in firsts]
    assert same and all(same)


def test_controls_leave_world_queen_hook_unset(case):
    for a in CONTROL_ARMS:
        assert run(a, case)["queen_hook_unset"] is True
