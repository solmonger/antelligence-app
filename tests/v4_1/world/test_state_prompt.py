"""Regression: the worker prompt must describe Stay as the spec does (keep the current target; no target = stand still),
and the state must carry the squad's position, its current target and the task (goal 4.2 "State per decision").
Found by the first live SW-B episode (BRATS_016): 183/192 decisions chose S because the prompt said 'stay here'."""

from __future__ import annotations

from antelligence.kernel.types import Intent
from antelligence.worlds.tumor_mri import TumorMRIWorld, build_request
from antelligence.worlds.tumor_mri.data import select_candidates
from antelligence.worlds.tumor_mri.world import ACTION_DECIDE
from tests.v4_1.world.phantom import make_phantom, phantom_config


def _world():
    w = TumorMRIWorld(make_phantom(), phantom_config(), seed=1)
    w.set_candidates(select_candidates(w.regions))
    return w


def test_stay_is_not_described_as_stand_still():
    w = _world()
    req = build_request("m", w.observe("squad-0", 0).data)
    stay = req.questions["target"].criteria["S"]
    assert "current target" in stay and "stay here" not in stay


def test_state_reports_position_task_and_empty_target_at_round_zero():
    w = _world()
    obs = w.observe("squad-0", 0).data
    assert obs["goal_dist_mm"] is None
    state = build_request("m", obs).state
    assert "Task: deliver drug to the tumour" in state
    assert f"voxel ({obs['position_vox'][0]}, {obs['position_vox'][1]}, {obs['position_vox'][2]})" in state
    assert "Current target: none, standing still." in state


def test_state_reports_target_after_a_route_is_chosen_and_stay_keeps_walking():
    w = _world()
    label = w.candidates_for(0)[0].label
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": label, "release": 0, "amount": 1}), 0)
    obs = w.observe("squad-0", 0).data
    assert obs["goal_dist_mm"] and obs["goal_dist_mm"] > 0
    assert "Current target: moving," in build_request("m", obs).state
    for t in range(5):
        w.step_environment(t)
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "S", "release": 0, "amount": 1}), 5)
    before = w.squad_positions()[0]
    w.step_environment(5)
    assert w.squad_positions()[0] != before   # S kept the target: the squad still walks


# Goal 4.2 wording, verbatim (the prompt text is hashed into config/cache keys; see questions.py).
SPEC_STAY = "keep the current target"
SPEC_EXPLORE = ("go to the nearest perceived region no squad has visited. If there is none, "
                "go to the nearest highest-uncertainty voxel cluster")


def test_s_and_x_criteria_match_goal_4_2_verbatim():
    crit = build_request("m", _world().observe("squad-0", 0).data).questions["target"].criteria
    assert crit["S"] == SPEC_STAY
    assert crit["X"] == SPEC_EXPLORE
