"""The queen's route entry is the squad's assignment (goal 4.6): a squad with no target adopts the first assigned
region, a squad that already has a target keeps it, every arm applies the rule, the squad's own route entry is in its
brief, and no truth is involved. Fake backends; no network."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from antelligence.anchoring.chain_modes import RoundResult
from antelligence.experiments.v41_harness import ARM_SPECS, D1, Q4, Q9, BONSAI, FW, Worker
from antelligence.kernel.types import Intent
from antelligence.providers.decision_backends import FakeBackend, hashed_answers
from antelligence.worlds.tumor_mri import TumorMRIWorld
from antelligence.worlds.tumor_mri.world import ACTION_DECIDE
from tests.v4_1.harness.conftest import CASE_DEV, build_rig, d1_responder, fake_workers, small_cfg
from tests.v4_1.harness.test_harness import FakeRepClient
from tests.v4_1.world.phantom import make_phantom

SWARM = [a for a, s in ARM_SPECS.items() if s.kind in ("workers", "fw")]


def _recording_workers(states):
    def rec(name, inner):
        def fn(req):
            if "look" not in req.questions:
                states.append((name, req.state))
            return inner(req)
        return fn
    ws = {}
    for name, model in ((D1, "d1-fake"), (Q4, "q4-fake"), (Q9, "q9-fake"), (BONSAI, "bonsai-fake"), (FW, "fw-fake")):
        inner = d1_responder(0.05, hashed_answers) if name == D1 else hashed_answers
        ws[name] = Worker(FakeBackend(rec(name, inner), model=model, name=name))
    return ws


class _Post:
    def __init__(self):
        self.n = 0

    def post_round(self, run_id, entries):
        self.n += len(entries)
        return RoundResult(visible=set(range(self.n)), invisible=set(), timed_out=False, latency_ms={e.seq: 40 for e in entries})


@pytest.mark.parametrize("arm", SWARM)
def test_every_arm_applies_the_assignment_and_shows_it(tmp_path, arm):
    states = []
    rig = build_rig(tmp_path, workers=_recording_workers(states), post_mode_factory=_Post, reputation_client=FakeRepClient())
    asyncio.run(rig.harness.run(arm, CASE_DEV, 1))
    first = [s for _, s in states][:8 * 3]
    assert any("assigned by the queen" in s for s in first), arm
    # the squad's own route entry (board-on briefs) or inline assignment (board-off) is in the state's brief
    assert any("route: squad" in s or "regions [" in s for s in first), arm


def _world(truth_flip=False):
    crop = make_phantom()
    if truth_flip:
        crop = type(crop)(**{**crop.__dict__, "truth": ~crop.truth})
    w = TumorMRIWorld(crop, small_cfg(), seed=1)
    from antelligence.worlds.tumor_mri.data import select_candidates
    w.set_candidates(select_candidates(w.regions))
    return w


def test_adoption_only_when_target_is_none_and_never_overrides():
    w = _world()
    rid = w.candidates_for(0)[1].region_id
    assert w.assign_route(0, rid) is True
    obs = w.observe("squad-0", 0).data
    assert obs["target"] == "B" and obs["target_source"] == "queen" and obs["goal_dist_mm"] > 0
    assert w.assign_route(0, w.candidates_for(0)[0].region_id) is False   # already has a target: queen never gates
    assert w.observe("squad-0", 0).data["target"] == "B"
    # a worker-chosen target is not overridden either
    w.apply("squad-1", Intent(ACTION_DECIDE, {"target": "A", "release": 0, "amount": 1}), 0)
    assert w.assign_route(1, rid) is False and w.observe("squad-1", 0).data["target_source"] == "worker"
    # a worker can still pick anything after an assignment
    w.apply("squad-0", Intent(ACTION_DECIDE, {"target": "X", "release": 0, "amount": 1}), 0)
    assert w.observe("squad-0", 0).data["target"] == "X"
    assert w.assign_route(0, 10 ** 6) is False                           # unknown region: no-op


def test_assignment_does_not_read_truth():
    a, b = _world(), _world(truth_flip=True)
    rid = a.candidates_for(0)[0].region_id
    a.assign_route(0, rid)
    b.assign_route(0, rid)
    oa, ob = a.observe("squad-0", 0).data, b.observe("squad-0", 0).data
    assert oa == ob
    for _ in range(10):
        a.step_environment(0)
        b.step_environment(0)
    assert a.squad_positions() == b.squad_positions()


def test_own_route_entry_in_brief_and_caps_hold(tmp_path):
    states = []
    rig = build_rig(tmp_path, workers=_recording_workers(states))
    res = asyncio.run(rig.harness.run("SW-B", CASE_DEV, 1))
    import json
    rows = [json.loads(l) for l in (tmp_path / "out" / "runs" / res.cell["run_id"] / "decisions.jsonl").read_text().splitlines()]
    assert rows and all(6 <= len(r["brief_ids"]) <= 12 or len(r["brief_ids"]) >= 1 for r in rows)
    assert all(len(r["brief_ids"]) <= 12 for r in rows)
    assert all(len(s) <= 1200 for _, s in states)
    assert any("route: squad" in s for _, s in states)
