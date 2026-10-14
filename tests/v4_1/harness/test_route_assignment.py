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


def _perturbed(kind):
    """A world over the same scan whose truth mask is flipped / shifted / emptied. Nothing else differs."""
    crop = make_phantom()
    truth = {"flip": ~crop.truth, "shift": np.roll(crop.truth, 7, axis=0), "empty": np.zeros_like(crop.truth)}[kind]
    w = TumorMRIWorld(type(crop)(**{**crop.__dict__, "truth": truth}), small_cfg(), seed=1)
    from antelligence.worlds.tumor_mri.data import select_candidates
    w.set_candidates(select_candidates(w.regions))
    return w


@pytest.mark.parametrize("kind", ["flip", "shift", "empty"])
def test_no_model_visible_thing_depends_on_truth(kind):
    """Observations, rendered prompts (state + question text), candidates and routes stay byte-identical across several
    ticks and every squad when the truth mask changes; the scorer's output is the only thing that moves."""
    import json
    from antelligence.worlds.tumor_mri import build_request
    a, b = _world(), _perturbed(kind)
    assert a.candidates_for(0) == b.candidates_for(0)
    n = len(a.agents())
    assert n > 1
    cands = a.candidates_for(0)
    for i in range(n):                       # the queen's route and the squads' own choices are the same in both worlds
        assert a.assign_route(i, cands[i % len(cands)].region_id) == b.assign_route(i, cands[i % len(cands)].region_id)
    seen = 0
    for tick in range(0, 30):
        if tick % a.config.decision_every == 0:
            for i, ag in enumerate(a.agents()):
                oa, ob = a.observe(ag, tick).data, b.observe(ag, tick).data
                assert json.dumps(oa, sort_keys=True, default=str) == json.dumps(ob, sort_keys=True, default=str)
                ra, rb = build_request("m", oa, "brief text", seed=3), build_request("m", ob, "brief text", seed=3)
                assert ra.state == rb.state and ra.questions == rb.questions
                seen += 1
                dec = {"target": "S" if tick else "A", "release": int(i % 2 == 0 and tick > 0), "amount": 3}
                assert a.apply(ag, Intent(ACTION_DECIDE, dec), tick).accepted == b.apply(ag, Intent(ACTION_DECIDE, dec), tick).accepted
        a.step_environment(tick)
        b.step_environment(tick)
        assert a.squad_positions() == b.squad_positions()
    assert seen >= n * 3
    assert np.array_equal(a.field.dose, b.field.dose)          # identical drug, because decisions never saw truth
    ma, mb = a.metrics(), b.metrics()
    assert ma["released_total"] > 0
    assert (ma["coverage"], ma["off_target_share"]) != (mb["coverage"], mb["off_target_share"])   # the scorer does see it


def test_own_route_entry_in_brief_and_caps_hold(tmp_path):
    states = []
    rig = build_rig(tmp_path, workers=_recording_workers(states))
    res = asyncio.run(rig.harness.run("SW-B", CASE_DEV, 1))
    import json
    rows = [json.loads(l) for l in (tmp_path / "out" / "runs" / res.cell["run_id"] / "decisions.jsonl").read_text().splitlines()]
    assert rows and all(1 <= len(r["brief_ids"]) <= 12 for r in rows)
    assert all(len(r["brief_ids"]) <= 12 for r in rows)
    assert all(len(s) <= 1200 for _, s in states)
    assert any("route: squad" in s for _, s in states)


# --- briefs: caps at the exact boundary, and the squad's own route under mode D --------------------------------------

def _brief_board(n_extra):
    """A board with a task, perception knowledge, ``n_extra`` evidence entries and squad 0's own route."""
    from antelligence.kernel.board import Board
    from tests.v4_1.vectors import generate_vectors as gv
    board = Board(gv.test_header(), mode="A")
    board.append("task", "queen", {"objective": "Treat the tumour core", "squads": [0, 1], "until_tick": 50}, 0)
    for r in range(3):
        board.append("knowledge", "perception", {"region_id": r, "centroid_dmm": [r, 2, 3], "cls": "ET", "p_tumour_ppm": 500000 + r,
                                                 "volume_mm3": 10, "unc_ppm": 1000, "source": "perception", "prov": "u@1"}, 1)
    own = board.append("route", "queen", {"squad": 0, "region_ids": [1], "priority": 1, "brief_ids": []}, 2)
    for i in range(n_extra):
        board.append("evidence", "squad-1", {"squad": 1, "pos_vox": [1, 2, 3], "look_p_ppm": 100 + i, "p_tumour_ppm": 200,
                                             "dose_ug": i, "released_ug": 0}, 3 + i)
    return board, own


DRAFT = {"squad": 0, "region_ids": [1], "priority": 1, "brief_ids": []}


def test_brief_entry_cap_exact_boundary_and_own_route_survives():
    from antelligence.kernel import briefs as B
    for hi in (6, 12):
        sizes = {}
        for n_extra in range(0, 20):
            board, own = _brief_board(n_extra)
            ids = B.build_brief(board, 0, DRAFT, cap=(6, hi), own_route=own).brief_ids
            sizes[n_extra] = len(ids)
            assert own.id in ids and len(ids) <= hi
        available = 1 + 3 + 1   # task, knowledge x3, own route
        assert sizes[hi - available - 1] == hi - 1      # one under the cap
        assert sizes[hi - available] == hi              # exactly at the cap
        assert sizes[hi - available + 1] == hi          # one over: trimmed, not rejected
        assert sizes[19] == hi
    board, own = _brief_board(0)
    assert len(B.build_brief(board, 0, DRAFT, cap=(6, 12), own_route=own).brief_ids) == 5   # fewer than 6 exist: all of them
    # the own route is dropped last of the optional entries: a cap of 2 keeps task + own route
    board, own = _brief_board(10)
    assert B.build_brief(board, 0, DRAFT, cap=(1, 2), own_route=own).brief_ids[:2] == (board.by_type("task")[0].id, own.id)
    off = B.build_brief_board_off(board, 0, DRAFT, cap=(6, 4), own_route=own)   # board-off: task, own route, signals
    assert len(off.brief_ids) == 4 and own.id in off.brief_ids


def test_brief_token_cap_exact_boundary_and_no_carry_over():
    from antelligence.kernel import briefs as B
    board, own = _brief_board(10)
    full = B.build_brief(board, 0, DRAFT, cap=(6, 12), own_route=own)
    at = B.build_brief(board, 0, DRAFT, cap=(6, 12), own_route=own, max_tokens=full.tokens)
    assert at.brief_ids == full.brief_ids and at.tokens == full.tokens           # at the cap: nothing dropped
    over = B.build_brief(board, 0, DRAFT, cap=(6, 12), own_route=own, max_tokens=full.tokens - 1)
    assert len(over.brief_ids) < len(full.brief_ids) and over.tokens <= full.tokens - 1   # one token over: tail dropped
    assert own.id in over.brief_ids
    again = B.build_brief(board, 0, DRAFT, cap=(6, 12), own_route=own)          # reset: a later brief is not shrunk by the earlier one
    assert again.brief_ids == full.brief_ids and again.text == full.text


@pytest.mark.parametrize("board_on", [True, False])
def test_mode_d_own_route_appears_only_once_included(board_on):
    from types import SimpleNamespace
    from antelligence.experiments.v41_harness import V41Harness
    board, own = _brief_board(4)
    ctx = SimpleNamespace(board=board, harness={"briefs": {"count": 12, "entry_types": ["evidence", "knowledge", "decision", "alarm", "route", "task"]}},
                          routes={0: dict(DRAFT)}, route_entries={0: own}, visible=None, reputation=None,
                          spec=SimpleNamespace(board_on=board_on))
    brief = lambda: V41Harness._brief(SimpleNamespace(_view=lambda c: V41Harness._view(None, c)), ctx, 0)
    everything = {e.seq for e in board.entries()}
    ctx.visible = everything - {own.seq}                  # mode D: the chain has not included the route yet
    assert own.id not in brief().brief_ids
    ctx.visible = everything                              # included: now it is in the brief
    assert own.id in brief().brief_ids
    ctx.visible = None                                    # modes without a chain: always readable
    assert own.id in brief().brief_ids
    ctx.visible = everything - {own.seq}                  # and it is withdrawn again if not visible
    assert own.id not in brief().brief_ids
