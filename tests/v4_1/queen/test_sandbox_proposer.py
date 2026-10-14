"""S23 (mistake-2041): the proposer prompt carries no example edit, key_effects is honest, repeats are rejected unrun."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
import yaml

from antelligence.experiments import v41_harness as harness_mod
from antelligence.experiments.v41_sandbox import ARMS, KEY_EFFECTS, PROPOSER_SYSTEM, EvalResult, key_effects, llm_proposer, run_sandbox
from antelligence.kernel import briefs
from antelligence.kernel.board import Board
from antelligence.kernel.policies import v41_queen
from antelligence.kernel.policies.v41_harness import harness_sha256, load_allowed_keys, load_harness
from antelligence.providers.wrappers import Cached, FakeProvider
from tests.v4_1.vectors import generate_vectors as gv

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = load_allowed_keys()
DEV_A = [f"A{i:02d}" for i in range(30)]
DEV_B = [f"B{i:02d}" for i in range(10)]
BASE_SHA = "540273f36628ff287e08f30347767aab9fbfaf580bd8e119629f036f36cbfc11"   # harness.yaml at 6731ab2


def patch(path, value, d=15000):
    return {"patch": [{"op": "replace", "path": path, "value": value}], "prediction": {"metric": "coverage", "delta_ppm": d}}


class Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, harness, arm, cases):
        self.calls.append((harness_sha256(harness), arm, tuple(cases)))
        return EvalResult({c: 0.5 for c in cases}, {})


def run(proposals):
    board = Board(gv.test_header(), mode="A")
    runner, seen, it = Runner(), [], iter(proposals)

    def propose(ctx):
        seen.append(ctx)
        return next(it, None)
    return run_sandbox(board, load_harness(), ALLOWED, DEV_A, DEV_B, runner, propose), board, runner, seen


def leaves(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from leaves(v)
    elif isinstance(node, list):
        yield node
        for v in node:
            yield from leaves(v)
    else:
        yield node


# --- prompt --------------------------------------------------------------------------------------

def test_prompt_has_no_concrete_path_or_value():
    doc = yaml.safe_load((ROOT / "config/v4_1/harness.yaml").read_text())
    for path in doc["allowed_keys"]:
        assert path not in PROPOSER_SYSTEM and path.rsplit("/", 1)[-1] not in PROPOSER_SYSTEM
    for v in leaves(doc["harness"]):
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            for lit in {repr(v), f"{v:g}", f"{v:.2f}"}:
                assert not re.search(rf"(?<![\w.]){re.escape(lit)}(?![\w.])", PROPOSER_SYSTEM), lit
        elif isinstance(v, str):
            assert v not in PROPOSER_SYSTEM, v
    assert "<one path from allowed_keys>" in PROPOSER_SYSTEM and "<signed non-zero integer>" in PROPOSER_SYSTEM
    for rule in ("exactly one edit", "previous_edits", "key_effects"):
        assert rule in PROPOSER_SYSTEM


# --- key_effects pinned to the code --------------------------------------------------------------

def effect(key, arm):
    return KEY_EFFECTS[key][arm][0]


def test_key_effects_cover_every_allowed_key_and_match_the_code():
    assert set(KEY_EFFECTS) == set(ALLOWED)
    view = key_effects(ALLOWED)
    assert set(view) == set(ALLOWED) and all(set(view[k]) == set(ARMS) for k in ALLOWED)
    assert view["/router/cost_penalty"]["SW-B"]["changes_behaviour"] is False

    # cost_penalty and fw_conf_min: read nowhere outside the schema validator (v41_harness.py docstring)
    reads = re.compile(r"""["'](cost_penalty|fw_conf_min)["']""")
    skip = {"kernel/policies/v41_harness.py"}
    for f in (ROOT / "antelligence").rglob("*.py"):
        if str(f.relative_to(ROOT / "antelligence")) not in skip:
            assert not reads.search(f.read_text()), f
    for arm in ARMS:
        assert not effect("/router/cost_penalty", arm) and not effect("/router/fw_conf_min", arm)

    # slm_conf_min: used in the shared decision path, on the non-calibration, non-FW branch, for both swarm arms
    src = inspect.getsource(harness_mod)
    assert src.count('ctx.harness["router"]["slm_conf_min"]') == 1
    assert {"SW-B", "SW-noBoard"} <= set(harness_mod.ROUTER_MODELS_BY_ARM)
    assert effect("/router/slm_conf_min", "SW-B") and effect("/router/slm_conf_min", "SW-noBoard")

    # entry_types: build_brief takes it; the board-off builder cannot see it
    assert "entry_types" in inspect.signature(briefs.build_brief).parameters
    assert "entry_types" not in inspect.signature(briefs.build_brief_board_off).parameters
    assert "entry_types" in inspect.getsource(harness_mod.V41Harness._brief).split("if ctx.spec.board_on:")[1].split("return briefs_mod.build_brief_board_off")[0]
    assert effect("/briefs/entry_types", "SW-B") and not effect("/briefs/entry_types", "SW-noBoard")

    # count: both builders cap on it
    brief_src = inspect.getsource(harness_mod.V41Harness._brief)
    assert brief_src.count('cap=(6, h["count"])') == 2
    assert effect("/briefs/count", "SW-B") and effect("/briefs/count", "SW-noBoard")

    # objective_template and the task in a brief: board-off keeps the task entry too
    assert 'harness["briefs"]["objective_template"]' in inspect.getsource(v41_queen)
    assert "chosen.append(task)" in inspect.getsource(briefs.build_brief_board_off)
    assert effect("/briefs/objective_template", "SW-B") and effect("/briefs/objective_template", "SW-noBoard")

    # cadence and playbook: the queen runs in both arms and reads both
    assert harness_mod.ARM_SPECS["SW-B"].uses_queen and harness_mod.ARM_SPECS["SW-noBoard"].uses_queen
    assert 'ctx.harness["queen"]["cadence"]' in inspect.getsource(harness_mod)
    assert 'self.harness["playbook_notes"]' in inspect.getsource(v41_queen)
    for key in ("/queen/cadence", "/playbook_notes"):
        assert effect(key, "SW-B") and effect(key, "SW-noBoard")


def test_proposer_context_carries_key_effects():
    _, _, _, seen = run([None])
    assert seen[0]["key_effects"] == key_effects(ALLOWED)


# --- duplicate and no-op guard -------------------------------------------------------------------

def test_duplicate_and_noop_proposals_are_rejected_without_episodes():
    cur = load_harness()["router"]["cost_penalty"]
    res, board, runner, seen = run([patch("/router/cost_penalty", 0.3),          # round 1: runs, reverted (no gain)
                                    patch("/router/cost_penalty", 0.3, d=-5000),   # round 2: same patch, new prediction: duplicate
                                    patch("/router/cost_penalty", cur),            # round 3: equals the current value: no-op
                                    patch("/router/cost_penalty", 0.3)])           # round 4: duplicate again
    r1, r2, r3, r4 = res.rounds
    assert not r1.rejected and r1.score_points == 0.0 and not r1.kept
    assert r2.rejected == "duplicate of round 1" and r2.score_points is None and not r2.kept
    assert "changes nothing" in r3.rejected and r3.score_points is None
    assert r4.rejected == "duplicate of round 1"
    # episodes: base + round-1 candidate, each on both arms over dev-A; nothing for the rejected rounds
    assert len(runner.calls) == 4 and res.evaluations == 4
    edits = board.by_type("edit")
    assert [e.payload["round"] for e in edits] == [1, 2, 3, 4] and [r.edit_entry_id for r in res.rounds] == [e.id for e in edits]
    assert [e.payload["kept"] for e in edits] == [0, 0, 0, 0]
    # previous_edits shows the rejection and the patch, so the proposer can see what not to repeat
    prev = seen[3]["previous_edits"]
    assert [p["round"] for p in prev] == [1, 2, 3]
    assert prev[1]["rejected"] == "duplicate of round 1"
    assert prev[1]["patch"] == [{"op": "replace", "path": "/router/cost_penalty", "value": 0.3}]
    assert prev[0]["patch"] == prev[1]["patch"] and "changes nothing" in prev[2]["rejected"]


def test_a_kept_edit_cannot_be_repeated_either():
    runner_gain = lambda h: 0.03 if h["router"]["cost_penalty"] > 0.15 else 0.0
    board = Board(gv.test_header(), mode="A")
    calls = []

    def evaluate(h, arm, cases):
        calls.append(arm)
        return {c: 0.5 + runner_gain(h) for c in cases}
    it = iter([patch("/router/cost_penalty", 0.3), patch("/router/cost_penalty", 0.3)])
    res = run_sandbox(board, load_harness(), ALLOWED, DEV_A, DEV_B, evaluate, lambda ctx: next(it, None))
    assert res.rounds[0].kept and res.rounds[1].rejected == "duplicate of round 1"


# --- determinism: a repeat after a rejection is a fresh request ----------------------------------

def test_repeat_after_rejection_is_a_fresh_request_not_a_cached_replay():
    first, second = patch("/router/cost_penalty", 0.3), patch("/queen/cadence", 36)

    def responder(req):
        ctx = json.loads(req.messages[1]["content"])
        return json.dumps(second if ctx["previous_edits"] else first)

    fake = FakeProvider(responder, model="fake")
    cached = Cached(fake)                                   # the reply cache the real run sits behind
    propose = llm_proposer(cached, "fake")
    board = Board(gv.test_header(), mode="A")
    runner = Runner()
    # round 1 proposes `first`; round 2's request differs (previous_edits non-empty) so it reaches the model
    res = run_sandbox(board, load_harness(), ALLOWED, DEV_A, DEV_B, runner, propose, max_rounds=2)
    assert len(fake.requests) == 2 and fake.requests[0].request_hash != fake.requests[1].request_hash
    assert [r.keys for r in res.rounds] == [("/router/cost_penalty",), ("/queen/cadence",)]

    # and a model that repeats itself is rejected, then sees the rejection in its next request
    stubborn = FakeProvider(lambda req: json.dumps(first) if len(json.loads(req.messages[1]["content"])["previous_edits"]) < 2
                            else json.dumps(second), model="fake")
    res2 = run_sandbox(Board(gv.test_header(), mode="A"), load_harness(), ALLOWED, DEV_A, DEV_B, Runner(),
                       llm_proposer(Cached(stubborn), "fake"), max_rounds=3)
    assert [bool(r.rejected) for r in res2.rounds] == [False, True, False]
    assert res2.rounds[1].rejected == "duplicate of round 1"
    hashes = {r.request_hash for r in stubborn.requests}
    assert len(stubborn.requests) == 3 == len(hashes)
    assert json.loads(stubborn.requests[2].messages[1]["content"])["previous_edits"][1]["rejected"] == "duplicate of round 1"


# --- base harness unchanged ----------------------------------------------------------------------

def test_base_harness_and_hash_unchanged():
    assert harness_sha256(load_harness()) == BASE_SHA
    assert load_allowed_keys() == ["/router/slm_conf_min", "/router/fw_conf_min", "/router/cost_penalty", "/briefs/count",
                                   "/briefs/entry_types", "/briefs/objective_template", "/queen/cadence", "/playbook_notes"]
    res, *_ = run([patch("/router/cost_penalty", 0.3)])
    assert harness_sha256(res.base_harness) == BASE_SHA == harness_sha256(res.final_harness)
