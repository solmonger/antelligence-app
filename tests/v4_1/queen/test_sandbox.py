import json

import pytest

from antelligence.experiments.v41_sandbox import (
    ARMS, EvalResult, PairedScore, PatchError, apply_patch, keep_rule, llm_proposer, paired_score, parse_proposal, run_sandbox,
)
from antelligence.kernel.board import Board
from antelligence.kernel.policies.v41_harness import harness_sha256, load_allowed_keys, load_harness
from antelligence.providers.wrappers import FakeProvider
from tests.v4_1.vectors import generate_vectors as gv

ALLOWED = load_allowed_keys()
DEV_A = [f"A{i:02d}" for i in range(30)]
DEV_B = [f"B{i:02d}" for i in range(10)]


def pred(d=15000):
    return {"metric": "coverage", "delta_ppm": d}


def patch(path, value, op="replace", d=15000):
    return {"patch": [{"op": op, "path": path, "value": value}], "prediction": pred(d)}


def score(points, win_rate=1.0):
    return PairedScore(points, 0, 0, 0, win_rate, {}, 60)


def test_patch_rejects_forbidden_keys():
    for path in ("/scorer/threshold", "/physics/decay", "/verifier/strict", "/conformal/alpha", "/budgets/calls",
                 "/keys/queen", "/caps/max_calls", "/allowed_keys", "/allowed_keys/-", "/router/caps", "/queen/budget"):
        with pytest.raises(PatchError, match="forbidden"):
            parse_proposal(patch(path, 1), ALLOWED)
    for path in ("/router/unknown", "/briefs", "/harness/router/cost_penalty"):
        with pytest.raises(PatchError, match="not an allowed key"):
            parse_proposal(patch(path, 1), ALLOWED)
    # shape errors, op restrictions and a missing or non-numeric prediction
    ok = patch("/router/cost_penalty", 0.2)
    parse_proposal(ok, ALLOWED)
    parse_proposal(json.dumps(ok), ALLOWED)
    bad_ops = [{"op": "remove", "path": "/router/cost_penalty", "value": 0}, {"op": "move", "path": "/router/cost_penalty", "value": 0}]
    for op in bad_ops:
        with pytest.raises(PatchError):
            parse_proposal({"patch": [op], "prediction": pred()}, ALLOWED)
    for p in (None, {"metric": "coverage"}, {"metric": "coverage", "delta_ppm": 0}, {"metric": "coverage", "delta_ppm": 0.5},
              {"metric": "coverage", "delta_ppm": "10"}, {"metric": "", "delta_ppm": 5}):
        with pytest.raises(PatchError, match="prediction"):
            parse_proposal({"patch": ok["patch"], "prediction": p}, ALLOWED)
    with pytest.raises(PatchError):
        parse_proposal({"patch": ok["patch"]}, ALLOWED)
    with pytest.raises(PatchError):
        parse_proposal("not json", ALLOWED)
    # values outside the allowed ranges fail the harness schema
    base = load_harness()
    for path, value in (("/briefs/count", 13), ("/briefs/count", 5), ("/queen/cadence", 29), ("/queen/cadence", 41),
                        ("/router/slm_conf_min", 1.5), ("/briefs/entry_types", ["truth"]), ("/briefs/objective_template", "{secret}")):
        with pytest.raises(PatchError, match="invalid"):
            apply_patch(base, parse_proposal(patch(path, value), ALLOWED))
    new = apply_patch(base, parse_proposal(patch("/playbook_notes/-", "Look near the core first.", op="add"), ALLOWED))
    assert new["playbook_notes"][-1] == "Look near the core first." and len(base["playbook_notes"]) == 2   # base untouched
    assert apply_patch(base, parse_proposal(patch("/briefs/count", 10), ALLOWED))["briefs"]["count"] == 10


def test_keep_rule_thresholds():
    assert keep_rule(score(1.0, 0.6), 10_000)
    assert keep_rule(score(2.5, 0.9), 1)
    assert not keep_rule(score(0.99, 0.9), 10_000)          # under +1 point
    assert not keep_rule(score(1.5, 0.59), 10_000)          # under 60% wins
    assert not keep_rule(score(1.5, 0.9), -10_000)          # wrong sign
    assert not keep_rule(score(-2.0, 0.9), -10_000)         # right sign, but a loss never keeps
    assert not keep_rule(score(1.5, 0.9), 0)
    # win rate counts only differing pairs
    cur = {a: {"c1": 0.5, "c2": 0.5, "c3": 0.5, "c4": 0.5, "c5": 0.5} for a in ARMS}
    new = {a: {"c1": 0.7, "c2": 0.7, "c3": 0.7, "c4": 0.4, "c5": 0.5} for a in ARMS}
    s = paired_score(cur, new)
    assert (s.wins, s.losses, s.ties) == (6, 2, 2) and s.win_rate == 0.75
    assert s.score_points == pytest.approx((0.2 * 3 - 0.1) / 5 * 100)


class Runner:
    """Synthetic world: coverage depends on cost_penalty and cadence; the noBoard arm ignores briefs.count."""

    def __init__(self, gain=lambda h, arm: 0.0):
        self.calls = []
        self.gain = gain

    def __call__(self, harness, arm, cases):
        self.calls.append((harness_sha256(harness), arm, tuple(cases)))
        base = 0.50 + self.gain(harness, arm)
        return EvalResult({c: min(1.0, base + 0.001 * (i % 3)) for i, c in enumerate(cases)}, {"defer_rate": 0.1})


def run(proposals, gain, **kw):
    board = Board(gv.test_header(), mode="A")
    runner = Runner(gain)
    it = iter(proposals)
    seen = []

    def propose(ctx):
        seen.append(ctx)
        return next(it, None)
    res = run_sandbox(board, load_harness(), ALLOWED, DEV_A, DEV_B, runner, propose, **kw)
    return res, board, runner, seen


def test_sandbox_scores_both_arms_paired():
    gain = lambda h, arm: 0.03 if h["router"]["cost_penalty"] > 0.15 else 0.0
    res, board, runner, seen = run([patch("/router/cost_penalty", 0.3)], gain)
    r = res.rounds[0]
    assert r.kept and r.score_points == pytest.approx(3.0) and r.win_rate == 1.0
    arms_run = {(sha, arm) for sha, arm, cases in runner.calls if cases == tuple(DEV_A)}
    assert {arm for _, arm in arms_run} == set(ARMS)
    assert len({sha for sha, _ in arms_run}) == 2                      # current + candidate, each on both arms
    # the current harness is evaluated once per arm and then cached across rounds
    base_sha = harness_sha256(load_harness())
    assert sum(1 for sha, arm, c in runner.calls if sha == base_sha and c == tuple(DEV_A)) == 2
    assert res.final_harness["router"]["cost_penalty"] == 0.3
    # a board-only key is a no-op for noBoard: score is the mean over both arms
    gain_board = lambda h, arm: 0.04 if (arm == "SW-B" and h["briefs"]["count"] == 12) else 0.0
    res2, *_ = run([patch("/briefs/count", 12)], gain_board)
    assert res2.rounds[0].score_points == pytest.approx(2.0)           # (4 + 0) / 2
    assert res2.rounds[0].win_rate == 1.0                              # the 30 tied noBoard pairs do not count against it


def test_proposer_sees_aggregates_only():
    gain = lambda h, arm: 0.02
    res, _, _, seen = run([patch("/router/cost_penalty", 0.3), patch("/queen/cadence", 35)], gain)
    text = json.dumps(seen)
    assert not any(c in text for c in DEV_A + DEV_B)
    ctx = seen[1]
    assert set(ctx) == {"round", "harness", "allowed_keys", "aggregates", "previous_edits"}
    assert set(ctx["aggregates"]) == set(ARMS) and ctx["aggregates"]["SW-B"]["n_cases"] == 30
    assert ctx["previous_edits"][0]["round"] == 1 and "keys" in ctx["previous_edits"][0]


def test_edit_becomes_board_entry():
    gain = lambda h, arm: 0.03 if h["router"]["cost_penalty"] > 0.15 else 0.0
    good = patch("/router/cost_penalty", 0.3)
    flat = patch("/queen/cadence", 35)                                  # no effect: reverted
    forbidden = patch("/scorer/x", 1)                                   # rejected, still logged
    res, board, _, _ = run([good, flat, forbidden], gain)
    edits = board.by_type("edit")
    assert len(edits) == 3 == len(res.rounds)
    assert [e.payload["round"] for e in edits] == [1, 2, 3]
    assert [e.payload["kept"] for e in edits] == [1, 0, 0]
    assert edits[0].payload["keys"] == ["/router/cost_penalty"] and edits[0].payload["prediction"] == pred()
    assert edits[0].payload["patch_sha256"] == parse_proposal(good, ALLOWED).patch_sha256
    assert all(e.author["agent"] == "harness" for e in edits)
    assert res.rounds[2].rejected and res.rounds[2].score_points is None
    assert [r.edit_entry_id for r in res.rounds] == [e.id for e in edits]
    assert res.final_harness["queen"]["cadence"] == 30                  # reverted


def test_max_four_rounds_and_revert():
    gain = lambda h, arm: 0.0
    props = [patch("/router/cost_penalty", 0.11 + i / 100) for i in range(8)]
    res, board, _, _ = run(props, gain, max_rounds=99)
    assert len(res.rounds) == 4 and len(board.by_type("edit")) == 4
    assert not any(r.kept for r in res.rounds)
    assert harness_sha256(res.final_harness) == harness_sha256(res.base_harness)


def test_dev_b_promotion_check():
    # the edit helps on dev-A but hurts on dev-B: kept, not a promotion candidate
    def gain(h, arm):
        return 0.03 if h["router"]["cost_penalty"] > 0.15 else 0.0
    runner_cases = {}

    def evaluate(harness, arm, cases):
        up = harness["router"]["cost_penalty"] > 0.15
        delta = (0.03 if cases[0].startswith("A") else -0.02) if up else 0.0
        return {c: 0.5 + delta for c in cases}
    board = Board(gv.test_header(), mode="A")
    it = iter([patch("/router/cost_penalty", 0.3)])
    res = run_sandbox(board, load_harness(), ALLOWED, DEV_A, DEV_B, evaluate, lambda ctx: next(it, None))
    assert res.rounds[0].kept and res.promotion_candidates == () and res.dev_b_scores[1] == pytest.approx(-2.0)
    res2, *_ = run([patch("/router/cost_penalty", 0.3)], gain)
    assert res2.promotion_candidates == (1,)


def test_llm_proposer_with_fake_provider():
    reply = json.dumps(patch("/router/cost_penalty", 0.3))
    provider = FakeProvider(lambda req: f"```json\n{reply}\n```", model="fake")
    propose = llm_proposer(provider, "fake", thinking_params={"thinking": {"type": "disabled"}})
    out = propose({"round": 1, "aggregates": {}})
    assert parse_proposal(out, ALLOWED).keys == ["/router/cost_penalty"]
    assert provider.requests[0].max_tokens == 800
    assert llm_proposer(FakeProvider(lambda r: "nope"), "fake")({"round": 1}) != None
