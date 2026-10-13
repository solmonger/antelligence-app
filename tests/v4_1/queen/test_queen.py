import inspect
import json

import pytest

from antelligence.kernel.policies import v41_queen
from antelligence.kernel.policies.v41_queen import (
    MAX_CALLS, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, QueenError, QueenToolError, V41Queen, coverage_estimate_ppm,
)
from antelligence.kernel.briefs import estimate_tokens
from antelligence.providers.base import ChatResponse
from tests.v4_1.queen.conftest import N_SQUADS, good_reply, make_queen, make_regions, make_squads


def rising():
    n = {"i": 0}

    def cov():
        n["i"] += 1
        return 100_000 * n["i"]
    return cov


def test_queen_tools_are_board_writes_only():
    queen, _, board = make_queen()
    assert set(queen.tools.ALLOWED) == {"task", "route", "alarm"}
    for bad in ("decision", "evidence", "knowledge", "edit"):
        with pytest.raises(QueenToolError):
            queen.tools.write(bad, {}, 0)
    step = queen.step(0, make_regions(), make_squads(), 0)
    authored = {e.type for e in board.entries() if e.author["agent"] == "queen"}
    assert authored <= {"task", "route", "alarm"} and {"task", "route"} <= authored
    assert len(step.routes) == N_SQUADS
    # no public way to touch the world: she holds a provider, a board and her ledgers, nothing else
    assert not any(hasattr(queen, n) for n in ("world", "set_action", "act", "decide"))


def test_queen_call_cap_6_with_replans():
    queen, provider, board = make_queen()
    regions, squads = make_regions(), make_squads()
    steps = [queen.step(10 * i, regions, squads, 0) for i in range(10)]   # flat coverage: stalls, so re-plans fire
    assert len(provider.requests) == MAX_CALLS == queen.calls
    assert [s.called for s in steps] == [True] * 6 + [False] * 4
    assert queen.progress.replans >= 1
    assert all(s.fell_back for s in steps[6:])
    assert any(e.type == "alarm" and "call cap" in e.payload["note"] for e in board.entries())
    # the cap can be tightened, never raised
    q2, p2, _ = make_queen(max_calls=99)
    assert q2.max_calls == MAX_CALLS


def test_token_caps_enforced():
    queen, provider, _ = make_queen()
    regions = make_regions(400)               # far over 6k tokens
    queen.step(0, regions, make_squads(), 0)
    req = provider.requests[0]
    assert req.max_tokens == MAX_OUTPUT_TOKENS == 800
    total = sum(estimate_tokens(m["content"]) for m in req.messages)
    assert total <= MAX_INPUT_TOKENS
    body = json.loads(req.messages[1]["content"])
    kept = [r["region_id"] for r in body["regions"]]
    ranked = [r["region_id"] for r in v41_queen.rank_regions(regions)]
    assert kept == ranked[: len(kept)] and 0 < len(kept) < 400        # truncation keeps the best-ranked prefix
    # deterministic: the same inputs give the same prompt
    q2, p2, _ = make_queen()
    q2.step(0, regions, make_squads(), 0)
    assert p2.requests[0].messages == req.messages
    # an over-long reply is rejected and falls back
    q3, _, board3 = make_queen(lambda r: ChatResponse(content=good_reply(), model="m", request_hash=r.request_hash,
                                                      prompt_tokens=10, completion_tokens=801, finish_reason="stop"))
    assert q3.step(0, make_regions(), make_squads(), 0).fell_back


def test_stall_forces_replan():
    queen, provider, _ = make_queen()
    regions, squads = make_regions(), make_squads()
    s0 = queen.step(0, regions, squads, 100_000)        # first estimate: no history, no stall
    s1 = queen.step(30, regions, squads, 100_000)       # stall 1
    assert not s0.replan and not s1.replan
    s2 = queen.step(60, regions, squads, 101_000)       # stall 2 -> forced re-plan
    assert s2.replan and queen.progress.replans == 1 and queen.progress.stalls == 0
    assert json.loads(provider.requests[2].messages[1]["content"])["replan"] is True
    assert json.loads(provider.requests[1].messages[1]["content"])["replan"] is False
    s3 = queen.step(90, regions, squads, 300_000)       # progress: no stall
    assert not s3.replan and queen.progress.stalls == 0


def test_never_sees_truth():
    sig = inspect.signature(V41Queen.step_async)
    assert set(sig.parameters) == {"self", "tick", "regions", "squads", "coverage_ppm"}
    queen, provider, _ = make_queen()
    regions = make_regions()
    regions[0]["truth_mask"] = "SECRET-TRUTH"           # even a smuggled field never reaches the prompt
    regions[0]["true_coverage"] = 999
    queen.step(0, regions, make_squads(), 0)
    prompt = "\n".join(m["content"] for m in provider.requests[0].messages)
    for banned in ("truth", "secret", "true_coverage", "mask", "ground"):
        assert banned not in prompt.lower()
    body = json.loads(provider.requests[0].messages[1]["content"])
    assert set(body["regions"][0]) <= {"region_id", "centroid_dmm", "cls", "p_tumour_ppm", "volume_mm3", "unc_ppm"}
    # the coverage estimate is p x volume weighted dose, from perception only
    regs = [{"region_id": 0, "p_tumour_ppm": 1_000_000, "volume_mm3": 100}, {"region_id": 1, "p_tumour_ppm": 500_000, "volume_mm3": 100}]
    assert coverage_estimate_ppm(regs, {0: 1.0}) == 666667
    assert coverage_estimate_ppm(regs, {0: 3.0, 1: 1.0}) == 1_000_000      # clipped at 1


def test_board_off_queen_sees_no_squad_evidence():
    on, p_on, _ = make_queen(board_on=True)
    off, p_off, board_off = make_queen(board_on=False)
    on.step(0, make_regions(), make_squads(), 0)
    off.step(0, make_regions(), make_squads(), 0)
    assert json.loads(p_on.requests[0].messages[1]["content"])["evidence"]
    assert json.loads(p_off.requests[0].messages[1]["content"])["evidence"] == []


def test_invalid_reply_falls_back_and_alarms():
    for bad in ("not json", "{}", json.dumps({"routes": []}), json.dumps({"routes": [{"squad": 0, "region_ids": [999]}]}),
                json.dumps({"routes": [{"squad": 9, "region_ids": [0]}]})):
        queen, _, board = make_queen(lambda r, b=bad: b)
        step = queen.step(0, make_regions(), make_squads(), 0)
        assert step.fell_back and step.alarm is not None and step.alarm.type == "alarm"
        assert step.alarm.payload["kind"] == "other" and len(step.routes) == N_SQUADS
        for e in step.routes:
            assert e.payload["region_ids"]
    # S-rule: stay on the previous route once there is one
    replies = iter([good_reply(), "garbage"])
    queen, _, _ = make_queen(lambda r: next(replies))
    first = queen.step(0, make_regions(), make_squads(), 0)
    second = queen.step(30, make_regions(), make_squads(), 0)
    assert second.fell_back
    assert [e.payload["region_ids"] for e in second.routes] == [e.payload["region_ids"] for e in first.routes]


def test_provider_failure_falls_back():
    def boom(req):
        raise RuntimeError("down")
    queen, _, board = make_queen(boom)
    step = queen.step(0, make_regions(), make_squads(), 0)
    assert step.fell_back and "RuntimeError" in step.alarm.payload["note"]


def test_advise_never_gate_and_fixed_squad_count():
    queen, _, _ = make_queen(lambda r: good_reply(candidates=[0, 1, 2, 3, 4, 0, 1], routes=[
        {"squad": s, "region_ids": [0], "priority": 1} for s in range(9)]))
    step = queen.step(0, make_regions(), make_squads(), 0)
    assert len(step.candidate_region_ids) <= 6
    assert step.options[-2:] == ("S", "X") and step.options[:-2] == tuple("ABCDE")
    assert len(step.routes) == N_SQUADS                    # squads 3..8 in the reply were ignored
    with pytest.raises(QueenError):
        queen.step(60, make_regions(), make_squads(N_SQUADS + 1), 0)
    # even a fallback offers S and X
    q2, _, _ = make_queen(lambda r: "nope")
    assert q2.step(0, make_regions(), make_squads(), 0).options[-2:] == ("S", "X")


def test_thinking_param_is_configurable():
    queen, provider, _ = make_queen()
    queen.step(0, make_regions(), make_squads(), 0)
    assert provider.requests[0].extra_body == {"reasoning": {"enabled": False}}
    assert provider.requests[0].payload()["reasoning"] == {"enabled": False}
    q2, p2, _ = make_queen(thinking_params={"thinking": {"type": "enabled", "budget_tokens": 5}})
    q2.step(0, make_regions(), make_squads(), 0)
    assert p2.requests[0].extra_body["thinking"]["type"] == "enabled"
    q3, p3, _ = make_queen(thinking_params=None)
    q3.step(0, make_regions(), make_squads(), 0)
    assert "reasoning" not in p3.requests[0].payload()


def test_brief_count_within_6_12():
    for count in (6, 9, 12):
        harness = v41_queen.load_harness()
        harness["briefs"]["count"] = count
        for board_on in (True, False):
            queen, _, board = make_queen(board_on=board_on, harness=harness)
            step = queen.step(0, make_regions(), make_squads(), 0)
            for s, brief in step.briefs.items():
                assert len(brief.entries) <= count
                if board_on:
                    assert 6 <= len(brief.entries) <= count
                route = step.routes[s]
                assert list(route.payload["brief_ids"]) == list(brief.brief_ids)   # route carries the read receipt
                assert 1 <= len(route.payload["brief_ids"]) <= 12
    with pytest.raises(Exception):
        bad = v41_queen.load_harness()
        bad["briefs"]["count"] = 13
        make_queen(harness=bad)


def test_board_off_briefs_hide_other_squads_evidence():
    queen, _, board = make_queen(board_on=False)
    step = queen.step(0, make_regions(), make_squads(), 0)
    for brief in step.briefs.values():
        assert all(e.type in ("task", "knowledge", "route") for e in brief.entries)


def test_static_queen_is_base_harness():
    from antelligence.kernel.policies.v41_harness import load_harness, harness_sha256
    queen, _, _ = make_queen()
    assert harness_sha256(queen.harness) == harness_sha256(load_harness())
