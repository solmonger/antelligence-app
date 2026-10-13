"""scripts/v41_sandbox.py: the sandbox loop wired to the harness through evaluate(), with fake backends and a fake queen."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antelligence.experiments.v41_harness import HarnessError
from antelligence.experiments.v41_sandbox import ARMS, MAX_ROUNDS, llm_proposer, run_sandbox
from antelligence.kernel.board import Board
from antelligence.kernel.policies.v41_harness import harness_sha256, load_allowed_keys, load_harness
from antelligence.providers.wrappers import FakeProvider
from tests.v4_1.harness.conftest import CASE_EVAL, build_rig
from tests.v4_1.vectors import generate_vectors as gv

ROOT = Path(__file__).resolve().parents[3]
SPLIT = json.loads((ROOT / "config/v4_1/split.json").read_text())
DEV_A, DEV_B = SPLIT["dev_a"][:30], SPLIT["dev_b"][:10]
EVAL_CASE = SPLIT["eval"][0]


def load_script():
    spec = importlib.util.spec_from_file_location("v41_sandbox_t", ROOT / "scripts" / "v41_sandbox.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sb = load_script()


def pred(d=15000):
    return {"metric": "coverage", "delta_ppm": d}


def patch(path, value, d=15000):
    return {"patch": [{"op": "replace", "path": path, "value": value}], "prediction": pred(d)}


class FakeHarness:
    """Stands in for V41Harness: run_many returns synthetic cells whose coverage depends on the promoted harness."""

    def __init__(self, gain=lambda h, arm: 0.0):
        self.gain = gain
        self.promoted_harness = load_harness()
        self.runs = []                     # (harness sha, arm, case, seed)

    async def run_many(self, requests, *, concurrency=4, calibration=False):
        assert calibration is False
        out = []
        for arm, case, seed in requests:
            self.runs.append((harness_sha256(self.promoted_harness), arm, case, seed))
            cov = 0.5 + self.gain(self.promoted_harness, arm) + 0.000137 * (sum(map(ord, case)) % 97)
            out.append(SimpleNamespace(cell={"coverage": min(1.0, cov), "defer_rate": 0.2, "router_top_share": 0.6,
                                             "off_target": 0.1, "decision_error_share": 0.0, "queen_calls": 4, "alarms": 0}))
        return out


def evaluator(h, loop=None, tmp=None, cases=None):
    cache = sb.ResultsCache(None, "ctx")
    return sb.make_evaluate(h, cache, loop or sb.EventLoop(), cases or [*DEV_A, *DEV_B]), cache


def sandbox(h, proposals, max_rounds=MAX_ROUNDS):
    loop = sb.EventLoop()
    try:
        evaluate, cache = evaluator(h, loop)
        it = iter(proposals)
        seen = []

        def propose(ctx):
            seen.append(ctx)
            return next(it, None)
        board = Board(gv.test_header(), mode="A")
        res = run_sandbox(board, load_harness(), load_allowed_keys(), DEV_A, DEV_B, evaluate, propose, max_rounds=max_rounds)
        return res, board, cache, seen
    finally:
        loop.close()


# --- keep rule, rounds, revert ---------------------------------------------------------------------------------------

def test_keep_rule_on_synthetic_coverage_through_the_real_evaluate_callback():
    h = FakeHarness(lambda cfg, arm: 0.02 if cfg["router"]["cost_penalty"] > 0.15 else 0.0)
    res, board, cache, seen = sandbox(h, [patch("/router/cost_penalty", 0.3)])
    r = res.rounds[0]
    assert r.kept and r.score_points == pytest.approx(2.0) and r.win_rate == 1.0
    assert set(r.per_arm_points) == set(ARMS) and r.wins == 60
    assert r.diff[0] == {"op": "replace", "path": "/router/cost_penalty", "before": 0.1, "after": 0.3}
    assert res.final_harness["router"]["cost_penalty"] == 0.3
    assert res.promotion_candidates == (1,) and res.dev_b_detail[1]["score_points"] == pytest.approx(2.0)
    assert {e.type for e in board.entries()} == {"edit"}
    # a tiny gain is reverted (under +1 point), even with every pair winning
    h2 = FakeHarness(lambda cfg, arm: 0.004 if cfg["router"]["cost_penalty"] > 0.15 else 0.0)
    res2, *_ = sandbox(h2, [patch("/router/cost_penalty", 0.3)])
    assert not res2.rounds[0].kept and res2.final_harness == res2.base_harness


def test_at_most_four_rounds_even_if_asked_for_more():
    h = FakeHarness()
    res, board, cache, seen = sandbox(h, [patch("/briefs/count", 6 + i) for i in range(10)], max_rounds=9)
    assert len(res.rounds) == MAX_ROUNDS == 4 and len(seen) == 4 and board.size() == 4
    with pytest.raises(SystemExit):
        sb.main(["--rounds", "5", "--router", "fake", "--thresholds", "none", "--backends", "fake"])


def test_wrong_prediction_sign_reverts_a_winning_edit():
    h = FakeHarness(lambda cfg, arm: 0.05 if cfg["queen"]["cadence"] == 36 else 0.0)
    res, *_ = sandbox(h, [patch("/queen/cadence", 36, d=-15000)])
    r = res.rounds[0]
    assert r.score_points == pytest.approx(5.0) and not r.kept
    assert res.final_harness == res.base_harness and res.promotion_candidates == ()


# --- the queen sees aggregates only -----------------------------------------------------------------------------------

def test_queen_prompt_has_no_per_case_truth():
    h = FakeHarness(lambda cfg, arm: 0.03 if cfg["router"]["cost_penalty"] > 0.15 else 0.0)
    provider = FakeProvider(lambda req: json.dumps(patch("/router/cost_penalty", 0.3)), model="fake-queen")
    loop = sb.EventLoop()
    try:
        evaluate, cache = evaluator(h, loop)
        board = Board(gv.test_header(), mode="A")
        propose = llm_proposer(provider, "deepseek/deepseek-v4.1-flash", thinking_params=sb.NOUS_EXTRA, run=loop.run)
        res = run_sandbox(board, load_harness(), load_allowed_keys(), DEV_A, DEV_B, evaluate, propose, max_rounds=2)
    finally:
        loop.close()
    assert len(provider.requests) == 2
    covs = {repr(r["coverage"]) for r in cache._rows.values() if len(repr(r["coverage"])) >= 8}
    assert len(covs) > 10
    for req in provider.requests:
        assert req.extra_body == {"reasoning": {"enabled": False}} and req.max_tokens <= 800
        prompt = " ".join(m["content"] for m in req.messages)
        body = json.loads(req.messages[1]["content"])
        assert set(body) == {"round", "harness", "allowed_keys", "aggregates", "previous_edits"}
        assert all(c not in prompt for c in [*DEV_A, *DEV_B]), "a case id reached the queen"
        assert not any(c in prompt for c in covs), "a per-case coverage value reached the queen"
        assert set(body["aggregates"]) == set(ARMS) and "n_cases" in body["aggregates"]["SW-B"]
        assert body["aggregates"]["SW-B"]["stats"]["defer_rate"] == 0.2
    second = json.loads(provider.requests[1].messages[1]["content"])
    assert second["previous_edits"][0]["kept"] is True and res.rounds[0].kept


# --- the cache --------------------------------------------------------------------------------------------------------

def test_current_harness_is_not_rerun():
    h = FakeHarness(lambda cfg, arm: 0.02 if cfg["briefs"]["count"] >= 10 else 0.0)
    res, board, cache, seen = sandbox(h, [patch("/briefs/count", 10), patch("/briefs/count", 11), patch("/briefs/count", 12)])
    assert len(h.runs) == len(set(h.runs)), "an episode was run twice"
    base = harness_sha256(load_harness())
    assert sum(1 for sha, *_ in h.runs if sha == base) == 2 * (30 + 10)         # base: both arms on dev-A and dev-B, once
    # round 1 kept (+2); rounds 2 and 3 add nothing over the new current harness: each costs only its candidate
    assert [r.kept for r in res.rounds] == [True, False, False]
    assert cache.runs == len(h.runs)
    # asking again for the same (harness, arm, case, seed) never reaches the harness
    n = len(h.runs)
    loop = sb.EventLoop()
    try:
        ev = sb.make_evaluate(h, cache, loop, [*DEV_A, *DEV_B])
        ev(load_harness(), "SW-B", DEV_A)
    finally:
        loop.close()
    assert len(h.runs) == n and cache.hits == 30


def test_results_cache_persists_and_is_context_keyed(tmp_path):
    path = tmp_path / "c.jsonl"
    c1 = sb.ResultsCache(path, "ctx-a")
    c1.put("s" * 64, "SW-B", "BRATS_016", 1, 0.5, {"defer_rate": 0.2})
    assert sb.ResultsCache(path, "ctx-a").get("s" * 64, "SW-B", "BRATS_016", 1)["coverage"] == 0.5
    assert sb.ResultsCache(path, "ctx-b").get("s" * 64, "SW-B", "BRATS_016", 1) is None     # another router: not reused
    assert sb.ResultsCache(path, "ctx-a").get("s" * 64, "SW-B", "BRATS_016", 2) is None


# --- dev cases only ---------------------------------------------------------------------------------------------------

def test_non_dev_cases_are_refused():
    h = FakeHarness()
    loop = sb.EventLoop()
    try:
        ev = sb.make_evaluate(h, sb.ResultsCache(None, "x"), loop, [*DEV_A, *DEV_B])
        with pytest.raises(sb.DevOnlyError, match="development cases only"):
            ev(load_harness(), "SW-B", [DEV_A[0], EVAL_CASE])
        with pytest.raises(sb.DevOnlyError):
            ev(load_harness(), "SW-B", [SPLIT["reserve"][0]])
        with pytest.raises(HarnessError):
            ev(load_harness(), "FW", DEV_A[:2])
    finally:
        loop.close()
    assert h.runs == []
    with pytest.raises(sb.DevOnlyError):
        sb.dev_sets(SPLIT, override_a=[EVAL_CASE])
    with pytest.raises(sb.DevOnlyError):
        sb.dev_sets(SPLIT, override_a=[SPLIT["dev_b"][0]])                      # dev-B is not dev-A
    with pytest.raises(sb.DevOnlyError):
        sb.dev_sets(SPLIT, n_a=31)
    a, b = sb.dev_sets(SPLIT)
    assert a == DEV_A and b == DEV_B and len(a) == 30 and len(b) == 10
    with pytest.raises(SystemExit, match="development cases"):
        sb.main(["--router", "fake", "--thresholds", "none", "--backends", "fake", "--dev-a", EVAL_CASE])


# --- cost projection ---------------------------------------------------------------------------------------------------

def _row(cost, role="queen"):
    return {"utc": "2026-10-13T00:00:00+00:00", "role": role, "backend": "nous", "model": "deepseek/deepseek-v4.1-flash",
            "cost_usd": cost}


def test_projection_uses_measured_cost_per_queen_call(tmp_path, capsys):
    ledger = tmp_path / "usage.jsonl"
    ledger.write_text("".join(json.dumps(_row(c)) + "\n" for c in (0.0004, 0.0006)) + json.dumps(_row(0.5, "worker")) + "\n")
    p = sb.project_cost(4, 30, 10, usage_ledger=ledger, pod_costs=tmp_path / "none.jsonl", paid=True)
    assert p["max_episodes"] == 400 and p["queen_calls"] == 400 * 6 + 4
    assert p["cost_per_call_usd"] == pytest.approx(0.0005)
    assert p["projected_nous_usd"] == pytest.approx(2404 * 0.0005)
    assert "measured over 2" in p["basis"] and p["fits"]
    bound = sb.project_cost(3, 30, 10, usage_ledger=tmp_path / "missing.jsonl", pod_costs=tmp_path / "x", paid=True)
    assert "upper bound" in bound["basis"] and bound["cost_per_call_usd"] > 0
    ledger.write_text(json.dumps(_row(7.49)) + "\n" + json.dumps(_row(0.0005)) + "\n")      # no headroom: refuse to start
    code = sb.main(["--router", "r.json", "--thresholds", "t.json", "--usage-ledger", str(ledger), "--pod-costs",
                    str(tmp_path / "none"), "--out", str(tmp_path / "o.json"), "--work-dir", str(tmp_path / "w")])
    out = capsys.readouterr()
    assert code == 2 and "projected sandbox cost" in out.out and "refused" in out.err and not (tmp_path / "o.json").exists()


# --- end to end on the real harness (fake backends, fake queen, mock anchors) ------------------------------------------

def test_end_to_end_real_harness_writes_sandbox_json(tmp_path):
    rig = build_rig(tmp_path / "rig")
    seq = iter([patch("/router/cost_penalty", 0.3), patch("/briefs/count", 10)])
    out = tmp_path / "phase0" / "sandbox.json"
    code = sb.main(["--backends", "fake", "--anchors", "mock", "--router", "fake", "--thresholds", "none", "--rounds", "2",
                    "--n-a", "2", "--n-b", "1", "--concurrency", "2", "--out", str(out), "--work-dir", str(tmp_path / "w")],
                   harness=rig.harness, proposer=lambda ctx: next(seq, None))
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["status"] == "complete" and doc["anchors"] == "mock" and doc["backends"] == rig.harness.backends_kind
    assert doc["dev_a"] == DEV_A[:2] and doc["dev_b"] == DEV_B[:1]
    assert len(doc["rounds"]) == 2 and doc["seed"] == 1
    for r in doc["rounds"]:
        assert r["proposal"] and r["prediction"]["delta_ppm"] == 15000 and r["diff"]
        assert set(r["dev_a"]["per_arm_points"]) == set(ARMS) and r["dev_a"]["n_pairs"] == 4
        assert r["kept"] == (r["dev_b"] is not None)
        assert r["promotion_candidate"] == (r["kept"] and r["dev_b"]["score_points"] >= -1e-9)
    assert doc["board"]["entries"] == 2 and doc["board"]["anchors"], "the edit entries are anchored"
    assert "Nothing is promoted" in doc["promotion"]
    assert {c["round"] for c in doc["promotion_candidates"]} <= {r["round"] for r in doc["rounds"] if r["kept"]}
    # every candidate episode went through the real harness: SW-B / SW-noBoard cells for dev cases only
    assert doc["episodes_run"] > 0 and (tmp_path / "w" / "results-cache.jsonl").exists()
    assert all(r["case"] in DEV_A[:2] + DEV_B[:1] and r["seed"] == 1 for r in
               map(json.loads, (tmp_path / "w" / "results-cache.jsonl").read_text().splitlines()))
    # re-running resumes from the results cache: no new episodes
    code = sb.main(["--backends", "fake", "--anchors", "mock", "--router", "fake", "--thresholds", "none", "--rounds", "2",
                    "--n-a", "2", "--n-b", "1", "--out", str(out), "--work-dir", str(tmp_path / "w")],
                   harness=build_rig(tmp_path / "rig2").harness,
                   proposer=lambda ctx: {"patch": [{"op": "replace", "path": "/router/cost_penalty", "value": 0.3}],
                                         "prediction": pred()} if ctx["round"] == 1 else None)
    assert code == 0 and json.loads(out.read_text())["episodes_run"] == 0


def test_fake_backends_cli_builds_its_own_harness(tmp_path):
    (tmp_path / "w.yaml").write_text(json.dumps(__import__("tests.v4_1.harness.conftest", fromlist=["x"]).small_cfg().to_dict()))
    out = tmp_path / "s.json"
    code = sb.main(["--backends", "fake", "--anchors", "mock", "--router", "none", "--thresholds", "none", "--rounds", "1",
                    "--n-a", "1", "--n-b", "1", "--out", str(out), "--work-dir", str(tmp_path / "w"), "--world-yaml",
                    str(tmp_path / "w.yaml"), "--loader", "tests.v4_1.harness.test_dry_run:phantom_loader"])
    assert code == 0 and json.loads(out.read_text())["rounds"][0]["round"] == 1
