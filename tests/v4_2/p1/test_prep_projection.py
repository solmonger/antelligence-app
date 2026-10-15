"""Goal 11 item 4 and the prep script: projection arithmetic on synthetic inputs, plus the real inputs when they are on disk."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import v42_phase1_prep as prep  # noqa: E402

FRONTIER = {"cost_usd": 0.04, "calls": 196, "usd_per_call": 0.04 / 196, "wall_s": 240.0, "round_s": 10.0, "tokens_in": 1, "tokens_out": 1}


def stats(**kw):
    base = {"episodes": 4, "decisions": 192.0, "skipped": 0.0, "answer_turns": 0.0, "asks_sent": 0.0, "asks_answered": 0.0, "asks_refused": 0.0,
            "answers_refused": 0.0}
    return {**base, **kw}


def inputs(skip=0.05):
    st = {"S-solo": stats(), "S-marks": stats(), "S-talk": stats(asks_sent=60.0, asks_answered=60.0, answer_turns=60.0),
          "S-marks-quiet": stats(decisions=192.0 * (1 - skip), skipped=192.0 * skip)}
    lat = {"d1": prep.point(5.0), "qwen3.5-4b": prep.point(10.0), "qwen3.5-9b": prep.point(20.0)}
    c0 = {"wall_1": 100.0, "wall_16": 125.0, "efficiency": 0.8, "episodes_per_hour_16": 1.0}
    return prep.Inputs(st, lat, c0, FRONTIER, round_ratio=1.0)


ARM = {a["name"]: a for a in prep.ARMS}


def test_expected_max_of_points_and_of_a_spread():
    assert prep.expected_max([prep.point(3.0)] * 8) == pytest.approx(3.0)
    two = [(1.0, 0.5), (3.0, 0.5)]
    assert prep.expected_max([two]) == pytest.approx(2.0)
    assert prep.expected_max([two, two]) == pytest.approx(1 * 0.25 + 3 * 0.75)             # P(max=1) = 1/4
    assert prep.expected_max([two] * 8) > prep.expected_max([two] * 2)
    assert prep.expected_max([two] * 3, active=0.0) == 0.0


def test_pod_hours_bill_whole_batches_of_16():
    assert prep.pod_hours(16, 900.0, 0.9) == pytest.approx(1000.0 / 3600)
    assert prep.pod_hours(17, 900.0, 0.9) == pytest.approx(2000.0 / 3600)
    assert prep.pod_hours(80, 900.0, 0.9) == pytest.approx(5000.0 / 3600)


def test_answer_phase_share_is_a_probability():
    assert prep.answer_phase_share(0.0) == 0.0
    assert 0.0 < prep.answer_phase_share(24.0) < 1.0
    assert prep.answer_phase_share(48.0) > prep.answer_phase_share(24.0)


def test_solo_makes_one_call_per_turn_and_no_answers():
    r = prep.project_arm(inputs(), ARM["M-solo"], ["qwen3.5-9b"] * 8, False)
    assert r["calls_total"] == pytest.approx(192) and r["answers"] == 0 and r["answer_s"] == 0
    assert r["round_s"] == pytest.approx(20.0) and r["wall_s"] == pytest.approx(24 * 20.0 + prep.ENGINE_S)
    assert r["nous_usd"] == 0 and r["pod_hours"] > 0


def test_talk_adds_one_call_per_answer_and_asks_are_free():
    r = prep.project_arm(inputs(), ARM["M-talk (scripted queen)"], ["qwen3.5-4b"] * 8, False)
    assert r["asks"] == pytest.approx(60) and r["answers"] == pytest.approx(60)
    assert r["calls_total"] == pytest.approx(192 + 60)
    assert r["wall_s"] > 24 * 10.0 + prep.ENGINE_S


def test_quiet_rule_scales_calls_by_the_skip_share_and_barely_moves_wall_time():
    inp = inputs(skip=0.05)
    a = prep.project_arm(inp, ARM["M-talk (scripted queen)"], ["qwen3.5-9b"] * 8, False)
    b = prep.project_arm(inp, ARM["M-talk (scripted queen)"], ["qwen3.5-9b"] * 8, True)
    assert b["calls_total"] == pytest.approx(a["calls_total"] * 0.95)
    assert b["asks"] == pytest.approx(a["asks"] * 0.95) and b["decisions"] == pytest.approx(182.4)
    assert b["wall_s"] <= a["wall_s"] and (a["wall_s"] - b["wall_s"]) / a["wall_s"] < 0.05


def test_escalation_adds_calls_to_the_bigger_model_and_frontier_costs_dollars():
    inp = inputs()
    r = prep.project_arm(inp, ARM["M-escalate"], ["qwen3.5-4b"] * 8, False)
    assert r["escalations"] == pytest.approx(192 * 60 / 192)
    assert r["calls"]["qwen3.5-9b"] == pytest.approx(r["escalations"]) and r["nous_usd"] == 0
    big = prep.project_arm(inp, ARM["M-escalate"], ["qwen3.5-9b"] * 8, False)
    assert big["calls"]["frontier"] == pytest.approx(big["escalations"])
    assert big["nous_usd"] == pytest.approx(big["escalations"] * FRONTIER["usd_per_call"] * 80)
    assert big["wall_s"] > r["wall_s"] > prep.project_arm(inp, ARM["M-solo"], ["qwen3.5-4b"] * 8, False)["wall_s"]


def test_queen_rows():
    inp = inputs()
    none = prep.project_arm(inp, ARM["M-talk, no queen"], ["qwen3.5-9b"] * 8, False)
    scripted = prep.project_arm(inp, ARM["M-talk (scripted queen)"], ["qwen3.5-9b"] * 8, False)
    llm = prep.project_arm(inp, ARM["M-talk, LLM queen"], ["qwen3.5-9b"] * 8, False)
    assert none["calls_total"] == scripted["calls_total"] and none["wall_s"] == scripted["wall_s"] and none["nous_usd"] == 0
    assert llm["queen_calls"] == 4 and llm["calls"]["frontier"] == 4
    assert llm["nous_usd"] == pytest.approx(prep.LLM_QUEEN_USD * 80) and llm["wall_s"] == pytest.approx(scripted["wall_s"] + 4 * FRONTIER["round_s"])


def test_mixed_colony_waits_for_its_slowest_model():
    inp = inputs()
    mixed = prep.project_arm(inp, ARM["M-talk (scripted queen)"], prep.MIXED, False)
    nine = prep.project_arm(inp, ARM["M-talk (scripted queen)"], ["qwen3.5-9b"] * 8, False)
    four = prep.project_arm(inp, ARM["M-talk (scripted queen)"], ["qwen3.5-4b"] * 8, False)
    assert mixed["wall_s"] == pytest.approx(nine["wall_s"]) and mixed["wall_s"] > four["wall_s"]
    assert mixed["calls"]["d1"] == pytest.approx(mixed["calls_total"] * 3 / 8)
    assert mixed["calls_total"] == pytest.approx(nine["calls_total"])


def test_frontier_reference_has_no_pod():
    r = prep.project_frontier_reference(inputs(), False)
    assert r["pod_hours"] == 0 and r["nous_usd"] == pytest.approx(252 * FRONTIER["usd_per_call"] * 80)


def test_independent_bound_is_at_least_the_batched_headline():
    inp = prep.Inputs(inputs().stats, {m: prep.from_quantiles([4.0 + 0.1 * i for i in range(101)]) for m in prep.MODELS}, inputs().c0, FRONTIER, 1.0)
    for name in ("M-solo", "M-talk (scripted queen)"):
        h = prep.project_arm(inp, ARM[name], ["qwen3.5-9b"] * 8, False)
        u = prep.project_arm(inp, ARM[name], ["qwen3.5-9b"] * 8, False, mode="independent")
        assert u["wall_s"] > h["wall_s"] * 0.95


def test_load_arm_stats_reads_bundles(tmp_path):
    for i, (dec, skip, ans) in enumerate(((10, 2, 3), (14, 0, 5))):
        d = tmp_path / "S-x" / f"BRATS_{i:03d}" / "seed1"
        d.mkdir(parents=True)
        log = [{"kind": "decision"}] * dec + [{"kind": "skipped"}] * skip + [{"kind": "answer"}] * ans
        (d / "bundle.json").write_text(json.dumps({"extra": {"colony": {"action_log": log, "counters": {
            "asks_sent": ans, "asks_answered": ans, "asks_refused": 1, "answers_refused": 0}}}}))
    s = prep.load_arm_stats("S-x", tmp_path)
    assert s["episodes"] == 2 and s["decisions"] == 12 and s["skipped"] == 1 and s["asks_sent"] == 4 and s["answer_turns"] == 4
    with pytest.raises(FileNotFoundError):
        prep.load_arm_stats("S-none", tmp_path)


def test_token_counts_refuse_non_dev_cases():
    from antelligence.experiments.v42_harness import EvalRefused
    with pytest.raises(EvalRefused):
        prep.cmd_token_counts(cases=["NOT_A_DEV_CASE"])


def test_real_projection_renders_when_inputs_exist():
    if not (prep.VERDICT / "S-talk").exists() or not (prep.V41_RUNS / "plumbing3" / "cells.jsonl").exists():
        pytest.skip("verdict1 / v4.1 results not on this machine")
    stats = {a: prep.load_arm_stats(a) for a in ("S-solo", "S-marks", "S-talk", "S-marks-quiet")}
    inp = prep.Inputs(stats, prep.load_latencies(), prep.load_c0_efficiency(), prep.load_frontier(), prep.load_round_ratio())
    assert 1.0 <= inp.round_ratio < 1.5 and 0.8 < inp.c0["efficiency"] < 1.0
    r = prep.project_arm(inp, ARM["M-talk (scripted queen)"], ["qwen3.5-9b"] * 8, False)
    assert all(math.isfinite(r[k]) for k in ("wall_s", "pod_hours", "calls_total"))
    assert r["asks"] == pytest.approx(stats["S-talk"]["asks_sent"])


def test_projection_document_has_every_arm_and_no_nan():
    doc = (prep.OUT / "PROJECTION.md")
    if not doc.exists():
        pytest.skip("run `v42_phase1_prep.py projection` first")
    t = doc.read_text()
    for arm in ("M-solo", "M-marks", "M-talk (scripted queen)", "M-escalate", "M-talk, no queen", "M-talk, LLM queen", "Frontier reference", "Mixed colony"):
        assert arm in t
    assert "nan" not in t.lower().replace("finance", "") and "[A1]" in t and "[A13]" in t


def test_scripted_episode_states_render_within_the_limit_and_keep_the_barrier():
    """One real scripted S-talk episode (dev case, no model): every state a model ant would see builds a request at the default limit."""
    if not (prep.CASES_ROOT / "BRATS_016" / "case.npz").exists():
        pytest.skip("dev case data not on this machine")
    from antelligence.kernel.decision import MAX_STATE_CHARS
    from antelligence.worlds.tumor_mri.colony_obs import ALLOWED
    from antelligence.worlds.tumor_mri.world import OBS_KEYS
    from antelligence.worlds.tumor_mri.questions import build_request, render_state
    rows = []
    prep.capture_states("S-talk", "BRATS_016", 3, rows)
    assert len(rows) == 192
    allowed = set(OBS_KEYS) | set(ALLOWED["S-talk"])
    for r in rows:
        assert set(r["fields"]) == allowed                                  # the information barrier holds on what we render
        assert len(r["fields"]["candidates"]) <= 6
        assert len(render_state(r["fields"])) <= MAX_STATE_CHARS
        build_request("m", r["fields"], with_mark=True)
    assert any("Marks near you:" in render_state(r["fields"]) for r in rows)
