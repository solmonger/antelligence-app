import json

import pytest

from antelligence.experiments.v42_controls import shuffled_other
from antelligence.experiments.v42_metrics import (Timings, claim_accuracy_by_ant, effect_study, effect_summary, honesty,
                                                  mark_quality, pearson, run_metrics, sample_items, sample_runs)
from antelligence.experiments.v42_report import render, write_report
from antelligence.kernel.colony.episode import run_colony_episode
from antelligence.kernel.signal import SignalDraft
from tests.v4_2.colony.fake_world import ARMS, BASE_KEYS, FakeWorld, Scripted

LOOKS = {(0, 0, 0): 0.9, (10, 0, 0): 0.7, (20, 0, 0): 0.1}


def mark(kind, pos):
    return SignalDraft(kind=kind, pos=tuple(float(c) for c in pos), ttl=1)


def talker(view):
    i = int(view.agent_id.split("-")[1])
    if i == 0 and view.tick in (0, 10):
        return ({"release": 0}, [mark("tumour_here", (5, 0, 0))],
                {"question": "is_tumour", "pos_vox": [0, 0, 0], "note": "x"} if view.tick == 0 else None)
    if i == 1 and view.tick == 0:
        return ({"release": 0}, [mark("not_tumour", (5, 0, 0))], None)
    if i == 1 and view.observation.get("messages_in"):
        return ({"release": 1}, [], None)
    return ({"release": 0}, [], None)


class CovWorld(FakeWorld):
    def metrics(self):
        return {**super().metrics(), "coverage": (30.0 - sum(self.payload.values())) / 30.0}


def episode(arm="S-talk", **kw):
    w = CovWorld(looks=LOOKS)
    res = run_colony_episode(w, lambda a: Scripted(talker, []), arm=ARMS[arm], base_keys=BASE_KEYS, **kw)
    return res


def test_run_metrics_on_a_fake_episode():
    res = episode()
    m = {"coverage": 0.4, "off_target_share": 0.25, "released_total": 8.0, "time_to_50_coverage": None,
         "coverage_at_queen": {0: 0.0, 30: 0.3}}
    t = Timings(clock=iter([0.0, 0.002, 1.0, 1.004]).__next__)
    with t.time("look_ms"):
        pass
    with t.time("queen_ms"):
        pass
    out = run_metrics(m, res.colony, timings=t)
    assert out["coverage"] == 0.4 and out["drug_on_non_tumour"] == pytest.approx(2.0)
    assert out["coverage_at_queen"] == {"0": 0.0, "30": 0.3} and out["time_to_50_coverage"] is None
    assert out["timings"]["look_ms"]["mean"] == pytest.approx(2.0) and out["timings"]["queen_ms"]["mean"] == pytest.approx(4.0)
    c = out["colony"]
    assert c["marks_by_kind"]["not_tumour"] == 1 and c["marks_by_kind"]["tumour_here"] >= 1
    assert c["asks_sent"] == 1 and c["asks_answered"] == 1 and c["asks_lapsed"] == 0 and c["asks_refused"] == 0
    assert c["turns_skipped"] == 0 and c["quorum_events"] == res.colony["counters"]["quorum_events"]
    assert 0.0 <= c["share_strong"] <= 1.0
    assert c["share_disputed"] > 0                         # tumour_here and not_tumour within 4 mm of one spot


def test_mark_quality_strong_and_disputed_hand_built():
    colony = {"params": {}, "marks": [
        {"mark_kind": "tumour_here", "ant": 0, "pos_vox": [5, 0, 0], "amount_milli": 1000, "reading_ppm": 0, "tick": 0},
        {"mark_kind": "tumour_here", "ant": 1, "pos_vox": [5, 1, 0], "amount_milli": 1000, "reading_ppm": 0, "tick": 0},
        {"mark_kind": "need_more", "ant": 2, "pos_vox": [50, 0, 0], "amount_milli": 1000, "reading_ppm": 0, "tick": 3}]}
    q = mark_quality(colony)
    assert q["by_kind"] == {"need_more": 1, "tumour_here": 2} and q["share_strong"] == pytest.approx(2 / 3)
    assert q["share_disputed"] == 0.0
    assert mark_quality({"marks": []})["share_strong"] == 0.0


def test_honesty_and_claim_accuracy():
    res = episode()
    acc = claim_accuracy_by_ant(res.colony, lambda pos: pos[0] >= 8)           # truth: tumour only for x >= 8
    assert acc["ant-00"]["checked"] >= 1 and acc["ant-00"]["correct"] == 0           # ant 0 says tumour at x=5: wrong
    assert acc["ant-01"] == {"correct": 1, "checked": 1}                              # ant 1 says not_tumour there: right
    cards = {f"ant-0{i}": {"in_run": {"claims": 5, "confirmed": c, "contradicted": 5 - c}} for i, c in enumerate([5, 4, 1, 0])}
    colony = {"cards_after": cards}
    accuracy = {f"ant-0{i}": {"correct": c, "checked": 5} for i, c in enumerate([5, 4, 1, 0])}
    h = honesty(colony, accuracy)
    assert h["n_ants"] == 4 and h["correlation"] > 0.95
    assert honesty(colony, {})["correlation"] is None and honesty(colony, {})["n_ants"] == 0
    assert pearson([1, 1, 1], [1, 2, 3]) is None and pearson([1, 2], [1, 2]) is None


def test_quiet_rule_metrics_report_turns_skipped():
    from antelligence.kernel.colony.episode import ColonyConfig
    res = episode("S-marks", config=ColonyConfig(quiet_rule=True))
    out = run_metrics({"coverage": 0.1}, res.colony)
    assert out["colony"]["turns_skipped"] == sum(res.colony["turns_skipped"].values())


def test_effect_sampling_is_seeded_and_capped():
    keys = [(f"c{i}", s) for i in range(10) for s in (3, 4)]
    r = sample_runs("S-marks", keys, 10)
    assert r == sample_runs("S-marks", keys, 10) and len(r) == 10 and r != sample_runs("S-talk", keys, 10)
    colony = {"marks": [{}] * 100, "messages": [{"msg_id": i} for i in range(1, 50)]}
    it = sample_items("S-talk", "c0", 3, colony, 30)
    assert len(it) == 30 and it == sample_items("S-talk", "c0", 3, colony, 30)
    assert all(i["kind"] == "mark" for i in sample_items("S-marks", "c0", 3, colony, 200))
    assert len(sample_items("S-marks", "c0", 3, colony, 200)) == 100


def test_effect_study_with_real_removal_replay_on_the_fake_world():
    base = episode().colony
    baselines = {("fake", 3): base}

    def replay_fn(key, bundle, drop):
        w = CovWorld(looks=LOOKS)
        return run_colony_episode(w, lambda a: Scripted(talker, []), arm=ARMS["S-talk"], base_keys=BASE_KEYS,
                                  replay=bundle, drop=drop)

    out = effect_study("S-talk", baselines, replay_fn, n_runs=10, max_items=30)
    assert out["runs"] == 1 and out["n_items"] == len(base["marks"]) + len(base["messages"]) - out["unapplied"]
    msg_rows = [r for r in out["rows"] if r["item"]["kind"] == "message" and r["item"]["id"] == 1]
    assert msg_rows and msg_rows[0]["changed"] is True            # ant 1 acted on the ask; removing it changes its next action
    assert 0.0 <= out["spam_rate"] <= 1.0 and out["effect_rate"] > 0


def test_effect_summary_values():
    rows = [{"changed": True, "no_ant_changed": False, "coverage_delta": -0.02},
            {"changed": False, "no_ant_changed": True, "coverage_delta": 0.0},
            {"changed": False, "no_ant_changed": True, "coverage_delta": 0.0},
            {"changed": True, "no_ant_changed": False, "coverage_delta": 0.04}]
    s = effect_summary(rows)
    assert s["effect_rate"] == 0.5 and s["spam_rate"] == 0.5
    assert s["mean_delta"] == pytest.approx(0.005) and s["median_delta"] == 0.0
    assert effect_summary([])["effect_rate"] is None


def _results():
    runs = []
    for i in range(40):
        case = f"case{i:02d}"
        for s in (3, 4):
            base = 0.3 + 0.004 * i
            for arm, lift in (("S-solo", 0), ("S-marks", 0.06), ("S-talk", 0.07), ("S-marks-displaced", 0.0), ("S-talk-shuffled", 0.01),
                              ("S-marks-perfect", 0.1)):
                runs.append({"arm": arm, "case_id": case, "seed": s, "split": "dev-A" if i < 30 else "dev-B",
                             "coverage": base + lift + 0.001 * s, "off_target_share": 0.2, "time_to_50_coverage": 40 + i,
                             "coverage_at_queen": {"0": 0.0, "30": base}, "drug_on_non_tumour": 1.0,
                             "colony": None if arm == "S-solo" else {"marks_total": 4, "share_strong": 0.5, "share_disputed": 0.1,
                                                                    "asks_sent": 2, "asks_answered": 1, "asks_lapsed": 1, "asks_refused": 0,
                                                                    "quorum_events": 1, "turns_skipped": 0, "marks_by_kind": {"dosed": 3, "tumour_here": 1},
                                                                    "honesty": {"correlation": 0.5, "n_ants": 6}},
                             "timings": {"look_ms": {"mean": 1.5}, "queen_ms": {"mean": 3.0}}})
    return {"meta": {"seeds": [3, 4]}, "runs": runs,
            "effect": {"S-marks": {"runs": 10, "n_items": 300, "effect_rate": 0.4, "mean_delta": 0.001, "median_delta": 0.0, "spam_rate": 0.5, "unapplied": 0}}}


def test_report_renders_tables_from_results_json(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(_results()))
    out = tmp_path / "out" / "REPORT.md"
    text = write_report(p, out)
    assert out.read_text() == text
    for needle in ("## Primary metrics", "## Colony metrics", "## Message effect", "## Verdict statistics", "PASS",
                   "Diagnostic: S-marks-perfect", "S-talk - S-marks", "Sharing helps scripts"):
        assert needle in text
    assert text == write_report(p, out)                       # deterministic


def test_report_without_all_arms_says_so():
    r = _results()
    r["runs"] = [x for x in r["runs"] if x["arm"] in ("S-solo", "S-marks")]
    assert "Not computed" in render(r)
