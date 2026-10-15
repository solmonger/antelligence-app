"""C: replication verdict, headroom report, patchiness curve, S' ceiling check. All on fabricated results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from antelligence.experiments import v43_stats as S
from antelligence.experiments.v42_stats import holm
from tests.v4_3.c.helpers import case_names, fab_runs, keys, script

N = 400                      # bootstrap resamples (fast; the stream is seeded so the result is deterministic)


def replication_results(effect_a=0.06, effect_b=0.06, control_gap=0.05, *, jitter=0.01, seeds=(7, 8), arm="S-marks", control="S-marks-displaced",
                        n_a=30, n_b=10):
    """40 cases x 2 seeds. S-solo ~ 0.30; ``arm`` = solo + effect (A and B cases); ``control`` = arm - control_gap."""
    splits = case_names(n_a, n_b)
    g = np.random.default_rng(3)
    solo, arm_v, ctl = {}, {}, {}
    for c, s in keys(splits, seeds):
        base = 0.30 + 0.05 * g.random()
        eff = effect_a if splits[c] == "dev-A" else effect_b
        solo[(c, s)] = base
        arm_v[(c, s)] = base + eff + jitter * (g.random() - 0.5)
        ctl[(c, s)] = arm_v[(c, s)] - control_gap + jitter * (g.random() - 0.5)
    return fab_runs({"S-solo": solo, arm: arm_v, control: ctl}, splits)


def test_a_clean_replication_passes_the_full_bar():
    doc = S.replication_verdict(replication_results(), "S-marks", "S-marks-displaced", n=N, expected_pairs=80)
    v = doc["verdict"]
    assert doc["n_pairs"] == 80 and doc["n_cases"] == 40 and doc["seeds"] == [7, 8]
    assert v["median_ge_0.03"] and v["holm_p_lt_0.05"] and v["same_sign_dev_a_b"] and v["beats_control"] and v["pairs_complete"] and v["pass"]
    assert doc["boot"]["median"] == pytest.approx(0.06, abs=0.01)
    assert doc["pass"] is True


def test_talk_pair_uses_its_own_control():
    res = replication_results(arm="S-talk", control="S-talk-shuffled")
    assert S.replication_verdict(res, "S-talk", "S-talk-shuffled", n=N, expected_pairs=80)["pass"]
    with pytest.raises(ValueError, match="missing"):
        S.replication_verdict(res, "S-marks", "S-marks-displaced", n=N)


def test_fails_when_the_median_gain_is_below_0_03_and_only_that_condition_fails():
    doc = S.replication_verdict(replication_results(effect_a=0.02, effect_b=0.02), "S-marks", "S-marks-displaced", n=N, expected_pairs=80)
    v = doc["verdict"]
    assert not v["median_ge_0.03"] and not v["pass"]
    assert v["holm_p_lt_0.05"] and v["same_sign_dev_a_b"] and v["beats_control"]


def test_fails_when_dev_a_and_dev_b_disagree_in_sign_and_only_that_condition_fails():
    doc = S.replication_verdict(replication_results(effect_a=0.07, effect_b=-0.03), "S-marks", "S-marks-displaced", n=N, expected_pairs=80)
    v = doc["verdict"]
    assert doc["by_split"]["dev-A"] > 0 > doc["by_split"]["dev-B"]
    assert not v["same_sign_dev_a_b"] and not v["pass"]
    assert v["median_ge_0.03"] and v["holm_p_lt_0.05"] and v["beats_control"]


def test_fails_when_the_arm_does_not_beat_its_own_control():
    doc = S.replication_verdict(replication_results(control_gap=-0.02), "S-marks", "S-marks-displaced", n=N, expected_pairs=80)    # control is better
    v = doc["verdict"]
    assert not v["beats_control"] and not v["pass"]
    assert v["median_ge_0.03"] and v["holm_p_lt_0.05"] and v["same_sign_dev_a_b"]
    assert doc["control_boot"]["median"] < 0


def test_fails_when_the_control_median_is_positive_but_its_interval_reaches_zero():
    res = replication_results(control_gap=0.0003, jitter=0.2)               # noisy, essentially no edge over the control
    v = S.replication_verdict(res, "S-marks", "S-marks-displaced", n=N)["verdict"]
    assert not v["beats_control"]


def test_fails_when_the_gain_is_not_significant_after_holm_and_only_that_condition_fails():
    """A median of +0.06 carried by 55% of the cases while the rest are negative: the median passes, the bootstrap p is large."""
    splits = case_names()
    solo = {k: 0.3 for k in keys(splits, (7, 8))}
    arm, ctl = {}, {}
    seen = {"dev-A": 0, "dev-B": 0}
    for c in sorted(splits):
        pos = seen[splits[c]] < {"dev-A": 16, "dev-B": 6}[splits[c]]       # 22 of 40 cases (55%) at +0.06, the rest at -0.05; both splits lean positive
        seen[splits[c]] += 1
        for s in (7, 8):
            arm[(c, s)] = 0.3 + (0.06 if pos else -0.05)
            ctl[(c, s)] = arm[(c, s)] - 0.05
    doc = S.replication_verdict(fab_runs({"S-solo": solo, "S-marks": arm, "S-marks-displaced": ctl}, splits), "S-marks", "S-marks-displaced", n=N)
    v = doc["verdict"]
    assert v["median_ge_0.03"] and not v["holm_p_lt_0.05"] and not v["pass"] and v["same_sign_dev_a_b"] and v["beats_control"]
    assert doc["holm_p"] > 0.05


def test_holm_gives_the_unreplicated_primary_p_equal_one(monkeypatch):
    """Adjusted p = min(1, 2p). Pin the arithmetic around the 0.025 edge by fixing the raw p the bootstrap returns."""
    assert holm({"replicated": 0.02, "not_replicated": 1.0})["replicated"] == pytest.approx(0.04)
    assert holm({"replicated": 0.03, "not_replicated": 1.0})["replicated"] == pytest.approx(0.06)
    assert holm({"replicated": 0.7, "not_replicated": 1.0})["replicated"] == 1.0
    real = S.cluster_bootstrap

    def fake(by_case, name, **kw):
        out = real(by_case, name, **kw)
        if name.endswith("S-marks-minus-S-solo"):
            out = {**out, "p": fake.p}
        return out

    monkeypatch.setattr(S, "cluster_bootstrap", fake)
    res = replication_results()
    for raw, passes in ((0.02, True), (0.0249, True), (0.0251, False), (0.04, False)):
        fake.p = raw
        doc = S.replication_verdict(res, "S-marks", "S-marks-displaced", n=N)
        assert doc["holm_p"] == pytest.approx(min(1.0, 2 * raw)) and doc["raw_p"] == raw
        assert doc["verdict"]["holm_p_lt_0.05"] is passes and doc["pass"] is passes
    fake.p = 0.9
    assert S.replication_verdict(res, "S-marks", "S-marks-displaced", n=N)["holm_p"] == 1.0


def test_an_incomplete_replication_fails_even_if_the_numbers_are_good():
    res = replication_results()
    res["runs"] = [r for r in res["runs"] if not (r["case_id"] == "A00" and r["seed"] == 8)]          # drop one case-seed of all three arms
    doc = S.replication_verdict(res, "S-marks", "S-marks-displaced", n=N, expected_pairs=80)
    assert doc["n_pairs"] == 79 and not doc["verdict"]["pairs_complete"] and not doc["pass"]
    assert doc["verdict"]["median_ge_0.03"] and doc["verdict"]["beats_control"]
    assert S.replication_verdict(res, "S-marks", "S-marks-displaced", n=N)["pass"]                      # no expectation given: just the bar


def test_the_seeds_filter_restricts_the_pairs():
    both = replication_results(seeds=(5, 6, 7, 8))
    doc = S.replication_verdict(both, "S-marks", "S-marks-displaced", n=N, expected_pairs=80, seeds=(7, 8))
    assert doc["seeds"] == [7, 8] and doc["n_pairs"] == 80 and doc["pass"]


def test_replication_is_deterministic():
    r = replication_results()
    a, b = (S.replication_verdict(r, "S-marks", "S-marks-displaced", n=N) for _ in range(2))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- headroom ------------------------------------------------------------------------------------------------------------------------

def headroom_results():
    splits = {"A00": "dev-A", "A01": "dev-A", "A02": "dev-A", "A03": "dev-A", "B00": "dev-B"}
    gaps = {"A00": 0.30, "A01": 0.10, "A02": 0.0999, "A03": 0.02, "B00": 0.20}
    solo = {(c, 1): 0.30 for c in splits}
    prime = {(c, 1): 0.30 + g for c, g in gaps.items()}
    return fab_runs({"S-solo": solo, "S'": prime, "S-marks": {(c, 1): 0.9 for c in splits}}, splits), gaps


def test_headroom_share_mean_and_median_gap_and_no_pass_flag():
    res, gaps = headroom_results()
    h = S.headroom_report(res)
    assert h["n_pairs"] == 5 and h["threshold"] == 0.10
    assert h["share_pairs_ge_threshold"] == pytest.approx(3 / 5)             # 0.30, 0.10 (the edge counts) and 0.20; 0.0999 and 0.02 do not
    assert h["mean_gap"] == pytest.approx(np.mean(list(gaps.values()))) and h["median_gap"] == pytest.approx(0.10)
    assert h["share_cases_ge_threshold"] == pytest.approx(3 / 5)
    assert not any("pass" in k for k in h) and "report only" in h["note"]
    assert S.headroom_report(res, split="dev-A")["n_pairs"] == 4 and S.headroom_report(res, split="dev-B")["share_pairs_ge_threshold"] == 1.0


def test_headroom_needs_both_arms_and_uses_only_the_chosen_seeds():
    with pytest.raises(ValueError):
        S.headroom_report(fab_runs({"S-solo": {("A00", 1): 0.3}}, {"A00": "dev-A"}))
    splits = {"A00": "dev-A"}
    res = fab_runs({"S-solo": {("A00", 1): 0.3, ("A00", 2): 0.3}, "S'": {("A00", 1): 0.5, ("A00", 2): 0.31}}, splits)
    assert S.headroom_report(res, seeds=[1])["share_pairs_ge_threshold"] == 1.0
    assert S.headroom_report(res, seeds=[2])["share_pairs_ge_threshold"] == 0.0
    assert S.headroom_report(res)["share_pairs_ge_threshold"] == 0.5


def test_headroom_script_writes_a_report_without_a_pass_flag(tmp_path):
    res, _ = headroom_results()
    (tmp_path / "results.json").write_text(json.dumps(res))
    mod = script("v43_headroom")
    assert mod.main(["--results", str(tmp_path), "--split", "dev-A"]) == 0
    doc = json.loads((tmp_path / "headroom.json").read_text())
    assert doc["n_pairs"] == 4 and doc["split"] == "dev-A" and doc["source"].endswith("results.json")
    assert mod.main(["--results", str(tmp_path / "results.json"), "--out", str(tmp_path / "h2.json")]) == 0 and (tmp_path / "h2.json").is_file()


# --- curve ---------------------------------------------------------------------------------------------------------------------------

def curve_results(d_marks, d_talk, *, level=None, jitter=0.0, seeds=(5, 6)):
    splits = case_names(6, 2)
    g = np.random.default_rng(11)
    cov = {a: {} for a in ("S", "S-solo", "S-marks", "S-talk", "S'")}
    for c, s in keys(splits, seeds):
        solo = 0.30 + 0.02 * g.random()
        cov["S"][(c, s)] = 0.25
        cov["S-solo"][(c, s)] = solo
        cov["S-marks"][(c, s)] = solo + d_marks + jitter * (g.random() - 0.5)
        cov["S-talk"][(c, s)] = solo + d_talk + jitter * (g.random() - 0.5)
        cov["S'"][(c, s)] = 0.8
    res = fab_runs(cov, splits)
    if level:
        res["meta"]["level"] = level
    return res


def test_curve_level_reports_means_medians_intervals_and_arm_means():
    lv = S.curve_level(curve_results(0.05, -0.01), n=N, level="mid")
    m, t = lv["vs_s_solo"]["S-marks"], lv["vs_s_solo"]["S-talk"]
    assert m["n_pairs"] == 16 and m["n_cases"] == 8
    assert m["mean"] == pytest.approx(0.05) and m["median"] == pytest.approx(0.05)
    assert m["mean_ci"] == pytest.approx([0.05, 0.05]) and m["median_ci"] == pytest.approx([0.05, 0.05])       # no spread, so a point interval
    assert t["mean"] == pytest.approx(-0.01)
    assert set(lv["arm_means"]) == {"S", "S-solo", "S-marks", "S-talk", "S'"} and lv["arm_means"]["S'"] == pytest.approx(0.8)
    assert not any("pass" in k for k in lv) and "pass" not in m


def test_curve_intervals_widen_with_noise_and_contain_the_estimate():
    lv = S.curve_level(curve_results(0.05, 0.0, jitter=0.2), n=N, level="low")
    m = lv["vs_s_solo"]["S-marks"]
    assert m["mean_ci"][0] < m["mean"] < m["mean_ci"][1] and m["median_ci"][0] < m["median_ci"][1]
    assert m["mean_ci"][1] - m["mean_ci"][0] > 0.01


def test_curve_over_three_levels_and_markdown():
    doc = S.curve({"low": curve_results(0.01, 0.0), "mid": curve_results(0.04, 0.0), "high": curve_results(0.08, 0.01)}, n=N)
    assert list(doc["levels"]) == ["low", "mid", "high"]
    assert [doc["levels"][k]["vs_s_solo"]["S-marks"]["mean"] for k in ("low", "mid", "high")] == pytest.approx([0.01, 0.04, 0.08])
    md = S.curve_markdown(doc)
    assert md.startswith("## Patchiness curve") and "| high | S-marks |" in md and "| mid | S-talk |" in md and "No pass/fail" in md


def test_curve_handles_a_missing_arm():
    res = curve_results(0.05, 0.0)
    res["runs"] = [r for r in res["runs"] if r["arm"] != "S-talk"]
    lv = S.curve_level(res, n=N)
    assert lv["vs_s_solo"]["S-talk"] is None and lv["vs_s_solo"]["S-marks"] is not None
    assert "n/a" in S.curve_markdown(S.curve({"mid": res}, n=N))


def test_curve_script_writes_json_and_markdown_and_checks_levels(tmp_path):
    mod = script("v43_curve")
    for lvl, d in (("low", 0.01), ("mid", 0.04), ("high", 0.08)):
        (tmp_path / lvl).mkdir()
        (tmp_path / lvl / "results.json").write_text(json.dumps(curve_results(d, 0.0, level=lvl)))
    out = tmp_path / "out"
    assert mod.main(["--low", str(tmp_path / "low"), "--mid", str(tmp_path / "mid"), "--high", str(tmp_path / "high"), "--out", str(out),
                     "--resamples", str(N)]) == 0
    doc = json.loads((out / "curve.json").read_text())
    assert set(doc["sources"]) == {"low", "mid", "high"} and doc["levels"]["high"]["vs_s_solo"]["S-marks"]["mean"] == pytest.approx(0.08)
    assert "Source:" in (out / "curve.md").read_text()
    assert mod.main(["--low", str(tmp_path / "mid"), "--mid", str(tmp_path / "mid"), "--high", str(tmp_path / "high"), "--out", str(tmp_path / "o2")]) == 2


# --- ceiling check -------------------------------------------------------------------------------------------------------------------------

def test_ceiling_check_logs_violations_and_the_share():
    runs = [{"case_id": f"c{i}", "seed": 1, "arm": a, "coverage": v} for i in range(20)
            for a, v in (("S'", 0.7), ("S", 0.5), ("S-marks", 0.6 if i else 0.9))]
    chk = S.ceiling_check(runs, ["S", "S-marks", "S'"])
    assert chk["share_s_prime_ge_all"] == 0.95 and chk["ok"] and chk["violations"][0]["case_id"] == "c0"
    runs[4]["coverage"] = 0.99
    assert not S.ceiling_check(runs, ["S", "S-marks", "S'"])["ok"]
