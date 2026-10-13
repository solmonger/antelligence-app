import json
import math

import pytest

from antelligence.experiments import v41_analysis as A
from antelligence.experiments import v41_cells as C

from .conftest import make_cell, synth


def test_cells_validator():
    assert C.validate_cell(make_cell("SW-B", "c", 0, 0.5)) == []
    assert C.validate_cell({**make_cell("SW-B", "c", 0, 0.5), "bogus": 1})
    assert C.validate_cell(make_cell("SW-B", "c", 0, 1.5))
    assert C.validate_cell(make_cell("NOPE", "c", 0, 0.5))
    assert C.validate_cell({**make_cell("SW-B", "c", 0, 0.5), "coverage": None})
    assert C.validate_cells([make_cell("SW-B", "c", 0, 0.5)] * 2)  # duplicate
    from .conftest import ROOT
    doc = (ROOT / "docs" / "v4.1" / "CELLS.md").read_text()
    assert all(f"`{n}`" in doc for n in C.FIELD_NAMES)


def test_p1_wilcoxon_and_bca_on_synthetic():
    cells = synth(effect=0.05)
    r = A.p1(cells)
    assert r["n_cases"] == 30 and r["passed"] is True and r["p_value"] < 0.001
    assert r["ci_method"] == "BCa" and r["ci_low"] < r["mean_diff"] < r["ci_high"] and r["ci_low"] > 0
    assert A.p1(cells) == r  # fixed seed -> reproducible
    # case-level mean over seeds: each case contributes once
    assert r["n_cases"] == len({c["case"] for c in cells})
    null = A.p1(synth(effect=0.0, noise=0.03, rng_seed=5))
    assert null["passed"] is False


def test_p1_requires_positive_mean():
    r = A.p1(synth(effect=-0.05))
    assert r["p_value"] < A.ALPHA and r["mean_diff"] < 0 and r["passed"] is False


def test_p2_gated_on_p1():
    good = synth(effect=0.05)  # FW = SW-B + 0.01 -> noninferior
    r1 = A.p1(good)
    r2 = A.p2(good, r1)
    assert r2["noninferior"] and r2["passed"] and r2["status"] == "confirmatory"
    failed_p1 = {**r1, "passed": False}
    r2b = A.p2(good, failed_p1)
    assert r2b["noninferior"] is True and r2b["passed"] is False and r2b["status"] == "exploratory"
    bad = [c if c["arm"] != "FW" else {**c, "coverage": c["coverage"] + 0.2} for c in good]  # FW far better
    assert A.p2(bad, r1)["noninferior"] is False


def test_s9_ratio_of_means_filter():
    cells = []
    # case a: gap 0.4 (kept), b: gap 0.1 (dropped), c: gap 0.2 (kept)
    for case, s, sp, x in (("a", 0.2, 0.6, 0.4), ("b", 0.5, 0.6, 0.0), ("c", 0.3, 0.5, 0.5)):
        for arm, v in (("S", s), ("S-prime", sp), ("SW-B", x)):
            cells.append(make_cell(arm, case, 0, v))
    r = A.s9_normalised(cells, ["SW-B"])
    assert r["n_cases_gap"] == 2
    # ratio of means: mean(0.2, 0.2) / mean(0.4, 0.2) = 0.2/0.3
    assert r["arms"]["SW-B"]["ratio"] == pytest.approx(0.2 / 0.3)
    assert r["arms"]["SW-B"]["ratio"] != pytest.approx((0.2 / 0.4 + 0.2 / 0.2) / 2)  # not mean of ratios


def test_power_mdd():
    mdd = A.minimum_detectable_diff(0.1, 80)
    z = 1.959963984540054 + 0.8416212335729143
    assert mdd == pytest.approx(z * 0.1 / math.sqrt(80 * 3 / math.pi))
    assert mdd > z * 0.1 / math.sqrt(80)  # Wilcoxon needs a larger effect than the t-test
    assert A.minimum_detectable_diff(0.1, 0) is None
    dev = synth(split="dev", effect=0.05)
    pw = A.power_analysis(dev + synth(split="eval"))
    assert pw["dev_cases"] == 30 and pw["mdd_p1"] == pytest.approx(A.minimum_detectable_diff(pw["dev_paired_sd"], 80))


def test_explained_null_flags():
    def run(dev_ece=0.05, **kw):
        runs = [make_cell("SW-B", f"c{i}", 0, 0.5, **kw) for i in range(10)]
        dev = [make_cell("SW-B", f"d{i}", 0, 0.5, split="dev", perception_ece=dev_ece) for i in range(3)]
        return A.explained_null(runs, dev_cells=dev)
    assert not run()["any_flagged"]
    items = run(0.2, unread_frac=0.6, brief_age_median=31.0, defer_rate=0.9, router_top_share=0.9)["items"]
    assert all(i["flagged"] for i in items.values()) and all(i["basis"] for i in items.values())
    edge = run(0.10, unread_frac=0.5, brief_age_median=30.0, router_top_share=0.8)["items"]
    assert not any(i["flagged"] for i in edge.values())  # strict >
    mixed = A.explained_null([make_cell("SW-B", f"c{i}", 0, 0.5, defer_rate=0.9 if i < 3 else 0.2) for i in range(10)])
    assert mixed["items"]["defer_out_of_band"]["flagged"]  # 30% > 25%
    assert mixed["items"]["perception_ece"]["value"] is None  # no development runs given


def test_secondary_matches_goal_section_7():
    cells = synth(arms={"SW-B", "SW-noBoard", "S", "S-prime", "FW", "SW-D"})
    for arm in ("SW-noQueen", "SW-staticQueen", "SW-Bonsai", "SW-Byz-local", "SW-Byz-chain", "SW-B-missingFLAIR"):
        cells += [make_cell(arm, f"BRATS_eval_{i:03d}", 0, 0.4) for i in range(5)]
    sec = A.secondary(A.usable(cells), latency={"d1_c1_s_per_decision": 1.5})
    assert list(sec) == [f"S{i}" for i in range(1, 13)]
    assert sec["S1"]["comparisons"]["SW-noQueen_vs_SW-B"]["n_cases"] == 5
    assert set(sec["S4"]["comparisons"]) == {"SW-Byz-local_vs_SW-Byz-chain", "SW-Byz-local_vs_SW-B", "SW-Byz-chain_vs_SW-B"}
    assert sec["S4"]["comparisons"]["SW-Byz-local_vs_SW-Byz-chain"]["mean_diff"] == pytest.approx(0.0)
    assert set(sec["S5"]["fields"]) == {"perception_dice", "perception_ece", "locallook_ece", "truth_outside_crop"}
    assert {"defer_rate", "alarms", "audit_conformal_coverage"} == set(sec["S6"]["fields"])
    assert {"citation_validity", "uncited_decisions", "unread_frac"} == set(sec["S7"]["fields"])
    assert "seconds_to_50" in sec["S8"]["fields"] and "off_target" in sec["S8"]["fields"]
    assert {"tokens_in", "tokens_out", "pod_minutes", "gas", "cost_usd"} == set(sec["S11"]["fields"])
    assert sec["S12"]["available"] and sec["S12"]["metrics"] == {"d1_c1_s_per_decision": 1.5}
    assert sec["S10"]["comparisons"]["SW-B-missingFLAIR_vs_SW-B"]["mean_diff"] is not None


def test_single_seed_arm_pairs_on_its_own_seed():
    cells = [make_cell("SW-B", "c", 0, 0.9), make_cell("SW-B", "c", 1, 0.5), make_cell("FW", "c", 1, 0.5)]
    _, d = A.paired_case_diffs(cells, "SW-B", "FW")
    assert list(d) == [0.0]


def test_defer_band_and_live_check():
    assert A.NULL_DEFER_BAND == (0.05, 0.40) and A.LIVE_CHECK_DEFER_BAND == (0.10, 0.30)
    runs = [make_cell("SW-B", f"c{i}", 0, 0.5, defer_rate=0.35 if i < 2 else 0.2) for i in range(10)]
    assert not A.explained_null(runs)["items"]["defer_out_of_band"]["flagged"]
    assert A.live_check(runs)["in_band"] is True
    assert A.live_check([make_cell("SW-B", "c", 0, 0.5, defer_rate=0.35)])["in_band"] is False


def test_analyze_json_roundtrip(tmp_path):
    res = A.analyze(synth() + synth(split="dev", rng_seed=9), "eval")
    A.write_results(res, tmp_path / "results.json")
    assert json.loads((tmp_path / "results.json").read_text())["p1"]["passed"] is True
    assert res["p4"]["significance_tested"] is False and res["p4"]["seconds_to_50_overhead"]["n_cases"] == 30


def test_decision_error_sensitivity_only_when_an_arm_exceeds_five_percent():
    clean = synth(n_cases=12, arms={"SW-B", "SW-noBoard", "FW"})
    quiet = A.decision_error_sensitivity(A.usable(clean))
    assert not quiet["triggered"] and quiet["sensitivity"] is None and set(quiet["per_arm"]) == {"SW-B", "SW-noBoard", "FW"}
    assert quiet["per_arm"]["FW"]["mean_share"] == 0.0 and quiet["caveat"]
    noisy = []
    for c in clean:
        bad = c["arm"] == "FW" and c["case"].endswith(("0", "1", "2", "3"))
        noisy.append({**c, "decision_errors": 4 if bad else 0, "decision_error_share": 0.2 if bad else 0.0,
                      "decision_error_alarm": int(bad)})
    out = A.decision_error_sensitivity(A.usable(noisy))
    fw = out["per_arm"]["FW"]
    assert out["triggered"] and fw["mean_share"] > A.DECISION_ERROR_SHARE_MAX and fw["runs_alarmed"] == fw["runs_with_errors"] > 0
    sens = out["sensitivity"]
    assert sens["excluded_runs"] == fw["runs_with_errors"] and sens["kept_runs"] == len(noisy) - sens["excluded_runs"]
    assert sens["p1"]["n_cases"] > 0 and sens["p2"]["status"] == "exploratory" and sens["p2"]["n_cases"] < 12
    # the full payload carries it as a secondary; nothing else changes
    res = A.analyze(noisy + synth(n_cases=12, split="dev", arms={"SW-B", "SW-noBoard"}), "eval")
    assert res["decision_errors"]["triggered"] and res["p1"]["n_cases"] == 12
