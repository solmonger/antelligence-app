import json
import math

import numpy as np
import pytest

from antelligence.experiments import v41_analysis as A
from antelligence.experiments import v41_cells as C

from .conftest import make_cell, synth


def test_cells_validator():
    assert C.validate_cell(make_cell("SW-B", "c", 0, 0.5)) == []
    assert C.validate_cell({**make_cell("SW-B", "c", 0, 0.5), "bogus": 1})
    assert C.validate_cell(make_cell("SW-B", "c", 0, 1.5))
    assert C.validate_cell(make_cell("NOPE", "c", 0, 0.5))
    assert C.validate_cell({**make_cell("SW-B", "c", 0, 0.5), "score": None})
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
    bad = [c if c["arm"] != "FW" else {**c, "score": c["score"] + 0.2} for c in good]  # FW far better
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
    def run(**kw):
        return A.explained_null([make_cell("SW-B", f"c{i}", 0, 0.5, **kw) for i in range(10)])
    assert not run()["any_flagged"]
    items = run(unread_frac=0.6, brief_age_median=31.0, defer_rate=0.9, router_top_share=0.9, perception_ece=0.2)["items"]
    assert all(i["flagged"] for i in items.values())
    edge = run(unread_frac=0.5, brief_age_median=30.0, router_top_share=0.8, perception_ece=0.10)["items"]
    assert not any(i["flagged"] for i in edge.values())  # strict >
    mixed = A.explained_null([make_cell("SW-B", f"c{i}", 0, 0.5, defer_rate=0.9 if i < 3 else 0.2) for i in range(10)])
    assert mixed["items"]["defer_out_of_band"]["flagged"]  # 30% > 25%


def test_analyze_json_roundtrip(tmp_path):
    res = A.analyze(synth() + synth(split="dev", rng_seed=9), "eval")
    A.write_results(res, tmp_path / "results.json")
    assert json.loads((tmp_path / "results.json").read_text())["p1"]["passed"] is True
    assert res["p4"]["significance_tested"] is False and res["p4"]["seconds_to_50_overhead"]["n_cases"] == 30
