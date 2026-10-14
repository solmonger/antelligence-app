import numpy as np
import pytest

from antelligence.experiments.v42_stats import (analyse, cluster_bootstrap, group_by_case, holm, paired_diffs,
                                                primary_verdict)

GOOD = dict(median_diff=0.05, holm_p=0.01, median_dev_a=0.04, median_dev_b=0.06, control_median=0.03, control_ci_low=0.01)


def test_bootstrap_deterministic_and_seeded_by_name():
    by_case = {f"c{i}": [0.01 * i, 0.02 * i] for i in range(12)}
    a = cluster_bootstrap(by_case, "x", n=500)
    assert a == cluster_bootstrap(by_case, "x", n=500)
    assert a["ci_low"] != cluster_bootstrap(by_case, "y", n=500)["ci_low"] or a["p"] != cluster_bootstrap(by_case, "y", n=500)["p"]


def test_bootstrap_known_answer():
    same = cluster_bootstrap({f"c{i}": [0.1, 0.1] for i in range(10)}, "k", n=1000)
    assert (same["median"], same["ci_low"], same["ci_high"]) == (pytest.approx(0.1),) * 3
    assert same["p"] == pytest.approx(2 / 1001)                          # every resample > 0
    zero = cluster_bootstrap({f"c{i}": [0.0, 0.0] for i in range(10)}, "z", n=1000)
    assert zero["p"] == 1.0 and zero["median"] == 0.0
    # sign-symmetric data: interval straddles 0, p large
    sym = cluster_bootstrap({f"c{i}": [(-1) ** i * 0.1] * 2 for i in range(10)}, "s", n=2000)
    assert sym["ci_low"] < 0 < sym["ci_high"] and sym["p"] > 0.3


def test_bootstrap_unequal_cluster_sizes_and_empty():
    r = cluster_bootstrap({"a": [0.1], "b": [0.1, 0.1, 0.1]}, "u", n=200)
    assert r["median"] == pytest.approx(0.1) and r["n_pairs"] == 4
    assert cluster_bootstrap({}, "e")["p"] is None


def test_resamples_clusters_not_pairs():
    # one case with a huge value: a case-level resample either has it or not, so the median CI stays near the bulk
    by_case = {f"c{i}": [0.1, 0.1] for i in range(9)}
    by_case["big"] = [5.0, 5.0]
    r = cluster_bootstrap(by_case, "cl", n=500)
    assert r["median"] == pytest.approx(0.1) and r["ci_high"] <= 5.0


def test_holm():
    assert holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert holm({"a": 0.04, "b": 0.03}) == {"a": 0.06, "b": 0.06}       # monotone: 0.03*2=0.06, max(0.06, 0.04)
    assert holm({"a": 0.9, "b": 0.8})["a"] == 1.0
    assert holm({"a": 0.2}) == {"a": 0.2}


def test_verdict_pass_and_each_edge():
    assert primary_verdict(**GOOD)["pass"]
    assert primary_verdict(**{**GOOD, "median_diff": 0.03})["pass"]                 # boundary is inclusive
    assert not primary_verdict(**{**GOOD, "median_diff": 0.0299})["pass"]
    assert not primary_verdict(**{**GOOD, "holm_p": 0.05})["pass"]                  # strictly below
    assert primary_verdict(**{**GOOD, "holm_p": 0.0499})["pass"]
    assert not primary_verdict(**{**GOOD, "median_dev_b": -0.01})["pass"]           # sign flip
    assert not primary_verdict(**{**GOOD, "median_dev_b": 0.0})["pass"]             # zero is not same sign
    assert not primary_verdict(**{**GOOD, "median_dev_a": None})["pass"]
    v = primary_verdict(**{**GOOD, "median_dev_a": -0.02, "median_dev_b": -0.03})
    assert v["same_sign_dev_a_b"] and not v["pass"]
    assert not primary_verdict(**{**GOOD, "control_ci_low": 0.0})["pass"]           # interval must be above 0
    assert not primary_verdict(**{**GOOD, "control_median": 0.0, "control_ci_low": 0.01})["pass"]
    assert not primary_verdict(**{**GOOD, "control_median": None})["pass"]


def test_paired_by_case_and_seed():
    vals = {"x": {("a", 3): 0.5, ("a", 4): 0.6, ("b", 3): 0.2}, "y": {("a", 3): 0.4, ("a", 4): 0.4}}
    d = paired_diffs(vals, "x", "y")
    assert d == {("a", 3): pytest.approx(0.1), ("a", 4): pytest.approx(0.2)}      # ("b",3) has no partner
    assert group_by_case(d) == {"a": [pytest.approx(0.1), pytest.approx(0.2)]}


def _values(lift_marks, lift_talk, ctrl_gap=0.05):
    g = np.random.default_rng(0)
    v = {a: {} for a in ("S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled")}
    splits = {}
    for i in range(40):
        case = f"case{i:02d}"
        splits[case] = "dev-A" if i < 30 else "dev-B"
        for s in (3, 4):
            base = 0.4 + 0.05 * g.standard_normal()
            v["S-solo"][(case, s)] = base
            v["S-marks"][(case, s)] = base + lift_marks + 0.005 * g.standard_normal()
            v["S-talk"][(case, s)] = base + lift_talk + 0.005 * g.standard_normal()
            v["S-marks-displaced"][(case, s)] = base + lift_marks - ctrl_gap
            v["S-talk-shuffled"][(case, s)] = base + lift_talk - ctrl_gap
    return v, splits


def test_analyse_end_to_end():
    v, splits = _values(0.08, 0.0)
    out = analyse(v, splits, n=500)
    assert out["primary"]["a"]["verdict"]["pass"] and not out["primary"]["b"]["verdict"]["pass"]
    assert out["sharing_helps_scripts"]
    assert out["primary"]["a"]["holm_p"] >= out["primary"]["a"]["boot"]["p"]
    v, splits = _values(0.08, 0.08, ctrl_gap=0.0)                       # lift not above the control -> no pass
    out = analyse(v, splits, n=500)
    assert not out["primary"]["a"]["verdict"]["pass"] and not out["sharing_helps_scripts"]
