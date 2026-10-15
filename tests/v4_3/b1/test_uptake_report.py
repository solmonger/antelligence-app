"""B1 section 7: the realised-u report script (aggregates only) and its committed output."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from antelligence.worlds.tumor_mri.foraging import foraging_uptake, realised_means
from tests.v4_3.b1.test_foraging_field import sphere_volume, tiny

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "research" / "v4.3-20261015" / "uptake" / "realised_u.json"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("v43_uptake_report", REPO / "scripts" / "v43_uptake_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_case_stats_agree_with_realised_means(mod):
    wt, tc, et = sphere_volume(n=48, r_wt=20, r_tc=14, r_et_in=9)
    u = foraging_uptake(wt, tc, et, "c", "mid")
    s = mod.summarise(mod.case_stats(u, wt, tc, et))
    rm = realised_means(u, wt, tc, et)
    for p in ("rim", "core", "edema", "wt"):
        assert s[p]["mean_u"] == pytest.approx(rm[p]) and s[p]["n_voxels"] == rm[f"n_{p}"]
    assert s["wt"]["share_clipped_at_1.0"] == pytest.approx(rm["clipped_high"]) and s["wt"]["share_clipped_at_0.05"] == pytest.approx(rm["clipped_low"])


def test_pooled_mean_is_voxel_weighted(mod):
    a, b = tiny(), sphere_volume(n=40, r_wt=16, r_tc=11, r_et_in=7)
    stats = [mod.case_stats(foraging_uptake(*v, cid, "high"), *v) for v, cid in ((a, "a"), (b, "b"))]
    pooled = mod.pooled(stats)
    n = sum(s["wt"]["n"] for s in stats)
    assert pooled["wt"]["n_voxels"] == n
    assert pooled["wt"]["mean_u"] == pytest.approx(sum(s["wt"]["sum_u"] for s in stats) / n)


@pytest.mark.skipif(not OUT.is_file(), reason="report not generated")
def test_committed_report_is_aggregates_only():
    doc = json.loads(OUT.read_text())
    assert set(doc["levels"]) == {"low", "mid", "high"} and doc["cases"] == {"dev_a": 30, "dev_b": 10} and doc["skipped"] == []

    def lists(o, path=""):
        if isinstance(o, dict):
            return [x for k, v in o.items() for x in lists(v, f"{path}/{k}")]
        return [path] if isinstance(o, list) else []

    assert lists(doc["levels"]) == []                                  # no arrays of any kind below `levels`: scalars per case and part
    for lvl, d in doc["levels"].items():
        assert d["n_cases"] == 40 and len(d["per_case"]) == 40
        rim = d["overall_voxel_weighted"]["rim"]["mean_u"]
        assert {"low": 0.92, "mid": 0.84, "high": 0.76}[lvl] == pytest.approx(rim, abs=0.03)
        for row in d["per_case"].values():
            assert set(row) == {"group", "rim", "core", "edema", "wt"}
            assert all(set(row[p]) == {"n_voxels", "mean_u", "share_clipped_at_0.05", "share_clipped_at_1.0"} for p in ("rim", "core", "edema", "wt"))
