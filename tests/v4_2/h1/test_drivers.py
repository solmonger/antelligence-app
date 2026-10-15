"""Tune / verdict / effect drivers: refusals, selection rule, grid, resumability, pool limit."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from antelligence.experiments import v42_drive as drive
from antelligence.experiments.v42_harness import EvalRefused
from antelligence.experiments.v41_gate import load_checked_split

REPO = Path(__file__).resolve().parents[3]


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tune, verdict, effect = script("v42_tune"), script("v42_verdict"), script("v42_effect")
HAVE_DATA = drive.DEFAULT_CASES_ROOT.is_dir() and (drive.DEFAULT_CASES_ROOT / "BRATS_016" / "case.npz").is_file()
CFG = REPO / "config" / "v4_2" / "world.yaml"


def test_grid_is_12_points_and_within_bounds():
    g = tune.grid()
    assert len(g) == 12 <= tune.MAX_POINTS
    assert {p["u_core"] for p in g} == {0.3, 0.5, 0.7} and {p["f3_decay"] for p in g} == {0.04, 0.08} and {p["f3_share"] for p in g} == {0.1, 0.2}


@pytest.mark.parametrize("arm", ["S", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S-marks-perfect", "S-marks-quiet"])
def test_tune_refuses_every_sharing_arm(arm, tmp_path):
    with pytest.raises(tune.TuneArmRefused):
        tune.check_arms([arm])
    with pytest.raises(tune.TuneArmRefused):
        tune.make_tasks(tune.grid(), ["BRATS_016"], ["S-solo", arm], config=CFG, cases_root=None, ticks=None, done=set())
    assert tune.main(["--out", str(tmp_path / "t"), "--arms", f"S-solo,{arm}"]) == 2
    assert not (tmp_path / "t").exists()


def test_tune_tasks_are_solo_and_prime_only():
    tasks = tune.make_tasks(tune.grid(), ["BRATS_016"], list(tune.TUNE_ARMS), config=CFG, cases_root=None, ticks=None, done=set())
    assert {t["arm"] for t in tasks} == {"S-solo", "S'"} and len(tasks) == 24 and all(t["seed"] == 1 for t in tasks)


def test_tune_refuses_eval_and_dev_b(tmp_path):
    doc = load_checked_split(REPO)
    for c in (doc["eval"][0], doc["dev_b"][0], "BRATS_nope"):
        assert tune.main(["--out", str(tmp_path / "t"), "--cases", c]) == 2


def _results(i, solo, prime):
    return {(i, "S-solo", "c"): {"coverage": solo, "timing_s": {"wall": 1}}, (i, "S'", "c"): {"coverage": prime, "timing_s": {"wall": 1}}}


def test_selection_rule_and_flag():
    pts = [{"u_core": 0.3, "f3_decay": 0.04, "f3_share": 0.1}, {"u_core": 0.5, "f3_decay": 0.04, "f3_share": 0.1}]
    res = {**_results(0, 0.30, 0.70), **_results(1, 0.60, 0.90)}       # point 1: both medians out of band
    rows = tune.summarise(pts, ["c"], res)
    assert rows[0]["qualifies"] and not rows[1]["qualifies"]
    assert tune.select(rows)["selected"] == 0
    res = {**_results(0, 0.10, 0.70), **_results(1, 0.60, 0.90)}
    sel = tune.select(tune.summarise(pts, ["c"], res))
    assert sel["selected"] is None and "NO POINT QUALIFIES" in sel["flag"]


def test_selection_prefers_larger_share_of_gap_cases():
    pts = [{"u_core": 0.3, "f3_decay": 0.04, "f3_share": 0.1}, {"u_core": 0.5, "f3_decay": 0.04, "f3_share": 0.1}]
    res = {}
    for i, gaps in ((0, (0.1, 0.1, 0.3)), (1, (0.2, 0.2, 0.3))):
        for j, g in enumerate(gaps):
            res[(i, "S-solo", f"c{j}")] = {"coverage": 0.40, "timing_s": {"wall": 1}}
            res[(i, "S'", f"c{j}")] = {"coverage": 0.40 + g, "timing_s": {"wall": 1}}
    rows = tune.summarise(pts, ["c0", "c1", "c2"], res)
    assert rows[0]["share_gap_ge_0.15"] < rows[1]["share_gap_ge_0.15"]
    assert tune.select(rows)["selected"] in (None, 1)


@pytest.mark.skipif(not HAVE_DATA, reason="dev case data not mounted")
def test_tune_end_to_end_resumable_and_never_writes_config(tmp_path, capsys):
    before = hashlib.sha256(CFG.read_bytes()).hexdigest()
    args = ["--out", str(tmp_path), "--cases", "BRATS_016", "--workers", "2", "--ticks", "30"]
    assert tune.main(args) == 0
    doc = json.loads((tmp_path / "tune_results.json").read_text())
    assert len(doc["points"]) == 12 and doc["arms"] == ["S-solo", "S'"] and "NOT written" in doc["note"]
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 24
    capsys.readouterr()
    assert tune.main(args) == 0
    assert "0 episodes to run, 24 already done" in capsys.readouterr().out
    assert hashlib.sha256(CFG.read_bytes()).hexdigest() == before
    assert not list((tmp_path).glob("**/world.yaml"))


def test_ceiling_check():
    runs = [{"case_id": f"c{i}", "seed": 1, "arm": a, "coverage": v} for i in range(20)
            for a, v in (("S'", 0.7), ("S", 0.5), ("S-marks", 0.6 if i else 0.9))]
    chk = verdict.ceiling_check(runs, ["S", "S-marks", "S'"])
    assert chk["share_s_prime_ge_all"] == 0.95 and chk["ok"] and chk["violations"][0]["case_id"] == "c0"
    runs[4]["coverage"] = 0.99
    assert not verdict.ceiling_check(runs, ["S", "S-marks", "S'"])["ok"]


def test_verdict_refuses_eval_workers_and_unfrozen_config(tmp_path):
    doc = load_checked_split(REPO)
    assert verdict.main(["--out", str(tmp_path / "v"), "--cases", doc["eval"][0]]) == 2
    assert verdict.main(["--out", str(tmp_path / "v"), "--cases", doc["reserve"][0]]) == 2
    assert verdict.main(["--out", str(tmp_path / "v"), "--cases", "BRATS_016", "--workers", "7"]) == 2
    assert verdict.main(["--out", str(tmp_path / "v"), "--cases", "BRATS_016"]) == 2       # facts all off in the repo config
    assert not (tmp_path / "v").exists()


def test_effect_refuses_eval_baseline(tmp_path):
    ev = load_checked_split(REPO)["eval"][0]
    d = tmp_path / "S-marks" / ev / "seed1"
    d.mkdir(parents=True)
    (d / "record.json").write_text(json.dumps({"complete": True}))
    (d / "bundle.json").write_text(json.dumps({"extra": {"colony": {"metrics": {}}}}))
    assert effect.main(["--run-dir", str(tmp_path)]) == 2


def test_pool_limit():
    with pytest.raises(ValueError):
        drive.run_pool([], 7, lambda t, r: None)
    with pytest.raises(ValueError):
        drive.run_pool([], 0, lambda t, r: None)


@pytest.mark.skipif(not HAVE_DATA, reason="dev case data not mounted")
def test_verdict_end_to_end_resumable_with_effect(tmp_path, capsys):
    args = ["--out", str(tmp_path), "--cases", "BRATS_016", "--seeds", "1", "--f2", "0.5", "--f3", "0.04,0.1", "--ticks", "30",
            "--workers", "2", "--resamples", "50"]
    assert verdict.main(args) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert {r["arm"] for r in res["runs"]} == set(verdict.DEFAULT_ARMS) and "ceiling_check" in res["meta"]
    assert (tmp_path / "REPORT.md").is_file()
    capsys.readouterr()
    assert verdict.main(args) == 0
    assert capsys.readouterr().out.count("skip ") == 9
    assert effect.main(["--run-dir", str(tmp_path), "--runs", "1", "--items", "3", "--workers", "1"]) == 0
    assert "effect" in json.loads((tmp_path / "results.json").read_text())
