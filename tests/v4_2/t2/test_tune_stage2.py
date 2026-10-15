"""Stage-2 tuning (k re-tune): grid, limits, resume keys, no config write, sharing arms refused."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from antelligence.experiments import v42_drive as drive

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("v42_tune", REPO / "scripts" / "v42_tune.py")
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)
CFG = REPO / "config" / "v4_2" / "world.yaml"
HAVE_DATA = drive.DEFAULT_CASES_ROOT.is_dir() and (drive.DEFAULT_CASES_ROOT / "BRATS_016" / "case.npz").is_file()
K = 0.3559008534420454


def test_baseline_k_read_from_config():
    assert tune.baseline_k(CFG) == K


def test_stage2_grid_pairs_and_k():
    g = tune.stage2_grid([1.5, 2.0, 2.5, 3.0], [0, 4, 8], K)
    assert len(g) == 12
    assert [p["facts_index"] for p in g] == [0] * 4 + [4] * 4 + [8] * 4
    assert g[0]["k"] == pytest.approx(K * 1.5) and g[3]["k"] == pytest.approx(K * 3.0)
    base = tune.grid()
    assert all(p["u_core"] == base[p["facts_index"]]["u_core"] for p in g)


def test_stage2_hard_fails_above_12_and_bad_index():
    with pytest.raises(ValueError):
        tune.stage2_grid([1.5, 2.0, 2.5, 3.0, 3.5], [0, 4, 8], K)
    with pytest.raises(ValueError):
        tune.stage2_grid([1.5], [12], K)


def test_main_refuses_over_12(tmp_path):
    assert tune.main(["--out", str(tmp_path / "t"), "--stage", "2", "--k-mults", "1,2,3,4,5", "--facts-points", "0,4,8"]) == 2
    assert tune.main(["--out", str(tmp_path / "t"), "--stage", "2"]) == 2
    assert tune.main(["--out", str(tmp_path / "t"), "--k-mults", "2"]) == 2


def test_tasks_carry_k_and_stage():
    pts = tune.stage2_grid([2.0], [0], K)
    t = tune.make_tasks(pts, ["BRATS_016"], list(tune.TUNE_ARMS), config=CFG, cases_root=None, ticks=None, done=set(), stage=2)
    assert {x["arm"] for x in t} == {"S-solo", "S'"} and all(x["stage"] == 2 and x["drug_scale_k"] == pytest.approx(2 * K) for x in t)


def test_sharing_arm_still_refused(tmp_path):
    assert tune.main(["--out", str(tmp_path / "t"), "--stage", "2", "--k-mults", "2", "--facts-points", "0", "--arms", "S-solo,S-marks"]) == 2


def _rec(**kw):
    r = {"stage": 1, "point": 0, "k_mult": 1.0, "ticks": None, "config_hash": "h", "arm": "S-solo", "case_id": "c"}
    return {**r, **kw}


def test_resume_keys_distinguish_stage_k_ticks_hash(tmp_path):
    p = tmp_path / "e.jsonl"
    recs = [_rec(), _rec(stage=2), _rec(k_mult=2.0), _rec(ticks=5), _rec(config_hash="g")]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert len(tune.load_done(p)) == 5


def test_stage2_ignores_stage1_episodes(tmp_path):
    # stage-1 records (no k_mult/stage in old files, or stage 1) never satisfy a stage-2 key
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"point": 0, "arm": "S-solo", "case_id": "c"}) + "\n" + json.dumps(_rec()) + "\n")
    done = tune.load_done(p)
    assert (2, 0, 1.5, None, "h", "S-solo", "c") not in done and (1, 0, 1.0, None, "h", "S-solo", "c") in done


def test_selection_same_rule_on_stage2_rows():
    pts = tune.stage2_grid([1.5, 2.0], [0], K)
    res = {}
    for i, (a, b) in enumerate([(0.3, 0.7), (0.3, 0.8)]):
        for arm, v in (("S-solo", a), ("S'", b)):
            res[(i, arm, "c")] = {"coverage": v, "timing_s": {"wall": 1.0}}
    rows = tune.summarise(pts, ["c"], res)
    assert tune.select(rows)["selected"] == 1                  # tie on share; larger median gap wins


@pytest.mark.skipif(not HAVE_DATA, reason="dev case data not mounted")
def test_dry_run_does_not_write_config_and_reuses_nothing_from_stage1(tmp_path):
    before = hashlib.sha256(CFG.read_bytes()).hexdigest()
    base = ["--out", str(tmp_path / "o"), "--cases", "BRATS_016", "--ticks", "10", "--workers", "1"]
    assert tune.main(base) == 0                                # stage 1 episodes (grid of 12)
    assert tune.main(base + ["--stage", "2", "--k-mults", "2.0", "--facts-points", "0"]) == 0
    doc = json.loads((tmp_path / "o" / "tune_results.json").read_text())
    assert doc["stage"] == 2 and doc["points"][0]["k"] == pytest.approx(2 * K) and doc["points"][0]["n_cases"] == 1
    n = sum(1 for _ in (tmp_path / "o" / "episodes.jsonl").open())
    assert n == 24 + 2
    assert hashlib.sha256(CFG.read_bytes()).hexdigest() == before
