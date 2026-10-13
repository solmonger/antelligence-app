"""scripts/v41_fit.py and antelligence/experiments/v41_fit.py on synthetic calibration logs (dev cases only)."""
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments import v41_analysis, v41_fit
from antelligence.experiments.v41_conformal import ConformalDeferral, ConformalThresholds
from antelligence.experiments.v41_router import ROUTER_MODELS, KNNRouter, TrainRow, best_model

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "config/v4_1/split.json"
_SPLIT = json.loads(SPLIT.read_text())
DEV = list(_SPLIT["dev_a"]) + list(_SPLIT["dev_b"])
EVAL = list(_SPLIT["eval"])
TARGETS = list("ABCDEFSX")


def _dist(rng, labels, hit, sharp):
    """A probability vector over ``labels`` whose argmax is ``hit`` with strength ``sharp`` (noisy)."""
    raw = {l: rng.random() * 0.3 for l in labels}
    raw[hit] += sharp
    z = sum(raw.values())
    return {l: v / z for l, v in raw.items()}


def _answer(rng, truth_t, truth_r, truth_a, p_right):
    t = truth_t if rng.random() < p_right else rng.choice(TARGETS)
    r = truth_r if rng.random() < p_right else 1 - truth_r
    a = truth_a if rng.random() < p_right else rng.randint(1, 5)
    return t, r, a


def _cf(rng, truth, p_right, sharp_lo, sharp_hi):
    t, r, a = _answer(rng, *truth, p_right)
    sharp = rng.uniform(sharp_lo, sharp_hi)
    probs = {"target": _dist(rng, TARGETS, t, sharp), "release": _dist(rng, ["no", "yes"], "yes" if r else "no", sharp),
             "amount": _dist(rng, [str(i) for i in range(5)], str(a - 1), sharp)}
    return {"target": t, "release": r, "amount": a, "probs": probs,
            "call": {"worker": "x", "server_ms": 10, "network_ms": 0, "queue_ms": 0}}


def make_run(root: Path, case, arm="SW-B", *, seed=1, rows=40, rng_seed=0, driver="qwen3.5-9b", split="dev",
             calibration=True, with_features=True, wall_s=10.0, mismatch=None):
    rng = random.Random(f"{case}|{arm}|{rng_seed}")
    run_id = f"{abs(hash((case, arm, rng_seed))) % 10**12:064d}"
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    out = []
    for n in range(rows):
        truth = (rng.choice(TARGETS), rng.randint(0, 1), rng.randint(1, 5))
        t, r, a = _answer(rng, *truth, 0.8)
        sp = {"target": rng.choice(TARGETS), "release": rng.choice(["yes", "no"]), "amount": str(rng.randint(1, 5))}
        row = {"kind": "decision", "run_id": run_id, "arm": f"{arm}:calibration", "case": mismatch or case, "seed": seed,
               "squad": n % 8, "round": n // 8, "tick": n, "backend": driver, "target": t, "release": r, "amount": a,
               "calls": [{"worker": driver, "server_ms": 100, "network_ms": 0, "queue_ms": 0, "tokens_in": 1, "tokens_out": 1,
                          "cost_usd": 0.0}],
               "counterfactual": {"d1": _cf(rng, truth, 0.7, 0.2, 1.5), "qwen3.5-4b": _cf(rng, truth, 0.6, 0.2, 1.0)},
               "sprime": sp, "error": False}
        if with_features:
            row["features"] = [rng.random() for _ in range(11)]
        out.append(row)
    (d / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    (d / "run_log.json").write_text(json.dumps({"run_id": run_id, "arm": f"{arm}:calibration", "case": case, "seed": seed,
                                                "split": split, "calibration": calibration, "wall_s": wall_s}))
    (d / "cell.json").write_text(json.dumps({"run_id": run_id, "arm": arm, "case": case, "seed": seed, "split": split,
                                             "wall_s": wall_s, "coverage": rng.random()}))
    return d


def make_set(root: Path, cases=None, rows=40, **kw):
    for c in cases or DEV:
        for arm in ("SW-B", "SW-noBoard"):
            make_run(root, c, arm, rows=rows, **kw)
    return root


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return make_set(tmp_path_factory.mktemp("calib"), rows=40)


def _fit(synthetic, **kw):
    runs = [v41_fit.load_run(d, json.loads(SPLIT.read_text())) for d in v41_fit.discover([synthetic])]
    samples, skipped = v41_fit.samples_from(runs, **kw)
    return runs, samples, skipped


def test_joint_defer_rate_is_about_20_percent(synthetic):
    _, samples, skipped = _fit(synthetic)
    assert len(samples) == 40 * 2 * 40 and not skipped
    th = v41_fit.fit_conformal(samples)
    assert abs(th.joint_dev_defer_rate - 0.20) <= 0.02 and th.joint_on_target
    # recomputing the joint rate by applying the frozen thresholds agrees with what was frozen
    dec = ConformalDeferral(th)
    rate = np.mean([dec.decide(s.probs).deferred for s in samples])
    assert rate == pytest.approx(th.joint_dev_defer_rate)
    rep = v41_fit.conformal_report(th, samples)
    assert set(rep["defer_rate_by_arm"]) == {"SW-B", "SW-noBoard"} and rep["sha256"] == th.sha256


def test_router_ties_go_to_the_cheaper_model():
    lab = {"target": "A", "release": "yes", "amount": "3"}
    same = {"target": "A", "release": True, "amount": 3}
    assert best_model({"d1": same, "qwen3.5-4b": same, "qwen3.5-9b": same}, lab) == "d1"
    worse = {"target": "B", "release": True, "amount": 3}
    assert best_model({"d1": worse, "qwen3.5-4b": same, "qwen3.5-9b": same}, lab) == "qwen3.5-4b"
    # all models agree everywhere -> every training label is d1 and the router picks d1
    rng = random.Random(3)
    rows = [TrainRow([rng.random() for _ in range(11)], "d1") for _ in range(30)]
    assert KNNRouter.fit(rows).predict([0.5] * 11) == "d1"
    # a 2-vs-2 vote tie among neighbours goes to the cheaper of the tied models
    tie = [TrainRow([float(i)] + [0.0] * 10, m) for i, m in enumerate(["qwen3.5-9b", "qwen3.5-4b"] * 2)]
    assert KNNRouter.fit(tie, k=4).predict([1.5] + [0.0] * 10) == "qwen3.5-4b"


def test_vectorised_predict_matches_router_predict():
    rng = random.Random(5)
    rows = [TrainRow([rng.random() for _ in range(11)], rng.choice(ROUTER_MODELS)) for _ in range(200)]
    router = KNNRouter.fit(rows)
    x = np.array([[rng.random() for _ in range(11)] for _ in range(50)])
    assert v41_fit.predict_many(router, x) == [router.predict(r) for r in x]


def test_fit_router_reports(synthetic):
    _, samples, _ = _fit(synthetic)
    router, rep = v41_fit.fit_router(samples)
    assert router.k == 15 and rep["models"] == list(ROUTER_MODELS)
    assert sum(rep["in_sample_pick_shares"].values()) == pytest.approx(1.0)
    cv = rep["cross_validated"]
    assert cv["folds"] == 40 and 0.0 <= cv["agreement"] <= 1.0 and sum(cv["pick_shares"].values()) == pytest.approx(1.0)
    assert set(rep["sprime_match_share_all_three"]) == set(ROUTER_MODELS)
    # trained on both arms
    assert rep["n_decisions"] == 40 * 2 * 40


def test_frozen_files_reload_to_identical_predictions(synthetic, tmp_path):
    _, samples, _ = _fit(synthetic)
    th = v41_fit.fit_conformal(samples)
    router, _ = v41_fit.fit_router(samples)
    cp, rp = tmp_path / "conformal.json", tmp_path / "router.json"
    assert th.save(cp) == th.sha256 and router.save(rp) == router.sha256
    th2, router2 = ConformalThresholds.load(cp, th.sha256), KNNRouter.load(rp, router.sha256)
    assert th2.sha256 == th.sha256 and router2.sha256 == router.sha256
    d1, d2 = ConformalDeferral(th), ConformalDeferral(th2)
    assert [d1.decide(s.probs) for s in samples] == [d2.decide(s.probs) for s in samples]
    feats = np.array([s.features for s in samples])
    assert v41_fit.predict_many(router, feats) == v41_fit.predict_many(router2, feats)
    doc = json.loads(cp.read_text())
    doc["body"]["per_question"]["target"]["alpha"] = 0.5
    cp.write_text(json.dumps(doc))
    with pytest.raises(Exception):
        ConformalThresholds.load(cp)


def test_refuses_evaluation_logs(tmp_path):
    split_doc = json.loads(SPLIT.read_text())
    # run labelled eval
    d = make_run(tmp_path / "a", DEV[0], split="eval")
    with pytest.raises(v41_fit.FitError, match="dev"):
        v41_fit.load_run(d, split_doc)
    # run claims dev but the case is an evaluation case in the committed split
    d = make_run(tmp_path / "b", EVAL[0])
    with pytest.raises(v41_fit.FitError, match="eval case"):
        v41_fit.load_run(d, split_doc)
    # a case missing from the split is refused too
    d = make_run(tmp_path / "c", "BRATS_999999")
    with pytest.raises(v41_fit.FitError, match="not in the split"):
        v41_fit.load_run(d, split_doc)
    # not a calibration run
    d = make_run(tmp_path / "d", DEV[0], calibration=False)
    with pytest.raises(v41_fit.FitError, match="calibration"):
        v41_fit.load_run(d, split_doc)
    # a row from another case hidden in a dev run
    d = make_run(tmp_path / "e", DEV[0], mismatch=EVAL[0])
    with pytest.raises(v41_fit.FitError, match="names case"):
        v41_fit.load_run(d, split_doc)
    # eval cells in cells.jsonl
    cells = tmp_path / "cells.jsonl"
    cells.write_text(json.dumps({"arm": "SW-B", "case": EVAL[0], "split": "eval", "coverage": 0.5}) + "\n")
    with pytest.raises(v41_fit.FitError, match="dev cells only"):
        v41_fit.load_cells(cells, split_doc)
    # and run_fit as a whole, with nothing frozen
    with pytest.raises(v41_fit.FitError):
        v41_fit.run_fit([tmp_path / "a"], tag="t", out_conformal=tmp_path / "c.json", split_file=SPLIT)
    assert not (tmp_path / "c.json").exists()


def test_bonsai_variant(tmp_path):
    make_set(tmp_path, DEV[:6], rows=30, driver="bonsai-8b")
    with pytest.raises(v41_fit.FitError, match="--bonsai"):
        v41_fit.run_fit([tmp_path], tag="t", split_file=SPLIT)
    rep = v41_fit.run_fit([tmp_path], tag="b", bonsai=True, split_file=SPLIT, out_conformal=tmp_path / "c.json",
                          out_router=tmp_path / "r.json")
    assert rep["variant"] == "bonsai" and rep["router"]["models"] == ["d1", "qwen3.5-4b", "bonsai-8b"]
    assert ConformalThresholds.load(tmp_path / "c.json").target_model == "bonsai-8b"
    assert KNNRouter.load(tmp_path / "r.json").models == ("d1", "qwen3.5-4b", "bonsai-8b")


def test_logs_without_features_still_freeze_conformal(tmp_path):
    make_set(tmp_path, DEV[:4], rows=30, with_features=False)
    rep = v41_fit.run_fit([tmp_path], tag="nf", split_file=SPLIT, out_conformal=tmp_path / "c.json",
                          out_router=tmp_path / "r.json")
    assert rep["status"] == "router_not_fitted" and "features" in rep["router"]["error"]
    assert (tmp_path / "c.json").exists() and not (tmp_path / "r.json").exists()


def test_power_and_throughput(tmp_path):
    rng = random.Random(2)
    cells = []
    diffs = []
    for i, c in enumerate(DEV):
        base = rng.random() * 0.5
        d = rng.gauss(0.02, 0.05)
        diffs.append(d)
        cells += [{"arm": "SW-noBoard", "case": c, "seed": 1, "split": "dev", "coverage": base},
                  {"arm": "SW-B", "case": c, "seed": 1, "split": "dev", "coverage": base + d}]
    rep = v41_fit.power_report(cells)
    sd = float(np.std(diffs, ddof=1))
    z = 1.959963984540054 + 0.8416212335729143
    assert rep["dev_paired_sd"] == pytest.approx(sd) and rep["dev_cases"] == 40 and rep["eval_cases"] == 80
    assert rep["mdd_p1"] == pytest.approx(z * sd / math.sqrt(80 * 3 / math.pi), rel=1e-6)
    assert rep["mdd_p1_points"] == pytest.approx(rep["mdd_p1"] * 100)
    make_set(tmp_path, DEV[:3], rows=24, wall_s=8.0)
    runs = [v41_fit.load_run(d, None) for d in v41_fit.discover([tmp_path])]
    t = v41_fit.throughput_report(runs, concurrency=4)
    assert t["runs"] == 6 and t["decisions"] == 144 and t["seconds_per_episode_mean"] == 8.0
    assert t["decisions_per_second_per_run"] == pytest.approx(144 / 48.0)
    assert t["decisions_per_second_at_concurrency"] == pytest.approx(144 / 48.0 * 4)
    assert t["server_ms_median_by_model"]["qwen3.5-9b"] == 100


def test_cli_end_to_end_and_no_silent_overwrite(tmp_path):
    spec = importlib.util.spec_from_file_location("v41_fit_cli", ROOT / "scripts/v41_fit.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    make_set(tmp_path / "in", DEV[:5], rows=30)
    cells = tmp_path / "cells.jsonl"
    cells.write_text("".join(json.dumps(json.loads(p.read_text())) + "\n" for p in sorted((tmp_path / "in").rglob("cell.json"))))
    args = [str(tmp_path / "in"), "--cells", str(cells), "--tag", "t", "--split-file", str(SPLIT),
            "--out-conformal", str(tmp_path / "conformal.json"), "--out-router", str(tmp_path / "router.json"),
            "--report", str(tmp_path / "fit_t.json")]
    assert cli.main(args) == 0
    rep = json.loads((tmp_path / "fit_t.json").read_text())
    assert rep["status"] == "ok" and rep["inputs"]["dev_only"] and rep["inputs"]["cells_file_lines"] == 10
    assert not any(c in json.dumps(rep) for c in DEV)                       # aggregates only: no case ids
    assert cli.main(args) == 1                                              # frozen files are not overwritten ...
    assert cli.main(args + ["--force"]) == 0                                # ... without --force
