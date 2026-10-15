"""A1: combined scan-and-look fit. Synthetic data only: no MRI, no case store."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from antelligence.worlds.tumor_mri import combined_fit as cf

REPO = Path(__file__).resolve().parents[3]


def _script():
    spec = importlib.util.spec_from_file_location("v43_combined_fit_under_test", REPO / "scripts" / "v43_combined_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(7)
    n = 200_000
    scan = rng.uniform(0.02, 0.98, n)
    look = rng.uniform(0.02, 0.98, n)
    true = np.array([-0.3, 0.8, 0.5, 0.1])
    p = cf.predict(true, scan, look)
    y = (rng.uniform(size=n) < p).astype(float)
    coef, info = cf.fit_logistic(cf.design(scan, look), y)
    assert info["converged"] and info["ridge"] is False
    assert np.allclose(coef, true, atol=0.05)   # n = 200k: standard errors are about 0.01 to 0.02


def test_fit_flags_ridge_when_singular():
    X = np.column_stack([np.ones(10), np.arange(10.0), np.arange(10.0), np.arange(10.0) * 0])   # duplicate + zero column
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=float)
    coef, info = cf.fit_logistic(X, y)
    assert info["ridge"] is True and np.all(np.isfinite(coef))


def test_ece_hand_checked():
    assert cf.ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == 0.0           # perfect
    assert cf.ece([1.0, 1.0, 0.0, 0.0], [0, 0, 1, 1]) == 1.0           # reversed
    # one bin (0.5 up to, not including, 0.6): mean prediction 0.525, observed rate 0.5 -> 0.025
    assert cf.ece([0.5, 0.55], [1, 0]) == pytest.approx(0.025)
    # two bins, weights 2/4 each: |0.15 - 0.0| and |0.85 - 1.0| -> 0.15
    assert cf.ece([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(0.15)
    assert cf.ece([], []) is None


def test_auroc_hand_checked():
    assert cf.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0          # perfect
    assert cf.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0          # reversed
    assert cf.auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5          # all ties
    # pos {0.4, 0.8}, neg {0.4, 0.2}: pairs 0.4>0.4 tie .5, 0.4>0.2 1, 0.8>0.4 1, 0.8>0.2 1 -> 3.5 / 4
    assert cf.auroc([0.4, 0.8, 0.4, 0.2], [1, 1, 0, 0]) == pytest.approx(0.875)
    assert cf.auroc([0.1, 0.9], [1, 1]) is None


def test_clipping_never_gives_inf_or_nan():
    x = np.array([0.0, 1.0, 0.5, 1e-12, 1 - 1e-12])
    z = cf.clip_logit(x)
    assert np.all(np.isfinite(z))
    assert z[0] == pytest.approx(np.log(0.001 / 0.999)) and z[1] == pytest.approx(-z[0])
    d = cf.design(x, x[::-1])
    assert d.shape == (5, 4) and np.all(np.isfinite(d)) and np.all(d[:, 0] == 1.0)
    pr = cf.predict([5.0, 5.0, 5.0, 5.0], x, x)
    assert np.all(np.isfinite(pr)) and np.all((pr >= 0) & (pr <= 1))
    huge = cf.predict([1e4, 1e4, 1e4, 1e4], np.array([0.0, 1.0]), np.array([0.0, 1.0]))   # exp overflow guard
    assert np.all(np.isfinite(huge))


def test_coefficients_round_trip_and_schema(tmp_path):
    doc = cf.coefficients_doc([0.1, 1.2, 0.7, -0.05], split="dev_a", n_looks=100, n_cases=30, ridge=False, inputs_sha256="ab" * 32)
    assert list(doc) == ["schema", "features", "coef", "clip", "fit"]
    assert doc["schema"] == "combined_v4_3" and doc["features"] == ["1", "logit_scan", "logit_look", "logit_scan*logit_look"]
    assert doc["clip"] == [0.001, 0.999]
    assert list(doc["fit"]) == ["split", "n_looks", "n_cases", "ridge", "inputs_sha256"]
    path = tmp_path / "sub" / "coefficients.json"
    cf.save_json(doc, path)
    assert cf.load_coefficients(path) == doc
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**doc, "schema": "other"}))
    with pytest.raises(ValueError):
        cf.load_coefficients(bad)


def test_fit_pipeline_on_synthetic_rows():
    mod = _script()
    rng = np.random.default_rng(3)
    rows = []
    for i in range(600):
        label = int(rng.uniform() < 0.5)
        band = int(rng.choice([3, 4])) if label else int(rng.choice([0, 1, 2]))
        scan = int(np.clip(rng.normal(0.8 if label else 0.2, 0.15), 0.001, 0.999) * 1e6)
        look = int(np.clip(rng.normal(0.7 if label else 0.3, 0.3), 0.001, 0.999) * 1e6)
        rows.append({"run": f"r{i % 7}", "case": f"C{i % 4}", "split": "dev_a" if i % 4 < 3 else "dev_b", "squad": i % 3,
                     "pos": (i % 5, i % 7, i % 11), "look_ppm": look, "scan_ppm": scan, "label": label, "band": band})
    coef_doc, cal = mod.run_fit({"rows": rows, "drops": {"dev_a": {}, "dev_b": {}}, "seen": {}, "runs": 7})
    assert coef_doc["fit"]["split"] == "dev_a" and coef_doc["fit"]["n_cases"] == 3
    assert coef_doc["fit"]["n_looks"] == sum(r["split"] == "dev_a" for r in rows)
    assert cal["splits"]["dev_b"]["in_sample"] is False
    for s in ("dev_a", "dev_b"):
        o = cal["splits"][s]["overall"]
        assert set(o) >= {"n", "combined", "scan", "look"} and 0.5 < o["combined"]["auroc"] <= 1.0
        assert set(cal["splits"][s]["by_band"]["per_band"]) == set(mod.BAND_NAMES)
    assert mod.inputs_sha256(rows) == mod.inputs_sha256(list(reversed(rows)))   # order-independent


def _fake_run_dir(root: Path) -> Path:
    """RUN_DIR/S-solo/<case>/seed<N>/bundle.json with looks, a scan-only reading, and one bad look."""
    for case, seed, looks in [
        ("CA", 0, [[3, 3, 3, 0.9], [1, 1, 1, 0.1], [3, 3, 3, 0.8]]),   # repeats voxel (3,3,3)
        ("CB", 1, [[2, 2, 2, 0.7], [0, 0, 0, 0.2]]),
    ]:
        d = root / "S-solo" / case / f"seed{seed}"
        d.mkdir(parents=True)
        log = [{"tick": t, "pos_vox": [x, y, z], "kind": "look", "value": v, "band": 2, "word": None} for t, (x, y, z, v) in enumerate(looks)]
        log.append({"tick": 99, "pos_vox": [0, 0, 0], "kind": "scan", "value": 0.5, "band": 2, "word": None})
        if case == "CA":
            log.append({"tick": 100, "pos_vox": [9, 9, 9], "kind": "look", "value": 0.5, "band": 2, "word": None})   # outside crop
        (d / "bundle.json").write_text(json.dumps({"extra": {"colony": {"reading_logs": {"0": log, "1": []}}}}))
    return root


def test_circularity_mode_on_fake_bundles(tmp_path):
    mod = _script()
    run_dir = _fake_run_dir(tmp_path / "run")
    shape = (4, 4, 4)
    p_map = np.full(shape, 0.5)
    truth = np.zeros(shape, dtype=bool)
    truth[2:4, 2:4, 2:4] = True
    p_map[truth] = 0.85
    p_map[~truth] = 0.15
    loaded = []

    def loader(case):
        loaded.append(case)
        return p_map, truth

    coef_doc = cf.coefficients_doc([0.0, 0.6, 0.6, 0.0], split="dev_a", n_looks=10, n_cases=2, ridge=False, inputs_sha256="0" * 64)
    res = mod.run_circularity(run_dir, coef_doc, loader, {"dev_a": {"CA"}, "dev_b": {"CB"}})
    assert res["overall"]["n"] == 5 and res["overall"]["n_unique_voxels"] == 4
    assert res["dev_a"]["n"] == 3 and res["dev_a"]["n_unique_voxels"] == 2 and res["dev_b"]["n"] == 2
    assert res["dropped"] == {"pos_outside_crop": 1}
    assert sorted(loaded) == ["CA", "CB"]                                  # each case loaded once
    o = res["overall"]
    assert o["scan"]["auroc"] == 1.0 and o["look"]["auroc"] == 1.0 and o["combined"]["auroc"] == 1.0
    assert o["combined"]["ece"] is not None
    # a case outside the dev split is refused before it is loaded
    with pytest.raises(SystemExit):
        mod.run_circularity(run_dir, coef_doc, loader, {"dev_a": {"CA"}, "dev_b": set()})


def test_circularity_cli_writes_json(tmp_path, monkeypatch):
    mod = _script()
    run_dir = _fake_run_dir(tmp_path / "run")
    truth = np.zeros((4, 4, 4), dtype=bool)
    truth[2:4, 2:4, 2:4] = True
    p_map = np.where(truth, 0.85, 0.15)
    coef_path = tmp_path / "coefficients.json"
    cf.save_json(cf.coefficients_doc([0.0, 0.6, 0.6, 0.0], split="dev_a", n_looks=1, n_cases=1, ridge=False, inputs_sha256="0" * 64), coef_path)
    monkeypatch.setattr(mod, "default_case_loader", lambda root: (lambda case: (p_map, truth)))
    monkeypatch.setattr(mod, "split_sets", lambda *a, **k: {"dev_a": {"CA"}, "dev_b": {"CB"}})
    out = tmp_path / "out"
    assert mod.main(["--circularity", str(run_dir), "--cases-root", str(tmp_path), "--out", str(out), "--coefficients", str(coef_path)]) == 0
    doc = json.loads((out / "circularity.json").read_text())
    assert doc["overall"]["n"] == 5 and set(doc) >= {"overall", "dev_a", "dev_b"}


def test_out_dir_guard_refuses_frozen_folders():
    mod = _script()
    for bad in ("config/v4_2", "docs/research/v4.2-20261014/x", "config/v4_1"):
        with pytest.raises(SystemExit):
            mod._refuse_forbidden_out(REPO / bad)
    mod._refuse_forbidden_out(REPO / "docs/research/v4.3-20261015/combined")
