"""L1 store -> tumor_mri world adapter (L2c): round trip, truth channel order, outside-crop count, split gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from antelligence.experiments import v41_split as sp
from antelligence.experiments.v41_gate import GateError
from antelligence.perception import crop as l1_crop
from antelligence.perception import pipeline, regions, store
from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, WorldConfig, run_episode
from antelligence.worlds.tumor_mri.store_adapter import AdapterError, load_case

N = 24
TX = "0x" + "ab" * 32


def _arrays(seed: int = 3):
    g = np.indices((N, N, N)).astype(float)
    c = (N - 1) / 2
    brain = ((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= 11.5 ** 2
    wt = ((g[0] - 14) ** 2 + (g[1] - 11) ** 2 + (g[2] - 12) ** 2) <= 4.5 ** 2
    tc = ((g[0] - 14) ** 2 + (g[1] - 11) ** 2 + (g[2] - 12) ** 2) <= 3.0 ** 2
    et = ((g[0] - 14) ** 2 + (g[1] - 11) ** 2 + (g[2] - 12) ** 2) <= 1.5 ** 2
    rng = np.random.default_rng(seed)
    p = np.clip(wt * 0.8 + rng.normal(0, 0.02, wt.shape), 0, 1).astype(np.float32) * brain
    return dict(brain_mask=brain, p_tumour=p, unc=(4 * p * (1 - p)).astype(np.float32),
                flair=rng.random(wt.shape).astype(np.float32), t1c=rng.random(wt.shape).astype(np.float32),
                truth=np.stack([wt, tc, et]),  # L1 order: WT, TC, ET
                p_tc=p * 0.5, p_et=p * 0.2, origin_vox=np.asarray([3, 4, 5]), spacing_mm=np.asarray([2.0] * 3))


def _write_case(root: Path, case: str, outside: int | None = 17):
    a = _arrays()
    store.save_case(case, root, **a)
    payloads, _ = regions.extract_regions(a["p_tumour"], 2.0, unc=a["unc"], p_tc=a["p_tc"], p_et=a["p_et"])
    m = {"case": case, "split": "dev_a", "crop_origin_vox": [3, 4, 5], "crop_shape_vox": [N, N, N], "regions": payloads,
         "truth_metrics_computed": True}
    if outside is not None:
        m["truth_wt_outside_crop_voxels"] = outside
    (root / case / "metrics.json").write_text(json.dumps(m))
    return a, payloads


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture()
def repo(tmp_path):
    """A study repo with a committed split, preregistration and anchor receipt (the lock is satisfiable)."""
    doc = sp.compute_split([f"C{i:03d}" for i in range(150)])
    r = tmp_path / "repo"
    (r / "config" / "v4_1").mkdir(parents=True)
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    split = r / "config" / "v4_1" / "split.json"
    split.write_text(sp.dump(doc))
    prereg = r / "PREREGISTRATION.md"
    prereg.write_text(f"locked\n| split file sha256 | `{_sha(split)}` |\n")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(tx_hash=TX, status=1, readback_ok=True, chain_id=300, prereg_sha256=_sha(prereg))))
    g1 = tmp_path / "G1_APPROVED"
    g1.write_text("approved")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "lock")
    return dict(doc=doc, repo=r, lock=dict(prereg_path=prereg, anchor_receipt=receipt, g1_file=g1), root=tmp_path / "cases")


def test_store_adapter_roundtrip(repo):
    case = repo["doc"]["dev_a"][0]
    a, payloads = _write_case(repo["root"], case, outside=17)
    lc = load_case(case, root=repo["root"], repo_root=repo["repo"])
    assert lc.split == "dev"
    c = lc.crop
    assert c.case_id == case and c.voxel_mm == 2.0 and c.shape == (N, N, N)
    assert np.array_equal(c.brain, a["brain_mask"]) and np.array_equal(c.truth, a["truth"][0])  # WT is channel 0
    assert not np.array_equal(c.truth, a["truth"][1])  # ... not TC
    assert np.allclose(c.p_tumour, a["p_tumour"]) and np.allclose(c.unc, a["unc"])
    assert np.array_equal(c.flair, a["flair"]) and np.array_equal(c.t1c, a["t1c"])
    assert c.outside_crop_truth_voxels == 17
    # regions come from the knowledge payloads, in the crop grid: centroid_dmm / 20
    assert len(lc.regions) == len(payloads) >= 1
    for r, p in zip(lc.regions, payloads):
        assert r.region_id == p["region_id"] and r.volume_mm3 == p["volume_mm3"]
        assert r.centroid_vox == pytest.approx(tuple(x / 20.0 for x in p["centroid_dmm"]))
        assert all(0 <= x < N for x in r.centroid_vox)
    # the world accepts it, scores against truth in the crop plus the outside count, and runs
    cfg = WorldConfig(total_drug=300.0, dose_threshold=4.0, entry_min_dist_mm=6.0)
    w = TumorMRIWorld(c, cfg, seed=1, regions=lc.regions)
    assert w.describe()["outside_crop_truth_voxels"] == 17
    assert w.scorer_truth().denominator == int(a["truth"][0].sum()) + 17
    assert 0.0 <= run_episode(w, lambda _: ScriptS())["coverage"] <= 1.0


def test_adapter_computes_outside_voxels_from_raw_when_metrics_lack_it(repo):
    case = repo["doc"]["dev_b"][0]
    _write_case(repo["root"], case, outside=None)
    shp = (40, 40, 40)
    truth = np.zeros((3, *shp), bool)
    truth[0, 5:20, 10:20, 10:20] = True  # crop box is [3:27, 4:28, 5:29]: x 5..19 inside, nothing outside in x ...
    truth[0, 0:3, 10:12, 10:12] = True  # ... plus 3*2*2 = 12 voxels below the crop origin in x
    raw = repo["root"].parent / "raw"
    raw.mkdir()
    np.savez_compressed(raw / f"{case}.npz", truth=truth)
    lc = load_case(case, root=repo["root"], repo_root=repo["repo"])
    assert lc.crop.outside_crop_truth_voxels == 12
    assert l1_crop.truth_voxels_outside(truth[0], (slice(3, 27), slice(4, 28), slice(5, 29))) == 12
    other = repo["doc"]["dev_a"][1]
    _write_case(repo["root"], other, outside=None)
    with pytest.raises(AdapterError, match="raw bundle"):  # neither the metrics key nor a raw bundle to compute it from
        load_case(other, root=repo["root"], repo_root=repo["repo"], raw_dir=raw / "nope")


def test_backfill_outside_voxels_roundtrip(tmp_path):
    shp = (20, 20, 20)
    truth = np.zeros((3, *shp), bool)
    truth[0, 2:8, 6:10, 6:10] = True
    raw = tmp_path / "raw"
    raw.mkdir()
    np.savez_compressed(raw / "C.npz", truth=truth)
    out = tmp_path / "out"
    (out / "C").mkdir(parents=True)
    meta = {"crop_origin_vox": [5, 5, 5], "crop_shape_vox": [10, 10, 10], "truth_metrics_computed": True}
    (out / "C" / "metrics.json").write_text(json.dumps(meta))
    n = pipeline.backfill_outside_voxels("C", raw, out)
    assert n == int(truth[0].sum() - truth[0][5:15, 5:15, 5:15].sum()) > 0
    assert json.loads((out / "C" / "metrics.json").read_text())["truth_wt_outside_crop_voxels"] == n
    meta["truth_metrics_computed"] = False  # eval metrics are withheld: never backfilled
    (out / "C" / "metrics.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="withheld"):
        pipeline.backfill_outside_voxels("C", raw, out)


def test_adapter_refuses_eval_without_gate(repo):
    ev, rs = repo["doc"]["eval"][0], repo["doc"]["reserve"][0]
    for c in (ev, rs):
        _write_case(repo["root"], c)
    kw = dict(root=repo["root"], repo_root=repo["repo"])
    with pytest.raises(GateError, match="v41_gate"):
        load_case(ev, **kw)  # no lock paths at all
    for missing in repo["lock"]:  # any one of the three missing
        part = {k: v for k, v in repo["lock"].items() if k != missing}
        with pytest.raises(GateError):
            load_case(ev, **kw, **part)
    with pytest.raises(GateError, match="reserve|v41_gate"):
        load_case(rs, **kw)
    with pytest.raises(GateError, match="reserve"):
        load_case(rs, **kw, **repo["lock"])  # a satisfied lock does not open the reserve set
    with pytest.raises(GateError, match="not in the split"):
        load_case("UNKNOWN", **kw)
    # a broken lock is refused by the gate itself
    bad = dict(repo["lock"], g1_file=repo["lock"]["g1_file"].with_name("missing"))
    with pytest.raises(GateError, match="G1"):
        load_case(ev, **kw, **bad)
    # with the lock satisfied the eval case loads, flagged as eval
    assert load_case(ev, **kw, **repo["lock"]).split == "eval"
