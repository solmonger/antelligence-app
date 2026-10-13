"""Unit tests for the v4.1 perception pipeline. Synthetic phantoms only; no network, no real MRI."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from antelligence.perception import calibrate, crop, infer, model, msd, regions, store

MSD_DATASET = {
    "name": "BRATS",
    "modality": {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"},
    "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"},
    "training": [{"image": "./imagesTr/BRATS_001.nii.gz", "label": "./labelsTr/BRATS_001.nii.gz"}],
}


def tiny_model():
    torch.manual_seed(0)
    net = model.build_segresnet({"blocks_down": [1, 1], "blocks_up": [1], "init_filters": 8,
                                 "in_channels": 4, "out_channels": 3, "dropout_prob": 0.0})
    return net.eval()


def test_channel_reorder_from_dataset_json(tmp_path):
    (tmp_path / "dataset.json").write_text(json.dumps(MSD_DATASET))
    ds = msd.read_dataset_json(tmp_path)
    assert msd.channel_names(ds) == ["FLAIR", "T1", "T1c", "T2"]
    assert msd.assert_msd_channel_order(ds) == [2, 1, 3, 0]
    # a different layout is derived, not assumed
    other = {**MSD_DATASET, "modality": {"0": "T2w", "1": "FLAIR", "2": "T1w", "3": "t1gd"}}
    assert msd.bundle_reorder_indices(msd.channel_names(other)) == [3, 2, 0, 1]
    with pytest.raises(msd.MSDError):
        msd.assert_msd_channel_order(other)
    # label definition comes from the json too
    ld = msd.label_definition(ds)
    assert ld.wt == (1, 2, 3) and ld.tc == (2, 3) and ld.et == (3,)
    t = msd.truth_channels(np.array([0, 1, 2, 3]), ld)
    assert t.tolist() == [[False, True, True, True], [False, False, True, True], [False, False, False, True]]


def test_weights_only_load_refuses_model_ts(tmp_path):
    net = tiny_model()
    ts = tmp_path / "model.ts"
    ts.write_bytes(b"not read")
    with pytest.raises(model.UnsafeWeightsError):
        model.load_weights(net, ts)
    # model.pt is a plain state dict loaded with weights_only=True
    pt = tmp_path / "model.pt"
    torch.save(net.state_dict(), pt)
    loaded = model.load_weights(tiny_model(), pt)
    for a, b in zip(net.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    # a pickled arbitrary object is rejected by weights_only
    bad = tmp_path / "bad.pt"
    torch.save({"model": object()}, bad)
    with pytest.raises(Exception):
        model.load_weights(tiny_model(), bad)


def test_tta_mean_std_shapes():
    net = tiny_model()
    x = np.random.RandomState(0).randn(4, 16, 16, 16).astype(np.float32)
    assert len(infer.ALL_FLIPS) == 8
    mean, std = infer.predict_tta(net, x, "cpu", roi_size=(16, 16, 16), overlap=0.25)
    assert mean.shape == std.shape == (3, 16, 16, 16)
    assert (mean >= 0).all() and (mean <= 1).all() and (std >= 0).all()
    assert std.max() > 0  # flips of a random net genuinely disagree
    # single identity pass has zero spread and equals the plain forward sigmoid
    m1, s1 = infer.predict_tta(net, x, "cpu", roi_size=(16, 16, 16), overlap=0.25, flips=((),))
    with torch.no_grad():
        ref = torch.sigmoid(net(torch.from_numpy(x)[None]))[0].numpy()
    assert np.allclose(m1, ref, atol=1e-5) and np.allclose(s1, 0, atol=1e-6)


def test_ece_brier_known_values():
    assert calibrate.ece(np.full(4, 0.5), np.array([1, 1, 0, 0])) == pytest.approx(0.0)
    assert calibrate.ece(np.full(10, 0.9), np.array([1] * 5 + [0] * 5)) == pytest.approx(0.4)
    assert calibrate.ece(np.array([1.0, 1.0]), np.array([1, 1])) == pytest.approx(0.0)  # p=1 in last bin
    assert calibrate.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert calibrate.brier(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)
    assert calibrate.dice(np.array([1, 1, 0], bool), np.array([1, 0, 0], bool)) == pytest.approx(2 / 3)
    # temperature fit shrinks an over-confident predictor
    rng = np.random.RandomState(1)
    true_p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(float)
    over = calibrate.apply_temperature(true_p, 0.25)
    c, rep = calibrate.fit_calibrator(over, y)
    assert c.method in ("temperature", "isotonic") and c.temperature > 1.0
    assert rep[c.method]["ece"] < rep["none"]["ece"]


def test_crop_margin_and_clip():
    p = np.zeros((60, 60, 60), np.float32)
    p[30:34, 30:34, 30:34] = 0.9
    brain = np.zeros(p.shape, bool)
    brain[5:55, 5:55, 28:50] = True
    box = crop.compute_crop(p, brain, spacing_mm=2.0)  # margin 30 mm = 15 voxels
    assert (box[0].start, box[0].stop) == (15, 49)  # 30-15 .. 33+15+1
    assert (box[1].start, box[1].stop) == (15, 49)
    assert (box[2].start, box[2].stop) == (28, 49)  # clipped to brain z-range [28, 50) by lo; hi 49 < 50
    assert crop.compute_crop(np.zeros_like(p), brain, 2.0) is None
    # resample 1 mm -> 2 mm halves the grid
    assert crop.resample(np.ones((10, 10, 9), np.float32), (1, 1, 1), 2.0).shape == (5, 5, 4)


def test_truth_share_outside_crop():
    truth = np.zeros((20, 20, 20), bool)
    truth[2:6, 2:6, 2:6] = True  # 64 voxels
    box = (slice(0, 20), slice(0, 20), slice(0, 4))  # keeps z 2,3 -> half
    assert crop.truth_share_outside(truth, box) == pytest.approx(0.5)
    assert crop.truth_share_outside(truth, (slice(0, 20),) * 3) == 0.0
    assert crop.truth_share_outside(truth, None) == 1.0
    assert crop.truth_share_outside(np.zeros_like(truth), box) == 0.0


def _blob(shape, lo, hi, v=0.8):
    p = np.zeros(shape, np.float32)
    p[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = v
    return p


def test_regions_min_volume_and_split():
    sp = 2.0  # 8 mm3 voxels
    p = np.zeros((60, 60, 60), np.float32)
    p[2:4, 2:4, 2:5] = 0.9      # 12 vox = 96 mm3 -> dropped (<100)
    p[10:12, 10:12, 10:13] = 0.9  # 12 vox... make it 13 voxels worth: add one
    p[12, 10, 10] = 0.9         # 13 vox = 104 mm3 -> kept
    p[20:35, 20:35, 20:35] = 0.9  # 3375 vox = 27000 mm3 -> split into ~5 supervoxels
    p[40, 40, 40] = 0.9; p[41, 41, 41] = 0.9  # diagonal only: 6-connectivity keeps them apart, each tiny
    pay, lab = regions.extract_regions(p, sp)
    vols = [r["volume_mm3"] for r in pay]
    assert 104 in vols and 96 not in vols
    big = [v for v in vols if v > 1000]
    assert len(big) == 5 and abs(sum(big) - 27000) == 0 and all(3000 < v < 8000 for v in big)
    assert len(pay) == 6  # 1 small kept + 5 supervoxels; diagonal specks dropped
    assert lab.max() == len(pay)
    again, _ = regions.extract_regions(p, sp)
    assert again == pay  # deterministic


def test_region_payload_integer_units():
    p = _blob((40, 40, 40), (10, 10, 10), (20, 20, 20), 0.873)
    unc = _blob((40, 40, 40), (10, 10, 10), (20, 20, 20), 0.05)
    et = _blob((40, 40, 40), (10, 10, 10), (20, 20, 20), 0.9)
    pay, _ = regions.extract_regions(p, 2.0, unc=unc, p_tc=et, p_et=et)
    assert len(pay) == 1
    r = pay[0]
    assert tuple(r) == regions.PAYLOAD_KEYS
    assert r["p_tumour_ppm"] == 873000 and r["unc_ppm"] == 50000
    assert r["volume_mm3"] == 1000 * 8 and r["cls"] == "ET" and r["source"] == "perception"
    assert r["centroid_dmm"] == [int(round(14.5 * 20))] * 3
    for key in ("region_id", "p_tumour_ppm", "volume_mm3", "unc_ppm"):
        assert type(r[key]) is int
    assert all(type(v) is int for v in r["centroid_dmm"])
    from antelligence.kernel.canonical import canonical_json
    canonical_json(r)  # no floats anywhere: encoder would raise


def test_local_patch_shape(tmp_path):
    rs = np.random.RandomState(0)
    shp = (50, 40, 30)
    arrs = dict(brain_mask=np.ones(shp, bool), p_tumour=np.zeros(shp, np.float32), unc=np.zeros(shp, np.float32),
                flair=rs.randn(*shp).astype(np.float32), t1c=rs.randn(*shp).astype(np.float32),
                truth=np.zeros((3, *shp), bool))
    store.save_case("C1", tmp_path, **arrs)
    patch = store.local_patch("C1", (25, 20, 10), tmp_path)
    assert patch.shape == (64, 64, 2) and patch.dtype == np.float32
    assert np.array_equal(patch[32, 32, 0], arrs["flair"][25, 20, 10])
    assert np.array_equal(patch[32, 32, 1], arrs["t1c"][25, 20, 10])
    edge = store.local_patch("C1", (0, 0, 0), tmp_path)  # zero padded off-crop
    assert edge.shape == (64, 64, 2) and edge[0, 0].tolist() == [0.0, 0.0]
    with pytest.raises(IndexError):
        store.local_patch("C1", (0, 0, 99), tmp_path)


def test_missing_flair_zeroes_channel():
    rs = np.random.RandomState(3)
    img = rs.rand(4, 8, 8, 8).astype(np.float32) + 0.5  # MSD order, FLAIR = channel 0
    names = ["FLAIR", "T1", "T1c", "T2"]
    z = infer.zero_channel(img, names)
    assert (z[0] == 0).all() and np.array_equal(z[1:], img[1:]) and (img[0] != 0).all()  # copy, not in place
    x = infer.prepare_input(img, [2, 1, 3, 0], names)  # bundle order: FLAIR last
    assert (x[3] == 0).all() and x[:3].std() > 0.5
    full = infer.prepare_input(img, [2, 1, 3, 0])
    assert abs(float(full[3].mean())) < 1e-5 and abs(float(full[3].std()) - 1) < 1e-4
