"""v4.31 A.2: the fine scoring lattice (fine_grid.py, scoring.py) and the geometric off-target measures (geometry.py). Synthetic data only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import ndimage

from antelligence.worlds.tumor_mri import ScriptS, TumorMRIWorld, WorldConfig, run_episode
from antelligence.worlds.tumor_mri import fine_grid as FG
from antelligence.worlds.tumor_mri import geometry as GEO
from antelligence.worlds.tumor_mri import physical as P
from antelligence.worlds.tumor_mri.scoring import release_log, score_dose
from antelligence.worlds.tumor_mri.store_adapter import DevCaseInTestError
from tests.v4_1.world.phantom import make_phantom

REPO = Path(__file__).resolve().parents[2]
SPLIT = json.loads((REPO / "config/v4_1/split.json").read_text())
DEV = SPLIT["dev_a"][0]


def _ref_cfg(ticks: int = 60, thr: float = 23.5) -> WorldConfig:
    doc = yaml.safe_load((REPO / "config/v4_31/world_units_k_ref.yaml").read_text())["world"]
    doc["physics"] = {**doc["physics"], "policy_threshold_umol_h_L": thr}
    return WorldConfig.from_dict(doc).with_(ticks=ticks)


def _aligned_fine_case(crop, origin=(0, 0, 0)) -> FG.FineCase:
    """A 1 mm FineCase that sits exactly on a 2 mm phantom crop (even sizes, scale 2.0 on every axis), truth upsampled from the phantom's."""
    n = crop.shape
    full = tuple(2 * (o + s) for o, s in zip(origin, n))
    up = lambda a: np.repeat(np.repeat(np.repeat(a, 2, axis=0), 2, axis=1), 2, axis=2)                  # noqa: E731
    wt = up(crop.truth)
    arrays = {"truth": np.stack([wt, wt, wt]), "brain_mask": up(crop.brain), "flair": np.ones(wt.shape, np.float32), "t1": np.ones(wt.shape, np.float32),
              "box": np.asarray([[2 * o, 2 * (o + s)] for o, s in zip(origin, n)]), "origin_2mm": np.asarray(origin), "full_shape": np.asarray(full)}
    return FG.FineCase("phantom", arrays, 0)


def test_the_world_dose_equals_the_block_mean_of_the_fine_score_on_a_real_episode():
    """End to end: play an episode on the 2 mm phantom, score its release log on the 1 mm corner lattice, average 2x2x2 blocks: it is the world's own 2 mm dose."""
    crop = make_phantom(36)
    cfg = _ref_cfg(ticks=60)
    w = TumorMRIWorld(crop, cfg, seed=3)
    run_episode(w, lambda a: ScriptS())
    fine = _aligned_fine_case(crop)
    dose1 = score_dose(release_log(w.field), cfg.physical_params(), fine, 1.0)
    block = dose1.reshape(36, 2, 36, 2, 36, 2).mean(axis=(1, 3, 5))
    big = w.field.dose > 1.0
    assert big.sum() > 200
    assert np.abs(block[big] - w.field.dose[big]).max() / w.field.dose[big].max() < 2e-3
    assert block.sum() == pytest.approx(w.field.dose.sum(), rel=2e-3)


@pytest.mark.parametrize("sp", [1.0, 0.5])
def test_scored_exposure_conserves_mass(sp):
    crop = make_phantom(36)
    fine = _aligned_fine_case(crop)
    p = P.PhysicalParams.carmustine_wafer("ref")
    rel = [((18, 18, 18), 4.0, 0), ((10, 20, 15), 2.5, 20)]
    dose = score_dose(rel, p, fine, sp) if sp == 1.0 else score_dose(rel, p, fine, sp)
    total = float(dose.sum()) * (sp ** 3) * p.k_clear_s / p.exposure_to_umol_h_L
    want = sum(m * (-math.expm1(-p.k_clear_s * (p.ticks - t0) * p.tick_seconds)) for _, m, t0 in rel)
    assert total == pytest.approx(want, rel=3e-3)


def test_axis_scales_and_release_lattice_follow_the_pipelines_resample():
    assert FG.axis_scales((240, 240, 155)) == pytest.approx((2.0, 2.0, 155 / 78))
    fine = _aligned_fine_case(make_phantom(36), origin=(3, 2, 1))
    lat = fine.release_lattice([(0, 0, 0), (5, 6, 7)], 1.0)
    assert lat.tolist() == [[1, 1, 1], [11, 13, 15]]                               # a release at a 2 mm voxel centre is the corner at (2i + 1) mm
    assert fine.release_lattice([(5, 6, 7)], 0.5).tolist() == [[22, 26, 30]]


def test_fine_spacings_and_the_outside_count_scale_with_the_lattice():
    fine = _aligned_fine_case(make_phantom(36))
    fine.outside = 7
    t1, b1, o1 = fine.at_spacing(1.0)
    t5, b5, o5 = fine.at_spacing(0.5)
    t25, _, o25 = fine.at_spacing(0.25)
    assert t5.shape[1:] == tuple(2 * s for s in t1.shape[1:]) and t25.shape[1] == 4 * t1.shape[1]
    assert (o1, o5, o25) == (7, 56, 448) and int(t5[0].sum()) == 8 * int(t1[0].sum())
    with pytest.raises(ValueError):
        fine.at_spacing(2.0)


def test_make_fine_arrays_is_native_and_aligned_with_a_two_mm_store():
    from antelligence.perception import crop as crp
    shape = (60, 56, 31)
    g = np.indices(shape).astype(float)
    label = np.zeros(shape, np.int16)
    r2 = (g[0] - 30) ** 2 + (g[1] - 28) ** 2 + (g[2] - 15) ** 2
    label[r2 <= 81] = 1
    label[r2 <= 25] = 3
    image = np.stack([np.ones(shape, np.float32) * (i + 1) for i in range(4)])
    truth_fn = lambda lab: np.stack([lab > 0, lab >= 2, lab == 3])                                       # noqa: E731
    t2 = crp.resample((label > 0).astype(np.float32), (1.0, 1.0, 1.0), 2.0, 0) >= 0.5
    origin, shape2 = (4, 4, 1), (20, 18, 12)
    sl = tuple(slice(o, o + n) for o, n in zip(origin, shape2))
    a = FG.make_fine_arrays(image, label, ["FLAIR", "T1", "T1c", "T2"], truth_fn, t2[sl][None], origin, shape2)
    assert a["agreement"] == pytest.approx(1.0) and "p_tumour" not in a                                   # no perception on the fine grid
    box = tuple(slice(int(lo), int(hi)) for lo, hi in a["box"])
    assert np.array_equal(a["truth"][0], truth_fn(label)[0][box]) and a["flair"].max() == 1.0 and a["t1"].max() == 2.0


def test_fine_loader_refuses_real_stores_in_a_test_and_roundtrips_a_temp_store(tmp_path):
    from antelligence.experiments.v41_gate import GateError
    with pytest.raises(DevCaseInTestError):
        FG.load_fine_case(DEV)
    with pytest.raises(GateError):
        FG.load_fine_case(SPLIT["eval"][0])
    crop = make_phantom(36)
    f = _aligned_fine_case(crop)
    FG.save_fine_case(DEV, {"truth": f.truth, "brain_mask": f.brain, "flair": f.flair, "t1": f.t1, "box": np.asarray([[0, 72]] * 3), "origin_2mm": np.zeros(3, int),
                            "full_shape": np.asarray([72] * 3), "outside": 5, "agreement": 1.0}, tmp_path)
    got = FG.load_fine_case(DEV, root=tmp_path, repo_root=REPO)
    assert got.outside == 5 and got.shape == (72, 72, 72) and got.truth.shape[0] == 3


# --- geometry ---------------------------------------------------------------------------------------------------------------------------------------------


def _ball(n=72, r=30.0):
    g = np.indices((n, n, n)).astype(float)
    c = (n - 1) / 2
    return ((g[0] - c) ** 2 + (g[1] - c) ** 2 + (g[2] - c) ** 2) <= r * r


def test_surface_distance_is_the_distance_to_the_mask_boundary_and_crop_faces_are_not_pia():
    b = _ball()
    d = GEO.surface_distance_mm(b, 1.0)
    assert d[b].max() == pytest.approx(29.5, abs=1.0) and d[~b].max() == 0.0
    slab = np.zeros((20, 20, 20), bool)
    slab[:, :, 5:15] = True                                  # brain touching the crop faces in x and y
    d2 = GEO.surface_distance_mm(slab, 1.0)
    assert d2[10, 10, 10] == pytest.approx(5.0, abs=0.51) and d2[0, 10, 10] >= 5.0                    # the x face is not a surface: only z is


def test_precise_rules_count_the_mass_that_breaks_each_rule():
    b = _ball()
    surf = GEO.surface_distance_mm(b, 1.0)
    vent = np.zeros_like(b)
    vent[34:38, 34:38, 34:38] = True
    vd = GEO.distance_to_mask_mm(vent, 1.0)
    lat = np.array([[36, 36, 36], [14, 36, 36], [8, 36, 36], [3, 36, 36]])        # centre (29 mm deep); 8 mm deep; 2 mm deep; outside the mask
    m = [4.0, 1.0, 2.0, 1.0]
    r = GEO.precise_rules(lat, m, surf, vd)
    assert r["depth_lt_25mm"] == pytest.approx(4.0 / 8.0)                          # everything but the centre release is shallower than 25 mm
    assert r["pia_lt_5mm"] == pytest.approx(3.0 / 8.0)                             # the 2 mm deep release and the one outside the mask
    assert r["ependyma_lt_10mm_proxy"] == pytest.approx(4.0 / 8.0)                 # only the centre release is within 10 mm of the (proxy) ventricle
    assert GEO.precise_rules(lat, m, surf, None)["ependyma_lt_10mm_proxy"] is None
    assert GEO.precise_rules(lat, [0.0] * 4, surf, vd)["depth_lt_25mm"] == 0.0


def test_exposure_shares_near_the_pia_and_the_proxy_ependyma():
    b = _ball()
    surf = GEO.surface_distance_mm(b, 1.0)
    dose = np.where(b, 1.0, 0.0)
    s = GEO.exposure_shares(dose, b, surf, None)
    frac = float(((surf < 5.0) & b).sum() / b.sum())
    assert s["within_5mm_of_pia"] == pytest.approx(frac) and s["within_10mm_of_ependyma_proxy"] is None
    vent = np.zeros_like(b)
    vent[32:40, 32:40, 32:40] = True
    s2 = GEO.exposure_shares(dose, b, surf, GEO.distance_to_mask_mm(vent, 1.0))
    assert 0.0 < s2["within_10mm_of_ependyma_proxy"] < 0.2


def test_ventricle_proxy_finds_a_dark_interior_blob_and_ignores_sulcal_edges_and_tumour():
    b = _ball(72, 28.0)
    fl, t1 = np.full(b.shape, 100.0), np.full(b.shape, 100.0)
    g = np.indices(b.shape)
    blob = (np.abs(g[0] - 36) < 6) & (np.abs(g[1] - 36) < 5) & (np.abs(g[2] - 36) < 8)
    fl[blob], t1[blob] = 10.0, 20.0
    edge = b & (GEO.surface_distance_mm(b, 1.0) < 3)
    fl[edge], t1[edge] = 10.0, 20.0                                                      # dark rim (sulcal CSF): must be excluded by the interior rule
    wt = np.zeros_like(b)
    wt[44:50, 32:40, 32:40] = True
    fl[wt], t1[wt] = 10.0, 20.0                                                         # dark tumour core: must be excluded by the tumour rule
    v = GEO.ventricle_proxy(fl, t1, b, wt, 1.0)
    assert v[blob].mean() > 0.95 and not (v & edge).any() and not (v & wt).any()
    assert not GEO.ventricle_proxy(np.full(b.shape, 100.0), np.full(b.shape, 100.0), b, wt, 1.0).any()
