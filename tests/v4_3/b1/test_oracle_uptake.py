"""B1 section 6: oracle S' reads the world's continuous ``field.uptake`` (no two-level assumption) in its projected dose."""

import numpy as np
import pytest
from scipy import ndimage

from antelligence.worlds.tumor_mri import Crop, TumorMRIWorld, oracle_for, run_episode
from antelligence.worlds.tumor_mri.config import FactsConfig
from tests.v4_1.world.phantom import oracle_config
from tests.v4_2.q1.helpers import PHANTOM
from tests.v4_3.b1.helpers import channels


@pytest.fixture(scope="module")
def crop():
    return Crop.from_npz(PHANTOM)


def make(crop, level="mid", seed=1):
    cfg = oracle_config(facts=FactsConfig(f2=True, f2_model="foraging", uptake_level=level))
    w = TumorMRIWorld(crop, cfg, seed=seed, scorer_channels=channels(crop))
    return w, oracle_for(w)


def test_oracle_holds_the_worlds_own_continuous_field(crop):
    w, o = make(crop)
    assert o._dose is w.field and o._dose.uptake is w.field.uptake
    assert len(np.unique(w.field.uptake[crop.truth])) > 20 and w.field.uptake[crop.truth].min() >= 0.05


def test_commit_scales_the_puff_by_u_voxel_by_voxel(crop):
    w, o = make(crop)
    v = tuple(int(c) for c in np.argwhere(crop.truth)[len(np.argwhere(crop.truth)) // 2])
    kern = o._puff_kernel(v, 40)
    proj = np.zeros(crop.shape)
    o._commit(proj, v, kern, 7.0)
    sl, ks = o._window(proj.shape, v)
    assert np.allclose(proj[sl], 7.0 * kern[ks] * w.field.uptake[sl])
    inside = crop.truth[sl] & (kern[ks] > 0)
    ratio = proj[sl][inside] / (7.0 * kern[ks][inside])
    assert np.allclose(ratio, w.field.uptake[sl][inside]) and len(np.unique(np.round(ratio, 6))) > 10   # not two levels


def test_expose_projects_the_flight_dose_through_u(crop):
    w, o = make(crop)
    conc = np.zeros((9, 9, 9), dtype=np.float32)
    conc[4, 4, 4] = 1.0
    u = np.linspace(0.05, 1.0, 729, dtype=np.float32).reshape(9, 9, 9)
    plain = o._expose(conc, 1, coarse=False)
    with_u = o._expose(conc, 1, coarse=False, u=u)
    assert np.allclose(with_u, plain * u, rtol=1e-6)                               # dose += u * conc, per voxel
    coarse = o._expose(conc, 3, coarse=True, u=u)
    assert coarse.shape == conc.shape and coarse.sum() < o._expose(conc, 3, coarse=True).sum()


def test_gains_use_u_exactly(crop):
    w, o = make(crop)
    v = tuple(int(c) for c in np.argwhere(crop.truth)[100])
    kern = o._puff_kernel(v, 40)
    masses = np.array([2.0, 8.0, 30.0])
    proj = np.full(crop.shape, 0.9 * w.config.dose_threshold)             # close to D_t, so a small puff lifts some voxels past it
    counts, frac = o._gains(proj, v, kern, masses)
    sl, ks = o._window(proj.shape, v)
    d_t = w.config.dose_threshold
    m = crop.truth[sl] & (proj[sl] < d_t)
    k = kern[ks][m].astype(np.float64) * w.field.uptake[sl][m]
    want = ((d_t - proj[sl][m])[None, :] <= masses[:, None] * k[None, :]).sum(axis=1)
    assert np.array_equal(counts, want.astype(float))
    # a field of 1s would lift more voxels: u really matters
    w.field.uptake[:] = 1.0
    counts_one, _ = o._gains(proj, v, kern, masses)
    assert counts.sum() > 0 and counts_one.sum() > counts.sum()


def test_oracle_runs_clean_on_every_level(crop):
    for level in ("low", "mid", "high"):
        w, o = make(crop, level)
        m = run_episode(w, lambda a: o)
        assert abs(m["mass_residual"]) < 1e-9 and 0.0 < m["coverage"] <= 1.0
