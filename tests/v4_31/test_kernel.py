"""v4.31 Step B: the exposure kernel AUC(r) = M / (4 pi D r) exp(-r/L), voxel-averaged and episode-truncated. Synthetic only.

Check 1 of the goal (exact): the closed form matches a direct numeric time integral of (4 pi D t)^-1.5 exp(-r^2/(4Dt)) exp(-kt)
to 1e-6 relative at r = 0.5, 1, 2 and 5 mm. A mismatch here is our bug, not the formula's."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import tplquad

from antelligence.worlds.tumor_mri import physical as P

D = P.PhysicalParams.carmustine_wafer("low").d_star_mm2_s            # same D for both ends of the pair
KS = {"low": P.K_CLEAR_LOW_S, "high": P.K_CLEAR_HIGH_S}


@pytest.mark.parametrize("k", ["low", "high"])
@pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0])
def test_closed_form_matches_numeric_time_integral(k, r):
    num = P.numeric_time_integral(r, D, KS[k])
    assert float(P.point_exposure_inf(r, D, KS[k])) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("k", ["low", "high"])
@pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("T", [1800.0, 7200.0, 216000.0])
def test_finite_horizon_closed_form_matches_numeric(k, r, T):
    """The episode truncation is exact, not a correction term."""
    assert float(P.point_exposure_T(r, D, KS[k], T)) == pytest.approx(P.numeric_time_integral(r, D, KS[k], T), rel=1e-6)


def test_truncation_is_monotone_and_limits():
    r, k = 3.0, KS["low"]
    vals = [float(P.point_exposure_T(r, D, k, T)) for T in (1800.0, 3600.0, 18000.0, 216000.0, 2.16e7)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(float(P.point_exposure_inf(r, D, k)), rel=1e-9)
    assert float(P.point_exposure_T(r, D, k, 0.0)) == 0.0


def test_total_exposure_integrates_to_M_over_k():
    """Mass balance of the kernel: the exposure integrated over all space is exactly M / k (every unit of mass is cleared at rate k)."""
    k = KS["high"]
    r = np.linspace(1e-6, 60.0, 600001)
    tot = np.trapezoid(P.point_exposure_inf(r, D, k) * 4.0 * math.pi * r * r, r)
    assert tot == pytest.approx(1.0 / k, rel=1e-5)


def test_box_inverse_r_integral_is_exact():
    assert float(P.box_inv_r_integral(np.array([[-.5, -.5, -.5]]), np.array([[.5, .5, .5]]))[0]) == pytest.approx(P.CUBE_CENTRE_INV_R, rel=1e-13)
    lo, hi = np.array([[1.0, 0.0, -0.5]]), np.array([[3.0, 1.0, 0.5]])
    ref = tplquad(lambda z, y, x: 1.0 / math.sqrt(x * x + y * y + z * z), 1.0, 3.0, 0.0, 1.0, -0.5, 0.5, epsabs=1e-12, epsrel=1e-12)[0]
    assert float(P.box_inv_r_integral(lo, hi)[0]) == pytest.approx(ref, rel=1e-10)


def test_voxel_average_matches_brute_force_time_integral_average():
    """A voxel away from the source: the voxel-averaged kernel equals the plain Gauss-Legendre average of the NUMERIC time integral."""
    k, s, off = KS["high"], 2.0, (3, 2, 1)
    got = float(P.voxel_average_kernel(np.array([off]), s, D, k, math.inf, 6)[0])
    x, w = np.polynomial.legendre.leggauss(6)
    w = w / 2.0
    tot = 0.0
    for i in range(6):
        for j in range(6):
            for l in range(6):
                p = np.array(off) * s + 0.5 * s * np.array([x[i], x[j], x[l]])
                tot += w[i] * w[j] * w[l] * P.numeric_time_integral(float(np.linalg.norm(p)), D, k)
    assert got == pytest.approx(tot, rel=1e-6)


@pytest.mark.parametrize("k", ["low", "high"])
def test_subsampling_convergence(k):
    """Sub-sampling converges: 3x3x3 is already close, 5x5x5 closer, and the release voxel (singularity removed exactly) converges too."""
    kk = KS[k]
    for off in [(1, 0, 0), (1, 1, 0), (2, 0, 0), (3, 2, 1)]:
        ref = P.reference_voxel_average(off, 2.0, D, kk, math.inf, n_ref=20)
        errs = [abs(float(P.voxel_average_kernel(np.array([off]), 2.0, D, kk, math.inf, n)[0]) - ref) / ref for n in (3, 5, 7)]
        assert errs[0] < 2e-3 and errs[1] < 1e-4 and errs[2] <= errs[1] + 1e-12, (off, errs)
    own = [float(P.voxel_average_kernel(np.zeros((1, 3)), 2.0, D, kk, math.inf, n)[0]) for n in (4, 24, 64)]
    assert abs(own[1] - own[2]) / own[2] < 1e-5            # the release voxel at the production sub-sampling is converged to 1e-5


def test_centre_sampling_is_wrong_near_the_source_and_averaging_is_not():
    """Why voxel-averaging: at the adjacent voxel the centre value (r = 2 mm) is off by a lot; the average is the physical answer."""
    k = KS["high"]
    centre = float(P.point_exposure_inf(2.0, D, k))
    avg = float(P.voxel_average_kernel(np.array([[1, 0, 0]]), 2.0, D, k, math.inf, 7)[0])
    assert abs(centre - avg) / avg > 0.1


def test_release_voxel_share_is_a_fraction_and_matches_a_direct_sum():
    for k in ("low", "high"):
        sh = P.release_voxel_share(2.0, D, KS[k], 10)
        assert 0.0 < sh < 0.8
        # direct: the exposure integrated over a large cube of voxels is almost all of M/k; the release voxel's own part over that sum
        w = 14
        uo, inv = P._unique_offsets(w)
        vals = P.voxel_average_kernel(uo, 2.0, D, KS[k], math.inf, 5)[inv]
        total = float(vals.sum()) * 8.0 * KS[k]
        assert sh == pytest.approx(float(vals[w, w, w]) * 8.0 * KS[k], rel=1e-12)
        assert 0.9 < total <= 1.0 + 1e-3 if k == "high" else 0.6 < total <= 1.0 + 1e-3


@pytest.mark.parametrize("k", ["low", "high"])
def test_tables_are_symmetric_cumulative_and_saturate(k):
    p = P.PhysicalParams.carmustine_wafer(k)
    tb = P.build_tables(p, n_sub=5)
    c = tb.radius_vox
    last = tb.cum[-1]
    assert np.allclose(last, last[::-1, :, :], rtol=1e-6) and np.allclose(last, np.transpose(last, (1, 0, 2)), rtol=1e-6)
    d = np.diff(tb.cum[:, c, c, c])
    assert np.all(d >= 0) and np.all(d[:3] > 0)                        # more ticks, more exposure (float32 ties once saturated)
    assert tb.cum[0].max() == 0.0
    assert (tb.cum[-1][c, c, c] - tb.cum[-2][c, c, c]) / tb.cum[-1][c, c, c] < 1e-6      # saturated
    # table value equals the direct kernel at the same horizon
    n = 5
    direct = float(P.voxel_average_kernel(np.array([[2, 1, 0]]), 2.0, p.d_star_mm2_s, p.k_clear_s, n * p.tick_seconds, 5)[0]) * p.exposure_to_umol_h_L
    assert float(tb.cum[n][c + 2, c + 1, c]) == pytest.approx(direct, rel=1e-5)


def test_table_radius_covers_the_budget_to_the_stated_tolerance():
    for k in ("low", "high"):
        p = P.PhysicalParams.carmustine_wafer(k)
        w = P.kernel_radius_vox(p, 0.1)
        beyond = p.budget_mg * p.exposure_to_umol_h_L * float(P.point_exposure_inf((w + 0.5) * p.voxel_mm, p.d_star_mm2_s, p.k_clear_s))
        assert beyond < 0.1
        assert p.budget_mg * p.exposure_to_umol_h_L * float(P.point_exposure_inf((w - 0.5) * p.voxel_mm, p.d_star_mm2_s, p.k_clear_s)) >= 0.1


def test_mass_fraction_cube_is_a_unit_gaussian_mass():
    f = P.mass_fraction_cube(30, 2.0, D, 1800.0)
    assert f.sum() == pytest.approx(1.0, abs=1e-12)
    f0 = P.mass_fraction_cube(3, 2.0, D, 0.0)
    assert f0[3, 3, 3] == 1.0 and f0.sum() == 1.0
