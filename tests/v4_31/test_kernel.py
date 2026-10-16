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


@pytest.mark.parametrize("k", ["low", "high"])
def test_tables_have_no_negative_roundoff(k):
    """Regression: far from the source the kernel was computed as 1/(4 pi D r) minus 1/(4 pi D r), which cancelled to roundoff noise of
    +/-6e-9 umol*h/L and left negative table cells. The planner's `score >= score.max() * (1 - 1e-9)` then selected nothing at the last decision
    tick and crashed (found on a real development case during Step D). Exposure cannot be negative: the far field is now sampled directly."""
    p = P.PhysicalParams.carmustine_wafer(k)
    tb = P.build_tables(p, n_sub=5)
    assert float(tb.cum.min()) >= 0.0
    # and nothing was merely clipped: no cell is a noise-level value where the exact kernel is far larger or smaller
    c = tb.radius_vox
    n = 5
    for off in ((4, 0, 0), (7, 3, 1), (tb.radius_vox, 0, 0)):
        want = float(P.voxel_average_kernel(np.array([off]), 2.0, p.d_star_mm2_s, p.k_clear_s, n * p.tick_seconds, 7)[0]) * p.exposure_to_umol_h_L
        got = float(tb.cum[n][c + off[0], c + off[1], c + off[2]])
        assert got == pytest.approx(want, rel=1e-4, abs=1e-12)


def test_far_field_direct_sampling_agrees_with_the_near_field_method_at_the_switch():
    k, s = KS["high"], 2.0
    for T in (math.inf, 9000.0):
        for off in ((3, 3, 3), (4, 3, 3)):                          # NEAR_VOX = 3: (3,3,3) is subtracted, (4,3,3) is direct
            ref = P.reference_voxel_average(off, s, D, k, T, n_ref=30)
            assert float(P.voxel_average_kernel(np.array([off]), s, D, k, T, 5)[0]) == pytest.approx(ref, rel=1e-6)


# --- A.2: the reference clearance, corner-placement tables and the scoring lattice ---------------------------------------------------------------------------


@pytest.mark.parametrize("vox,corner", [(2.0, False), (1.0, True), (0.5, True)])
def test_ref_tables_are_nonnegative_conserve_mass_and_are_symmetric(vox, corner):
    p = P.PhysicalParams.carmustine_wafer("ref").with_voxel(vox)
    tb = P.build_tables(p, n_sub=5, corner=corner)
    last = tb.cum[-1].astype(np.float64)
    assert float(tb.cum.min()) >= 0.0 and tb.corner == corner
    assert last.shape[0] == (2 * tb.radius_vox if corner else 2 * tb.radius_vox + 1)
    total = float(last.sum()) * p.voxel_volume_mm3 * p.k_clear_s / p.exposure_to_umol_h_L             # all the exposure / (M / k): 1 up to the window tail
    assert total == pytest.approx(1.0, abs=2e-3)
    assert np.allclose(last, last[::-1], rtol=1e-5) and np.allclose(last, np.transpose(last, (1, 0, 2)), rtol=1e-5)


def test_release_voxel_shares_for_the_reference_clearance_match_the_a2_goal():
    D = P.D_STAR_CM2_S * P.MM2_PER_CM2
    s2 = P.release_voxel_share(2.0, D, P.K_CLEAR_REF_S, 10, 5)
    s1 = P.release_voxel_share(1.0, D, P.K_CLEAR_REF_S, 10, 5)
    assert s2 == pytest.approx(0.69, abs=0.01) and s1 == pytest.approx(0.34, abs=0.01)         # the goal: about 69% at 2 mm, about 0.34 at 1 mm
    assert P.release_voxel_share(0.5, D, P.K_CLEAR_REF_S, 10, 5) < 0.15


def test_the_one_mm_corner_table_block_averages_to_the_two_mm_centre_table():
    """The exposure averaged over a 2 mm voxel centred on the release is exactly the mean of the eight 1 mm voxels around that corner: the scorer is the same physics, finer."""
    p2 = P.PhysicalParams.carmustine_wafer("ref")
    p1 = p2.with_voxel(1.0)
    t2, t1 = P.build_tables(p2, 5), P.build_tables(p1, 5, corner=True)
    n = 3
    c2 = t2.cumulative(n).astype(np.float64)                  # (2w2+1)^3, centred
    c1 = t1.cumulative(n).astype(np.float64)                  # (2w1)^3, corner at the centre of the cube
    w2, w1 = t2.radius_vox, t1.radius_vox
    for off in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]:
        sl = [slice(w1 + 2 * o - 1, w1 + 2 * o + 1) for o in off]       # the eight 1 mm voxels that make up the 2 mm voxel at this offset
        block = c1[sl[0], sl[1], sl[2]].mean()
        assert block == pytest.approx(float(c2[w2 + off[0], w2 + off[1], w2 + off[2]]), rel=5e-4)


def test_ref_subsampling_converges_on_the_scoring_lattices():
    p = P.PhysicalParams.carmustine_wafer("ref")
    D, k = p.d_star_mm2_s, p.k_clear_s
    for vox in (1.0, 0.5):
        worst = 0.0
        for off in [(0.5, 0.5, 1.5), (1.5, 0.5, 0.5), (1.5, 1.5, 0.5), (2.5, 1.5, 0.5), (4.5, 0.5, 0.5)]:
            ref = P.reference_voxel_average(off, vox, D, k, math.inf, n_ref=24)
            worst = max(worst, abs(float(P.voxel_average_kernel(np.array([off]), vox, D, k, math.inf, 5)[0]) - ref) / ref)
        assert worst < 5e-4
        own = [float(P.voxel_average_kernel(np.full((1, 3), 0.5), vox, D, k, math.inf, n)[0]) for n in (24, 96)]       # a voxel that touches the release corner
        assert abs(own[0] - own[1]) / own[1] < 2e-4
