"""A2r1: the move-on helper on a small synthetic scan p map (nearest voxel, 6 mm exclusion, ties, zone limits, none left)."""

import itertools
import math

import numpy as np

from antelligence.worlds.tumor_mri.colony_policies import ScanInfo
from antelligence.worlds.tumor_mri.data import Region
from antelligence.worlds.tumor_mri.move_on import nearest_new_spot, zone_candidates

VM = 2.0
REGION = Region(0, (20.0, 20.0, 20.0), 0.8, 0.1, 4000)       # zone radius 9.85 + 6 = 15.85 mm = 7.9 voxels


def pmap(block=(15, 25), low=0.2, shape=(60, 40, 40)):
    p = np.full(shape, low)
    lo, hi = block
    p[lo:hi + 1, lo:hi + 1, lo:hi + 1] = 0.9
    return p


def scan(p):
    return ScanInfo((REGION,), VM, p)


def brute(p, scan_, from_pos, known, gap_mm=6.0, p_min=0.5):
    """Reference: every voxel of the array, no box, plain loops."""
    r = scan_.zone_radius_mm(REGION)
    best = None
    for v in itertools.product(*(range(s) for s in p.shape)):
        if p[v] < p_min or VM * math.dist(v, REGION.centroid_vox) > r * (1 + 1e-9):
            continue
        if any(VM * math.dist(v, k) < gap_mm * (1 - 1e-9) for k in known):
            continue
        key = (math.dist(v, from_pos), v)
        if best is None or key < best:
            best = key
    return None if best is None else best[1]


def test_nearest_voxel_ties_to_the_lexicographically_smallest():
    s = scan(pmap())
    spot = nearest_new_spot(s.zone_voxels(0), (20, 20, 20), [(20, 20, 20)], VM, 6.0)
    # six voxels at exactly 3 voxels = 6 mm tie; (17, 20, 20) is the smallest tuple
    assert spot == (17, 20, 20)
    assert spot == brute(s.p_tumour, s, (20, 20, 20), [(20, 20, 20)])


def test_six_mm_exclusion_is_inclusive_at_exactly_six_and_exclusive_below():
    cands = np.array([[18, 18, 20], [17, 20, 20], [23, 20, 20]])         # 5.66 mm, 6.0 mm, 6.0 mm from (20, 20, 20)
    assert nearest_new_spot(cands, (20, 20, 20), [(20, 20, 20)], VM, 6.0) == (17, 20, 20)
    assert nearest_new_spot(cands[:1], (20, 20, 20), [(20, 20, 20)], VM, 6.0) is None
    assert nearest_new_spot(cands, (20, 20, 20), [(20, 20, 20), (17, 20, 20), (23, 20, 20)], VM, 6.0) is None


def test_every_known_spot_is_respected_and_matches_the_brute_force_reference():
    s = scan(pmap())
    rng = np.random.default_rng(7)
    for _ in range(12):
        known = [tuple(int(c) for c in rng.integers(14, 27, 3)) for _ in range(int(rng.integers(1, 7)))]
        pos = tuple(int(c) for c in rng.integers(14, 27, 3))
        got = nearest_new_spot(s.zone_voxels(0), pos, known, VM, 6.0)
        assert got == brute(s.p_tumour, s, pos, known)
        if got is not None:
            assert all(VM * math.dist(got, k) >= 6.0 * (1 - 1e-9) for k in known)


def test_zone_limits_p_threshold_and_bounding_box():
    p = pmap()
    p[40:46, 20:26, 20:26] = 0.95                      # a bright block 20+ voxels away: outside the zone sphere
    p[20, 20, 21] = 0.49                               # just under the 0.5 threshold, inside the zone
    s = scan(p)
    zv = s.zone_voxels(0)
    assert len(zv) and (zv[:, 0] < 40).all()
    assert not any((zv == [20, 20, 21]).all(axis=1))
    centre = np.array(REGION.centroid_vox)
    assert (VM * np.linalg.norm(zv - centre, axis=1) <= s.zone_radius_mm(REGION) * (1 + 1e-9)).all()
    # standing next to the outside block does not pull the pick out of the zone
    spot = nearest_new_spot(zv, (39, 22, 22), [], VM, 6.0)
    assert spot[0] < 40 and VM * math.dist(spot, centre) <= s.zone_radius_mm(REGION)
    assert zone_candidates(p, (20, 20, 20), 15.85, VM).shape == zv.shape


def test_zone_voxels_are_cached_and_empty_without_a_map():
    s = scan(pmap())
    assert s.zone_voxels(0) is s.zone_voxels(0)
    assert len(ScanInfo((REGION,), VM).zone_voxels(0)) == 0 and len(s.zone_voxels(99)) == 0


def test_none_left_when_the_known_spots_cover_the_zone():
    p = np.full((60, 40, 40), 0.1)
    p[19:22, 19:22, 19:22] = 0.9                       # a 3x3x3 blob: all within 6 mm of its centre
    s = scan(p)
    assert nearest_new_spot(s.zone_voxels(0), (20, 20, 20), [(20, 20, 20)], VM, 6.0) is None
    assert nearest_new_spot(np.zeros((0, 3), dtype=np.int64), (0, 0, 0), [], VM, 6.0) is None


def test_scan_map_does_not_change_scaninfo_equality_or_hash():
    a, b = ScanInfo((REGION,), VM), scan(pmap())
    assert a == b and hash(a) == hash(b)
