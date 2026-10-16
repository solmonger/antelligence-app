"""v4.31 A.2: score an episode on a fine lattice. Additive; the world, the policies and the 2 mm field are untouched.

An episode is played on the 2 mm world (perception, targets, release points, S and S' decisions all at 2 mm). Its releases (position at a 2 mm voxel centre, mass, tick) are
logged by :class:`UnitsField`. Because the exposure of a release is analytic, the SAME releases can be scored on any lattice: the exposure each release adds to a 1 mm or 0.5 mm voxel
is the voxel-averaged closed form, a function of the offset between the release and the voxel and of the time left in the episode. Scoring finer than the world is therefore a free
numerical choice, not a new world.

Placement. A release at a 2 mm voxel centre is the corner shared by eight voxels of a 1 mm (or 0.5 mm) lattice, so the corner-placement tables are used (physical.build_tables(..., corner=True)).
On the 2 mm lattice itself the release is at a voxel centre and the world's own field is the score.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from antelligence.worlds.tumor_mri.fine_grid import FineCase
from antelligence.worlds.tumor_mri.physical import PhysicalParams, build_tables


def release_log(field) -> List[Tuple[Tuple[int, int, int], float, int]]:
    """The releases of a finished :class:`UnitsField`: (2 mm voxel position, mass in mg, release tick)."""
    return [(tuple(int(c) for c in pos), float(m), int(t0)) for pos, m, t0 in field._rel]


def score_dose(releases: Sequence[Tuple[Tuple[int, int, int], float, int]], params: PhysicalParams, fine: FineCase, spacing_mm: float, *, n_sub: int = 5,
               tol_umol_h_L: float = 0.1) -> np.ndarray:
    """Exposure (umol*h/L) on the ``spacing_mm`` lattice of the fine case's crop box, at the end of the episode, from a release log."""
    truth, _, _ = fine.at_spacing(spacing_mm)
    shape = truth.shape[1:]
    ps = params.with_voxel(spacing_mm)
    tb = build_tables(ps, n_sub=n_sub, tol_umol_h_L=tol_umol_h_L, corner=True)
    w = tb.radius_vox
    dose = np.zeros(shape, dtype=np.float64)
    if not releases:
        return dose
    lat = fine.release_lattice([r[0] for r in releases], spacing_mm)
    for (pos, mass, t0), u in zip(releases, lat):
        n = params.ticks - t0
        if n < 1:
            continue
        cube = tb.cumulative(n)
        lo = [int(u[a]) - w for a in range(3)]
        hi = [int(u[a]) + w for a in range(3)]
        dst = tuple(slice(max(lo[a], 0), min(hi[a], shape[a])) for a in range(3))
        if any(d.start >= d.stop for d in dst):
            continue
        src = tuple(slice(dst[a].start - lo[a], dst[a].stop - lo[a]) for a in range(3))
        dose[dst] += mass * cube[src]
    return dose
