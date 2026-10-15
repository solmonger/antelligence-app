"""v4.3 controls for the foraging world (goal section 4, design readings 19). Fake-backend safe, domain-light.

* ``DisplaceMarksV43``   -> ``mark_pos_fn`` (S-marks-displaced). Every mark moves as in v4.2 (``DisplaceMarks``: a seeded random
  brain voxel of its own zone, stream ``displace``) EXCEPT the uptake marks (``good_uptake``, ``slow_uptake``): they move to a
  seeded random voxel of a seeded random ZONE, any zone, uniform (stream ``displace_zone``: case, seed, mark_index), the voxel
  uniform over that zone's brain voxels (stream ``displace``). So the queen's per-zone uptake counts are displaced too.
* ``PerfectMarksV43``    -> S-marks-perfect diagnostic. Only what OTHER ants read changes. The v4.2 rewrites stay (judgement marks
  from the true label, washout marks from the true delta). An automatic uptake mark takes the TRUE class of the field at its
  voxel: ``good_uptake`` if true u >= 0.7, ``slow_uptake`` if u < 0.4, dropped if middling; its reading ppm is the true u. A
  reading that produced no mark (middling to the author) still produces none: marks are re-classed, never invented.
* ``shuffled_other`` and ``quiet_pair`` are the v4.2 ones, re-exported (S-talk-shuffled and S-marks-quiet are unchanged).

``truth`` objects are scorer-side and passed in by the harness hook; this module imports no world and no store.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from antelligence.experiments.v42_controls import (SLOW_U, DisplaceMarks, PerfectMarks, Truth, _ivox, _ppm,  # noqa: F401
                                                   quiet_pair, shuffled_other)
from antelligence.kernel.colony import rng
from antelligence.kernel.colony.quorum import ZoneSpec

UPTAKE_KINDS = ("good_uptake", "slow_uptake")
GOOD_AT, POOR_BELOW = 0.7, 0.4            # the sensor words of goal B1 (reading >= 0.7 good, < 0.4 poor)

Vox = Tuple[int, int, int]
AutoMark = Tuple[str, Sequence[int], float, int]


class DisplaceMarksV43(DisplaceMarks):
    """v4.2 displacement, except that uptake marks go to a random voxel of a random zone."""

    def __init__(self, zones: Sequence[ZoneSpec], brain_voxels: np.ndarray, voxel_mm: float, case_id: str, seed: int, **kw: Any) -> None:
        super().__init__(zones, brain_voxels, voxel_mm, case_id, seed, **kw)
        self._pool: Optional[List[int]] = None

    def zone_pool(self) -> List[int]:
        """Region ids of the zones that hold at least one brain voxel, sorted. A zone with none cannot receive a mark."""
        if self._pool is None:
            self._pool = [z.region_id for z in self.zones if len(self.zone_voxels(z.region_id)) > 0]
        return self._pool

    def __call__(self, mark_index: int, ant: int, kind: str, pos: Sequence[int]) -> Vox:
        if kind not in UPTAKE_KINDS:
            return super().__call__(mark_index, ant, kind, pos)
        pool = self.zone_pool()
        if not pool:                                           # no zone has a brain voxel: nowhere to move to
            new = _ivox(pos)
            self.log.append({"mark_index": mark_index, "ant": ant, "kind": kind, "from": list(new), "to": list(new), "zone": None,
                             "to_zone": None})
            return new
        zid = pool[int(rng("displace_zone", self.case_id, self.seed, mark_index).integers(len(pool)))]
        cand = self.zone_voxels(zid)
        new = tuple(int(c) for c in cand[int(rng("displace", self.case_id, self.seed, mark_index).integers(len(cand)))])
        self.log.append({"mark_index": mark_index, "ant": ant, "kind": kind, "from": list(_ivox(pos)), "to": list(new),
                         "zone": self.zone_of(pos), "to_zone": zid})
        return new  # type: ignore[return-value]


def true_uptake_class(u: float) -> Optional[str]:
    """``good_uptake`` (u >= 0.7), ``slow_uptake`` (u < 0.4) or None (middling)."""
    if u >= GOOD_AT:
        return "good_uptake"
    return "slow_uptake" if u < POOR_BELOW else None


class PerfectMarksV43(PerfectMarks):
    """v4.2's perfect marks with the v4.3 uptake classes. ``slow_u`` of the parent is unused for uptake marks here."""

    def wrap_auto_rule(self, rule: Callable[[Mapping[str, Any]], Iterable[AutoMark]]
                       ) -> Callable[[Mapping[str, Any]], List[AutoMark]]:
        uptake: List[AutoMark] = []

        def without_uptake(ctx: Mapping[str, Any]) -> List[AutoMark]:
            uptake.clear()
            kept: List[AutoMark] = []
            for m in rule(ctx):
                (uptake if m[0] in UPTAKE_KINDS else kept).append(m)
            return kept

        base = super().wrap_auto_rule(without_uptake)         # v4.2 behaviour for dosed / dose_ok / washout

        def perfect(ctx: Mapping[str, Any]) -> List[AutoMark]:
            out = list(base(ctx))
            for _kind, pos, amount, _ppm_in in uptake:
                u = float(self.truth.u(_ivox(pos)))
                new_kind = true_uptake_class(u)
                if new_kind is None:
                    self.dropped += 1
                    continue
                if new_kind != _kind:
                    self.rewritten += 1
                out.append((new_kind, pos, amount, _ppm(u)))
            return out

        return perfect
