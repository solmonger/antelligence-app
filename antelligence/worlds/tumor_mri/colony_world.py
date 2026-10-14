"""``ColonyTumorWorld``: the v4.1 tumour world plus what the colony kernel needs (goal 6.9, 2.8).

A subclass, so ``world.py`` is untouched. With ``arm="S"`` it behaves exactly like :class:`TumorMRIWorld` (colony off ==
v4.1). In every other colony arm the two visit sets are PER ANT: ``_visited_regions`` (which regions an ant has been near)
and the walked-voxel map ``_visited``. Explore's uncertainty-cluster fallback and ``candidates[].visited`` read the asking
ant's own sets; sharing arms learn about other ants' visits only from marks (goal 2.8).

``observe`` still returns only the v4.1 ``OBS_KEYS``; the colony keys are added by ``run_colony_episode`` from the arm's
allowed set. Sensor readings are produced at the ant's own voxel in :meth:`read_sensors`. The F1 look comes from
``look_fn``; uptake and washout readings arrive through the optional ``uptake_fn`` / ``washout_fn`` stand-ins until the W1
physics lands (no hidden field is read here).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from antelligence.kernel.colony.quorum import ZoneSpec
from antelligence.kernel.colony.wake import scan_band
from antelligence.kernel.types import Observation
from antelligence.worlds.tumor_mri.config import WorldConfig
from antelligence.worlds.tumor_mri.data import Crop, Region, select_candidates
from antelligence.worlds.tumor_mri.physics import Vox
from antelligence.worlds.tumor_mri.world import LookFn, QueenHook, TumorMRIWorld, _Squad

ZONE_MARGIN_MM = 6.0
SensorFn = Callable[..., Optional[Dict[str, Any]]]


class ColonyTumorWorld(TumorMRIWorld):
    def __init__(self, crop: Crop, config: Optional[WorldConfig] = None, seed: int = 0, *, arm: str = "S",
                 look_fn: Optional[LookFn] = None, queen_hook: Optional[QueenHook] = None,
                 regions: Optional[Sequence[Region]] = None, uptake_fn: Optional[SensorFn] = None,
                 washout_fn: Optional[SensorFn] = None) -> None:
        super().__init__(crop, config, seed, look_fn=look_fn, queen_hook=queen_hook, regions=regions)
        self.arm = arm
        self.per_ant_visits = arm != "S"
        self.uptake_fn, self.washout_fn = uptake_fn, washout_fn
        self.decision_every = self.config.decision_every
        self.queen_every = self.config.queen_every
        if self.per_ant_visits:
            self._vis_arrays: List[np.ndarray] = []
            self._vis_regions: List[set] = [set() for _ in self._squads]
            for sq in self._squads:
                arr = np.zeros(self.shape, dtype=bool)
                arr[sq.pos] = True
                self._vis_arrays.append(arr)

    # --- per-ant visit sets ----------------------------------------------------------------------------------------

    @contextmanager
    def _as_ant(self, i: int) -> Iterator[None]:
        saved = (self._visited, self._visited_regions)
        self._visited, self._visited_regions = self._vis_arrays[i], self._vis_regions[i]
        try:
            yield
        finally:
            self._visited, self._visited_regions = saved

    def observe(self, agent_id: str, tick: int) -> Observation:
        if not self.per_ant_visits:
            return super().observe(agent_id, tick)
        with self._as_ant(self._idx(agent_id)):
            return super().observe(agent_id, tick)

    def _explore_goal(self, i: int, sq: _Squad) -> Optional[Vox]:
        if not self.per_ant_visits:
            return super()._explore_goal(i, sq)
        with self._as_ant(i):
            return super()._explore_goal(i, sq)

    def _mark_visits(self) -> None:
        if not self.per_ant_visits:
            return super()._mark_visits()
        r2 = (self.config.visit_radius_mm / self.voxel_mm) ** 2
        for sq in self._squads:
            seen = self._vis_regions[sq.index]
            for r in self.regions:
                if r.region_id not in seen and sum((a - b) ** 2 for a, b in zip(sq.pos, r.centroid_vox)) <= r2:
                    seen.add(r.region_id)

    def step_environment(self, tick: int) -> None:
        super().step_environment(tick)
        if self.per_ant_visits:
            for sq in self._squads:
                self._vis_arrays[sq.index][sq.pos] = True

    # --- ColonyWorld protocol (kernel/colony/episode.py) -------------------------------------------------------------

    def start_episode(self) -> None:
        """v4.1 ``run_episode`` default: with no queen hook the candidates are the top six regions by p x volume."""
        if self.queen_hook is None:
            self.set_candidates(select_candidates(self.regions))

    def ant_pos(self, i: int) -> Vox:
        return self._squads[i].pos

    def ant_goal(self, i: int) -> Optional[Vox]:
        return self._squads[i].goal

    def ant_payload(self, i: int) -> float:
        return self._squads[i].payload

    def zones(self) -> List[ZoneSpec]:
        """A zone is a perceived region plus everything within 6 mm of it (INTERFACES 0). Only the region's centroid and
        volume are kept, so a region is approximated by the sphere of equal volume."""
        out = []
        for r in self.regions:
            radius = (3.0 * r.volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0) + ZONE_MARGIN_MM
            out.append(ZoneSpec(r.region_id, tuple(r.centroid_vox), self._contains(tuple(r.centroid_vox), radius)))
        return out

    def _contains(self, centre: Sequence[float], radius_mm: float) -> Callable[[Sequence[float]], bool]:
        vmm = self.voxel_mm
        return lambda pos: vmm * math.dist(pos, centre) <= radius_mm * (1 + 1e-9)

    def read_sensors(self, i: int, tick: int) -> List[Dict[str, Any]]:
        sq = self._squads[i]
        out: List[Dict[str, Any]] = []
        if self.look_fn is not None:
            look = self.look_fn(i, sq.pos, tick)
            if look is not None:
                out.append({"kind": "look", "value": float(look), "band": scan_band(float(self._percept.p_tumour[sq.pos]))})
        out.append({"kind": "dose", "value": float(self.field.dose[sq.pos]) / self.config.dose_threshold})
        if self.uptake_fn is not None and sq.last_release is not None:
            r = self.uptake_fn(i, tick, dict(sq.last_release))
            if r is not None:
                out.append({"kind": "uptake", **r})
        if self.washout_fn is not None:
            r = self.washout_fn(i, tick, tuple(sq.pos))
            if r is not None:
                out.append({"kind": "washout", **r})
        return out

    def release_fraction(self, i: int, released: float) -> float:
        return float(released) / (self.total_drug / self.config.n_squads)

    def visited_self(self, i: int) -> Dict[str, Any]:
        return {"regions": sorted(self._vis_regions[i]), "walked_count": int(self._vis_arrays[i].sum())}
