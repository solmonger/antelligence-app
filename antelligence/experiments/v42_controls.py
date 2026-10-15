"""v4.2 M1 controls (goal section 8): transforms plugged into the K1 episode-driver hooks. Fake-backend safe, domain-light.

* ``DisplaceMarks``      -> ``mark_pos_fn`` (S-marks-displaced). Stream ``displace`` (case, seed, mark_index).
* ``shuffled_other``     -> a recipient picker for S-talk-shuffled (stream ``shuffle``); never the asker.
* ``PerfectMarks``       -> S-marks-perfect diagnostic. Only what OTHER ants read is changed: the policy's emitted
  judgement marks and the automatic ``slow_uptake`` / ``washout`` marks are rewritten from truth at the mark's spot. The
  observation, the reading log and the author's own decisions are untouched (the wrappers run after ``decide``).
* ``quiet_pair``         -> the S-marks / S-marks-quiet config pair.

``truth`` objects are scorer-side and must be passed in by the harness; this module never imports a world or a store.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from antelligence.kernel.colony import rng
from antelligence.kernel.colony.episode import ColonyConfig
from antelligence.kernel.colony.messages import shuffled_recipient  # noqa: F401  (K1's rule-avoiding variant, re-exported)
from antelligence.kernel.colony.quorum import ZoneSpec

Vox = Tuple[int, int, int]
OUTSIDE_RADIUS_MM = 6.0
SLOW_U = 0.7          # goal 5.2: uptake reading < 0.7 reads "slow"


def _ivox(pos: Sequence[float]) -> Vox:
    return tuple(int(round(c)) for c in pos)  # type: ignore[return-value]


# --- S-marks-displaced -----------------------------------------------------------------------------------------------

class DisplaceMarks:
    """``mark_pos_fn(mark_index, ant, kind, pos) -> pos``.

    The mark moves to a seeded random brain voxel in its zone. Zone of a spot: the zone(s) that contain it; overlapping
    zones -> the one whose centroid is nearest (ties -> lower region_id). Outside every zone: a seeded random brain voxel
    within 6 mm of the spot (the spot itself if the brain has none there). Kind, amount and time are not touched here: the
    driver only lets us change the position.
    """

    def __init__(self, zones: Sequence[ZoneSpec], brain_voxels: np.ndarray, voxel_mm: float, case_id: str, seed: int,
                 *, outside_radius_mm: float = OUTSIDE_RADIUS_MM) -> None:
        self.zones = sorted(zones, key=lambda z: z.region_id)
        self.brain = np.asarray(brain_voxels, dtype=np.int64).reshape(-1, 3)
        self.vmm, self.case_id, self.seed = float(voxel_mm), str(case_id), int(seed)
        self.outside_mm = float(outside_radius_mm)
        self._zone_cache: Dict[int, np.ndarray] = {}
        self.log: List[Dict[str, Any]] = []

    def zone_of(self, pos: Sequence[float]) -> Optional[int]:
        best = None
        for z in self.zones:
            if z.contains(pos):
                key = (round(self.vmm * math.dist(pos, z.centroid_vox), 9), z.region_id)
                if best is None or key < best:
                    best = key
        return None if best is None else best[1]

    def zone_voxels(self, region_id: int) -> np.ndarray:
        if region_id not in self._zone_cache:
            z = next(z for z in self.zones if z.region_id == region_id)
            keep = [i for i in range(len(self.brain)) if z.contains(tuple(self.brain[i]))]
            self._zone_cache[region_id] = self.brain[keep]
        return self._zone_cache[region_id]

    def candidates(self, pos: Sequence[float]) -> Tuple[Optional[int], np.ndarray]:
        zid = self.zone_of(pos)
        if zid is not None:
            return zid, self.zone_voxels(zid)
        d = self.vmm * np.linalg.norm(self.brain - np.asarray(pos, dtype=float), axis=1) if len(self.brain) else np.zeros(0)
        return None, self.brain[d <= self.outside_mm * (1 + 1e-9)]

    def __call__(self, mark_index: int, ant: int, kind: str, pos: Sequence[int]) -> Vox:
        zid, cand = self.candidates(pos)
        if len(cand) == 0:
            new = _ivox(pos)
        else:
            new = tuple(int(c) for c in cand[int(rng("displace", self.case_id, self.seed, mark_index).integers(len(cand)))])
        self.log.append({"mark_index": mark_index, "ant": ant, "kind": kind, "from": list(_ivox(pos)), "to": list(new),
                         "zone": zid})
        return new  # type: ignore[return-value]


# --- S-talk-shuffled -------------------------------------------------------------------------------------------------

def shuffled_other(case_id: str, seed: int, msg_id: int, from_ant: int, n_ants: int) -> int:
    """A seeded random ant other than the asker (stream ``shuffle``). The receiver then answers from its own log as usual."""
    if n_ants < 2:
        raise ValueError("need at least two ants to shuffle a recipient")
    pool = [i for i in range(n_ants) if i != from_ant]
    return pool[int(rng("shuffle", case_id, seed, msg_id).integers(len(pool)))]


# --- S-marks-perfect -------------------------------------------------------------------------------------------------

class Truth(Protocol):
    def label(self, pos: Sequence[int]) -> bool: ...      # exact tumour label at the spot
    def u(self, pos: Sequence[int]) -> float: ...         # F2 uptake factor
    def delta(self, pos: Sequence[int]) -> float: ...     # F3 extra decay


def _ppm(x: float) -> int:
    return int(min(1_000_000, max(0, round(float(x) * 1_000_000))))


class PerfectMarks:
    """What OTHER ants read comes from truth at the mark's spot: exact label, u and delta.

    * ``wrap_policy_for(policy_for)``: judgement drafts ``tumour_here`` / ``not_tumour`` get the true label's kind
      (``need_more`` is a request, not a claim, and is left alone). Runs on the policy OUTPUT, so the author's own
      observation and decision are exactly those of S-marks.
    * ``wrap_auto_rule(rule)``: an automatic ``slow_uptake`` mark is kept (with reading = true u) only where true u < 0.7,
      a ``washout`` mark only where true delta >= ``fast_delta`` (reading = true delta); a mark truth disagrees with is
      dropped. ``dosed`` / ``dose_ok`` are world facts and pass unchanged. The judgement mark's ``reading_ppm`` is set by
      the driver from the author's own look and cannot be rewritten through the hooks (logged limitation).
    """

    def __init__(self, truth: Truth, *, fast_delta: float = 0.02, slow_u: float = SLOW_U) -> None:
        self.truth, self.fast_delta, self.slow_u = truth, float(fast_delta), float(slow_u)
        self.rewritten = 0
        self.dropped = 0

    def _draft(self, d: Any) -> Any:
        kind, pos = (d["kind"], d["pos_vox"]) if isinstance(d, Mapping) else (d.kind, d.pos)
        if kind not in ("tumour_here", "not_tumour"):
            return d
        true_kind = "tumour_here" if self.truth.label(_ivox(pos)) else "not_tumour"
        if true_kind == kind:
            return d
        self.rewritten += 1
        if isinstance(d, Mapping):
            return {**d, "kind": true_kind}
        return dataclasses.replace(d, kind=true_kind)

    def wrap_policy_for(self, policy_for: Callable[[str], Any]) -> Callable[[str], Any]:
        outer = self

        class _P:
            def __init__(self, inner: Any) -> None:
                self.inner = inner

            def decide(self, view: Any) -> Any:
                intent = self.inner.decide(view)
                return dataclasses.replace(intent, emit=tuple(outer._draft(d) for d in intent.emit))

            def __getattr__(self, name: str) -> Any:
                return getattr(self.inner, name)

        return lambda agent: _P(policy_for(agent))

    def wrap_auto_rule(self, rule: Callable[[Mapping[str, Any]], Iterable[Tuple[str, Sequence[int], float, int]]]
                       ) -> Callable[[Mapping[str, Any]], List[Tuple[str, Sequence[int], float, int]]]:
        def perfect(ctx: Mapping[str, Any]) -> List[Tuple[str, Sequence[int], float, int]]:
            out = []
            for kind, pos, amount, ppm in rule(ctx):
                p = _ivox(pos)
                if kind == "slow_uptake":
                    u = float(self.truth.u(p))
                    if u >= self.slow_u:
                        self.dropped += 1
                        continue
                    ppm = _ppm(u)
                elif kind == "washout":
                    dl = float(self.truth.delta(p))
                    if dl < self.fast_delta:
                        self.dropped += 1
                        continue
                    ppm = _ppm(dl)
                out.append((kind, pos, amount, ppm))
            return out

        return perfect


# --- quiet-rule pair -------------------------------------------------------------------------------------------------

def quiet_pair(base: Optional[ColonyConfig] = None) -> Dict[str, ColonyConfig]:
    """``S-marks`` (quiet rule off, as in every verdict arm) and ``S-marks-quiet`` (on). Everything else is identical."""
    base = base or ColonyConfig()
    return {"S-marks": dataclasses.replace(base, quiet_rule=False), "S-marks-quiet": dataclasses.replace(base, quiet_rule=True)}
