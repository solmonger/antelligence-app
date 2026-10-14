"""Quorum from marks only (goal 6.3, INTERFACES v4.2 section 4).

A zone becomes ``done`` when ``k`` distinct ants have deposited ``dose_ok`` within 6 mm of its centroid, and
``not_tumour`` when ``k`` distinct ants have deposited ``not_tumour`` within the zone. Default k = 2. Nothing but marks
is read, so the result carries nothing an S-solo ant cannot see; the status is sticky and written once. Status is shown
only to the sharing arms and never removes Stay or Explore.

If both thresholds are met in the same evaluation, ``done`` wins (listed first); a zone never changes status afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

from antelligence.kernel.colony.marks import MarkField, ant_of

STATUS_OPEN = "open"
NEED_LEVELS = ("none", "low", "medium", "high")


@dataclass(frozen=True)
class ZoneSpec:
    """One perceived zone. ``contains`` answers "is this voxel within the zone" (region plus 6 mm); the world supplies it."""

    region_id: int
    centroid_vox: Tuple[float, float, float]
    contains: Callable[[Sequence[float]], bool]


@dataclass(frozen=True)
class QuorumEvent:
    region_id: int
    status: str            # "done" | "not_tumour"
    k: int
    tick_first: int

    def to_payload(self) -> Dict[str, Any]:
        return {"subtype": "status", "region_id": self.region_id, "status": self.status, "k": self.k,
                "tick_first": self.tick_first}


def need_payload(region_id: int, need: str, need_milli: int, delivered_milli: int) -> Dict[str, Any]:
    if need not in NEED_LEVELS:
        raise ValueError(f"bad need level {need!r}")
    return {"subtype": "need", "region_id": int(region_id), "need": need, "need_milli": int(need_milli),
            "delivered_milli": int(delivered_milli)}


class Quorum:
    def __init__(self, zones: Sequence[ZoneSpec], marks: MarkField, *, k: int = 2, centroid_radius_mm: float = 6.0) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.zones = sorted(zones, key=lambda z: z.region_id)
        self.marks = marks
        self.k = int(k)
        self.centroid_radius_mm = float(centroid_radius_mm)
        self.status: Dict[int, str] = {z.region_id: STATUS_OPEN for z in self.zones}
        self.events: List[QuorumEvent] = []

    def authors(self, zone: ZoneSpec, kind: str, tick: int) -> Set[int]:
        """Distinct ants with a live, visible ``kind`` mark that counts for ``zone``."""
        vmm = self.marks.params.voxel_mm
        if kind == "dose_ok":
            return {ant_of(s.sender) for s in self.marks.live(kind, tick)
                    if vmm * math.dist(s.pos, zone.centroid_vox) <= self.centroid_radius_mm * (1 + 1e-9)}
        return {ant_of(s.sender) for s in self.marks.live(kind, tick) if zone.contains(s.pos)}

    def evaluate(self, tick: int) -> List[QuorumEvent]:
        """Check every still-open zone against the marks visible at ``tick``; returns the new events (written once)."""
        new: List[QuorumEvent] = []
        for z in self.zones:
            if self.status[z.region_id] != STATUS_OPEN:
                continue
            for kind, status in (("dose_ok", "done"), ("not_tumour", "not_tumour")):
                who = self.authors(z, kind, tick)
                if len(who) >= self.k:
                    self.status[z.region_id] = status
                    ev = QuorumEvent(z.region_id, status, len(who), tick)
                    self.events.append(ev)
                    new.append(ev)
                    break
        return new
