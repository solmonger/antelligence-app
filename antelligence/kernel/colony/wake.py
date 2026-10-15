"""The quiet rule (goal 6.5, INTERFACES v4.2 section 7). OFF by default and OFF in every verdict arm.

When on, an ant gets a decision turn on a decision tick only if something happened since its last turn: a new or
strengthened mark within the read radius (sharing arms), a message to it, a sensor surprise (its close-look band differs
from the scan's p band at the same voxel, cuts 0.2 / 0.5 / 0.8, or a reading is slow / fast), it reached its goal voxel,
its payload changed, or its zone's need changed. A heartbeat gives a turn every ``heartbeat_rounds``. An ant without a
turn keeps its goal and does not release. Sensors are read on every decision tick: a skipped turn skips only the
decision. Answer turns never touch this tracker, so they never reset the change detector. Skipped turns are counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

BAND_CUTS = (0.2, 0.5, 0.8)
TRIGGERS = ("mark", "message", "surprise", "goal", "payload", "need")


def scan_band(p: float) -> int:
    """Band index 0..3 for a probability against the cuts 0.2 / 0.5 / 0.8."""
    return sum(1 for c in BAND_CUTS if p >= c)


def sensor_surprise(readings: Sequence[Mapping[str, Any]]) -> bool:
    """A look whose band differs from the scan band at that voxel, or a slow uptake / fast washout reading."""
    for r in readings:
        if r["kind"] == "look" and r.get("band") is not None and scan_band(float(r["value"])) != r["band"]:
            return True
        if (r["kind"] == "uptake" and r.get("word") in ("slow", "poor")) or (r["kind"] == "washout" and r.get("word") == "fast"):
            return True
    return False


@dataclass(frozen=True)
class WakeDecision:
    turn: bool
    reasons: tuple


class WakeRule:
    def __init__(self, n_ants: int, *, enabled: bool = False, heartbeat_rounds: int = 3) -> None:
        if heartbeat_rounds < 1:
            raise ValueError("heartbeat_rounds must be >= 1")
        self.enabled = bool(enabled)
        self.heartbeat_rounds = int(heartbeat_rounds)
        self.skipped: Dict[int, int] = {i: 0 for i in range(n_ants)}
        self._last_round: Dict[int, int] = {}

    def last_round(self, ant: int) -> Optional[int]:
        return self._last_round.get(ant)

    def decide(self, ant: int, round_idx: int, triggers: Mapping[str, bool]) -> WakeDecision:
        """Does ``ant`` take its decision turn this round? ``triggers`` holds the TRIGGERS flags (missing = False)."""
        if not self.enabled:
            return WakeDecision(True, ("off",))
        last = self._last_round.get(ant)
        if last is None:
            return WakeDecision(True, ("first",))
        reasons: List[str] = [t for t in TRIGGERS if triggers.get(t)]
        if round_idx - last >= self.heartbeat_rounds:
            reasons.append("heartbeat")
        return WakeDecision(bool(reasons), tuple(reasons))

    def record(self, ant: int, round_idx: int, turn: bool) -> None:
        if turn:
            self._last_round[ant] = round_idx
        else:
            self.skipped[ant] += 1
