"""Marks: field deposits with our own distinct-author aggregation (goal 6.1, INTERFACES v4.2 section 2).

A mark is a :class:`SignalDraft` stamped into a :class:`Signal` and held in a scoped :class:`GridField` (scope = arm
name, so arms never read each other). It is visible from the next tick (kernel rule ``emitted_at < tick``).

Aggregation is deliberately NOT ``GridField.intensity_at`` (linear distance falloff, sums repeat deposits). For one kind
at a query voxel: per author take the latest deposit within ``confirm_radius``; strength is the sum over those authors of
``amount * 0.5 ** ((t - tick) / half_life)``. A repeat deposit by the same author refreshes its own age and adds nothing.
Candidates are pulled with an explicit ``SenseQuery.max_results`` (the default of 64 would silently truncate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from antelligence.kernel.board import MARK_KINDS
from antelligence.kernel.field import GridField, SenseQuery
from antelligence.kernel.signal import Signal, SignalDraft

KIND_CODES: Dict[str, int] = {k: i + 1 for i, k in enumerate(MARK_KINDS)}
JUDGEMENT_KINDS = ("tumour_here", "not_tumour", "need_more")
AUTOMATIC_KINDS = ("dosed", "dose_ok", "slow_uptake", "washout")
DEFAULT_HALF_LIFE: Dict[str, float] = {
    "dosed": 30, "dose_ok": 40, "slow_uptake": 60, "washout": 60, "tumour_here": 40, "not_tumour": 80, "need_more": 15,
}
HALF_LIVES_TO_TTL = 8          # strength is negligible after 8 half-lives
MAX_RESULTS = 1 << 16          # explicit SenseQuery.max_results; a truncated pull raises instead of undercounting
STRONG_AUTHORS = 2
STRONG_STRENGTH = 1.0
_EPS = 1e-9

Vox = Tuple[int, int, int]


class MarkError(ValueError):
    pass


def ant_sender(ant: int) -> str:
    """INTERFACES section 0: ``ant-NN`` <-> world agent ``squad-N``; the signal sender is the squad id."""
    return f"squad-{int(ant)}"


def ant_of(sender: str) -> int:
    return int(sender.rsplit("-", 1)[1])


@dataclass(frozen=True)
class MarkParams:
    voxel_mm: float = 1.0
    read_radius_mm: float = 6.0
    confirm_radius_mm: float = 4.0
    half_life: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_HALF_LIFE))

    def __post_init__(self) -> None:
        if not (self.voxel_mm > 0 and self.read_radius_mm > 0 and self.confirm_radius_mm > 0):
            raise MarkError("voxel size and radii must be positive")
        hl = dict(self.half_life)
        if set(hl) != set(MARK_KINDS) or not all(v > 0 and math.isfinite(v) for v in hl.values()):
            raise MarkError(f"half_life needs a positive entry for exactly {MARK_KINDS}")
        object.__setattr__(self, "half_life", hl)

    def ttl(self, kind: str) -> int:
        return int(math.ceil(HALF_LIVES_TO_TTL * self.half_life[kind]))


@dataclass(frozen=True)
class MarkSummary:
    kind: str
    pos_vox: Vox
    strength: float
    distinct_authors: int
    authors: Tuple[int, ...]
    strong: bool

    def to_dict(self, disputed: bool = False) -> Dict[str, Any]:
        return {"kind": self.kind, "pos_vox": list(self.pos_vox), "strength": round(self.strength, 6),
                "distinct_authors": self.distinct_authors, "strong": self.strong, "disputed": bool(disputed),
                "authors": list(self.authors)}


def _vox(pos: Sequence[float]) -> Vox:
    return tuple(int(round(c)) for c in pos)  # type: ignore[return-value]


class MarkField:
    """One arm's marks. ``records`` is the append-only deposit log (board payload plus ``tick``), in deposit order."""

    def __init__(self, scope: str, params: Optional[MarkParams] = None, *, max_live: int = 100_000) -> None:
        self.scope = scope
        self.params = params or MarkParams()
        self._field = GridField(max_live)
        self.records: List[Dict[str, Any]] = []

    # --- writes -----------------------------------------------------------------------------------------------

    def deposit(self, ant: int, kind: str, pos_vox: Sequence[int], tick: int, *, amount: float = 1.0,
                reading_ppm: int = 0) -> Optional[Dict[str, Any]]:
        """Stamp and store one mark. Returns the record (board payload + ``tick``), or None for an exact duplicate.

        ``amount`` is quantised to milli-units so the signal, the board entry and a replay agree exactly.
        """
        if kind not in KIND_CODES:
            raise MarkError(f"unknown mark kind {kind!r}")
        if not (math.isfinite(amount) and amount >= 0):
            raise MarkError("amount must be finite and >= 0")
        if not (isinstance(reading_ppm, int) and 0 <= reading_ppm <= 1_000_000):
            raise MarkError("reading_ppm must be an int in [0, 1_000_000]")
        pos = _vox(pos_vox)
        milli = int(round(amount * 1000))
        draft = SignalDraft(kind=kind, pos=tuple(float(c) for c in pos),
                            payload={"ant": int(ant), "amount": milli / 1000.0, "tick": int(tick)},
                            ttl=self.params.ttl(kind))
        signal = Signal.stamp(draft, sender=ant_sender(ant), scope=self.scope, tick=int(tick), revision=0)
        if not self._field.deposit(signal):
            return None
        rec = {"subtype": "mark", "mark_kind": kind, "ant": int(ant), "pos_vox": list(pos), "amount_milli": milli,
               "reading_ppm": int(reading_ppm), "tick": int(tick)}
        self.records.append(rec)
        return dict(rec)

    def expire(self, tick: int) -> None:
        """Drop marks that are no longer live at ``tick``."""
        self._field.expire(tick - 1)

    # --- reads (all take the reader's tick; same-tick deposits are invisible) --------------------------------

    def _dist_mm(self, a: Sequence[float], b: Sequence[float]) -> float:
        return self.params.voxel_mm * math.dist(a, b)

    def near(self, pos_vox: Sequence[float], tick: int, radius_mm: float,
             kinds: Optional[Iterable[str]] = None) -> List[Signal]:
        """Live marks within ``radius_mm`` (millimetres) of ``pos_vox``, nearest first."""
        r_vox = radius_mm / self.params.voxel_mm
        query = SenseQuery(pos=tuple(float(c) for c in pos_vox), radius=r_vox * (1 + _EPS) + _EPS,
                           kinds=frozenset(kinds) if kinds is not None else None, max_results=MAX_RESULTS)
        hits = self._field.sense(self.scope, tick, query)
        if len(hits) >= MAX_RESULTS:
            raise MarkError("mark pull hit max_results; refusing to undercount")
        return [s for s in hits if self._dist_mm(s.pos, pos_vox) <= radius_mm * (1 + _EPS)]

    def live(self, kind: str, tick: int) -> List[Signal]:
        return [s for s in self._field.snapshot(self.scope, tick) if s.kind == kind]

    def aggregate(self, kind: str, pos_vox: Sequence[float], tick: int) -> MarkSummary:
        """Distinct-author strength of ``kind`` around ``pos_vox`` (confirm radius). See the module docstring."""
        latest: Dict[int, Signal] = {}
        for s in self.near(pos_vox, tick, self.params.confirm_radius_mm, (kind,)):
            a = ant_of(s.sender)
            cur = latest.get(a)
            if cur is None or (s.emitted_at, s.id) > (cur.emitted_at, cur.id):
                latest[a] = s
        hl = self.params.half_life[kind]
        strength = 0.0
        for a in sorted(latest):
            s = latest[a]
            strength += float(s.payload["amount"]) * 0.5 ** ((tick - s.emitted_at) / hl)
        authors = tuple(sorted(latest))
        strong = len(authors) >= STRONG_AUTHORS and strength >= STRONG_STRENGTH
        return MarkSummary(kind, _vox(pos_vox), strength, len(authors), authors, strong)

    def disputed_at(self, pos_vox: Sequence[float], tick: int) -> bool:
        """A ``tumour_here`` and a ``not_tumour`` both within the confirm radius of this spot."""
        return bool(self.near(pos_vox, tick, self.params.confirm_radius_mm, ("tumour_here",))
                    and self.near(pos_vox, tick, self.params.confirm_radius_mm, ("not_tumour",)))

    def marks_near(self, pos_vox: Sequence[float], tick: int) -> List[Dict[str, Any]]:
        """The ``marks_near`` observation value: one row per (kind, mark voxel) inside the read radius."""
        spots = sorted({(MARK_KINDS.index(s.kind), _vox(s.pos)) for s in self.near(pos_vox, tick, self.params.read_radius_mm)})
        rows = []
        for ki, spot in spots:
            rows.append(self.aggregate(MARK_KINDS[ki], spot, tick).to_dict(self.disputed_at(spot, tick)))
        return rows

    def disputed_spots(self, pos_vox: Sequence[float], radius_mm: float, tick: int) -> List[Vox]:
        spots = sorted({_vox(s.pos) for s in self.near(pos_vox, tick, radius_mm, ("tumour_here", "not_tumour"))})
        return [p for p in spots if self.disputed_at(p, tick)]

    def latest_by_other(self, pos_vox: Sequence[float], tick: int, exclude_ant: int) -> Optional[int]:
        """Author of the most recent mark within the read radius, ignoring ``exclude_ant``; ties go to the lower ant."""
        best: Optional[Tuple[int, int]] = None
        for s in self.near(pos_vox, tick, self.params.read_radius_mm):
            a = ant_of(s.sender)
            if a == exclude_ant:
                continue
            key = (-s.emitted_at, a)
            if best is None or key < best:
                best = key
        return None if best is None else best[1]

    def has_own(self, ant: int, kind: str, pos_vox: Sequence[float], tick: int, radius_mm: Optional[float] = None) -> bool:
        r = self.params.read_radius_mm if radius_mm is None else radius_mm
        return any(ant_of(s.sender) == ant for s in self.near(pos_vox, tick, r, (kind,)))

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.records:
            out[r["mark_kind"]] = out.get(r["mark_kind"], 0) + 1
        return dict(sorted(out.items()))
