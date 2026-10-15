"""The scripted queen (goal-setter), v4.2 goal section 7.

She acts at tick 0 and every 30 ticks and is passed to the colony driver as ``queen=``. ``world.queen_hook`` (the v4.1
path, outside the information barrier) stays UNSET in every colony arm.

What she reads (board entries only, INTERFACES 14): the ``dosed``, ``dose_ok``, ``tumour_here`` and ``not_tumour`` marks
(only ``dosed`` and the judgement marks are used) and the quorum events. She never reads ``world.field.dose``, the scorer
or the F1-F3 fields. In S-solo the driver gives her no board, so her need stays at the tick-0 scan value.

What she computes, per zone (a perceived region):

* allotment = total drug x zone scan mass / total scan mass, scan mass = p x volume (the region's mean calibrated p times its
  volume, the perception payload);
* delivered share = min(1, sum of the zone's ``dosed`` amounts / allotment), from board marks only (a ``dosed`` amount is the
  fraction of one squad's share, so drug = amount x total_drug / n_squads);
* need = scan mass x (1 - delivered share), normalised by the maximum over zones. Bands: none < 0.05, low < 0.25,
  medium < 0.6, high >= 0.6.
* v4.3 B3 (``uptake_scaling=True``, sharing arms only; the harness sets it): the RAW need (before the normalisation) is
  multiplied by 1.25 when more than half of the zone's uptake-mark strength is ``good_uptake``, by 0.5 when more than half
  is ``slow_uptake``, else by 1 (also 1 with no uptake marks). Strength = sum over the live marks inside the zone of
  amount x 0.5 ** ((tick - mark tick) / half_life[kind]). She still reads board marks only. The factor is returned per zone
  in ``need_rows`` (``uptake_factor_milli``) and only when the scaling is on.

Alarms (board ``alarm`` type, ``kind: "other"``): ``stall`` when the summed delivered share is unchanged over 2 queen steps (fires when a stall begins; again only after the
delivered share has changed),
``dispute:<region_id>`` when a zone has been disputed for more than 30 ticks. The colony note is a fixed template. She never
sets options, routes or gates: she returns only ``zone_need``, ``colony_note`` and ``need_rows``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.kernel.colony.marks import DEFAULT_HALF_LIFE, complete_half_life
from antelligence.worlds.tumor_mri.colony_policies import ScanInfo
from antelligence.worlds.tumor_mri.data import Region

NEED_CUTS = ((0.05, "none"), (0.25, "low"), (0.6, "medium"))      # band = first cut the value is below, else "high"
DISPUTE_TICKS = 30
STALL_STEPS = 2
CONFIRM_MM = 4.0
QUEEN_KINDS = ("dosed", "tumour_here", "not_tumour")             # of the four mark kinds she may read, the ones she uses
NOTE_MAX = 200
UPTAKE_KINDS = ("good_uptake", "slow_uptake")                    # read only when the B3 scaling is on
UPTAKE_GOOD_FACTOR, UPTAKE_POOR_FACTOR = 1.25, 0.5


def need_band(x: float) -> str:
    for cut, name in NEED_CUTS:
        if x < cut:
            return name
    return "high"


class ScriptedQueen:
    """``queen(tick, inputs) -> {"zone_need", "colony_note", "need_rows"}``. Deterministic; draws nothing.

    ``regions`` and ``total_drug`` are the scan side of the world (perception only); ``board`` is optional and receives the
    alarms (the driver posts the need rows itself)."""

    def __init__(self, regions: Sequence[Region], total_drug: float, n_squads: int, voxel_mm: float, *,
                 board: Any = None, half_life: Optional[Mapping[str, float]] = None,
                 uptake_scaling: bool = False) -> None:
        if total_drug <= 0 or n_squads < 1:
            raise ValueError("total_drug and n_squads must be positive")
        self.scan = ScanInfo(tuple(regions), float(voxel_mm))
        self.total_drug, self.n_squads, self.vmm = float(total_drug), int(n_squads), float(voxel_mm)
        self.board = board
        self.half_life = complete_half_life(half_life or DEFAULT_HALF_LIFE)
        self.uptake_scaling = bool(uptake_scaling)
        self.mass = {r.region_id: float(r.p_tumour) * float(r.volume_mm3) for r in self.scan.regions}
        self.total_mass = sum(self.mass.values())
        self.alarms: List[Dict[str, Any]] = []
        self._delivered_milli: List[int] = []                 # summed delivered share per queen step, milli
        self._stall_fired = False                              # a stall alarm is out; re-armed when the delivered share changes
        self._dispute_since: Dict[int, int] = {}
        self._dispute_alarmed: set = set()
        self.reads: List[Tuple[int, Tuple[str, ...]]] = []    # (tick, input keys read): the barrier test walks this

    # --- computation ---------------------------------------------------------------------------------------------------

    def allotment(self, region_id: int) -> float:
        return self.total_drug * self.mass[region_id] / self.total_mass if self.total_mass > 0 else 0.0

    def delivered_share(self, marks: Sequence[Mapping[str, Any]]) -> Dict[int, float]:
        drug: Dict[int, float] = {rid: 0.0 for rid in self.mass}
        per_squad = self.total_drug / self.n_squads
        for m in marks:
            if m["mark_kind"] != "dosed":
                continue
            rid = self.scan.zone_at(m["pos_vox"])
            if rid is not None:
                drug[rid] += m["amount_milli"] / 1000.0 * per_squad
        out = {}
        for rid, d in drug.items():
            a = self.allotment(rid)
            out[rid] = 1.0 if a <= 0 else min(1.0, d / a)
        return out

    def uptake_factors(self, marks: Sequence[Mapping[str, Any]], tick: int) -> Dict[int, float]:
        """B3: per zone, 1.25 / 0.5 / 1 from the strength of the live uptake marks inside it (see the module docstring)."""
        strength: Dict[int, Dict[str, float]] = {rid: {k: 0.0 for k in UPTAKE_KINDS} for rid in self.mass}
        for m in marks:
            k = m["mark_kind"]
            if k not in UPTAKE_KINDS or m["tick"] >= tick or tick - m["tick"] > math.ceil(8 * self.half_life[k]):
                continue
            rid = self.scan.zone_at(m["pos_vox"])
            if rid is not None:
                strength[rid][k] += m["amount_milli"] / 1000.0 * 0.5 ** ((tick - m["tick"]) / self.half_life[k])
        out: Dict[int, float] = {}
        for rid, s in strength.items():
            total = s["good_uptake"] + s["slow_uptake"]
            out[rid] = (UPTAKE_GOOD_FACTOR if s["good_uptake"] > total / 2 else
                        UPTAKE_POOR_FACTOR if s["slow_uptake"] > total / 2 else 1.0) if total > 0 else 1.0
        return out

    def _disputed_zones(self, marks: Sequence[Mapping[str, Any]], tick: int) -> Dict[int, int]:
        """Zones with a live ``tumour_here`` and ``not_tumour`` within the confirm radius of each other, mapped to the tick
        the (earliest) pair began: the later of the two deposits."""
        live = {k: [m for m in marks if m["mark_kind"] == k
                    and tick - m["tick"] <= math.ceil(8 * self.half_life[k])] for k in ("tumour_here", "not_tumour")}
        out: Dict[int, int] = {}
        for a in live["tumour_here"]:
            for b in live["not_tumour"]:
                if self.vmm * math.dist(a["pos_vox"], b["pos_vox"]) <= CONFIRM_MM * (1 + 1e-9):
                    rid = self.scan.zone_at(a["pos_vox"])
                    if rid is not None:
                        began = max(a["tick"], b["tick"])
                        out[rid] = min(out.get(rid, began), began)
        return out

    def _alarm(self, tick: int, value_ppm: int, note: str) -> None:
        rec = {"tick": tick, "kind": "other", "value_ppm": int(value_ppm), "note": note}
        self.alarms.append(rec)
        if self.board is not None:
            self.board.append("alarm", "queen", {"kind": "other", "value_ppm": int(value_ppm), "note": note}, tick)

    # --- the driver's queen interface ----------------------------------------------------------------------------------

    def __call__(self, tick: int, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        keys = tuple(sorted(inputs))
        self.reads.append((tick, keys))
        marks = [m for m in inputs.get("marks", ()) if m["mark_kind"] in QUEEN_KINDS and m["tick"] < tick]
        events = list(inputs.get("quorum_events", ()))
        share = self.delivered_share(marks)
        raw = {rid: self.mass[rid] * (1.0 - share[rid]) for rid in self.mass}
        factor: Dict[int, float] = {}
        if self.uptake_scaling:
            factor = self.uptake_factors([m for m in inputs.get("marks", ()) if m["mark_kind"] in UPTAKE_KINDS], tick)
            raw = {rid: v * factor[rid] for rid, v in raw.items()}
        top = max(raw.values(), default=0.0)
        norm = {rid: (v / top if top > 0 else 0.0) for rid, v in raw.items()}
        rows = [{"region_id": rid, "need": need_band(norm[rid]), "need_milli": int(round(norm[rid] * 1000)),
                 "delivered_milli": int(round(share[rid] * 1000))} for rid in sorted(self.mass)]
        if self.uptake_scaling:
            for r in rows:
                r["uptake_factor_milli"] = int(round(factor[r["region_id"]] * 1000))
        zone_need = {r["region_id"]: r["need"] for r in rows}

        summed = int(round(sum(share.values()) * 1000))
        self._delivered_milli.append(summed)
        h = self._delivered_milli
        if len(h) > 1 and h[-1] != h[-2]:
            self._stall_fired = False                                 # delivered share changed since: a new stall may alarm
        stalled = len(h) > STALL_STEPS and h[-1] == h[-1 - STALL_STEPS] and h[-1] < 1000 * len(self.mass)
        if stalled and not self._stall_fired:
            self._stall_fired = True                                  # fire once when the stall begins, not at every step
            self._alarm(tick, summed * 1000, "stall")                 # value_ppm = summed delivered share, ppm
        disputed = self._disputed_zones(marks, tick)
        for rid in list(self._dispute_since):
            if rid not in disputed:
                self._dispute_since.pop(rid)
                self._dispute_alarmed.discard(rid)
        for rid, began in sorted(disputed.items()):
            self._dispute_since.setdefault(rid, began)
            if tick - self._dispute_since[rid] > DISPUTE_TICKS and rid not in self._dispute_alarmed:
                self._dispute_alarmed.add(rid)
                self._alarm(tick, tick - self._dispute_since[rid], f"dispute:{rid}")   # value_ppm = ticks disputed (raw)

        count = lambda band: sum(1 for r in rows if r["need"] == band)
        total_allot = sum(self.allotment(rid) for rid in self.mass)
        pct = 0 if total_allot <= 0 else int(round(100 * sum(share[r] * self.allotment(r) for r in self.mass) / total_allot))
        done = sum(1 for e in events if e["status"] == "done")
        nt = sum(1 for e in events if e["status"] == "not_tumour")
        note = (f"Colony note, tick {tick}: {pct}% of the drug allotment delivered. Need: high {count('high')}, "
                f"medium {count('medium')}, low {count('low')}, none {count('none')}. Zones done {done}, not tumour {nt}.")
        return {"zone_need": zone_need, "colony_note": note[:NOTE_MAX], "need_rows": rows}
