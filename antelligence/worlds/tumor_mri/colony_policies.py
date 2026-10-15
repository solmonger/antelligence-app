"""The colony scripts: S-solo, S-marks, S-talk share ONE decision function (v4.2 goal section 7).

Fairness rule. The three arms run the same code, :func:`decide_turn`. They differ only in which NAMED inputs the
observation carries. The shared inputs are ``marks`` (obs key ``marks_near``), ``region_status`` and ``messages``
(obs key ``messages_in``); in S-solo they are absent (``None``) and every rule that uses one has an own-readings
fallback, so with the inputs absent the function IS the S-solo script. Nothing branches on the arm's name. Decision
turns are identical across the arms (the quiet rule is off in verdict arms; ``S-marks-quiet`` is its own pair).

What one turn does, in order:

1. Settle the plan chosen at the previous turn (job thresholds, INTERFACES 10): dose succeeds if the release was made on a
   look >= 0.5 and the uptake reading about it is not slow; explore if it found a look >= 0.65 where the scan said < 0.5 or
   <= 0.3 where the scan said >= 0.5; verify if the mean of its logged looks within 4 mm of the spot left [0.35, 0.65].
   Success: theta - 0.1 (floor 0.1); a turn without success: theta + 0.05 (cap 1.0).
2. S-talk only: fold in the answer to its pending ask, ``mean(estimate, 0.8 yes / 0.2 no)`` (unknown is ignored). On THIS
   turn the combined estimate REPLACES the look for that pending target in the release test (release needs estimate >= 0.5,
   the look rule's threshold; every other release condition is unchanged) and in dose-target eligibility (the target is
   eligible only at estimate >= 0.5). Nothing is dropped for good and no constant is added. Judgement marks keep the raw
   look. S-solo / S-marks never receive an answer, so for them this step does nothing.
3. Stimuli, each in [0, 1]: dose = max over zones of normalised need x scan p; explore = share of perceived regions not
   visited; verify = 1 if a grey or disputed spot lies within 20 mm. ``P_j = s_j^2 / (s_j^2 + theta_j^2)``, highest wins,
   ties dose > verify > explore.
4. Release rule (dose job; the explore job releases only if the look >= 0.65; verify never): release if the look is >= 0.5
   and its own dose reading is below D_t. 50% of the payload, 75% if its own uptake reading near this zone was slow.
   Skip releasing and move on if its own washout reading here is fast and the look is below 0.65 (the only move-on).
5. Judgement marks in every arm (read only in sharing arms): ``tumour_here`` (look >= 0.65 where scan p < 0.5, or look
   >= 0.8 anywhere), ``not_tumour`` (look <= 0.3 where scan p >= 0.5), ``need_more`` (own dose below D_t, look >= 0.5
   and payload empty).
6. S-talk only: when unsure (target estimate in [0.35, 0.65], or target disputed) it takes the S-marks action and also
   sends one ``is_tumour`` ask about its target.

S-marks adjustments (inputs that are None in S-solo): a strong ``not_tumour`` (region status) or ``done`` zone is skipped;
a strong ``dosed`` or ``dose_ok`` at the spot means "dose already high here"; ``tumour_here`` and ``need_more`` spots from
other ants are extra dose targets; others' ``slow_uptake`` and ``washout`` marks count exactly as own readings; a region
counts as visited if its status is not open or a visible mark lies in it.

v4.3. Every constant above is read from an :class:`AntRules` carried by ``AntState`` (``AntRules.v42()`` by default, which
reproduces everything described here exactly). With ``AntRules.v43()`` the ant works on the combined estimate ``c`` (the ``est``
of its own look at this tick, else the raw look), releases only at ``c >= 0.6`` (explore also needs scan p >= 0.5), lets a
fold-in only lower ``c``, bounds the verify job, and moves a job threshold only on a turn whose outcome was resolved. There is
still ONE decision function for every arm; no branch reads an arm's name.

A2r1 (``rules.dose_move_on``). A dose-job ant that still has payload but stands where its own dose reading is already >= D_t
(release reason ``dose_ok``) moves on inside the zone of its target: to the nearest voxel with scan p >= 0.5 that is >= 6 mm
from every spot it knows is dosed (:meth:`ScanInfo.zone_voxels`, ``move_on.nearest_new_spot``). That spot is a persistent dose
goal (``AntState.dose_goal``) that replaces the zone centroid as the dose target until the ant is within 1 voxel of it, 6
decision turns have passed, or the zone is skipped. With no spot left the zone is skipped. Known spots are the ant's own
release voxels, its own dose readings >= 1, its voxel, and, through ``marks`` (absent in S-solo), visible ``dosed`` /
``dose_ok`` marks. It needs the scan map (``ScanInfo.p_tumour``); without it the switch has no effect.

B2 (``rules.foraging``, :func:`uptake_info`, :func:`_forage_forced`, :func:`_forage_release`, :func:`_poor_response`). The probe-and-hold
rule: at a tumour spot (the ordinary release rule says ok, dose or explore job) with NO uptake information within 6 mm the ant
releases a 10% probe and holds on the voxel (``AntState.hold``); at its next decision turn the uptake reading is in its log and the
job is forced to dose there: good tissue (reading >= 0.7) 50%, middling 25% (and ``AntState.stay`` keeps dosing that voxel at that
amount while its own dose < D_t, c >= 0.6 and payload is left), poor (< 0.4) no release, the zone's value halved once in the ant's own
ranking (``AntState.zone_mult``) and the nearest unprobed voxel of the zone becomes the A2r1 dose goal (none left: the zone is
skipped). Uptake information is the nearest of an own uptake reading and, ONLY when the ``marks`` input exists (sharing arms), a
foreign ``good_uptake`` / ``slow_uptake`` mark; a foreign mark that saves a probe is counted (``probes_saved_by_marks``). No new input.

Judgement calls and defaults chosen where the goal text is silent are listed in ``DEFAULTS_CHOSEN`` and in the Q1 report.
Every random draw in the colony is seeded per INTERFACES 1; this module draws none.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from antelligence.kernel.colony.thresholds import Thresholds
from antelligence.kernel.signal import SignalDraft
from antelligence.kernel.types import Intent, LocalView
from antelligence.worlds.tumor_mri.ant_rules import AntRules
from antelligence.worlds.tumor_mri.data import Region
from antelligence.worlds.tumor_mri.move_on import nearest_new_spot, zone_candidates
from antelligence.worlds.tumor_mri.questions import EXPLORE, STAY
from antelligence.worlds.tumor_mri.world import ACTION_DECIDE

Vox = Tuple[int, int, int]

READ_MM = 6.0                 # read radius (INTERFACES 2.1)
CONFIRM_MM = 4.0              # confirm radius
NEAR_MM = 20.0                # verify stimulus: a grey or disputed spot within 20 mm (goal 6.8)
GREY = (0.35, 0.65)           # unsure / grey band (goal 6.8, 7)
RELEASE_LOOK = 0.5
EXPLORE_RELEASE_LOOK = 0.65
WASHOUT_RELEASE_LOOK = 0.65
TUMOUR_NEW_LOOK, TUMOUR_ANY_LOOK, NOT_TUMOUR_LOOK, SCAN_HALF = 0.65, 0.8, 0.3, 0.5
LEVEL_HALF, LEVEL_THREE_QUARTERS = 3, 4          # amount levels 3 and 4 are 50% and 75% (AMOUNT_FRACTIONS)
ASK_EVERY_ROUNDS = 2                              # INTERFACES 3.2 budget; the script stays inside it
ANSWER_VALUE = {"yes": 0.8, "no": 0.2}
ZONE_MARGIN_MM = 6.0
MOVE_ON_P_MIN = 0.5           # A2r1: a move-on spot needs scan p >= this
MOVE_ON_TURNS = 6             # A2r1: a dose goal lapses after this many decision turns
MOVE_ON_ARRIVE_VOX = 1.0      # A2r1: ... and is reached within this many voxels

# --- defaults the goal text did not give (see the Q1 report) -----------------------------------------------------------
EMPTY_FRACTION = 0.05         # "payload empty": payload_fraction below this (a squad never reaches exactly 0)
NEED_NORM = {"none": 0.0, "low": 0.15, "medium": 0.425, "high": 0.8}   # midpoints of the need bands (<.05, <.25, <.6, >=.6)
DEFAULTS_CHOSEN = {
    "empty_payload_fraction": EMPTY_FRACTION,
    "need_band_to_number": NEED_NORM,
    "washout_here_radius_mm": CONFIRM_MM,
    "uptake_near_zone_radius_mm": READ_MM,
    "look_missing_fallback": "scan p",
    "mark_spot_value": "need band of its zone (high outside any zone) x min(1, mark strength)",
}


@dataclass(frozen=True)
class ScanInfo:
    """What the scan tells every ant, arm-independent: the perceived regions (perception only, never truth)."""

    regions: Tuple[Region, ...]
    voxel_mm: float
    # The scan's own p map (perception, every ant sees it; no truth). Only the dose move-on reads it; None switches that rule off.
    p_tumour: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    _zone_cache: Dict[Tuple[int, float], np.ndarray] = field(default_factory=dict, compare=False, repr=False)

    def zone_voxels(self, region_id: int, p_min: float = MOVE_ON_P_MIN) -> np.ndarray:
        """(n, 3) voxels of the zone of ``region_id`` with scan p >= ``p_min`` (cached: a pure function of the scan)."""
        key = (region_id, p_min)
        hit = self._zone_cache.get(key)
        if hit is None:
            r = self.region(region_id)
            if self.p_tumour is None or r is None:
                hit = np.zeros((0, 3), dtype=np.int64)
            else:
                hit = zone_candidates(self.p_tumour, r.centroid_vox, self.zone_radius_mm(r), self.voxel_mm, p_min)
            self._zone_cache[key] = hit
        return hit

    def zone_radius_mm(self, r: Region) -> float:
        """Equal-volume sphere radius plus 6 mm: the same approximation the kernel uses for a zone (INTERFACES 14)."""
        return (3.0 * r.volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0) + ZONE_MARGIN_MM

    def zone_at(self, pos: Sequence[float]) -> Optional[int]:
        best: Optional[Tuple[float, int]] = None
        for r in self.regions:
            d = self.voxel_mm * math.dist(pos, r.centroid_vox)
            if d <= self.zone_radius_mm(r) * (1 + 1e-9) and (best is None or (d, r.region_id) < best):
                best = (d, r.region_id)
        return None if best is None else best[1]

    def region(self, region_id: int) -> Optional[Region]:
        return next((r for r in self.regions if r.region_id == region_id), None)


@dataclass(frozen=True)
class TurnInputs:
    """The named inputs of one decision turn. The last three are the SHARED inputs: ``None`` in S-solo."""

    tick: int
    round: int
    pos: Vox
    voxel_mm: float
    payload_fraction: float
    scan_p: float
    look: Optional[float]
    dose_rel: float
    candidates: Sequence[Mapping[str, Any]]
    reading_log: Sequence[Mapping[str, Any]]
    uptake: Optional[Mapping[str, Any]]
    visited_regions: FrozenSet = frozenset()
    zone_need: Mapping[int, str] = field(default_factory=dict)
    marks: Optional[Sequence[Mapping[str, Any]]] = None            # obs key marks_near       (sharing arms)
    region_status: Optional[Mapping[int, str]] = None              # obs key region_status    (sharing arms)
    messages: Optional[Sequence[Mapping[str, Any]]] = None         # obs key messages_in      (S-talk)
    est: Optional[float] = None                                    # v4.3: ``est`` of the own look logged at this tick


def _est_at_tick(log: Sequence[Mapping[str, Any]], tick: int) -> Optional[float]:
    """The ``est`` of the own look logged at ``tick`` (the log is in tick order), or None when there is none."""
    for e in reversed(log):
        if e["tick"] < tick:
            break
        if e["kind"] == "look" and e["tick"] == tick and e.get("est") is not None:
            return float(e["est"])
    return None


def inputs_from_observation(obs: Mapping[str, Any], voxel_mm: float) -> TurnInputs:
    """Read a colony observation into the named inputs. A key the arm may not see is absent and becomes ``None``."""
    status = obs.get("region_status")
    est = _est_at_tick(obs.get("reading_log", ()), int(obs["tick"]))
    return TurnInputs(
        tick=int(obs["tick"]), round=int(obs["round"]), pos=tuple(int(c) for c in obs["position_vox"]),  # type: ignore[arg-type]
        voxel_mm=float(voxel_mm), payload_fraction=float(obs["payload_fraction"]), scan_p=float(obs["local_p_tumour"]),
        look=obs.get("look", obs.get("local_look")), dose_rel=float(obs["local_dose_rel"]),
        candidates=obs["candidates"], reading_log=obs.get("reading_log", ()), uptake=obs.get("uptake_reading"),
        visited_regions=frozenset((obs.get("visited_self") or {}).get("regions", ())),
        zone_need={int(k): v for k, v in (obs.get("zone_need") or {}).items()},
        marks=obs.get("marks_near"), region_status=None if status is None else {int(k): v for k, v in status.items()},
        messages=obs.get("messages_in"), est=est)


@dataclass
class AntState:
    """Per-ant script memory. Reset by building a new one; never shared between ants."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    rules: AntRules = field(default_factory=AntRules.v42)
    skip_zones: Set[int] = field(default_factory=set)
    skip_spots: Set[Vox] = field(default_factory=set)
    pending: Optional[Dict[str, Any]] = None              # S-talk: {"pos", "region_id", "estimate", "round"}
    last_ask_round: Optional[int] = None
    prev: Optional[Dict[str, Any]] = None                  # the plan of the previous turn, settled at this turn
    counters: Counter = field(default_factory=Counter)
    decision_turns: int = 0                                 # this ant's own previous decision turns (v4.3 verify share)
    verify_turns: int = 0                                   # ... of which verify turns with a target spot
    verify_by_spot: Counter = field(default_factory=Counter)   # verify turns taken per spot (v4.3 per-spot cap)
    released_at: List[Vox] = field(default_factory=list)       # A2r1: voxels where this ant released (its own previous decisions)
    dose_goal: Optional[Dict[str, Any]] = None                 # A2r1: {"spot", "zone", "age"}, the persistent dose goal
    hold: Optional[Dict[str, Any]] = None                      # B2: {"pos", "age", "tick"}, a probe released and its reading awaited
    stay: Optional[Dict[str, Any]] = None                      # B2: {"pos", "cls"}, keep dosing this voxel at the class's amount
    zone_mult: Dict[int, float] = field(default_factory=dict)  # B2: zone -> factor on its value in this ant's own ranking


@dataclass(frozen=True)
class Decision:
    job: str
    target_label: str
    target_vox: Optional[Vox]
    release: bool
    level: int
    marks: Tuple[Tuple[str, Vox], ...]
    ask: Optional[Dict[str, Any]]
    info: Mapping[str, Any]


# --- small geometry helpers ---------------------------------------------------------------------------------------------

def _mm(vm: float, a: Sequence[float], b: Sequence[float]) -> float:
    return vm * math.dist(a, b)


def _vox(p: Sequence[float]) -> Vox:
    return tuple(int(round(c)) for c in p)  # type: ignore[return-value]


_V42 = AntRules.v42()


def _lv(e: Mapping[str, Any], rules: AntRules) -> float:
    """The value of a logged look under ``rules``: its combined estimate ``est`` when the rules use it, else the raw look."""
    return float(e["est"]) if rules.use_combined and e.get("est") is not None else float(e["value"])


def _looks_near(log: Iterable[Mapping[str, Any]], pos: Sequence[float], radius_mm: float, vm: float,
                rules: AntRules = _V42) -> List[float]:
    return [_lv(e, rules) for e in log if e["kind"] == "look" and _mm(vm, e["pos_vox"], pos) <= radius_mm * (1 + 1e-9)]


def _grey(x: float, grey: Tuple[float, float] = GREY) -> bool:
    return grey[0] <= x <= grey[1]


def _current_c(inp: TurnInputs, rules: AntRules, look: float) -> float:
    """The estimate ``c`` this turn acts on: under the combined rules the ``est`` of the own look logged at this tick (given
    directly, or found in the reading log); otherwise, and as the fallback, the raw look."""
    if not rules.use_combined:
        return look
    if inp.est is not None:
        return float(inp.est)
    est = _est_at_tick(inp.reading_log, inp.tick)
    return look if est is None else est


def _estimate(inp: TurnInputs, target: Sequence[float], look: float, rules: AntRules = _V42) -> Tuple[float, bool]:
    """The target estimate: mean of its own logged looks (their ``est`` under the combined rules) within 6 mm of the target,
    else the current ``c``. Returns (estimate, own_looks_exist)."""
    near = _looks_near(inp.reading_log, target, READ_MM, inp.voxel_mm, rules)
    return (sum(near) / len(near), True) if near else (look, False)


# --- readings that change the release rule ------------------------------------------------------------------------------

def _slow_uptake_near(inp: TurnInputs, pos: Sequence[float]) -> bool:
    """Own uptake reading near this zone was slow, or (sharing) another ant's ``slow_uptake`` mark within the read radius."""
    vm = inp.voxel_mm
    own = any(e["kind"] == "uptake" and e.get("word") == "slow" and _mm(vm, e["pos_vox"], pos) <= READ_MM * (1 + 1e-9)
              for e in inp.reading_log)
    return own or any(m["kind"] == "slow_uptake" for m in (inp.marks or ()) if _mm(vm, m["pos_vox"], pos) <= READ_MM * (1 + 1e-9))


def _washout_fast_here(inp: TurnInputs, pos: Sequence[float]) -> bool:
    vm = inp.voxel_mm
    own = any(e["kind"] == "washout" and e.get("word") == "fast" and _mm(vm, e["pos_vox"], pos) <= CONFIRM_MM * (1 + 1e-9)
              for e in inp.reading_log)
    return own or any(m["kind"] == "washout" for m in (inp.marks or ()) if _mm(vm, m["pos_vox"], pos) <= CONFIRM_MM * (1 + 1e-9))


def _shared_dose_high(inp: TurnInputs, pos: Sequence[float]) -> bool:
    """A strong ``dosed`` or ``dose_ok`` at the spot: 'dose already high here' (S-marks)."""
    vm = inp.voxel_mm
    return any(m["kind"] in ("dosed", "dose_ok") and m["strong"] and _mm(vm, m["pos_vox"], pos) <= CONFIRM_MM * (1 + 1e-9)
               for m in (inp.marks or ()))


def release_rule(inp: TurnInputs, job: str, look: float, pos: Sequence[float],
                 rules: AntRules = _V42) -> Tuple[bool, int, str]:
    """(release?, amount level, reason). Goal 7: look >= 0.5 and own dose below D_t; 50% of the payload, 75% if slow uptake;
    skip if own washout here is fast and the look is below 0.65. The explore job needs a look >= 0.65; verify never releases.
    ``look`` is the estimate the release test sees (v4.3: ``c``, possibly lowered by a fold-in); the thresholds are the
    rules' (v4.3: 0.6 for dose and explore, and the explore job also needs scan p >= 0.5)."""
    if job == "verify":
        return False, LEVEL_HALF, "verify"
    if inp.payload_fraction < EMPTY_FRACTION:
        return False, LEVEL_HALF, "payload_empty"
    if look < (rules.explore_release_c if job == "explore" else rules.release_c):
        return False, LEVEL_HALF, "look_low"
    if job == "explore" and inp.scan_p < rules.explore_scan_min:
        return False, LEVEL_HALF, "scan_low"
    if inp.dose_rel >= 1.0:
        return False, LEVEL_HALF, "dose_ok"
    if _shared_dose_high(inp, pos):
        return False, LEVEL_HALF, "shared_dose_high"
    if _washout_fast_here(inp, pos) and look < rules.washout_release_c:
        return False, LEVEL_HALF, "washout"
    return True, (LEVEL_THREE_QUARTERS if _slow_uptake_near(inp, pos) else LEVEL_HALF), "ok"


def judgement_marks(inp: TurnInputs, look: float, rules: AntRules = _V42) -> List[str]:
    """Judgement marks placed at the ant's own voxel, in every arm (goal 7). ``look`` is the ant's own estimate ``c``."""
    out: List[str] = []
    if ((look >= rules.tumour_here_c and inp.scan_p < rules.tumour_here_scan_max)
            or (rules.tumour_here_any_c is not None and look >= rules.tumour_here_any_c)):
        out.append("tumour_here")
    if look <= rules.not_tumour_c and inp.scan_p >= rules.not_tumour_scan_min:
        out.append("not_tumour")
    if inp.dose_rel < 1.0 and look >= rules.release_c and inp.payload_fraction < EMPTY_FRACTION:
        out.append("need_more")
    return out


# --- targets -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Target:
    kind: str                     # "zone" | "spot"
    label: str                    # candidate label for a zone, STAY for a spot
    region_id: Optional[int]
    pos: Vox
    value: float                  # normalised need x scan p
    dist_mm: float

    @property
    def score(self) -> float:
        return self.value / max(self.dist_mm, 1.0)


def _need(inp: TurnInputs, region_id: Optional[int]) -> float:
    """Normalised need of a zone from the queen's band. No queen step yet: 1.0 (the static scan need)."""
    if not inp.zone_need:
        return 1.0
    if region_id is None:
        return NEED_NORM["high"]                         # a spot outside every zone is treated as a high-need spot
    band = inp.zone_need.get(region_id)
    return 1.0 if band is None else NEED_NORM[band]


def _dose_targets(inp: TurnInputs, st: AntState, scan: ScanInfo, ant: int) -> List[_Target]:
    vm, status = inp.voxel_mm, inp.region_status or {}
    out: List[_Target] = []
    for c in inp.candidates:
        rid = int(c["region_id"])
        if rid in st.skip_zones or status.get(rid, "open") != "open":
            continue
        value = _need(inp, rid) * float(c["p_tumour"])
        if st.zone_mult:
            value *= st.zone_mult.get(rid, 1.0)
        if value > 0.0:
            out.append(_Target("zone", str(c["label"]), rid, _vox(c["centroid_vox"]), value, float(c["dist_mm"])))
    rows = list(inp.marks or ())
    for m in rows:                                  # S-marks: tumour_here / need_more spots from other ants are dose targets
        if m["kind"] not in ("tumour_here", "need_more") or m["disputed"] or list(m["authors"]) == [ant]:
            continue
        spot = _vox(m["pos_vox"])
        if spot in st.skip_spots:
            continue
        if any(o["strong"] and o["kind"] in ("dosed", "dose_ok", "not_tumour") and _mm(vm, o["pos_vox"], spot) <= CONFIRM_MM * (1 + 1e-9)
               for o in rows):
            continue
        rid = scan.zone_at(spot)
        if rid is not None and (status.get(rid, "open") != "open" or rid in st.skip_zones):
            continue
        value = _need(inp, rid) * min(1.0, float(m["strength"]))
        if st.zone_mult:
            value *= st.zone_mult.get(rid, 1.0)
        if value > 0.0:
            out.append(_Target("spot", STAY, rid, spot, value, _mm(vm, inp.pos, spot)))
    return sorted(out, key=lambda t: (-t.score, t.dist_mm, t.pos))


def _shared_visited(inp: TurnInputs, scan: ScanInfo) -> Set[int]:
    """Regions another ant has marked or the colony has closed. Empty in S-solo (the inputs are absent)."""
    seen: Set[int] = {rid for rid, s in (inp.region_status or {}).items() if s != "open"}
    for m in inp.marks or ():
        rid = scan.zone_at(m["pos_vox"])
        if rid is not None:
            seen.add(rid)
    return seen


def _verify_target(inp: TurnInputs, st: AntState) -> Optional[Tuple[Vox, Vox]]:
    """(spot, voxel to look at). A spot is a logged look whose local mean (4 mm) is grey and that still has an unlooked
    voxel within 4 mm, or (sharing arms) a disputed mark spot. Nearest within 20 mm, ties by position.
    Under the combined rules the grey test reads ``est`` and a spot this ant has already verified ``verify_per_spot_max`` times
    is skipped."""
    vm, rules = inp.voxel_mm, st.rules
    looked = {tuple(e["pos_vox"]) for e in inp.reading_log if e["kind"] == "look"}
    spots: Set[Vox] = set()
    for e in inp.reading_log:
        if e["kind"] == "look" and _grey(_lv(e, rules), rules.grey):
            near = _looks_near(inp.reading_log, e["pos_vox"], CONFIRM_MM, vm, rules)
            if _grey(sum(near) / len(near), rules.grey):
                spots.add(_vox(e["pos_vox"]))
    spots |= {_vox(m["pos_vox"]) for m in (inp.marks or ()) if m["disputed"]}
    cap = rules.verify_per_spot_max
    if cap is not None:
        spots = {s for s in spots if st.verify_by_spot[s] < cap}
    r = int(math.ceil(CONFIRM_MM / vm))
    best: Optional[Tuple[Tuple[float, Vox], Vox, Vox]] = None
    for s in sorted(spots):
        d = _mm(vm, inp.pos, s)
        if d > NEAR_MM * (1 + 1e-9):
            continue
        cands = sorted(((_mm(vm, (0, 0, 0), (dx, dy, dz)), (dx, dy, dz)) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                        for dz in range(-r, r + 1) if 0 < _mm(vm, (0, 0, 0), (dx, dy, dz)) <= CONFIRM_MM * (1 + 1e-9)))
        pick = next((v for _, off in cands if (v := (s[0] + off[0], s[1] + off[1], s[2] + off[2])) not in looked), None)
        if pick is None and s not in looked:
            pick = s                                     # a disputed spot nobody has looked at: look at the spot itself
        if pick is not None and (best is None or (d, s) < best[0]):
            best = ((d, s), s, pick)
    return None if best is None else (best[1], best[2])


# --- A2r1: the dose job moves on when its spot is already dosed ----------------------------------------------------------

def known_dosed_spots(inp: TurnInputs, st: AntState) -> List[Vox]:
    """Spots this ant knows are dosed: its own release voxels, its own logged dose readings >= 1 (a voxel at D_t), the voxel it
    stands on and, ONLY when the ``marks`` input exists (sharing arms), the visible ``dosed`` / ``dose_ok`` marks. In S-solo
    ``inp.marks`` is None, so the set is own-only. Nothing else enters."""
    out: Set[Vox] = set(st.released_at)
    out.update(_vox(e["pos_vox"]) for e in inp.reading_log if e["kind"] == "dose" and e["value"] >= 1.0)
    out.add(_vox(inp.pos))
    if inp.marks is not None:
        out.update(_vox(m["pos_vox"]) for m in inp.marks if m["kind"] in ("dosed", "dose_ok"))
    return sorted(out)


def find_move_on_spot(inp: TurnInputs, st: AntState, scan: ScanInfo, region_id: int) -> Optional[Vox]:
    """The nearest voxel of the zone ``region_id`` with scan p >= 0.5 that is >= ``rules.move_on_mm`` from every known dosed spot
    (ties to the lexicographically smallest voxel), or None when the zone has none left."""
    return nearest_new_spot(scan.zone_voxels(region_id), inp.pos, known_dosed_spots(inp, st), inp.voxel_mm, st.rules.move_on_mm)


def _dose_move_on(inp: TurnInputs, st: AntState, scan: ScanInfo, job: str, reason: str, target: Optional[_Target],
                  targets: List[_Target]) -> Optional[_Target]:
    """The A2r1 rule, applied to the dose target the turn would otherwise take (``target``). Returns the dose target to use.

    1. A standing goal ages one decision turn and lapses on arrival (within 1 voxel), after 6 turns, or when its zone is skipped
       or closed. 2. With no goal, a dose-job ant that has payload and reads its own dose >= D_t (reason ``dose_ok``) looks for
       a new spot in its target's zone; none left skips the zone and takes the next-best target. 3. A standing goal replaces the
       zone centroid as the dose target (kind ``spot``, label STAY) while the job is dose. A goal is never replaced while it
       stands: on the way, the ordinary release rule applies at every turn."""
    c, rules, vm = st.counters, st.rules, inp.voxel_mm
    g = st.dose_goal
    if g is not None:
        g["age"] += 1
        status = inp.region_status or {}
        if g["zone"] in st.skip_zones or status.get(g["zone"], "open") != "open":
            st.dose_goal, g = None, None
            c["dose_goal_zone_dropped"] += 1
        elif math.dist(inp.pos, g["spot"]) <= MOVE_ON_ARRIVE_VOX * (1 + 1e-9):
            st.dose_goal, g = None, None
            c["dose_goal_reached"] += 1
        elif g["age"] >= MOVE_ON_TURNS:
            st.dose_goal, g = None, None
            c["dose_goal_timeout"] += 1
    if job != "dose" or target is None:
        return target
    if g is None and reason == "dose_ok":
        if scan.p_tumour is None:
            c["move_on_no_scan_map"] += 1                 # no scan map given: the switch has no effect
            return target
        rid = target.region_id if target.kind == "zone" else scan.zone_at(inp.pos)
        if rid is None:
            return target
        spot = find_move_on_spot(inp, st, scan, rid)
        if spot is None:
            st.skip_zones.add(rid)
            c["zone_skipped_no_spot"] += 1
            rest = [t for t in targets if not (t.kind == "zone" and t.region_id == rid)]
            return rest[0] if rest else None
        st.dose_goal = g = {"spot": spot, "zone": rid, "age": 0}
        c["moved_on_dose_ok"] += 1
    if g is not None and any(t.kind == "zone" and t.region_id == g["zone"] for t in targets):
        return _Target("spot", STAY, g["zone"], g["spot"], target.value, _mm(vm, inp.pos, g["spot"]))
    return target


# --- B2: probe and hold (rules.foraging) -------------------------------------------------------------------------------------

UPTAKE_MARKS = {"good_uptake": "good", "slow_uptake": "poor"}      # a foreign mark carries good or poor, never middling


@dataclass
class _ForageOut:
    """What the foraging rule decided for the turn beyond (release, level, reason)."""

    stay_here: bool = False         # the decision keeps the ant on its own voxel (label STAY, target_vox = this voxel)
    poor: bool = False              # poor tissue: no release, halve the zone, move to the nearest unprobed voxel


def uptake_class(value: float, rules: AntRules) -> str:
    """good (>= good_u), poor (< poor_u), else middling: the sensor words of foraging.py as the ant reads them."""
    return "good" if value >= rules.good_u else "poor" if value < rules.poor_u else "middling"


def uptake_info(inp: TurnInputs, pos: Sequence[float], ant: int, rules: AntRules) -> Tuple[Optional[Dict[str, Any]], bool]:
    """(the nearest uptake information within ``info_mm`` of ``pos`` or None, whether an OWN reading lies within that distance).

    Candidates: an own ``uptake`` entry of the reading log (class from its ratio; its ``pos_vox`` is the voxel it is about) and,
    ONLY when the ``marks`` input exists (sharing arms), a visible ``good_uptake`` / ``slow_uptake`` mark that is not purely this
    ant's own (its own mark repeats its own reading). Nearest wins; ties: the later own reading, then own before foreign. A
    foreign mark never carries a middling class. Nothing else enters."""
    vm, lim2 = inp.voxel_mm, (rules.info_mm * (1 + 1e-9)) ** 2
    cands: List[Tuple[Tuple[int, int, int], Dict[str, Any]]] = []
    for e in inp.reading_log:
        if e["kind"] == "uptake":
            d2 = sum((int(a) - int(b)) ** 2 for a, b in zip(e["pos_vox"], pos))
            if vm * vm * d2 <= lim2:
                cands.append(((d2, 0, -int(e["tick"])), {"cls": uptake_class(float(e["value"]), rules), "src": "own"}))
    for m in inp.marks or ():
        cls = UPTAKE_MARKS.get(m["kind"])
        if cls is None or list(m["authors"]) == [ant]:
            continue
        d2 = sum((int(a) - int(b)) ** 2 for a, b in zip(m["pos_vox"], pos))
        if vm * vm * d2 <= lim2:
            cands.append(((d2, 1, 0), {"cls": cls, "src": "foreign"}))
    if not cands:
        return None, False
    return min(cands, key=lambda kv: kv[0])[1], any(k[1] == 0 for k, _ in cands)


def known_uptake_spots(inp: TurnInputs) -> List[Vox]:
    """Spots the ant has uptake information about: its own uptake readings and, with the ``marks`` input, visible uptake marks."""
    out: Set[Vox] = {_vox(e["pos_vox"]) for e in inp.reading_log if e["kind"] == "uptake"}
    if inp.marks is not None:
        out.update(_vox(m["pos_vox"]) for m in inp.marks if m["kind"] in UPTAKE_MARKS)
    return sorted(out)


def _hold_reading(inp: TurnInputs, hold: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The newest own uptake entry about the hold voxel (within 1 voxel) read after the probe's tick, else None."""
    hit = None
    for e in inp.reading_log:
        if (e["kind"] == "uptake" and int(e["tick"]) > int(hold["tick"])
                and math.dist(e["pos_vox"], hold["pos"]) <= MOVE_ON_ARRIVE_VOX * (1 + 1e-9)):
            hit = e
    return hit


def _forage_forced(inp: TurnInputs, st: AntState, look: float) -> Optional[Dict[str, Any]]:
    """Section 3: is this turn's job forced? ``{"kind": "hold"}`` (reading not yet here), ``{"kind": "dose", "cls", "src"}`` (act on
    a class at the held / stayed voxel; ``cls`` None when a stay just ended on ``dose_ok``, so the A2r1 move-on takes over), or None.
    Clears ``hold`` / ``stay`` when their conditions stop (payload empty, ant displaced, c dropped...)."""
    rules, c, pos = st.rules, st.counters, inp.pos
    payload_ok = inp.payload_fraction >= EMPTY_FRACTION
    h = st.hold
    if h is not None:
        e = _hold_reading(inp, h)
        if e is None:
            if h["age"] >= rules.hold_max_turns:
                st.hold = None
                c["hold_timeouts"] += 1
            else:
                h["age"] += 1
                c["hold_turns"] += 1
                return {"kind": "hold"}
        else:
            st.hold = None
            if math.dist(pos, h["pos"]) > MOVE_ON_ARRIVE_VOX * (1 + 1e-9) or not payload_ok:
                c["hold_dropped"] += 1
                return None
            c["holds_resolved"] += 1
            return {"kind": "dose", "cls": uptake_class(float(e["value"]), rules), "src": "reading"}
    s = st.stay
    if s is not None:
        if _vox(pos) != s["pos"]:
            st.stay = None
            c["stay_moved"] += 1
            return None
        ok, _, why = release_rule(inp, "dose", look, pos, rules)
        if ok:
            return {"kind": "dose", "cls": s["cls"], "src": "stay"}
        st.stay = None
        c["stay_ended_" + why] += 1
        return {"kind": "dose", "cls": None, "src": "stay"} if why == "dose_ok" else None
    return None


def _forage_release(inp: TurnInputs, st: AntState, ant: int, job: str, forced: Optional[Dict[str, Any]], release: bool, level: int,
                    reason: str) -> Tuple[bool, int, str, _ForageOut]:
    """Sections 2 and 3: the foraging rule applied to the ordinary release verdict ``(release, level, reason)``.

    A tumour spot is a voxel where the ordinary rule says ok, in the dose and explore jobs. No uptake information within 6 mm:
    probe (10%) and hold. Good: 50% (``ok_good``). Middling: 25% (``ok_middling``). Poor: no release (``poor_uptake``)."""
    rules, c, pos = st.rules, st.counters, inp.pos
    out = _ForageOut()
    if forced is not None and forced["kind"] == "hold":
        out.stay_here = True
        return False, level, "hold", out
    if forced is not None:
        if not release:
            return release, level, reason, out
        cls = forced["cls"]
    elif job in ("dose", "explore") and release:
        info, own = uptake_info(inp, pos, ant, rules)
        if info is None:
            st.hold = {"pos": _vox(pos), "age": 0, "tick": int(inp.tick)}
            c["probes"] += 1
            out.stay_here = True
            return True, rules.probe_level, "probe", out
        cls = info["cls"]
        if info["src"] == "foreign" and not own:
            c["probes_saved_by_marks"] += 1            # a probe this ant did not need: another ant's mark answered it
    else:
        return release, level, reason, out
    if cls == "poor":
        c["no_release_poor"] += 1
        out.poor = True
        return False, level, "poor_uptake", out
    if forced is not None:
        out.stay_here = True
        if forced["src"] == "reading":
            st.stay = {"pos": _vox(pos), "cls": cls}
    return True, (rules.good_level if cls == "good" else rules.middling_level), "ok_" + cls, out


def _poor_response(inp: TurnInputs, st: AntState, scan: ScanInfo, targets: List[_Target]) -> Optional[_Target]:
    """Section 4. Halve this zone's value in the ant's own ranking (once per zone), then send the ant to the nearest voxel of the
    SAME zone with scan p >= 0.5 that is >= ``info_mm`` from every known uptake-information spot; that spot becomes the A2r1
    persistent dose goal. No such voxel: the zone is exhausted and skipped. Returns the dose target to use."""
    rules, c, vm = st.rules, st.counters, inp.voxel_mm
    rid = scan.zone_at(inp.pos)
    if rid is None:
        c["poor_no_zone"] += 1
        return targets[0] if targets else None
    if rid not in st.zone_mult:
        st.zone_mult[rid] = rules.zone_halve
        c["zones_halved"] += 1
    if scan.p_tumour is None:
        c["forage_no_scan_map"] += 1
        return targets[0] if targets else None
    rest = [t for t in targets if t.region_id != rid]
    # "at least 6 mm" is taken on the strict side: a voxel exactly 6.0 mm from a known spot is still INSIDE the 6 mm information
    # radius (``uptake_info`` counts <= 6 mm), so the ant would arrive on tissue it already has a reading for and be sent on again.
    spot = nearest_new_spot(scan.zone_voxels(rid), inp.pos, known_uptake_spots(inp), vm, rules.info_mm * (1 + 1e-6))
    if spot is None:
        st.skip_zones.add(rid)
        if st.dose_goal is not None and st.dose_goal["zone"] == rid:
            st.dose_goal = None
        c["zone_exhausted"] += 1                       # exhausted, not dropped because of uptake
        return rest[0] if rest else None
    ref = next((t for t in targets if t.region_id == rid), None)
    st.dose_goal = {"spot": spot, "zone": rid, "age": 0}
    c["poor_moved_on"] += 1
    return _Target("spot", STAY, rid, spot, 0.0 if ref is None else ref.value, _mm(vm, inp.pos, spot))


# --- the one decision function ------------------------------------------------------------------------------------------

def _settle_resolved(inp: TurnInputs, st: AntState, c: float) -> None:
    """Step 1 under ``resolved_only`` (v4.3): the previous turn moves its job's threshold only if its outcome is RESOLVED.

    dose     the turn released and this turn has an uptake reading about a voxel within 6 mm of the release: ``good`` or
             ``middling`` succeeds, ``poor`` (legacy ``slow``) fails; anything else is unresolved;
    explore  this turn's position is not covered by an own look from an earlier tick within 4 mm: success if
             c >= release_c and scan p >= explore_scan_min, failure if c < grey low, else unresolved (a covered spot too);
    verify   the mean ``est`` of the own looks within 4 mm of the verified spot (this turn's included) is out of the grey band:
             success; still grey after ``verify_per_spot_max`` verify turns at that spot: failure; else unresolved.
    Success moves theta down, failure up (step sizes and bounds are the rules')."""
    prev = st.prev
    if prev is None:
        return
    rules, vm, job = st.rules, inp.voxel_mm, prev["job"]
    ok: Optional[bool] = None
    if job == "dose":
        u = inp.uptake
        if (prev["released"] and prev["pos"] is not None and u is not None
                and _mm(vm, u["about_vox"], prev["pos"]) <= READ_MM * (1 + 1e-9)):
            word = u.get("word")
            ok = True if word in ("good", "middling") else False if word in ("poor", "slow") else None
    elif job == "explore":
        covered = any(e["kind"] == "look" and e["tick"] < inp.tick and _mm(vm, e["pos_vox"], inp.pos) <= CONFIRM_MM * (1 + 1e-9)
                      for e in inp.reading_log)
        if not covered:
            if c >= rules.release_c and inp.scan_p >= rules.explore_scan_min:
                ok = True
            elif c < rules.grey[0]:
                ok = False
    else:
        spot = prev["spot"]
        near = [] if spot is None else _looks_near(inp.reading_log, spot, CONFIRM_MM, vm, rules)
        if near:
            if not _grey(sum(near) / len(near), rules.grey):
                ok = True
            elif rules.verify_per_spot_max is not None and st.verify_by_spot[spot] >= rules.verify_per_spot_max:
                ok = False
    if ok is None:
        st.counters["theta_unresolved"] += 1
        return
    st.thresholds.update(job, ok)


def _settle_previous(inp: TurnInputs, st: AntState, look: float) -> None:
    """Step 1: the plan chosen at the previous turn gets its success test, then its job's threshold moves."""
    prev = st.prev
    if prev is None:
        return
    if st.rules.resolved_only:
        return _settle_resolved(inp, st, look)
    vm, job = inp.voxel_mm, prev["job"]
    if job == "dose":
        slow = (inp.uptake is not None and inp.uptake.get("word") == "slow" and prev["pos"] is not None
                and _mm(vm, inp.uptake["about_vox"], prev["pos"]) <= READ_MM * (1 + 1e-9))
        ok = bool(prev["released"]) and prev["look"] >= RELEASE_LOOK and not slow
    elif job == "explore":
        ok = (look >= EXPLORE_RELEASE_LOOK and inp.scan_p < SCAN_HALF) or (look <= NOT_TUMOUR_LOOK and inp.scan_p >= SCAN_HALF)
    else:
        spot = prev["spot"]
        near = [] if spot is None else _looks_near(inp.reading_log, spot, CONFIRM_MM, vm)
        ok = bool(near) and not _grey(sum(near) / len(near))
    st.thresholds.update(job, ok)


def _fold_in_answer(inp: TurnInputs, st: AntState) -> Optional[Tuple[Dict[str, Any], float]]:
    """Step 2 (S-talk): the answer to the pending ask. mean(estimate, 0.8 yes | 0.2 no); unknown is ignored.
    Returns (the pending target, the combined estimate) when an answer was used, else None. The caller lets the combined
    estimate replace the look for that target on this turn only."""
    p = st.pending
    if p is None or inp.messages is None:
        return None
    st.pending = None                                   # an ask is settled at the next decision turn, answered or not
    vm = inp.voxel_mm
    answers = [m for m in inp.messages if m["type"] == "answer" and m["question"] == "is_tumour"
               and ((m["about"]["region_id"] is not None and m["about"]["region_id"] == p["region_id"])
                    or (m["about"]["pos_vox"] is not None and _mm(vm, m["about"]["pos_vox"], p["pos"]) <= CONFIRM_MM * (1 + 1e-9)))]
    if not answers:
        return None
    val = ANSWER_VALUE.get(answers[-1]["answer"])
    if val is None:
        return None
    st.counters["answers_used"] += 1
    return p, 0.5 * (p["estimate"] + val)


def decide_turn(inp: TurnInputs, st: AntState, scan: ScanInfo, ant: int) -> Decision:
    """One decision turn. See the module docstring."""
    vm, pos, rules = inp.voxel_mm, inp.pos, st.rules
    raw_look = float(inp.look) if inp.look is not None else inp.scan_p
    look = _current_c(inp, rules, raw_look)              # the estimate this turn acts on: c (combined rules) or the raw look
    _settle_previous(inp, st, look)
    folded = _fold_in_answer(inp, st)

    status = inp.region_status or {}
    targets = _dose_targets(inp, st, scan, ant)
    release_look = look                                  # the look the release test sees; the combined estimate replaces it
    if folded is not None:
        p, est = folded
        is_pending = lambda t: (p["region_id"] is not None and t.kind == "zone" and t.region_id == p["region_id"]) or (  # noqa: E731
            p["region_id"] is None and _mm(vm, t.pos, p["pos"]) <= CONFIRM_MM * (1 + 1e-9))
        here = ((p["region_id"] is not None and scan.zone_at(pos) == p["region_id"])
                or _mm(vm, pos, p["pos"]) <= READ_MM * (1 + 1e-9))
        if here:
            release_look = est if rules.fold_can_permit else min(look, est)     # v4.3: a fold-in can only LOWER the estimate
        if est < rules.fold_drop_c:                      # not eligible as a dose target on this turn (no permanent skip)
            kept = [t for t in targets if not is_pending(t)]
            if len(kept) != len(targets):
                st.counters["targets_dropped_by_answer"] += 1
            targets = kept
    n_regions = len(scan.regions)
    shared_seen = _shared_visited(inp, scan)
    seen = set(inp.visited_regions) | shared_seen
    unvisited = [r for r in scan.regions if r.region_id not in seen]
    verify = _verify_target(inp, st)
    if verify is not None and rules.verify_share_max is not None and not (
            (st.verify_turns + 1) <= rules.verify_share_max * (st.decision_turns + 1)):
        verify = None                                    # v4.3: verify is at most ~25% of this ant's decision turns
        st.counters["verify_blocked_by_share"] += 1
    payload_ok = inp.payload_fraction >= EMPTY_FRACTION
    stimuli = {"dose": max((t.value for t in targets), default=0.0) if payload_ok else 0.0,
               "explore": (len(unvisited) / n_regions) if n_regions else 0.0,
               "verify": 1.0 if verify is not None else 0.0}
    job = st.thresholds.choose(stimuli)
    forced = _forage_forced(inp, st, release_look) if rules.foraging else None
    if forced is not None:
        job = "dose"                                     # B2: a held probe or a stay forces the dose job at that voxel

    release, level, reason = release_rule(inp, job, release_look, pos, rules)
    if release_look != look and release != release_rule(inp, job, look, pos, rules)[0]:
        st.counters["release_changed_by_answer"] += 1
    fg: Optional[_ForageOut] = None
    if rules.foraging:
        release, level, reason, fg = _forage_release(inp, st, ant, job, forced, release, level, reason)
    target: Optional[_Target] = targets[0] if (job == "dose" and targets) else None
    if (target is not None and not release and reason == "washout"
            and _mm(vm, pos, target.pos) <= READ_MM * (1 + 1e-9)):
        # goal 7: the ONLY move-on of the release rule is a fast washout with a look below 0.65 (own reading, or another ant's
        # washout mark in sharing arms). Strong not_tumour / done zones are already dropped from the targets above. A dose_ok
        # reading at one voxel, a low look or a shared "dose high" never abandons a zone.
        (st.skip_zones.add(target.region_id) if target.kind == "zone" and target.region_id is not None
         else st.skip_spots.add(target.pos))
        st.counters["moved_on_" + reason] += 1
        rest = [t for t in targets if t is not target and not (t.kind == "zone" and t.region_id == target.region_id)]
        target = rest[0] if rest else None
    if rules.dose_move_on:
        target = _dose_move_on(inp, st, scan, job, reason, target, targets)
    if fg is not None and fg.poor:
        if job != "dose":
            job = "dose"                                 # an explorer on poor tissue goes on as a dose ant towards the new spot
            st.counters["poor_explore_to_dose"] += 1
        target = _poor_response(inp, st, scan, targets)

    label, goal = STAY, None
    spot = None
    ask_pos: Optional[Vox] = None
    ask_region: Optional[int] = None
    if job == "dose" and target is not None:
        label, goal = target.label, (target.pos if target.kind == "spot" else None)
        ask_pos, ask_region = target.pos, (target.region_id if target.kind == "zone" else None)
    elif job == "explore":
        label = EXPLORE
        if shared_seen - set(inp.visited_regions) and unvisited:      # shared visits changed the answer: aim there
            near = min(unvisited, key=lambda r: (math.dist(pos, r.centroid_vox), r.region_id))
            goal, ask_pos, ask_region = _vox(near.centroid_vox), _vox(near.centroid_vox), near.region_id
        elif unvisited:
            near = min(unvisited, key=lambda r: math.dist(pos, r.centroid_vox))     # the world's own tie rule (region order)
            ask_pos, ask_region = _vox(near.centroid_vox), near.region_id
    elif job == "verify" and verify is not None:
        spot, goal = verify[0], verify[1]
        label, ask_pos = EXPLORE, spot
        st.verify_turns += 1
        st.verify_by_spot[spot] += 1

    if fg is not None and fg.stay_here:
        label, goal = STAY, _vox(pos)                    # B2: a probe, a hold, or dosing a known voxel: stay on this voxel

    ask = None
    if inp.messages is not None and ask_pos is not None and st.pending is None and (
            st.last_ask_round is None or inp.round - st.last_ask_round >= ASK_EVERY_ROUNDS):
        est, own = _estimate(inp, ask_pos, look, rules)
        disputed = any(m["disputed"] and _mm(vm, m["pos_vox"], ask_pos) <= CONFIRM_MM * (1 + 1e-9) for m in (inp.marks or ()))
        if _grey(est, rules.grey) or disputed:
            # goal 7: with no own look near the target the estimate is the current look and the ask is still sent; the kernel
            # decides whether it has a reading to quote and counts a refusal (asks_refused) itself.
            ask = {"question": "is_tumour", **({"region_id": ask_region} if ask_region is not None else {"pos_vox": list(ask_pos)})}
            st.pending = {"pos": ask_pos, "region_id": ask_region, "estimate": est, "round": inp.round}
            st.last_ask_round = inp.round
            st.counters["asks"] += 1
            if not own:
                st.counters["asks_on_current_look"] += 1

    marks = tuple((k, pos) for k in judgement_marks(inp, look, rules))
    st.decision_turns += 1
    if release:
        st.released_at.append(_vox(pos))
    st.prev = {"job": job, "released": release, "look": release_look, "pos": pos if release else None, "spot": spot}
    return Decision(job, label, goal, release, level, marks, ask,
                    {"job": job, "reason": reason, "stimuli": {k: round(v, 6) for k, v in stimuli.items()},
                     "p": {k: round(v, 6) for k, v in st.thresholds.probabilities(stimuli).items()},
                     "status_seen": len(status)})


class ColonyScript:
    """The per-ant policy (kernel ``Policy`` protocol: ``decide(LocalView) -> Intent``)."""

    def __init__(self, scan: ScanInfo, ant: int, rules: Optional[AntRules] = None) -> None:
        self.scan, self.ant = scan, int(ant)
        self.rules = rules if rules is not None else AntRules.v42()      # None means the v4.2 ant
        self.state = AntState(thresholds=Thresholds(**self.rules.threshold_kwargs()), rules=self.rules)

    def decide(self, view: LocalView) -> Intent:
        inp = inputs_from_observation(view.observation, self.scan.voxel_mm)
        d = decide_turn(inp, self.state, self.scan, self.ant)
        params: Dict[str, Any] = {"target": d.target_label, "release": int(d.release), "amount": d.level}
        if d.target_vox is not None:
            params["target_vox"] = list(d.target_vox)
        if d.ask is not None:
            params["ask"] = d.ask
        emit = tuple(SignalDraft(kind=k, pos=tuple(float(c) for c in p), ttl=1) for k, p in d.marks)
        return Intent(ACTION_DECIDE, params, emit=emit, meta={"scripted": True, **d.info})

    def describe(self) -> Dict[str, Any]:
        return {"policy": "colony_script", "theta": dict(self.state.thresholds.theta)}


# --- automatic marks (INTERFACES 2.1; W1 sensor events become marks) ---------------------------------------------------

def automatic_marks(ctx: Mapping[str, Any]) -> List[Tuple[str, Sequence[int], float, int]]:
    """The mark placement rule passed to the driver as ``auto_mark_rule``.

    * ``dosed``: on every release, at the release voxel, amount = the fraction of one squad's share;
    * ``dose_ok``: the ant's own dose reading is >= D_t (``dose_rel`` >= 1) inside a zone;
    * ``slow_uptake``: the uptake reading is slow, at the RELEASE voxel the reading is about (not where the ant stands);
    * ``washout``: the washout reading is fast, at the ant's own voxel.
    The two sensor marks are exactly W1's ``pending_sensor_events`` (a test pins that).
    """
    out: List[Tuple[str, Sequence[int], float, int]] = []
    if ctx["release_fraction"] > 0:
        out.append(("dosed", ctx["pos_vox"], ctx["release_fraction"], 0))
    for r in ctx["readings"]:
        pos = r.get("pos_vox") or ctx["pos_vox"]
        ppm = int(min(1_000_000, max(0, round(float(r["value"]) * 1_000_000))))
        if r["kind"] == "dose" and ctx["zone"] is not None and r["value"] >= 1.0:
            out.append(("dose_ok", pos, 1.0, ppm))
        elif r["kind"] == "uptake" and r.get("word") == "slow":
            out.append(("slow_uptake", pos, 1.0, ppm))
        elif r["kind"] == "washout" and r.get("word") == "fast":
            out.append(("washout", pos, 1.0, ppm))
    return out
