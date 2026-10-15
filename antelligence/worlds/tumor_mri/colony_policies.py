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
2. S-talk only: fold in the answer to its pending ask, ``mean(estimate, 0.8 yes / 0.2 no)`` (unknown is ignored). A target
   whose combined estimate is <= 0.35 is dropped.
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

Judgement calls and defaults chosen where the goal text is silent are listed in ``DEFAULTS_CHOSEN`` and in the Q1 report.
Every random draw in the colony is seeded per INTERFACES 1; this module draws none.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from antelligence.kernel.colony.thresholds import Thresholds
from antelligence.kernel.signal import SignalDraft
from antelligence.kernel.types import Intent, LocalView
from antelligence.worlds.tumor_mri.data import Region
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


def inputs_from_observation(obs: Mapping[str, Any], voxel_mm: float) -> TurnInputs:
    """Read a colony observation into the named inputs. A key the arm may not see is absent and becomes ``None``."""
    status = obs.get("region_status")
    return TurnInputs(
        tick=int(obs["tick"]), round=int(obs["round"]), pos=tuple(int(c) for c in obs["position_vox"]),  # type: ignore[arg-type]
        voxel_mm=float(voxel_mm), payload_fraction=float(obs["payload_fraction"]), scan_p=float(obs["local_p_tumour"]),
        look=obs.get("look", obs.get("local_look")), dose_rel=float(obs["local_dose_rel"]),
        candidates=obs["candidates"], reading_log=obs.get("reading_log", ()), uptake=obs.get("uptake_reading"),
        visited_regions=frozenset((obs.get("visited_self") or {}).get("regions", ())),
        zone_need={int(k): v for k, v in (obs.get("zone_need") or {}).items()},
        marks=obs.get("marks_near"), region_status=None if status is None else {int(k): v for k, v in status.items()},
        messages=obs.get("messages_in"))


@dataclass
class AntState:
    """Per-ant script memory. Reset by building a new one; never shared between ants."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    skip_zones: Set[int] = field(default_factory=set)
    skip_spots: Set[Vox] = field(default_factory=set)
    pending: Optional[Dict[str, Any]] = None              # S-talk: {"pos", "region_id", "estimate", "round"}
    last_ask_round: Optional[int] = None
    prev: Optional[Dict[str, Any]] = None                  # the plan of the previous turn, settled at this turn
    counters: Counter = field(default_factory=Counter)


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


def _looks_near(log: Iterable[Mapping[str, Any]], pos: Sequence[float], radius_mm: float, vm: float) -> List[float]:
    return [float(e["value"]) for e in log if e["kind"] == "look" and _mm(vm, e["pos_vox"], pos) <= radius_mm * (1 + 1e-9)]


def _grey(x: float) -> bool:
    return GREY[0] <= x <= GREY[1]


def _estimate(inp: TurnInputs, target: Sequence[float], look: float) -> Tuple[float, bool]:
    """The target estimate: mean of its own logged looks within 6 mm of the target, else the current look.
    Returns (estimate, own_looks_exist)."""
    near = _looks_near(inp.reading_log, target, READ_MM, inp.voxel_mm)
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


def release_rule(inp: TurnInputs, job: str, look: float, pos: Sequence[float]) -> Tuple[bool, int, str]:
    """(release?, amount level, reason). Goal 7: look >= 0.5 and own dose below D_t; 50% of the payload, 75% if slow uptake;
    skip if own washout here is fast and the look is below 0.65. The explore job needs a look >= 0.65; verify never releases."""
    if job == "verify":
        return False, LEVEL_HALF, "verify"
    if inp.payload_fraction < EMPTY_FRACTION:
        return False, LEVEL_HALF, "payload_empty"
    if look < (EXPLORE_RELEASE_LOOK if job == "explore" else RELEASE_LOOK):
        return False, LEVEL_HALF, "look_low"
    if inp.dose_rel >= 1.0:
        return False, LEVEL_HALF, "dose_ok"
    if _shared_dose_high(inp, pos):
        return False, LEVEL_HALF, "shared_dose_high"
    if _washout_fast_here(inp, pos) and look < WASHOUT_RELEASE_LOOK:
        return False, LEVEL_HALF, "washout"
    return True, (LEVEL_THREE_QUARTERS if _slow_uptake_near(inp, pos) else LEVEL_HALF), "ok"


def judgement_marks(inp: TurnInputs, look: float) -> List[str]:
    """Judgement marks placed at the ant's own voxel, in every arm (goal 7)."""
    out: List[str] = []
    if (look >= TUMOUR_NEW_LOOK and inp.scan_p < SCAN_HALF) or look >= TUMOUR_ANY_LOOK:
        out.append("tumour_here")
    if look <= NOT_TUMOUR_LOOK and inp.scan_p >= SCAN_HALF:
        out.append("not_tumour")
    if inp.dose_rel < 1.0 and look >= RELEASE_LOOK and inp.payload_fraction < EMPTY_FRACTION:
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
    voxel within 4 mm, or (sharing arms) a disputed mark spot. Nearest within 20 mm, ties by position."""
    vm = inp.voxel_mm
    looked = {tuple(e["pos_vox"]) for e in inp.reading_log if e["kind"] == "look"}
    spots: Set[Vox] = set()
    for e in inp.reading_log:
        if e["kind"] == "look" and _grey(float(e["value"])):
            near = _looks_near(inp.reading_log, e["pos_vox"], CONFIRM_MM, vm)
            if _grey(sum(near) / len(near)):
                spots.add(_vox(e["pos_vox"]))
    spots |= {_vox(m["pos_vox"]) for m in (inp.marks or ()) if m["disputed"]}
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


# --- the one decision function ------------------------------------------------------------------------------------------

def _settle_previous(inp: TurnInputs, st: AntState, look: float) -> None:
    """Step 1: the plan chosen at the previous turn gets its success test, then its job's threshold moves."""
    prev = st.prev
    if prev is None:
        return
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


def _fold_in_answer(inp: TurnInputs, st: AntState) -> None:
    """Step 2 (S-talk): the answer to the pending ask. mean(estimate, 0.8 yes | 0.2 no); unknown is ignored."""
    p = st.pending
    if p is None or inp.messages is None:
        return
    st.pending = None                                   # an ask is settled at the next decision turn, answered or not
    vm = inp.voxel_mm
    answers = [m for m in inp.messages if m["type"] == "answer" and m["question"] == "is_tumour"
               and ((m["about"]["region_id"] is not None and m["about"]["region_id"] == p["region_id"])
                    or (m["about"]["pos_vox"] is not None and _mm(vm, m["about"]["pos_vox"], p["pos"]) <= CONFIRM_MM * (1 + 1e-9)))]
    if not answers:
        return
    val = ANSWER_VALUE.get(answers[-1]["answer"])
    if val is None:
        return
    est = 0.5 * (p["estimate"] + val)
    st.counters["answers_used"] += 1
    if est <= GREY[0]:
        st.counters["targets_dropped_by_answer"] += 1
        if p["region_id"] is not None:
            st.skip_zones.add(p["region_id"])
        else:
            st.skip_spots.add(p["pos"])


def decide_turn(inp: TurnInputs, st: AntState, scan: ScanInfo, ant: int) -> Decision:
    """One decision turn. See the module docstring."""
    vm, pos = inp.voxel_mm, inp.pos
    look = float(inp.look) if inp.look is not None else inp.scan_p
    _settle_previous(inp, st, look)
    _fold_in_answer(inp, st)

    status = inp.region_status or {}
    targets = _dose_targets(inp, st, scan, ant)
    n_regions = len(scan.regions)
    shared_seen = _shared_visited(inp, scan)
    seen = set(inp.visited_regions) | shared_seen
    unvisited = [r for r in scan.regions if r.region_id not in seen]
    verify = _verify_target(inp, st)
    payload_ok = inp.payload_fraction >= EMPTY_FRACTION
    stimuli = {"dose": max((t.value for t in targets), default=0.0) if payload_ok else 0.0,
               "explore": (len(unvisited) / n_regions) if n_regions else 0.0,
               "verify": 1.0 if verify is not None else 0.0}
    job = st.thresholds.choose(stimuli)

    release, level, reason = release_rule(inp, job, look, pos)
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

    ask = None
    if inp.messages is not None and ask_pos is not None and st.pending is None and (
            st.last_ask_round is None or inp.round - st.last_ask_round >= ASK_EVERY_ROUNDS):
        est, own = _estimate(inp, ask_pos, look)
        disputed = any(m["disputed"] and _mm(vm, m["pos_vox"], ask_pos) <= CONFIRM_MM * (1 + 1e-9) for m in (inp.marks or ()))
        if _grey(est) or disputed:
            # goal 7: with no own look near the target the estimate is the current look and the ask is still sent; the kernel
            # decides whether it has a reading to quote and counts a refusal (asks_refused) itself.
            ask = {"question": "is_tumour", **({"region_id": ask_region} if ask_region is not None else {"pos_vox": list(ask_pos)})}
            st.pending = {"pos": ask_pos, "region_id": ask_region, "estimate": est, "round": inp.round}
            st.last_ask_round = inp.round
            st.counters["asks"] += 1
            if not own:
                st.counters["asks_on_current_look"] += 1

    marks = tuple((k, pos) for k in judgement_marks(inp, look))
    st.prev = {"job": job, "released": release, "look": look, "pos": pos if release else None, "spot": spot}
    return Decision(job, label, goal, release, level, marks, ask,
                    {"job": job, "reason": reason, "stimuli": {k: round(v, 6) for k, v in stimuli.items()},
                     "p": {k: round(v, 6) for k, v in st.thresholds.probabilities(stimuli).items()},
                     "status_seen": len(status)})


class ColonyScript:
    """The per-ant policy (kernel ``Policy`` protocol: ``decide(LocalView) -> Intent``)."""

    def __init__(self, scan: ScanInfo, ant: int) -> None:
        self.scan, self.ant = scan, int(ant)
        self.state = AntState()

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
