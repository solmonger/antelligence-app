"""The colony episode driver (goal 6.9, INTERFACES v4.2 sections 7, 12).

``run_colony_episode`` runs one episode of a :class:`ColonyWorld`. Per tick, in this fixed order:

1. marks that have run their ttl are dropped; quorum is evaluated (marks only); the scripted queen runs on its cadence
   (S-solo: tick 0 only, and she is given no board); unanswered asks past 2 ticks lapse;
2. answer turns for every ant with a pending ask (any tick; they post straight to the post office and never call
   ``world.apply``, which refuses non-decision ticks);
3. on a decision tick, per ant in order: read the sensors (always, even if the turn is then skipped), apply the quiet
   rule, build the observation, ask the policy, apply the intent, deposit marks, post an ask;
4. ``world.step_environment``.

Everything a mark or message does becomes visible from the NEXT tick (kernel rule ``emitted_at < tick``), so the order of
ants inside a tick never matters.

Determinism and replay. Nothing here draws randomness except the S-talk-shuffled recipient (stream ``shuffle``). A run's
bundle (``ColonyResult.extra["colony"]``) carries each policy output, so ``replay=bundle`` re-plays the episode with no
policy and must reproduce the bundle exactly. ``drop={"mark": i}`` or ``{"message": msg_id}`` re-runs LIVE policies with
that one item absent from the world (a dropped message is still numbered and counted but never delivered) and reports
whether the receiver's next decision changed (the message-effect study).

Domain-free: the world is reached through :class:`ColonyWorld`; the observation key sets arrive as :class:`ArmSpec`.
Judgement-mark and automatic-mark PLACEMENT rules belong to the colony scripts (Q1); ``default_auto_marks`` is the plain
INTERFACES 2.1 table and is replaceable through ``auto_mark_rule``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from antelligence.kernel.board import Board
from antelligence.kernel.colony.marks import (DEFAULT_HALF_LIFE, JUDGEMENT_KINDS, MarkField, MarkParams,
                                              complete_half_life)
from antelligence.kernel.colony.messages import (QUESTIONS, Message, PostOffice, answer_question, choose_recipient,
                                                  message_payload, shuffled_recipient)
from antelligence.kernel.colony.quorum import Quorum, ZoneSpec, need_payload
from antelligence.kernel.colony.wake import WakeRule, sensor_surprise
from antelligence.kernel.types import Intent, LocalView

QUEEN_INPUT_KEYS = ("tick", "regions")
QUEEN_BOARD_KEYS = ("marks", "quorum_events")         # sharing arms only; S-solo's queen sees no board
# v4.3: the two uptake kinds join (the queen's B3 scaling reads them; with scaling off she ignores them, as before)
QUEEN_MARK_KINDS = ("dosed", "dose_ok", "tumour_here", "not_tumour", "good_uptake", "slow_uptake")
COLONY_KEY_NAMES = ("look", "uptake_reading", "washout_reading", "reading_log", "marks_near", "messages_in", "zone_need",
                    "colony_note", "region_status", "visited_self", "card_self", "cards_public", "notebook_self")
_EPS = 1e-9


class ColonyError(RuntimeError):
    pass


class ColonyBarrierError(ColonyError):
    """An observation or a queen input did not match the arm's allowed key set (goal 2.8)."""


class ColonyReplayError(ColonyError):
    pass


class ColonyWorld(Protocol):
    """What the driver needs from a world, beyond the kernel ``World`` protocol (observe / apply / step_environment /
    metrics / done / agents / revision). Positions are integer voxels; distances in the driver are millimetres.

    ``read_sensors(i, tick)`` returns this tick's readings for ant ``i`` at its own voxel: dicts with ``kind``
    (``look`` | ``dose`` | ``uptake`` | ``washout``), ``value``, ``band`` (look only), ``word`` (uptake / washout) and
    optionally ``pos_vox`` (uptake: the release voxel). Sensor events never leave this call.
    """

    seed: int
    case_id: str
    voxel_mm: float
    decision_every: int
    queen_every: int

    def ant_pos(self, i: int) -> Tuple[int, int, int]: ...
    def ant_goal(self, i: int) -> Optional[Tuple[int, int, int]]: ...
    def ant_payload(self, i: int) -> float: ...
    def zones(self) -> List[ZoneSpec]: ...
    def read_sensors(self, i: int, tick: int) -> List[Dict[str, Any]]: ...
    def release_fraction(self, i: int, released: float) -> float: ...
    def visited_self(self, i: int) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class ArmSpec:
    """Which colony inputs an arm may see. ``allowed_keys`` is the per-arm ALLOWED set (INTERFACES section 7)."""

    name: str
    colony: bool                         # False = plain v4.1 S: no marks, no messages, nothing added to the observation
    reads_marks: bool
    reads_talk: bool
    shuffled: bool = False
    allowed_keys: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_keys", frozenset(self.allowed_keys))
        if not set(self.allowed_keys) <= set(COLONY_KEY_NAMES):
            raise ValueError(f"unknown observation keys {sorted(set(self.allowed_keys) - set(COLONY_KEY_NAMES))}")
        if not self.colony and (self.allowed_keys or self.reads_marks or self.reads_talk):
            raise ValueError("a colony-off arm sees nothing")
        if self.reads_talk and not self.reads_marks:
            raise ValueError("talk arms read marks too")
        if self.reads_marks != ("marks_near" in self.allowed_keys) or self.reads_talk != ("messages_in" in self.allowed_keys):
            raise ValueError("reads_marks / reads_talk must match marks_near / messages_in in allowed_keys")


@dataclass(frozen=True)
class ColonyConfig:
    quiet_rule: bool = False
    heartbeat_rounds: int = 3
    quorum_k: int = 2
    read_radius_mm: float = 6.0
    confirm_radius_mm: float = 4.0
    half_life: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_HALF_LIFE))
    ask_every_rounds: int = 2
    answers_per_tick: int = 2
    ask_lapse_ticks: int = 2
    queen_every_ticks: int = 30
    answer_rule: str = "v42"            # "v42" | "v43" (see ``messages.answer_question``); omitted from ``to_dict`` at "v42"

    _IGNORED = ("enabled", "arm", "notebook", "ant_rules")

    def __post_init__(self) -> None:
        if self.answer_rule not in ("v42", "v43"):
            raise ValueError(f"answer_rule must be 'v42' or 'v43', got {self.answer_rule!r}")
        object.__setattr__(self, "half_life", complete_half_life(self.half_life))      # a seven-kind (v4.2) dict gains good_uptake

    def to_dict(self) -> Dict[str, Any]:
        d = {"quiet_rule": self.quiet_rule, "heartbeat_rounds": self.heartbeat_rounds, "quorum_k": self.quorum_k,
             "read_radius_mm": self.read_radius_mm, "confirm_radius_mm": self.confirm_radius_mm,
             "half_life": self._half_life_dict(), "ask_every_rounds": self.ask_every_rounds,
             "answers_per_tick": self.answers_per_tick, "ask_lapse_ticks": self.ask_lapse_ticks,
             "queen_every_ticks": self.queen_every_ticks}
        if self.answer_rule != "v42":                   # omitted at the legacy value: v4.2 board headers and hashes stay identical
            d["answer_rule"] = self.answer_rule
        return d

    def _half_life_dict(self) -> Dict[str, float]:
        """``good_uptake`` is omitted at its default so the v4.2 header (and its hash) stays identical."""
        hl = dict(self.half_life)
        if hl.get("good_uptake") == DEFAULT_HALF_LIFE["good_uptake"]:
            del hl["good_uptake"]
        return hl

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ColonyConfig":
        """The ``colony:`` block of ``config/v4_2/world.yaml``. ``enabled`` / ``arm`` / ``notebook`` belong to others."""
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known - set(cls._IGNORED)
        if extra:
            raise ValueError(f"unknown colony config keys: {sorted(extra)}")
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ColonyResult:
    metrics: Dict[str, Any]
    colony: Dict[str, Any]
    board: Optional[Board] = None
    effect: Optional[Dict[str, Any]] = None

    @property
    def extra(self) -> Dict[str, Any]:
        """Goes under ``bundle.json`` ``extra`` (INTERFACES 12)."""
        return {"colony": self.colony}


def plain(value: Any) -> Any:
    """Detached JSON-plain deep copy: tuples become lists and integer dict keys become strings (as in ``bundle.json``)."""
    return json.loads(json.dumps(value))


def _ppm(x: float) -> int:
    return int(min(1_000_000, max(0, round(float(x) * 1_000_000))))


def default_auto_marks(ctx: Mapping[str, Any]) -> List[Tuple[str, Sequence[int], float, int]]:
    """INTERFACES 2.1 automatic marks: ``dosed`` on a release (amount = fraction of the squad's share), ``dose_ok`` when
    the ant's own dose reads >= D_t inside a zone, ``slow_uptake`` / ``washout`` from a slow / fast reading."""
    out: List[Tuple[str, Sequence[int], float, int]] = []
    if ctx["release_fraction"] > 0:
        out.append(("dosed", ctx["pos_vox"], ctx["release_fraction"], 0))
    for r in ctx["readings"]:
        pos = r.get("pos_vox") or ctx["pos_vox"]
        if r["kind"] == "dose" and ctx["zone"] is not None and r["value"] >= 1.0:
            out.append(("dose_ok", pos, 1.0, _ppm(r["value"])))
        elif r["kind"] == "uptake" and r.get("word") == "slow":
            out.append(("slow_uptake", pos, 1.0, _ppm(r["value"])))
        elif r["kind"] == "washout" and r.get("word") == "fast":
            out.append(("washout", pos, 1.0, _ppm(r["value"])))
    return out


AutoMarkRule = Callable[[Mapping[str, Any]], Iterable[Tuple[str, Sequence[int], float, int]]]
Queen = Callable[[int, Mapping[str, Any]], Optional[Mapping[str, Any]]]


class _Run:
    def __init__(self, world: Any, policy_for: Optional[Callable[[str], Any]], arm: ArmSpec, cfg: ColonyConfig, *,
                 queen: Optional[Queen], providers: Mapping[str, Callable[..., Any]], auto_rule: AutoMarkRule,
                 board: Optional[Board], replay: Optional[Mapping[str, Any]], drop: Optional[Mapping[str, Any]],
                 base_keys: Optional[Iterable[str]], cards_fn: Optional[Callable[[], Any]],
                 mark_pos_fn: Optional[Callable[[int, int, str, Sequence[int]], Sequence[int]]]) -> None:
        self.world, self.arm, self.cfg = world, arm, cfg
        self.queen, self.providers, self.auto_rule, self.board = queen, dict(providers), auto_rule, board
        self.cards_fn, self.mark_pos_fn = cards_fn, mark_pos_fn
        self.base_keys = None if base_keys is None else frozenset(base_keys)
        self.agents: List[str] = world.agents()
        self.n = len(self.agents)
        self.vmm = float(world.voxel_mm)
        self.dec = int(world.decision_every)
        self.drop = dict(drop) if drop else None
        self.baseline = replay
        if drop and replay is None:
            raise ColonyError("a removal replay needs the baseline bundle: pass replay=bundle with drop=...")
        self.live_policies = policy_for is not None and (replay is None or bool(drop))
        if not self.live_policies and replay is None:
            raise ColonyError("policy_for is required unless replaying a bundle")
        self.policy_for = policy_for
        self.recorded: Dict[Tuple[int, int], Mapping[str, Any]] = {}
        if replay is not None and not drop:
            self.recorded = {(e["tick"], e["ant"]): e for e in replay["action_log"] if e["kind"] == "decision"}
            if replay["arm"] != arm.name:
                raise ColonyReplayError(f"bundle arm {replay['arm']!r} != {arm.name!r}")
        self.drop_mark_key = None
        self.drop_done = False
        drop_msg = None
        if drop:
            if set(drop) == {"mark"}:
                r = replay["marks"][int(drop["mark"])]
                self.drop_mark_key = (r["tick"], r["ant"], r["mark_kind"], tuple(r["pos_vox"]))
            elif set(drop) == {"message"}:
                drop_msg = int(drop["message"])
            else:
                raise ColonyError("drop must be {'mark': index} or {'message': msg_id}")
        self.marks = MarkField(arm.name, MarkParams(self.vmm, cfg.read_radius_mm, cfg.confirm_radius_mm, cfg.half_life))
        self.office = PostOffice(ask_every_rounds=cfg.ask_every_rounds, answers_per_tick=cfg.answers_per_tick,
                                 lapse_ticks=cfg.ask_lapse_ticks, drop_msg_id=drop_msg)
        self.zones: List[ZoneSpec] = world.zones() if arm.colony else []
        self.zone_by_id = {z.region_id: z for z in self.zones}
        self.quorum = Quorum(self.zones, self.marks, k=cfg.quorum_k)
        self.wake = WakeRule(self.n, enabled=cfg.quiet_rule and arm.colony, heartbeat_rounds=cfg.heartbeat_rounds)
        self.logs: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(self.n)}
        self.action_log: List[Dict[str, Any]] = []
        self.need_per_zone: List[Dict[str, Any]] = []
        self.zone_need: Dict[int, str] = {}
        self.colony_note = ""
        self.extra_counters = {"marks_refused": 0}
        self._records_at_eval = -1
        self.last_turn_tick: Dict[int, int] = {}
        self.last_payload: Dict[int, float] = {}
        self.last_at_goal: Dict[int, bool] = {}
        self.last_need: Dict[int, Optional[str]] = {}
        self._emit_drafts: List[Any] = []

    # --- board ----------------------------------------------------------------------------------------------------

    def _post(self, author: str, payload: Mapping[str, Any], tick: int) -> None:
        if self.board is not None:
            self.board.append("evidence", author, {k: v for k, v in payload.items() if k != "tick"}, tick)

    # --- geometry -------------------------------------------------------------------------------------------------

    def _mm(self, a: Sequence[float], b: Sequence[float]) -> float:
        return self.vmm * math.dist(a, b)

    def _zone_at(self, pos: Sequence[float]) -> Optional[int]:
        best = None
        for z in self.zones:
            if z.contains(pos):
                key = (round(self._mm(pos, z.centroid_vox), 9), z.region_id)
                if best is None or key < best:
                    best = key
        return None if best is None else best[1]

    def _about_pos(self, about: Mapping[str, Any]) -> Tuple[int, int, int]:
        if about["region_id"] is not None:
            c = self.zone_by_id[int(about["region_id"])].centroid_vox
            return tuple(int(round(x)) for x in c)  # type: ignore[return-value]
        return tuple(int(x) for x in about["pos_vox"])  # type: ignore[return-value]

    # --- marks ----------------------------------------------------------------------------------------------------

    def _deposit(self, ant: int, kind: str, pos: Sequence[int], tick: int, amount: float, reading_ppm: int) -> None:
        if self.mark_pos_fn is not None:
            pos = self.mark_pos_fn(len(self.marks.records), ant, kind, pos)
        pos = tuple(int(round(c)) for c in pos)
        milli = int(round(amount * 1000))
        if (not self.drop_done and self.drop_mark_key is not None
                and self.drop_mark_key == (tick, ant, kind, pos)):
            self.drop_done = True
            return
        rec = self.marks.deposit(ant, kind, pos, tick, amount=milli / 1000.0, reading_ppm=reading_ppm)
        if rec is not None:
            self._post(f"squad-{ant}", rec, tick)

    # --- queen ----------------------------------------------------------------------------------------------------

    def _queen_inputs(self, tick: int) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {"tick": tick, "regions": [
            {"region_id": z.region_id, "centroid_vox": list(z.centroid_vox)} for z in self.zones]}
        if self.arm.reads_marks:
            inputs["marks"] = [dict(r) for r in self.marks.records
                               if r["mark_kind"] in QUEEN_MARK_KINDS and r["tick"] < tick]
            inputs["quorum_events"] = [e.to_payload() for e in self.quorum.events]
        want = set(QUEEN_INPUT_KEYS) | (set(QUEEN_BOARD_KEYS) if self.arm.reads_marks else set())
        if set(inputs) != want:
            raise ColonyBarrierError(f"queen inputs {sorted(inputs)} != {sorted(want)}")
        return inputs

    def _run_queen(self, tick: int) -> None:
        if self.queen is None or tick % self.cfg.queen_every_ticks:
            return
        if not self.arm.reads_marks and tick > 0:
            return                                    # S-solo: the queen sees no board, need stays at its tick-0 value
        out = self.queen(tick, self._queen_inputs(tick)) or {}
        if "zone_need" in out:
            self.zone_need = {int(k): str(v) for k, v in out["zone_need"].items()}
        if "colony_note" in out:
            self.colony_note = str(out["colony_note"])[:200]
        for row in out.get("need_rows", ()):
            payload = need_payload(row["region_id"], row["need"], row["need_milli"], row["delivered_milli"])
            self.need_per_zone.append({"tick": tick, "region_id": payload["region_id"], "need": payload["need"],
                                       "need_milli": payload["need_milli"], "delivered_milli": payload["delivered_milli"],
                                       **({"uptake_factor_milli": int(row["uptake_factor_milli"])} if "uptake_factor_milli" in row else {})})
            self._post("queen", payload, tick)

    # --- tick start -----------------------------------------------------------------------------------------------

    def _begin_tick(self, tick: int) -> None:
        self.marks.expire(tick)
        if len(self.marks.records) != self._records_at_eval:
            self._records_at_eval = len(self.marks.records)
            for ev in self.quorum.evaluate(tick):
                self._post("harness", ev.to_payload(), tick)
        self._run_queen(tick)
        self.office.lapse(tick)
        if self.arm.reads_talk:
            for ant in range(self.n):
                for ask in self.office.pending_for(ant, tick):
                    pos = self._about_pos(ask.about)
                    ans, reading = answer_question(
                        ask.question, pos, self.logs[ant], voxel_mm=self.vmm, radius_mm=self.cfg.read_radius_mm,
                        has_own_need_more=self.marks.has_own(ant, "need_more", pos, tick),
                        answer_rule=self.cfg.answer_rule)
                    msg = self.office.answer(ask, tick, ans, reading)
                    if msg is None:
                        continue
                    self._post(f"squad-{ant}", message_payload(msg, self._centroid(msg)), tick)
                    self.action_log.append({"tick": tick, "ant": ant, "kind": "answer", "action": "answer",
                                            "params": {"msg_id": msg.msg_id, "reply_to": ask.msg_id, "answer": ans}})

    def _centroid(self, msg: Message) -> Optional[Tuple[int, int, int]]:
        return None if msg.about["pos_vox"] is not None else self._about_pos(msg.about)

    # --- decision tick --------------------------------------------------------------------------------------------

    def _read(self, i: int, tick: int) -> List[Dict[str, Any]]:
        pos = list(self.world.ant_pos(i))
        out = []
        for r in self.world.read_sensors(i, tick):
            e = {"tick": tick, "pos_vox": list(r.get("pos_vox") or pos), "kind": r["kind"], "value": float(r["value"]),
                 "band": r.get("band"), "word": r.get("word")}
            for extra in ("scan_p", "est"):             # optional look fields (v4.3 combined estimate); absent -> logs unchanged
                if r.get(extra) is not None:
                    e[extra] = float(r[extra])
            out.append(e)
            self.logs[i].append(e)
        return out

    def _triggers(self, i: int, tick: int, readings: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
        pos = self.world.ant_pos(i)
        since = self.last_turn_tick[i]
        goal = self.world.ant_goal(i)
        at_goal = goal is not None and tuple(goal) == tuple(pos)
        zone = self._zone_at(pos)
        return {
            "mark": self.arm.reads_marks and any(
                since <= r["tick"] < tick and r["ant"] != i and self._mm(pos, r["pos_vox"]) <= self.cfg.read_radius_mm * (1 + _EPS)
                for r in self.marks.records),
            "message": self.arm.reads_talk and self.office.has_new(i, tick),
            "surprise": sensor_surprise(readings),
            "goal": at_goal and not self.last_at_goal.get(i, False),
            "payload": self.world.ant_payload(i) != self.last_payload.get(i),
            "need": zone is not None and self.zone_need.get(zone) != self.last_need.get(i),
        }

    def _relevant_reading(self, i: int, question: str, about: Sequence[float]) -> Optional[float]:
        near = [e for e in self.logs[i] if self._mm(e["pos_vox"], about) <= self.cfg.read_radius_mm * (1 + _EPS)]
        if question == "dosed_enough":
            d = [e for e in near if e["kind"] == "dose"]
            return None if not d else min(1.0, max(0.0, float(max(d, key=lambda e: e["tick"])["value"])))
        v = [float(e.get("est", e["value"])) for e in near if e["kind"] == "look"]
        return None if not v else sum(v) / len(v)

    def _colony_obs(self, i: int, tick: int, readings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        A, pos = self.arm.allowed_keys, self.world.ant_pos(i)
        by_kind = {r["kind"]: r for r in readings}
        out: Dict[str, Any] = {}

        def reading(kind: str) -> Optional[Dict[str, Any]]:
            r = by_kind.get(kind)
            return None if r is None else {"value": r["value"], "word": r["word"], "about_vox": list(r["pos_vox"])}

        builders: Dict[str, Callable[[], Any]] = {
            "look": lambda: None if "look" not in by_kind else float(by_kind["look"]["value"]),
            "uptake_reading": lambda: reading("uptake"),
            "washout_reading": lambda: reading("washout"),
            "reading_log": lambda: [dict(e, pos_vox=list(e["pos_vox"])) for e in self.logs[i]],
            "marks_near": lambda: self.marks.marks_near(pos, tick),
            "messages_in": lambda: [m.quoted() for m in self.office.take_inbox(i, tick)],
            "zone_need": lambda: dict(self.zone_need),
            "colony_note": lambda: self.colony_note,
            "region_status": lambda: dict(self.quorum.status),
            "visited_self": lambda: self.world.visited_self(i),
        }
        for key in COLONY_KEY_NAMES:
            if key not in A:
                continue
            if key in builders:
                out[key] = builders[key]()
            elif key in self.providers:
                out[key] = self.providers[key](i, tick)
            else:
                out[key] = None if key == "card_self" else []
        return out

    def _observation(self, i: int, tick: int, readings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        obs = dict(self.world.observe(self.agents[i], tick).data)
        if self.arm.colony:
            obs.update(self._colony_obs(i, tick, readings))
        if self.base_keys is not None and set(obs) != self.base_keys | self.arm.allowed_keys:
            raise ColonyBarrierError(
                f"{self.arm.name}: observation keys differ from OBS_KEYS | ALLOWED: "
                f"extra={sorted(set(obs) - self.base_keys - self.arm.allowed_keys)} "
                f"missing={sorted((self.base_keys | self.arm.allowed_keys) - set(obs))}")
        return obs

    def _intent(self, i: int, tick: int, view: LocalView, policy: Any) -> Tuple[Intent, Optional[Mapping[str, Any]]]:
        """Returns the policy's intent split into the world intent and the colony ask."""
        if self.live_policies:
            intent = policy.decide(view)
            params = dict(intent.params)
            ask = params.pop("ask", None)
            world_intent = Intent(intent.action, params, meta=intent.meta)
            self._emit_drafts = list(intent.emit)
            return world_intent, ask
        rec = self.recorded.get((tick, i))
        if rec is None:
            raise ColonyReplayError(f"no recorded decision for ant {i} at tick {tick}")
        p = rec["params"]
        self._emit_drafts = [d for d in p["emit"]]
        return Intent(rec["action"], p["world"], meta=p["meta"]), p.get("ask")

    def _post_ask(self, i: int, tick: int, round_idx: int, ask: Mapping[str, Any]) -> None:
        question = ask.get("question")
        region, pos = ask.get("region_id"), ask.get("pos_vox")
        if question not in QUESTIONS or (region is None) == (pos is None) or (
                region is not None and int(region) not in self.zone_by_id):
            self.office.refuse_ask()
            return
        about = {"region_id": None if region is None else int(region),
                 "pos_vox": None if pos is None else [int(c) for c in pos]}
        about_pos = self._about_pos(about)
        sender = self._relevant_reading(i, question, about_pos)
        if sender is None:
            self.office.refuse_ask()                  # an ask must quote the asker's own reading (INTERFACES 3.1)
            return
        positions = {j: self.world.ant_pos(j) for j in range(self.n)}
        to = choose_recipient(i, about_pos, tick, self.marks, positions, self.logs, self.cfg.read_radius_mm)
        if self.arm.shuffled:
            to = shuffled_recipient(self.world.case_id, self.world.seed, len(self.office.messages) + 1, i, to, self.n)
        msg = self.office.ask(i, to, tick, round_idx, about, question, sender, str(ask.get("note", ""))[:120])
        if msg is not None:
            self._post(f"squad-{i}", message_payload(msg, self._centroid(msg)), tick)

    def _decision_tick(self, tick: int, policies: Mapping[str, Any]) -> None:
        round_idx = tick // self.dec
        for i, agent in enumerate(self.agents):
            readings = self._read(i, tick) if self.arm.colony else []
            turn, reasons = True, ()
            if self.arm.colony:
                trig = self._triggers(i, tick, readings) if self.wake.enabled and i in self.last_turn_tick else {}
                d = self.wake.decide(i, round_idx, trig)
                turn, reasons = d.turn, d.reasons
                self.wake.record(i, round_idx, turn)
            released = 0.0
            if turn:
                payload_before = self.world.ant_payload(i)
                obs = self._observation(i, tick, readings)
                view = LocalView(agent_id=agent, tick=tick, revision=self.world.revision, seed=self.world.seed,
                                 observation=obs)
                world_intent, ask = self._intent(i, tick, view, policies.get(agent))
                outcome = self.world.apply(agent, world_intent, tick)
                if self.arm.colony:
                    released = float(outcome.effects.get("released", 0.0)) if outcome.accepted else 0.0
                    drafts = self._record_marks(i, tick, readings)
                    if ask is not None and self.arm.reads_talk:
                        self._post_ask(i, tick, round_idx, ask)
                    params: Dict[str, Any] = {"world": dict(world_intent.params), "meta": dict(world_intent.meta),
                                              "emit": drafts, "ask": plain(ask) if ask is not None else None,
                                              "accepted": bool(outcome.accepted)}
                    if self.wake.enabled:
                        params["wake"] = list(reasons)
                    self.action_log.append({"tick": tick, "ant": i, "kind": "decision",
                                            "action": world_intent.action, "params": params})
                self.last_turn_tick[i] = tick
                self.last_payload[i] = payload_before
                goal = self.world.ant_goal(i)
                self.last_at_goal[i] = goal is not None and tuple(goal) == tuple(self.world.ant_pos(i))
                zone = self._zone_at(self.world.ant_pos(i)) if self.arm.colony else None
                self.last_need[i] = None if zone is None else self.zone_need.get(zone)
            elif self.arm.colony:
                self.action_log.append({"tick": tick, "ant": i, "kind": "skipped", "action": "skip", "params": {}})
            if self.arm.colony:
                self._auto_marks(i, tick, readings, self.world.release_fraction(i, released) if released else 0.0)

    def _record_marks(self, i: int, tick: int, readings: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Judgement marks from the intent's ``emit`` drafts (a mark must lie within the read radius of the ant)."""
        recorded = []
        pos = self.world.ant_pos(i)
        look = next((r for r in readings if r["kind"] == "look"), None)
        ppm = 0 if look is None else _ppm(look["value"])
        for d in self._emit_drafts:
            kind, dpos = (d["kind"], d["pos_vox"]) if isinstance(d, Mapping) else (d.kind, d.pos)
            cell = [int(round(c)) for c in dpos]
            recorded.append({"kind": kind, "pos_vox": cell})      # every draft is recorded, so a replay refuses the same ones
            if kind not in JUDGEMENT_KINDS or self._mm(pos, cell) > self.cfg.read_radius_mm * (1 + _EPS):
                self.extra_counters["marks_refused"] += 1
                continue
            self._deposit(i, kind, cell, tick, 1.0, ppm)
        return recorded

    def _auto_marks(self, i: int, tick: int, readings: Sequence[Mapping[str, Any]], fraction: float) -> None:
        pos = self.world.ant_pos(i)
        ctx = {"ant": i, "tick": tick, "pos_vox": list(pos), "readings": [dict(r) for r in readings],
               "release_fraction": fraction, "zone": self._zone_at(pos)}
        for kind, mpos, amount, ppm in self.auto_rule(ctx):
            self._deposit(i, kind, mpos, tick, amount, ppm)

    # --- the loop --------------------------------------------------------------------------------------------------

    def run(self) -> ColonyResult:
        w = self.world
        start = getattr(w, "start_episode", None)
        if start is not None:
            start()
        policies = {a: self.policy_for(a) for a in self.agents} if self.live_policies else {}
        cards_before = self.cards_fn() if self.cards_fn else {}
        hook = getattr(w, "queen_hook", None)
        tick = 0
        while not w.done(tick):
            if hook is not None and tick % w.queen_every == 0:
                hook(tick, w)
            if self.arm.colony:
                self._begin_tick(tick)
            if tick % self.dec == 0:
                self._decision_tick(tick, policies)
            w.step_environment(tick)
            tick += 1
        cards_after = self.cards_fn() if self.cards_fn else {}
        metrics = w.metrics()
        counters = dict(self.office.counters)
        counters.update({"marks": self.marks.counts(), "quorum_events": len(self.quorum.events), **self.extra_counters})
        colony = {
            "arm": self.arm.name,
            "params": {**self.cfg.to_dict(), "case_id": w.case_id, "seed": w.seed},
            "marks": self.marks.records,
            "messages": [m.to_dict() for m in self.office.messages],
            "reading_logs": {str(i): v for i, v in self.logs.items()},
            "cards_before": cards_before, "cards_after": cards_after,
            "turns_skipped": {f"ant-{i:02d}": n for i, n in self.wake.skipped.items()},
            "need_per_zone": self.need_per_zone,
            "action_log": self.action_log,
            "counters": counters,
            "region_status": dict(self.quorum.status),
            "metrics": metrics,
        }
        if self.drop:
            colony["replay"] = {"drop": dict(self.drop), "applied": self._dropped()}
        colony = plain(colony)
        effect = self._effect(colony) if self.drop else None
        return ColonyResult(metrics, colony, self.board, effect)

    def _dropped(self) -> bool:
        if "mark" in self.drop:
            return self.drop_done
        return any(m.msg_id == self.drop["message"] for m in self.office.messages)

    def _effect(self, colony: Mapping[str, Any]) -> Dict[str, Any]:
        base = self.baseline

        def key(e: Mapping[str, Any]) -> Any:
            return (e["tick"], e["ant"], e["kind"], e["action"], plain(e["params"]))

        a, b = base["action_log"], colony["action_log"]
        first = next((i for i in range(min(len(a), len(b))) if key(a[i]) != key(b[i])), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        div = None
        if first is not None:
            e = (a[first] if first < len(a) else b[first])
            div = {"tick": e["tick"], "ant": e["ant"]}
        eff: Dict[str, Any] = {"drop": dict(self.drop), "applied": self._dropped(), "receiver": None,
                               "baseline_next": None, "replay_next": None, "first_divergence": div,
                               "metrics_changed": plain(base["metrics"]) != colony["metrics"]}
        if "message" in self.drop:
            msg = next((m for m in base["messages"] if m["msg_id"] == self.drop["message"]), None)
            if msg is None:
                raise ColonyReplayError(f"baseline has no message {self.drop['message']}")
            eff["receiver"] = msg["to_ant"]

            def nxt(log: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
                for e in log:
                    if e["kind"] == "decision" and e["ant"] == msg["to_ant"] and e["tick"] > msg["tick"]:
                        return {"tick": e["tick"], "action": e["action"], "params": e["params"]["world"]}
                return None

            eff["baseline_next"], eff["replay_next"] = nxt(a), nxt(b)
            eff["changed"] = eff["baseline_next"] != eff["replay_next"]
        else:
            eff["changed"] = div is not None
        return eff


def run_colony_episode(world: Any, policy_for: Optional[Callable[[str], Any]], *, arm: ArmSpec,
                       config: Optional[ColonyConfig] = None, queen: Optional[Queen] = None,
                       obs_providers: Optional[Mapping[str, Callable[..., Any]]] = None,
                       auto_mark_rule: AutoMarkRule = default_auto_marks, board: Optional[Board] = None,
                       replay: Optional[Mapping[str, Any]] = None, drop: Optional[Mapping[str, Any]] = None,
                       base_keys: Optional[Iterable[str]] = None, cards_fn: Optional[Callable[[], Any]] = None,
                       mark_pos_fn: Optional[Callable[[int, int, str, Sequence[int]], Sequence[int]]] = None
                       ) -> ColonyResult:
    """Run one colony episode. See the module docstring.

    ``policy_for(agent_id)`` returns an object with ``decide(LocalView) -> Intent``; a colony ask rides in
    ``Intent.params["ask"]`` and judgement marks in ``Intent.emit``. ``obs_providers[key](ant, tick)`` supplies the keys
    this package does not own (``card_self``, ``cards_public``, ``notebook_self``). ``base_keys`` (the world's OBS_KEYS)
    turns on the runtime barrier check. ``mark_pos_fn(mark_index, ant, kind, pos)`` lets a control arm relocate a mark
    (S-marks-displaced). ``replay`` / ``drop``: see the module docstring.
    """
    return _Run(world, policy_for, arm, config or ColonyConfig(), queen=queen, providers=obs_providers or {},
                auto_rule=auto_mark_rule, board=board, replay=replay, drop=drop, base_keys=base_keys, cards_fn=cards_fn,
                mark_pos_fn=mark_pos_fn).run()
