"""Messages: ask / answer between ants (goal 6.2, INTERFACES v4.2 section 3).

Rules (fixed numbers): delivery next tick; an ask gives the receiver an *answer turn* at ``tick + 1`` on any tick, and an
answer turn never calls ``world.apply`` (the driver posts the answer straight to the office); unanswered asks lapse after
2 ticks; at most 1 ask per ant per 2 decision rounds and 2 answers per ant per tick, over-budget sends are refused and
counted. The ``note`` is free text, at most 120 characters, always quoted when shown, never followed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from antelligence.kernel.colony import rng
from antelligence.kernel.colony.marks import MarkField

QUESTIONS = ("is_tumour", "dosed_enough", "should_i_come")
QUESTION_CODES = {q: i + 1 for i, q in enumerate(QUESTIONS)}
ANSWER_CODES = {"no": 0, "yes": 1, "unknown": 2}
NOTE_MAX = 120
_EPS = 1e-9
Vox = Tuple[int, int, int]


class MessageError(ValueError):
    pass


@dataclass(frozen=True)
class Message:
    msg_id: int
    type: str                                   # "ask" | "answer"
    from_ant: int
    to_ant: int
    tick: int
    about: Mapping[str, Any]                    # {"region_id": int|None, "pos_vox": [x,y,z]|None}, exactly one set
    question: str
    sender_reading: float                       # REQUIRED, in [0, 1]
    reply_to: Optional[int] = None
    answer: Optional[str] = None
    answerer_reading: Optional[float] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.type not in ("ask", "answer"):
            raise MessageError(f"bad message type {self.type!r}")
        if self.question not in QUESTIONS:
            raise MessageError(f"bad question {self.question!r}")
        if self.from_ant == self.to_ant:
            raise MessageError("an ant never messages itself")
        about = {"region_id": self.about.get("region_id"), "pos_vox": self.about.get("pos_vox")}
        if (about["region_id"] is None) == (about["pos_vox"] is None):
            raise MessageError("about needs exactly one of region_id / pos_vox")
        if about["pos_vox"] is not None:
            about["pos_vox"] = [int(c) for c in about["pos_vox"]]
        object.__setattr__(self, "about", about)
        if not (isinstance(self.sender_reading, (int, float)) and 0.0 <= self.sender_reading <= 1.0):
            raise MessageError("sender_reading is required and must be in [0, 1]")
        if self.type == "ask" and (self.reply_to is not None or self.answer is not None or self.answerer_reading is not None):
            raise MessageError("an ask carries no reply_to / answer")
        if self.type == "answer" and (self.reply_to is None or self.answer not in ANSWER_CODES):
            raise MessageError("an answer needs reply_to and yes/no/unknown")
        if self.answerer_reading is not None and not 0.0 <= self.answerer_reading <= 1.0:
            raise MessageError("answerer_reading must be in [0, 1]")
        if not isinstance(self.note, str) or len(self.note) > NOTE_MAX:
            raise MessageError(f"note must be a string of at most {NOTE_MAX} characters")

    def to_dict(self) -> Dict[str, Any]:
        return {"msg_id": self.msg_id, "type": self.type, "from_ant": self.from_ant, "to_ant": self.to_ant,
                "tick": self.tick, "about": {"region_id": self.about["region_id"], "pos_vox": self.about["pos_vox"]},
                "question": self.question, "sender_reading": self.sender_reading, "reply_to": self.reply_to,
                "answer": self.answer, "answerer_reading": self.answerer_reading, "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Message":
        return cls(msg_id=d["msg_id"], type=d["type"], from_ant=d["from_ant"], to_ant=d["to_ant"], tick=d["tick"],
                   about=d["about"], question=d["question"], sender_reading=d["sender_reading"],
                   reply_to=d.get("reply_to"), answer=d.get("answer"), answerer_reading=d.get("answerer_reading"),
                   note=d.get("note", ""))

    def quoted(self) -> Dict[str, Any]:
        """The ``messages_in`` value: the note is wrapped so a reader sees it as quoted text, not an instruction."""
        d = self.to_dict()
        d["note"] = f'"{self.note}"' if self.note else ""
        return d


def _ppm(x: Optional[float]) -> int:
    return -1 if x is None else int(min(1_000_000, max(0, round(x * 1_000_000))))


def message_payload(msg: Message, centroid_vox: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Board ``evidence`` payload (INTERFACES 3.3). A region message needs the region centroid for ``about_pos_vox``."""
    pos = msg.about["pos_vox"]
    if pos is None:
        if centroid_vox is None:
            raise MessageError("a region message needs its centroid_vox for the board entry")
        pos = [int(round(c)) for c in centroid_vox]
    return {"subtype": "message", "msg_type": msg.type, "msg_id": msg.msg_id, "from_ant": msg.from_ant,
            "to_ant": msg.to_ant, "about_pos_vox": list(pos),
            "about_region": -1 if msg.about["region_id"] is None else int(msg.about["region_id"]),
            "question": QUESTION_CODES[msg.question], "sender_reading_ppm": _ppm(msg.sender_reading),
            "reply_to": -1 if msg.reply_to is None else int(msg.reply_to),
            "answer": -1 if msg.answer is None else ANSWER_CODES[msg.answer],
            "answerer_reading_ppm": _ppm(msg.answerer_reading), "note": msg.note}


# --- answers and recipients ---------------------------------------------------------------------------------------

def _within(log: Sequence[Mapping[str, Any]], kind: str, about: Sequence[float], radius_mm: float, voxel_mm: float):
    return [e for e in log if e["kind"] == kind
            and voxel_mm * math.dist(e["pos_vox"], about) <= radius_mm * (1 + _EPS)]


V43_YES, V43_NO = 0.6, 0.4          # v4.3 answer bands on the mean combined estimate: >= 0.6 yes, <= 0.4 no, else unknown


def _look_value(e: Mapping[str, Any]) -> float:
    """A look's value for the v4.3 rules: the combined estimate ``est`` when the world logged one, else the raw look."""
    return float(e.get("est", e["value"]))


def answer_question(question: str, about_pos: Sequence[float], log: Sequence[Mapping[str, Any]], *,
                    has_own_need_more: bool, voxel_mm: float, radius_mm: float = 6.0,
                    answer_rule: str = "v42") -> Tuple[str, Optional[float]]:
    """The answer rule from the answerer's own reading log (INTERFACES 3.2). Returns (answer, reading used).

    ``answer_rule="v42"`` is the v4.2 rule on the raw look values. ``"v43"`` reads the combined estimate (``est``, else the
    raw value) and uses the 0.6 / 0.4 bands with an ``unknown`` in between."""
    looks = _within(log, "look", about_pos, radius_mm, voxel_mm)
    v43 = answer_rule == "v43"
    if question == "is_tumour":
        if not looks:
            return "unknown", None
        if v43:
            mean = sum(_look_value(e) for e in looks) / len(looks)
            return ("yes" if mean >= V43_YES else "no" if mean <= V43_NO else "unknown"), mean
        mean = sum(float(e["value"]) for e in looks) / len(looks)
        return ("yes" if mean >= 0.5 else "no"), mean
    if question == "dosed_enough":
        doses = _within(log, "dose", about_pos, radius_mm, voxel_mm)
        if not doses:
            return "unknown", None
        latest = max(doses, key=lambda e: e["tick"])      # dose only accumulates, so the newest reading is the answer
        rel = float(latest["value"])
        return ("yes" if rel >= 1.0 else "no"), min(1.0, max(0.0, rel))
    # should_i_come: yes iff a need_more mark of its own within 6 mm of `about`, or a look >= 0.5 there (v4.3: est >= 0.6)
    if v43:
        top = max((_look_value(e) for e in looks), default=None)
        return ("yes" if has_own_need_more or (top is not None and top >= V43_YES) else "no"), top
    top = max((float(e["value"]) for e in looks), default=None)
    return ("yes" if has_own_need_more or (top is not None and top >= 0.5) else "no"), top


def choose_recipient(from_ant: int, about_pos: Sequence[float], tick: int, marks: MarkField,
                     positions: Mapping[int, Sequence[float]], logs: Mapping[int, Sequence[Mapping[str, Any]]],
                     radius_mm: float = 6.0) -> int:
    """Recipient rule (INTERFACES 3.2): (1) author of the most recent mark within the read radius of ``about``;
    (2) else the nearest ant whose reading log has an entry within 6 mm of ``about``; (3) else the nearest ant.
    Ties go to the lower ant id; never the sender."""
    voxel_mm = marks.params.voxel_mm
    a = marks.latest_by_other(about_pos, tick, exclude_ant=from_ant)
    if a is not None:
        return a
    others = sorted(i for i in positions if i != from_ant)
    if not others:
        raise MessageError("no other ant to ask")

    def dist(i: int) -> float:
        return round(voxel_mm * math.dist(positions[i], about_pos), 9)

    informed = [i for i in others if any(voxel_mm * math.dist(e["pos_vox"], about_pos) <= radius_mm * (1 + _EPS)
                                         for e in logs.get(i, ()))]
    pool = informed or others
    return min(pool, key=lambda i: (dist(i), i))


def shuffled_recipient(case_id: str, seed: int, msg_id: int, from_ant: int, rule_recipient: int, n_ants: int) -> int:
    """S-talk-shuffled: a random other ant (stream ``shuffle``), avoiding the rule's pick when there is a choice."""
    pool = [i for i in range(n_ants) if i != from_ant and i != rule_recipient] or \
           [i for i in range(n_ants) if i != from_ant]
    return pool[int(rng("shuffle", case_id, seed, msg_id).integers(len(pool)))]


# --- the office ---------------------------------------------------------------------------------------------------

COUNTER_KEYS = ("asks_sent", "asks_answered", "asks_lapsed", "asks_refused", "answers_refused")


class PostOffice:
    """Message ledger plus the budget, lapse and delivery rules. Holds no readings and never sees a world."""

    def __init__(self, *, ask_every_rounds: int = 2, answers_per_tick: int = 2, lapse_ticks: int = 2,
                 drop_msg_id: Optional[int] = None) -> None:
        self.ask_every_rounds = int(ask_every_rounds)
        self.answers_per_tick = int(answers_per_tick)
        self.lapse_ticks = int(lapse_ticks)
        self.drop_msg_id = drop_msg_id          # removal replay: this message exists but is never delivered
        self.messages: List[Message] = []
        self.counters: Dict[str, int] = {k: 0 for k in COUNTER_KEYS}
        self._pending: Dict[int, Message] = {}
        self._last_ask_round: Dict[int, int] = {}
        self._answers_at: Dict[Tuple[int, int], int] = {}
        self._unseen: Dict[int, List[int]] = {}

    def _next_id(self) -> int:
        return len(self.messages) + 1

    def _deliverable(self, msg_id: int) -> bool:
        return msg_id != self.drop_msg_id

    def ask(self, from_ant: int, to_ant: int, tick: int, round_idx: int, about: Mapping[str, Any], question: str,
            sender_reading: float, note: str = "") -> Optional[Message]:
        last = self._last_ask_round.get(from_ant)
        if last is not None and round_idx - last < self.ask_every_rounds:
            self.counters["asks_refused"] += 1
            return None
        msg = Message(self._next_id(), "ask", from_ant, to_ant, tick, about, question, sender_reading, note=note)
        self._last_ask_round[from_ant] = round_idx
        self.messages.append(msg)
        self.counters["asks_sent"] += 1
        if self._deliverable(msg.msg_id):
            self._pending[msg.msg_id] = msg
            self._unseen.setdefault(to_ant, []).append(msg.msg_id)
        return msg

    def refuse_ask(self) -> None:
        """An ask the driver could not even form (no reading to quote, ask not allowed here) counts as refused."""
        self.counters["asks_refused"] += 1

    def lapse(self, tick: int) -> List[Message]:
        gone = [m for m in self._pending.values() if tick - m.tick > self.lapse_ticks]
        for m in gone:
            del self._pending[m.msg_id]
            self.counters["asks_lapsed"] += 1
        return sorted(gone, key=lambda m: m.msg_id)

    def pending_for(self, ant: int, tick: int) -> List[Message]:
        return sorted((m for m in self._pending.values() if m.to_ant == ant and 1 <= tick - m.tick <= self.lapse_ticks),
                      key=lambda m: m.msg_id)

    def answer(self, ask: Message, tick: int, answer: str, answerer_reading: Optional[float]) -> Optional[Message]:
        """Post an answer turn's reply. Over the per-tick budget the answer is refused (counted) and the ask stays open."""
        key = (ask.to_ant, tick)
        if self._answers_at.get(key, 0) >= self.answers_per_tick:
            self.counters["answers_refused"] += 1
            return None
        self._answers_at[key] = self._answers_at.get(key, 0) + 1
        msg = Message(self._next_id(), "answer", ask.to_ant, ask.from_ant, tick, ask.about, ask.question,
                      0.0 if answerer_reading is None else answerer_reading, reply_to=ask.msg_id, answer=answer,
                      answerer_reading=answerer_reading)
        self.messages.append(msg)
        self._pending.pop(ask.msg_id, None)
        self.counters["asks_answered"] += 1
        if self._deliverable(msg.msg_id):
            self._unseen.setdefault(ask.from_ant, []).append(msg.msg_id)
        return msg

    def _ready(self, ant: int, tick: int) -> List[Message]:
        return [self.messages[i - 1] for i in self._unseen.get(ant, []) if self.messages[i - 1].tick < tick]

    def has_new(self, ant: int, tick: int) -> bool:
        return bool(self._ready(ant, tick))

    def take_inbox(self, ant: int, tick: int) -> List[Message]:
        """Messages to ``ant`` sent before ``tick`` and not shown to it yet (shown once)."""
        ready = self._ready(ant, tick)
        ids = {m.msg_id for m in ready}
        self._unseen[ant] = [i for i in self._unseen.get(ant, []) if i not in ids]
        return ready
