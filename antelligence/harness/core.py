"""The harness's basic shapes: a task (one request), a step (one worker's answer and what it cost), and the few
plain features of a task's text that the planner may look at."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LETTERS = "ABCDE"
_LABEL = re.compile(r"^\s*\(?([A-E])\)\s*")


@dataclass(frozen=True)
class Task:
    """One multiple-choice request. ``kind`` is a tag for reports only (the dev puzzles carry one); the planner never
    reads it, because real requests don't come labelled."""

    id: str
    passage: str
    stem: str
    options: Tuple[str, ...]           # "(A)..." strings, as the fair test shows them
    kind: str = ""

    @property
    def full(self) -> str:
        """Exactly what every worker is shown (the fair test's exam layout)."""
        parts = [self.passage] if self.passage else []
        parts += [self.stem, "Options:\n" + "\n".join(self.options)]
        return "\n\n".join(parts)

    @property
    def letters(self) -> Tuple[str, ...]:
        return tuple(LETTERS[: len(self.options)])

    def option_text(self, letter: str) -> str:
        return _LABEL.sub("", self.options[LETTERS.index(letter)]).strip()

    @classmethod
    def from_raw(cls, r: Mapping[str, Any]) -> "Task":
        for banned in ("label", "answer", "solution"):
            if banned in r:
                raise ValueError(f"task {r.get('id')} carries {banned!r}: answers stay out of the questions file")
        return cls(str(r["id"]), str(r.get("passage") or "").strip(), str(r["question"]).strip(),
                   tuple(str(o).strip() for o in r["options"]), str(r.get("task") or ""))


def load_tasks(path: Path) -> List[Task]:
    out = [Task.from_raw(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({t.id for t in out}) != len(out):
        raise ValueError("duplicate task ids")
    return out


def load_key(path: Path) -> Dict[str, str]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {str(r["id"]): str(r["label"]) for r in rows}


@dataclass(frozen=True)
class Step:
    """One worker's answer to one task, with its full bill.

    ``answer`` is a letter or None (nothing usable). ``sure`` is the worker's own sureness from 0 to 1 where it has
    one (a decision model's probability, a code check that passed exactly one option), else None. ``weight`` is a
    rough size of the model in GB, so tokens from a 1 GB ant and a 10 GB reader can also be compared by compute.
    ``passes`` (code checks only) lists the options the checking program said break no rule."""

    tool: str
    answer: Optional[str]
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    calls: int = 1
    weight: float = 0.0
    sure: Optional[float] = None
    passes: Optional[Tuple[str, ...]] = None
    note: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)   # e.g. the notes text; never scored

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def weighted_tokens(self) -> float:
        """Tokens times model size (GB): a rough compute bill. A step with no known size counts at size 1."""
        return self.tokens * (self.weight or 1.0)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["passes"] = list(self.passes) if self.passes is not None else None
        d["payload"] = dict(self.payload)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Step":
        d = dict(d)
        if d.get("passes") is not None:
            d["passes"] = tuple(d["passes"])
        d["payload"] = dict(d.get("payload") or {})
        return cls(**d)


# ------------------------------------------------------------------- features
#
# The planner looks only at these, computed from the text a user would send. They are deliberately coarse and
# general (they mean the same on an LSAT question as on a dev puzzle): is it a number answer, does it state a list of
# rules, are the rules about order, are they "all/no/some" statements, and what does the question ask.

_NUM = re.compile(r"^-?\$?\d[\d,]*(\.\d+)?%?$")
_QUANT = re.compile(r"^\s*(all|no|some|every|none of)\b", re.I | re.M)
_ORDER = re.compile(r"\b(before|after|first|second|third|fourth|fifth|sixth|earlier|later|immediately|consecutive)\b",
                    re.I)


def features(task: Task) -> Dict[str, Any]:
    opts = [task.option_text(l) for l in task.letters]
    lines = [ln for ln in task.passage.splitlines() if ln.strip()]
    rule_lines = [ln for ln in lines[1:] if len(ln.split()) <= 25]
    stem = task.stem.lower()
    if "could be" in stem or "could be true" in stem:
        ask = "could"
    elif "cannot" in stem or "except" in stem:
        ask = "cannot"
    elif "must" in stem:
        ask = "must"
    elif any(w in stem for w in ("how many", "what is", "how old", "what percent")):
        ask = "value"
    else:
        ask = "other"
    return {
        "numeric": all(_NUM.match(o.replace(" ", "")) for o in opts),
        "rules": len(rule_lines) >= 3,
        "order": len(_ORDER.findall(task.passage)) >= 3,
        "categorical": len(_QUANT.findall(task.passage)) >= 2,
        "ask": ask,
        "long": len(task.full.split()) > 220,
    }


def bucket(task: Task) -> str:
    """The planner's key: a short readable string, e.g. ``num`` or ``rules+order:could``."""
    f = features(task)
    if f["numeric"]:
        return "num"
    tags = [t for t in ("rules", "order", "categorical") if f[t]]
    return f"{'+'.join(tags) or 'text'}:{f['ask']}"


def vote(answers: Sequence[Optional[str]]) -> Optional[str]:
    """Plurality over usable letters; a tie goes to the tied letter given first (the fair test's rule)."""
    counts: Dict[str, int] = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return next(a for a in answers if a is not None and counts[a] == best)


def margin(answers: Sequence[Optional[str]]) -> int:
    """Votes for the leader minus votes for the runner-up (0 when nothing is usable)."""
    counts = sorted((sum(1 for a in answers if a == x) for x in set(answers) if x is not None), reverse=True)
    if not counts:
        return 0
    return counts[0] - (counts[1] if len(counts) > 1 else 0)
