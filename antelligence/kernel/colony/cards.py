"""Name cards (INTERFACES 8, goal 6.4). Domain-free.

A card is a public track record for one ant. Future mapping to the
ReputationRegistry (ERC-8004 style): ``cross_run.correct / checked`` would become
the feedback score, ``cross_run.runs`` the number of attested sessions, and
``public()`` the readable summary. Nothing is written on chain here; this module
only reads and writes local JSON under ``runs/<colony>/cards/<ant_id>.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

from antelligence.kernel.canonical import canonical_json

NEW_BELOW = 5
USUALLY_RIGHT = 0.7


class Card:
    def __init__(self, ant_id: str, policy: str, model: str = "script", rev: str = "v4.2") -> None:
        self.ant_id, self.policy, self.model, self.rev = ant_id, policy, model, rev
        self.thresholds = {"dose": 0.5, "explore": 0.5, "verify": 0.5}
        self.in_run = {"claims": 0, "confirmed": 0, "contradicted": 0}
        self.cross_run = {"claims": 0, "correct": 0, "checked": 0, "runs": 0}

    def begin_run(self) -> None:
        """Reset in-run counts and thresholds; cross-run counts are kept."""
        self.thresholds = {"dose": 0.5, "explore": 0.5, "verify": 0.5}
        self.in_run = {"claims": 0, "confirmed": 0, "contradicted": 0}

    def add_claim(self) -> None:
        self.in_run["claims"] += 1

    def confirm(self) -> None:
        self.in_run["confirmed"] += 1

    def contradict(self) -> None:
        self.in_run["contradicted"] += 1

    def end_run(self, correct: int = 0, checked: int = 0) -> None:
        """Fold the run into cross-run counts (correct/checked from the dev-case scorer)."""
        self.cross_run["claims"] += self.in_run["claims"]
        self.cross_run["correct"] += correct
        self.cross_run["checked"] += checked
        self.cross_run["runs"] += 1

    def band(self) -> str:
        c = self.in_run
        if c["claims"] < NEW_BELOW:
            return "new"
        resolved = c["confirmed"] + c["contradicted"]
        if resolved and c["confirmed"] / resolved >= USUALLY_RIGHT:
            return "usually right"
        return "mixed"

    def public(self) -> dict:
        return {"ant_id": self.ant_id, "policy": self.policy, "model": self.model,
                "in_run": dict(self.in_run), "band": self.band()}

    def to_dict(self) -> dict:
        return {"ant_id": self.ant_id, "policy": self.policy, "model": self.model, "rev": self.rev,
                "thresholds": dict(self.thresholds), "in_run": dict(self.in_run),
                "cross_run": dict(self.cross_run)}

    @classmethod
    def from_dict(cls, d: dict) -> "Card":
        c = cls(d["ant_id"], d["policy"], d.get("model", "script"), d.get("rev", "v4.2"))
        c.thresholds, c.in_run, c.cross_run = dict(d["thresholds"]), dict(d["in_run"]), dict(d["cross_run"])
        return c

    def save(self, colony_dir: Path | str) -> Path:
        """Write ``<colony_dir>/cards/<ant_id>.json`` (colony_dir = runs/<colony>)."""
        p = Path(colony_dir) / "cards" / f"{self.ant_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(canonical_json(self.to_dict()), encoding="utf-8")
        return p

    @classmethod
    def load(cls, colony_dir: Path | str, ant_id: str) -> "Card":
        p = Path(colony_dir) / "cards" / f"{ant_id}.json"
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
