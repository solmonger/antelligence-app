"""Per-ant notebook (INTERFACES 9, goal 6.7). Domain-free.

Design ideas (capped memory, archive-never-delete curation, frozen reads) are
inspired by Hermes agent/curator.py and tools/memory_tool_store.py (MIT). Ideas
only; no code was copied.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from antelligence.kernel.canonical import canonical_json

CAP_CHARS = 1500
MAX_TEXT = 200


class NotebookError(ValueError):
    pass


class Notebook:
    def __init__(self, ant_id: str, cap_chars: int = CAP_CHARS) -> None:
        self.ant_id, self.cap_chars = ant_id, cap_chars
        self.frozen, self.run_index = False, 0
        self.lessons: list[dict] = []

    # -- helpers
    def active_chars(self) -> int:
        return sum(len(l["text"]) for l in self.lessons if l["status"] == "active")

    def _guard(self) -> None:
        if self.frozen:
            raise NotebookError("notebook is frozen")

    def _find(self, lid: str) -> dict:
        for l in self.lessons:
            if l["id"] == lid:
                return l
        raise NotebookError(f"unknown lesson {lid}")

    @staticmethod
    def _check_text(text: str) -> None:
        if not text or len(text) > MAX_TEXT:
            raise NotebookError(f"text must be 1..{MAX_TEXT} chars")

    # -- writes
    def add(self, text: str, source_runs: list[str] | None = None) -> str:
        self._guard()
        self._check_text(text)
        if self.active_chars() + len(text) > self.cap_chars:
            raise NotebookError("cap exceeded")
        lid = f"L{len(self.lessons) + 1:03d}"
        self.lessons.append({"id": lid, "text": text, "source_runs": list(source_runs or []),
                             "uses": 0, "effect": 0.0, "status": "active",
                             "last_used_run": self.run_index, "pinned": False})
        return lid

    def patch(self, lid: str, text: str) -> None:
        self._guard()
        self._check_text(text)
        l = self._find(lid)
        if l["status"] != "active":
            raise NotebookError("lesson is archived")
        if self.active_chars() - len(l["text"]) + len(text) > self.cap_chars:
            raise NotebookError("cap exceeded")
        l["text"] = text

    def remove(self, lid: str) -> None:
        """Archive; the lesson stays in the file."""
        self._guard()
        self._find(lid)["status"] = "archived"

    def mark_used(self, lid: str, effect: float = 0.0) -> None:
        self._guard()
        l = self._find(lid)
        l["uses"] += 1
        l["effect"] = float(effect)
        l["last_used_run"] = self.run_index

    def pin(self, lid: str, pinned: bool = True) -> None:
        self._guard()
        self._find(lid)["pinned"] = pinned

    def curate(self, run_index: int, stale_runs: int = 10) -> list[str]:
        """Archive active, unpinned lessons unused for >= stale_runs. Never deletes."""
        self._guard()
        self.run_index = run_index
        gone = [l["id"] for l in self.lessons if l["status"] == "active" and not l["pinned"]
                and run_index - l["last_used_run"] >= stale_runs]
        for lid in gone:
            self._find(lid)["status"] = "archived"
        return gone

    def freeze(self) -> None:
        self.frozen = True

    # -- reads
    def snapshot(self) -> list[str]:
        """Frozen copy: active lesson texts only."""
        return [l["text"] for l in self.lessons if l["status"] == "active"]

    def export_skills(self) -> list[dict]:
        """agentskills.io shape: name <= 64, description <= 1024, body (markdown)."""
        out = []
        for l in self.lessons:
            if l["status"] != "active":
                continue
            name = re.sub(r"[^a-z0-9-]+", "-", f"{self.ant_id}-{l['id']}".lower()).strip("-")[:64]
            out.append({"name": name, "description": l["text"][:1024],
                        "body": f"# {name}\n\n{l['text']}\n\nSource runs: {', '.join(l['source_runs']) or 'none'}\n"})
        return out

    # -- persistence
    def to_dict(self) -> dict:
        return {"ant_id": self.ant_id, "cap_chars": self.cap_chars, "frozen": self.frozen,
                "run_index": self.run_index, "lessons": [dict(l) for l in self.lessons]}

    @classmethod
    def from_dict(cls, d: dict) -> "Notebook":
        n = cls(d["ant_id"], d["cap_chars"])
        n.frozen, n.run_index = d["frozen"], d["run_index"]
        n.lessons = [dict(l) for l in d["lessons"]]
        return n

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(canonical_json(self.to_dict()), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> "Notebook":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
