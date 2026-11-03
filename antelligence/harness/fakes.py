"""A fake model server for tests and the self-test: no network, no cost, known skill per model.

It speaks the provider contract (``complete(ChatRequest) -> ChatResponse``). It recognises which task a request is
about from the task's option lines, which every prompt contains, and answers right with the chance set for that
model (fixed per request, so a rerun gives the same reply). Tokens grow with the prompt's length; seconds grow with
the model's size. A code request gets back a program that prints the right letter as the only passing option, with
the same chance, or else a program that passes two options (so the checker is sent back to try again).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Optional, Sequence

from antelligence.harness.core import LETTERS, Task
from antelligence.harness.tools import CODE_SYSTEM
from antelligence.providers.base import ChatRequest, ChatResponse


def _coin(*parts: object) -> float:
    h = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


class FakeProvider:
    def __init__(self, tasks: Sequence[Task], key: Mapping[str, str], skill: Mapping[str, float],
                 size_gb: Optional[Mapping[str, float]] = None, default_skill: float = 0.5) -> None:
        self.sig = {"\n".join(t.options): t for t in tasks}
        self.key, self.skill, self.size = dict(key), dict(skill), dict(size_gb or {})
        self.default_skill = default_skill
        self.calls = 0

    def describe(self) -> Dict[str, str]:
        return {"provider": "fake"}

    def _task(self, req: ChatRequest) -> Optional[Task]:
        text = "\n".join(m["content"] for m in req.messages)
        for sig, t in self.sig.items():
            if sig in text:
                return t
        return None

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls += 1
        task = self._task(req)
        p = self.skill.get(req.model, self.default_skill)
        h = req.request_hash
        right = self.key.get(task.id) if task else None
        letters = task.letters if task else tuple(LETTERS)
        good = right is not None and _coin(h, "right") < p
        wrong = [l for l in letters if l != right] or list(letters)
        pick = right if good else wrong[int(_coin(h, "wrong") * len(wrong))]
        system = req.messages[0]["content"] if req.messages else ""
        user = req.messages[-1]["content"]
        if system == CODE_SYSTEM:
            if good:
                marks = {l: l == right for l in letters}
            else:
                marks = {l: l in (pick, wrong[-1]) for l in letters}
            content = f"```python\nprint({json.dumps(json.dumps(marks))})\n```"
        elif "Write notes for the team" in user:
            content = "The question asks which option fits every rule. Check each rule in turn; watch the order."
        else:
            content = json.dumps({"reason": "fake", "answer": pick})
        prompt = sum(len(m["content"]) for m in req.messages) // 4
        out = max(8, len(content) // 4)
        gb = self.size.get(req.model, 4.0)
        return ChatResponse(content=content, model=req.model, request_hash=h, prompt_tokens=prompt,
                            completion_tokens=out, finish_reason="stop", elapsed_s=round(0.05 * gb + out * 0.002 * gb, 4))
