"""A fake model server that also sends answer odds and prefix-cache counts, shaped like ``mlx_lm.server`` 0.32's.

For tests and dry runs of the borrowed switches (:mod:`antelligence.harness.borrowed`): no network, no cost.

* Asked for ``logprobs``, it sends back the reply's tokens with ``top_logprobs`` alternatives, in an arbitrary order
  as mlx does, and at the answer letter the option letters' chances: a right answer is usually given a high chance,
  a wrong one a lower chance, so a gate has something real to learn.
* It remembers each model's last 10 prompts and reports how much of a new prompt starts the same as one of them
  (``usage.prompt_tokens_details.cached_tokens``), as a prefix-caching server would.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Dict, List, Mapping, Optional, Sequence

from antelligence.harness.fakes import FakeProvider, _coin
from antelligence.harness.core import Task
from antelligence.providers import logprobs as lp
from antelligence.providers.base import ChatRequest, ChatResponse


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def fake_logprobs(reply: str, answer: Optional[str], sure: float, letters: Sequence[str] = "ABCDE"
                  ) -> Dict[str, object]:
    """An mlx-shaped ``logprobs`` block for ``reply``: one position per 4 characters, the answer letter's position
    holding every letter with ``answer`` at chance ``sure``."""
    content: List[Dict[str, object]] = []
    body = reply
    cut = reply.rfind(f'"{answer}"') if answer else -1
    head = body[:cut + 1] if cut >= 0 else body
    for j in range(0, len(head), 4):
        piece = head[j:j + 4]
        alts = [{"id": 1, "token": piece, "logprob": math.log(0.9)}, {"id": 2, "token": "ĠThe", "logprob": math.log(0.06)},
                {"id": 3, "token": "Ġis", "logprob": math.log(0.04)}]
        content.append({**alts[0], "top_logprobs": alts})
    if cut >= 0:
        rest = (1.0 - sure) / max(1, len(letters) - 1)
        alts = [{"id": 10 + k, "token": l, "logprob": math.log(sure if l == answer else max(rest, 1e-6))}
                for k, l in enumerate(reversed(letters))]          # not sorted, as mlx sends them
        content.append({**alts[0], "top_logprobs": alts})
        for piece in ('"}',):
            alts = [{"id": 99, "token": piece, "logprob": math.log(0.99)}]
            content.append({**alts[0], "top_logprobs": alts})
    return {"content": content}


class OddsFakeProvider(FakeProvider):
    def __init__(self, tasks: Sequence[Task], key: Mapping[str, str], skill: Mapping[str, float],
                 size_gb: Optional[Mapping[str, float]] = None, default_skill: float = 0.5) -> None:
        super().__init__(tasks, key, skill, size_gb, default_skill)
        self.recent: Dict[str, List[str]] = {}

    async def complete(self, req: ChatRequest) -> ChatResponse:
        from antelligence.harness.promptopt import REFLECT_SYSTEM

        if req.messages and req.messages[0]["content"] == REFLECT_SYSTEM:      # the prompt search's rewrite ask
            self.calls += 1
            n = 4 + int(_coin(req.request_hash, "words") * 20)
            content = json.dumps({"system": "Answer alone. Reply with one JSON object only.",
                                  "format": f'Reply as JSON: {{"reason": "<at most {n} words>", '
                                            f'"answer": "<one letter>"}}'})
            return ChatResponse(content=content, model=req.model, request_hash=req.request_hash,
                                prompt_tokens=sum(len(m["content"]) for m in req.messages) // 4,
                                completion_tokens=len(content) // 4, finish_reason="stop", elapsed_s=0.05)
        resp = await super().complete(req)
        extra = dict(resp.extra)
        text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in req.messages)
        prev = self.recent.setdefault(req.model, [])
        shared = max((_lcp(text, p) for p in prev), default=0)
        extra["cached_tokens"] = min(resp.prompt_tokens, shared // 4)
        prev.append(text)
        del prev[:-10]
        body = req.extra_body or {}
        if body.get("logprobs"):
            task = self._task(req)
            ans = None
            try:
                ans = json.loads(resp.content).get("answer")
            except (ValueError, AttributeError):
                pass
            right = task is not None and ans == self.key.get(task.id)
            u = _coin(req.request_hash, "sure")
            sure = 0.62 + 0.37 * u if right else 0.3 + 0.5 * u
            raw = fake_logprobs(resp.content, ans, sure, task.letters if task else "ABCDE")
            extra["logprobs"] = lp.summarize(raw)
        return dataclasses.replace(resp, extra=extra)
