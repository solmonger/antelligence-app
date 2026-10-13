"""Shared builders for the S3 worker tests: requests, fake chat replies with logprobs, mock transports."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import httpx

from antelligence.kernel.decision import DecisionRequest
from antelligence.worlds.tumor_mri.questions import build_questions

QWEN9 = ("RedHatAI/Qwen3.5-9B-FP8-dynamic", "790f0576d2d77dd5322aa0603a470bd9e3a3d1f6")
BONSAI = ("prism-ml/Ternary-Bonsai-8B-gguf", "c2aefbeb0000000000000000000000000000dead")
BRIEF = ("a" * 64, "b" * 64, "c" * 64)


def make_request(model: str, labels: Sequence[str] = ("A", "B", "C"), brief_ids: Sequence[str] = (),
                 images: Sequence[str] = ()) -> DecisionRequest:
    return DecisionRequest(model=model, state="Round 1 of 24, tick 0. Payload left: 100%.", seed=3,
                           questions=build_questions(labels), brief_ids=tuple(brief_ids), images=tuple(images))


def split_tokens(content: str, request: DecisionRequest) -> List[str]:
    """Hand tokenisation: each answer value is its own token; the text between values is one token each."""
    edges = {0, len(content)}
    for qid in request.questions:
        m = re.search(r'"' + re.escape(qid) + r'"\s*:\s*"', content)
        value_end = content.index('"', m.end())
        edges |= {m.end(), value_end}
    cuts = sorted(edges)
    return [content[a:b] for a, b in zip(cuts, cuts[1:])]


def chat_reply(request: DecisionRequest, chosen: Mapping[str, str], tops: Mapping[str, Mapping[str, float]],
               model: str, cites: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """A vLLM-style chat completion. ``tops[qid] = {token: prob}`` fills that question's top_logprobs."""
    body: Dict[str, Any] = {q: chosen[q] for q in request.questions}
    if cites is not None:
        body["cites"] = list(cites)
    content = json.dumps(body)
    tokens = split_tokens(content, request)
    value_at = {2 * i + 1: qid for i, qid in enumerate(request.questions)}  # segments alternate text, value, text, ...
    entries = []
    for i, tok in enumerate(tokens):
        entry: Dict[str, Any] = {"token": tok, "logprob": -0.01, "top_logprobs": [{"token": tok, "logprob": -0.01}]}
        if i in value_at and value_at[i] in tops:
            probs = tops[value_at[i]]
            entry["logprob"] = math.log(probs[tok]) if tok in probs else -5.0
            entry["top_logprobs"] = [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]
        entries.append(entry)
    return {"id": "x", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content},
                         "logprobs": {"content": entries}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": len(tokens)}}


def default_tops() -> Dict[str, Dict[str, float]]:
    return {"target": {"B": 0.6, "A": 0.3, "C": 0.05, "S": 0.03, "X": 0.01, "D": 0.005},  # 0.005 is outside the offered labels
            "release": {"yes": 0.8, "no": 0.15, "maybe": 0.05},
            "amount": {"3": 0.5, "2": 0.25, "4": 0.2, "1": 0.03, "5": 0.02}}


DEFAULT_CHOSEN = {"target": "B", "release": "yes", "amount": "3"}


class Recorder:
    """MockTransport handler: serves /version or /props and POSTs, recording what it received."""

    def __init__(self, post: Callable[[httpx.Request], httpx.Response], version: Optional[Mapping[str, Any]] = None,
                 props: Optional[Mapping[str, Any]] = None) -> None:
        self.post, self.version, self.props = post, version, props
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET" and request.url.path == "/version":
            return httpx.Response(200, json=self.version or {})
        if request.method == "GET" and request.url.path == "/props":
            return httpx.Response(200, json=self.props or {})
        return self.post(request)

    @property
    def posts(self) -> List[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)
