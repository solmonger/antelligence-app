"""Decision backends: Jev, Laya, an SLM on vLLM, a fake, plus cache, ledger and escalation wrappers.

All of them speak :class:`DecisionRequest` / :class:`DecisionResponse` from the
kernel contract. Stack for a run (innermost first)::

    EscalatingBackend(
        CachedDecision(LedgeredDecision(JevBackend(...), writer, role="worker"), cache_dir),
        CachedDecision(LedgeredDecision(SLMBackend(...), writer, role="escalation"), cache_dir),
    )

Keys are passed in by the caller and only ever used in the Authorization header;
they are scrubbed from every error message. Retry policy: transport errors,
HTTP 429/5xx and malformed bodies are retried ``retries`` times (default 3, so up
to 4 attempts), then :class:`DecisionError`. Other 4xx and a served-model mismatch
fail at once. No backend reads the environment or any file for credentials.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import httpx

from antelligence.kernel.canonical import canonical_json, content_hash
from antelligence.kernel.decision import (Answer, DecisionBackend, DecisionError, DecisionRequest, DecisionResponse,
                                          Question, QUESTION_TYPES, check_answers)
from antelligence.providers.base import BudgetExceeded, ChatRequest, Provider, ProviderError
from antelligence.providers.ledger import LedgerWriter, cost_usd

JEV_MODEL = "jev-1.13.0"
JEV_URL = "https://api.typesafe.ai"
LAYA_MODEL = "convaiinnovations/laya-typed-decisions"
LAYA_URL = "http://127.0.0.1:8520"
MAX_RESPONSE_BYTES = 2_000_000
logger = logging.getLogger(__name__)


class _Transient(Exception):
    """Worth retrying."""


class _Permanent(Exception):
    """Retrying cannot help."""


# --- wire <-> Answer -------------------------------------------------------


def _num(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"not a finite number: {value!r}")
    return float(value)


def parse_wire_answers(request: DecisionRequest, wire: Mapping[str, Any]) -> Dict[str, Answer]:
    """Jev/laya-serve ``answers`` -> Answers. Extra fields (answer_confidence, action, ...) are ignored."""
    if not isinstance(wire, Mapping):
        raise ValueError("answers is not an object")
    out: Dict[str, Answer] = {}
    for qid, question in request.questions.items():
        a = wire[qid]
        if a.get("type") != question.type:
            raise ValueError(f"{qid}: answer type {a.get('type')!r} != {question.type!r}")
        if question.type == "noul":
            p = _num(a["noul"])
            out[qid] = Answer("noul", p, {"yes": p, "no": 1.0 - p})
        else:
            confidence = None if a.get("confidence") is None else _num(a["confidence"])
            value = a["choice"] if question.type == "choice" else _num(a["score"])
            out[qid] = Answer(question.type, value, {str(k): _num(v) for k, v in a["probabilities"].items()},
                              confidence)
    return out


# --- HTTP backends ---------------------------------------------------------


class _TypedHTTPBackend:
    """POST /v1/systemone with retries, identity check, accounting."""

    name = "http"

    def __init__(self, base_url: str, *, model: str, api_key: Optional[str] = None, timeout_s: float = 60.0,
                 retries: int = 3, backoff_s: float = 0.5, price: Tuple[float, float] = (0.0, 0.0),
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be http(s)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout_s, self.retries, self.backoff_s, self.price = timeout_s, retries, backoff_s, price
        self._transport = transport
        self.calls = 0  # physical HTTP attempts

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "model": self.model, "base_url": self.base_url}

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _check_served(self, served: str) -> None:
        raise NotImplementedError

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        if request.model != self.model:
            raise DecisionError(f"request model {request.model!r} != backend model {self.model!r}")
        last = ""
        for attempt in range(self.retries + 1):
            try:
                return await self._once(request)
            except _Permanent as exc:
                raise DecisionError(self._scrub(f"{self.name}: {exc}")) from None
            except _Transient as exc:
                last = self._scrub(str(exc))
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff_s * (attempt + 1))
        raise DecisionError(f"{self.name}: failed after {self.retries + 1} attempts: {last}")

    async def _once(self, request: DecisionRequest) -> DecisionResponse:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self.calls += 1
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=False, trust_env=False,
                                         transport=self._transport) as client:
                reply = await client.post(f"{self.base_url}/v1/systemone", json=request.payload(), headers=headers)
        except httpx.HTTPError as exc:
            raise _Transient(f"transport error: {type(exc).__name__}") from None
        elapsed = time.monotonic() - start
        if reply.status_code == 429 or reply.status_code >= 500:
            raise _Transient(f"HTTP {reply.status_code}")
        if reply.status_code != 200:
            raise _Permanent(f"HTTP {reply.status_code}: {reply.text[:200]}")
        if len(reply.content) > MAX_RESPONSE_BYTES:
            raise _Permanent("response exceeds size limit")
        try:
            data = reply.json()
            served = data["model"]
            answers = parse_wire_answers(request, data["answers"])
            usage = data.get("usage") or {}
            tokens_in, tokens_out = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            raise _Transient(f"malformed response: {type(exc).__name__}: {exc}") from None
        if not isinstance(served, str) or not served:
            raise _Transient("malformed response: no served model")
        self._check_served(served)
        try:
            check_answers(request, answers)
        except DecisionError as exc:
            raise _Transient(str(exc)) from None
        return DecisionResponse(answers=answers, served_model=served, request_hash=request.request_hash,
                                input_tokens=tokens_in, output_tokens=tokens_out,
                                cost_usd=cost_usd(tokens_in, tokens_out, self.price), elapsed_s=elapsed)


class JevBackend(_TypedHTTPBackend):
    """TypeSafe Jev, pinned to ``jev-1.13.0`` (``jev-latest`` is refused). ``base_url`` is the origin."""

    name = "jev"

    def __init__(self, base_url: str = JEV_URL, api_key: str = "", *, model: str = JEV_MODEL, **kw: Any) -> None:
        if model != JEV_MODEL:
            raise ValueError(f"Jev is pinned to {JEV_MODEL!r}; got {model!r}")
        if not api_key:
            raise ValueError("api_key is required (pass it in; this module never reads the environment)")
        super().__init__(base_url, model=model, api_key=api_key, price=(0.042, 0.0), **kw)

    def _check_served(self, served: str) -> None:
        if served != JEV_MODEL:
            raise _Permanent(f"served model {served!r} is not {JEV_MODEL!r}")


class LayaBackend(_TypedHTTPBackend):
    """Local ``laya-serve``. Whatever model name it reports (e.g. ``laya-rl-agent``) is recorded."""

    name = "laya"

    def __init__(self, base_url: str = LAYA_URL, *, model: str = LAYA_MODEL, api_key: Optional[str] = None,
                 **kw: Any) -> None:
        super().__init__(base_url, model=model, api_key=api_key, **kw)

    def _check_served(self, served: str) -> None:  # any non-empty name; it is recorded in served_model
        return None


# --- SLM -------------------------------------------------------------------

_SLM_SYSTEM = (
    "You answer typed questions about a state. Reply with ONE JSON object and nothing else: no prose, no code "
    "fence. Probabilities in one question sum to 1; every listed option must appear (0 is allowed).")
SLM_TOKEN_CAP = 256  # per-request budget ceiling; the every-agent arms' 64-token cap does not apply here
_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
NORMALISE_RANGE = (0.5, 1.5)  # probability vectors summing inside this are rescaled to 1; outside is rejected


def slm_max_tokens(request: DecisionRequest, cap: int = SLM_TOKEN_CAP) -> int:
    """48 + 16 per noul + 14 per option of every choice/score question, at most ``cap``.

    Sized from Phase 0: DeepSeek V4.1 Flash was cut off on 16/300 replies at 24 + 12/noul + 10/option."""
    need = 48 + sum(16 if q.type == "noul" else 14 * len(q.options) for q in request.questions.values())
    return min(need, cap)


def _slm_prompt(request: DecisionRequest) -> str:
    questions, shape = {}, []
    for qid, q in request.questions.items():
        body: Dict[str, Any] = {"type": q.type, "instructions": q.instructions}
        keys = q.options if q.type == "choice" else tuple(str(i) for i in range(len(q.options)))
        if q.type == "choice":
            body["options"] = {o: q.criteria.get(o) for o in q.options}
        elif q.type == "score":
            body["levels"] = {str(i): text for i, text in enumerate(q.options)}
        questions[qid] = body
        probs = ", ".join(f"{json.dumps(k)}: p" for k in keys)
        shape.append(f"{json.dumps(qid)}: " + ("0.83" if q.type == "noul" else "{" + probs + "}"))
    return (json.dumps({"state": request.state, "questions": questions}, ensure_ascii=False, sort_keys=True)
            + "\nReply exactly in this shape (noul: P(yes) as one number, e.g. 0.83; choice/score: p is each "
            "option's probability, all options listed, sum 1):\n{" + ", ".join(shape) + "}")


def _unwrap(raw: Any) -> Any:
    """Models nest the value under the type name ({"noul": x}); strip one such wrapper."""
    if isinstance(raw, dict) and len(raw) == 1 and next(iter(raw)) in QUESTION_TYPES:
        return next(iter(raw.values()))
    return raw


def _p_yes(raw: Any) -> float:
    """Number, {"p_yes": x}, {"yes": x} or {"yes": a, "no": b} (a / (a + b)); anything else is rejected."""
    raw = _unwrap(raw)
    if not isinstance(raw, dict):
        return _num(raw)
    if set(raw) == {"p_yes"}:
        return _num(raw["p_yes"])
    if "yes" in raw and set(raw) <= {"yes", "no"}:
        yes, no = _num(raw["yes"]), _num(raw.get("no", 0.0))
        if yes < 0 or no < 0 or ("no" in raw and yes + no == 0):
            raise ValueError(f"yes/no weights {yes}, {no} are unusable")
        return yes / (yes + no) if "no" in raw else yes
    raise ValueError(f"cannot read P(yes) from {raw!r}")


def _distribution(raw: Any, keys: Tuple[str, ...]) -> Dict[str, float]:
    """Option map (maybe nested under the type name). Unknown names reject; omitted options count as 0 only
    when the listed ones already sum to >= 0.98 (otherwise the reply was truncated or confused)."""
    raw = _unwrap(raw)
    if not isinstance(raw, dict) or not raw or not {str(k) for k in raw} <= set(keys):
        raise ValueError(f"probabilities must be named from {list(keys)}")
    probs = {str(k): _num(v) for k, v in raw.items()}
    total = sum(probs.values())
    if len(probs) < len(keys) and total < 0.98:
        raise ValueError(f"options {sorted(set(keys) - set(probs))} missing and listed ones sum to {total:.3f}")
    if any(p < 0 for p in probs.values()) or not NORMALISE_RANGE[0] <= total <= NORMALISE_RANGE[1]:
        raise ValueError(f"probabilities sum to {total:.3f}")
    return {k: probs.get(k, 0.0) / total for k in keys}


def parse_slm_json(request: DecisionRequest, text: str) -> Dict[str, Answer]:
    """Strict parse of the SLM's JSON object (code fences allowed). Sums within 0.5..1.5 are normalised to 1."""
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise DecisionError(f"slm: reply is not JSON ({exc})") from None
    return parse_slm_object(request, raw)


def parse_slm_object(request: DecisionRequest, raw: Any) -> Dict[str, Answer]:
    """The checks of :func:`parse_slm_json` on an already-decoded reply object."""
    if not isinstance(raw, dict) or set(raw) != set(request.questions):
        raise DecisionError("slm: reply keys do not match the question ids")
    answers: Dict[str, Answer] = {}
    try:
        for qid, q in request.questions.items():
            if q.type == "noul":
                p = _p_yes(raw[qid])
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"{qid}: P(yes) {p} outside [0, 1]")
                answers[qid] = Answer("noul", p, {"yes": p, "no": 1.0 - p})
                continue
            keys = q.options if q.type == "choice" else tuple(str(i) for i in range(len(q.options)))
            probs = _distribution(raw[qid], keys)
            top = max(probs.values())
            if q.type == "choice":
                value: Any = next(k for k in keys if probs[k] == top)
            else:
                value = sum(int(k) * p for k, p in probs.items())
            answers[qid] = Answer(q.type, value, probs, top)
    except ValueError as exc:
        raise DecisionError(f"slm: {exc}") from None
    return answers


class SLMBackend:
    """Asks a chat model the same typed questions; non-thinking is the provider's extra_body.

    ``max_tokens`` caps the per-request budget from :func:`slm_max_tokens`. Only transport/provider errors are
    retried; a length cut-off or an unparseable reply raises at once.
    """

    def __init__(self, provider: Provider, model: str, *, max_tokens: int = SLM_TOKEN_CAP, retries: int = 3,
                 backoff_s: float = 0.5, name: str = "slm") -> None:
        self.provider, self.model, self.max_tokens = provider, model, max_tokens
        self.retries, self.backoff_s, self.name = retries, backoff_s, name

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "model": self.model, "provider": self.provider.describe()}

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        if request.model != self.model:
            raise DecisionError(f"request model {request.model!r} != backend model {self.model!r}")
        chat = ChatRequest(self.model, ({"role": "system", "content": _SLM_SYSTEM},
                                        {"role": "user", "content": _slm_prompt(request)}),
                           max_tokens=slm_max_tokens(request, self.max_tokens), temperature=0.0, seed=request.seed)
        last = ""
        for attempt in range(self.retries + 1):
            try:
                reply = await self.provider.complete(chat)
                break
            except BudgetExceeded as exc:
                raise DecisionError(f"{self.name}: {exc}") from None
            except ProviderError as exc:
                last = str(exc)
                if "(length)" in last:  # the provider's own length cut-off: deterministic, never retried
                    raise DecisionError(f"{self.name}: reply cut off at max_tokens={chat.max_tokens}") from None
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff_s * (attempt + 1))
        else:
            raise DecisionError(f"{self.name}: failed after {self.retries + 1} attempts: {last}")
        if reply.model != self.model:
            raise DecisionError(f"{self.name}: served model {reply.model!r} is not {self.model!r}")
        if reply.finish_reason == "length":  # deterministic at temperature 0: a retry would only repeat it
            raise DecisionError(f"{self.name}: reply cut off at max_tokens={chat.max_tokens}")
        answers = parse_slm_json(request, reply.content)  # malformed output is not retried: temperature 0
        check_answers(request, answers)
        reported = (reply.extra or {}).get("cost")
        return DecisionResponse(answers=answers, served_model=reply.model, request_hash=request.request_hash,
                                input_tokens=reply.prompt_tokens, output_tokens=reply.completion_tokens,
                                cost_usd=float(reported) if isinstance(reported, (int, float)) else 0.0,
                                elapsed_s=reply.elapsed_s)


# --- fake ------------------------------------------------------------------


def hashed_answers(request: DecisionRequest) -> Dict[str, Answer]:
    """Default FakeBackend responder: deterministic, varied, valid answers derived from the request hash."""
    out: Dict[str, Answer] = {}
    for qid, q in request.questions.items():
        h = int(content_hash([request.request_hash, qid])[:8], 16)
        if q.type == "noul":
            p = (h % 101) / 100
            out[qid] = Answer("noul", p, {"yes": p, "no": 1.0 - p})
            continue
        n = len(q.options)
        keys = q.options if q.type == "choice" else tuple(str(i) for i in range(n))
        top = h % n
        rest = 0.3 / (n - 1)
        probs = {k: (0.7 if i == top else rest) for i, k in enumerate(keys)}
        value: Any = keys[top] if q.type == "choice" else float(sum(i * probs[k] for i, k in enumerate(keys)))
        out[qid] = Answer(q.type, value, probs, 0.7)
    return out


class FakeBackend:
    """Offline backend: ``responder(request) -> {qid: Answer}`` (default :func:`hashed_answers`)."""

    def __init__(self, responder: Optional[Callable[[DecisionRequest], Mapping[str, Answer]]] = None, *,
                 model: str = "fake-1", served_model: Optional[str] = None, latency_s: float = 0.0,
                 name: str = "fake") -> None:
        self.responder = responder or hashed_answers
        self.model, self.served_model, self.latency_s, self.name = model, served_model or model, latency_s, name
        self.requests: list = []

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "model": self.model}

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        answers = dict(self.responder(request))
        check_answers(request, answers)
        return DecisionResponse(answers=answers, served_model=self.served_model, request_hash=request.request_hash,
                                input_tokens=max(1, len(request.state) // 4), output_tokens=len(answers) * 4,
                                elapsed_s=self.latency_s)


# --- cache / ledger / escalation -------------------------------------------


def response_to_dict(response: DecisionResponse) -> Dict[str, Any]:
    return {"answers": {q: a.to_dict() for q, a in response.answers.items()}, "served_model": response.served_model,
            "request_hash": response.request_hash, "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens, "cost_usd": response.cost_usd, "elapsed_s": response.elapsed_s,
            "escalated": list(response.escalated), "identity": dict(response.identity),
            "timing": dict(response.timing), "cites": list(response.cites), "prob_method": response.prob_method,
            "degraded": response.degraded, "reasoning_tokens": response.reasoning_tokens,
            "alarm": dict(response.alarm)}


def response_from_dict(data: Mapping[str, Any], *, cached: bool = False) -> DecisionResponse:
    answers = {q: Answer(a["type"], a["value"], a["probabilities"], a["confidence"])
               for q, a in data["answers"].items()}
    return DecisionResponse(answers=answers, served_model=data["served_model"], request_hash=data["request_hash"],
                            input_tokens=data["input_tokens"], output_tokens=data["output_tokens"],
                            cost_usd=data["cost_usd"], elapsed_s=data["elapsed_s"], cached=cached,
                            escalated=tuple(data["escalated"]), identity=dict(data.get("identity") or {}),
                            timing=dict(data.get("timing") or {}), cites=tuple(data.get("cites") or ()),
                            prob_method=data.get("prob_method", ""), degraded=bool(data.get("degraded", False)),
                            reasoning_tokens=int(data.get("reasoning_tokens", 0)), alarm=dict(data.get("alarm") or {}))


class CachedDecision:
    """Reply cache keyed by request hash, one JSON file per reply under ``cache_dir``.

    A hit returns the stored reply (``cached=True``) and calls nothing, so a
    wrapped :class:`LedgeredDecision` writes no row. ``offline=True`` raises on a miss.
    """

    def __init__(self, inner: DecisionBackend, cache_dir: Union[str, Path], *, offline: bool = False) -> None:
        self.inner, self.cache_dir, self.offline = inner, Path(cache_dir), offline
        self._locks: Dict[str, asyncio.Lock] = {}

    def describe(self) -> Dict[str, Any]:
        return self.inner.describe()  # transparent: a replay shares the live run's config

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        key = request.request_hash
        async with self._locks.setdefault(key, asyncio.Lock()):  # identical concurrent requests call once
            path = self._path(key)
            if path.exists():
                return response_from_dict(json.loads(path.read_text(encoding="utf-8")), cached=True)
            if self.offline:
                raise DecisionError(f"cache miss in offline mode ({key[:12]})")
            response = await self.inner.decide(request)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(canonical_json(response_to_dict(response)), encoding="utf-8")
            tmp.replace(path)
            return response


class LedgeredDecision:
    """One ledger row per physical ``decide`` (failures write an error row). Put it under the cache."""

    def __init__(self, inner: DecisionBackend, writer: LedgerWriter, *, role: str = "worker") -> None:
        self.inner, self.writer, self.role = inner, writer, role
        self.backend = str(inner.describe().get("backend", "unknown"))

    def describe(self) -> Dict[str, Any]:
        return self.inner.describe()

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        try:
            response = await self.inner.decide(request)
        except Exception as exc:
            self.writer.write(role=self.role, backend=self.backend, model=request.model, served_model=None,
                              request_hash=request.request_hash, error=f"{type(exc).__name__}: {exc}")
            raise
        self.writer.write(role=self.role, backend=self.backend, model=request.model,
                          served_model=response.served_model, request_hash=request.request_hash,
                          input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                          latency_s=response.elapsed_s, cost_usd=response.cost_usd)
        return response


@dataclasses.dataclass(frozen=True)
class EscalatedResponse(DecisionResponse):
    """A response that also names the questions whose escalation failed (the primary answer was kept)."""

    escalation_failed: Tuple[str, ...] = ()

    def accounting(self) -> Dict[str, Any]:
        out = super().accounting()
        return {**out, "escalation_failed": list(self.escalation_failed)} if self.escalation_failed else out


class EscalatingBackend:
    """Primary backend plus an optional escalation backend for the questions it is unsure about.

    A noul is uncertain when ``lo < P(yes) < hi`` (boundaries are confident); a
    choice when ``confidence < c_min`` or missing; scores never escalate. Only the
    uncertain questions are re-asked, in their original option order, and the
    escalation answers replace the primary's. If the escalation raises ``DecisionError`` the primary's answers
    stand and the response is an :class:`EscalatedResponse` naming the questions whose escalation failed.
    ``escalation`` must report its model in ``describe()["model"]``.
    """

    def __init__(self, primary: DecisionBackend, escalation: Optional[DecisionBackend] = None, *,
                 lo: float = 0.2, hi: float = 0.8, c_min: float = 0.5) -> None:
        if not 0.0 <= lo < hi <= 1.0 or not 0.0 <= c_min <= 1.0:
            raise ValueError("need 0 <= lo < hi <= 1 and 0 <= c_min <= 1")
        self.primary, self.escalation, self.lo, self.hi, self.c_min = primary, escalation, lo, hi, c_min

    def describe(self) -> Dict[str, Any]:
        return {"backend": "escalating", "model": self.primary.describe().get("model"),
                "primary": self.primary.describe(),
                "escalation": None if self.escalation is None else self.escalation.describe(),
                "lo": self.lo, "hi": self.hi, "c_min": self.c_min}

    def uncertain(self, question: Question, answer: Answer) -> bool:
        if question.type == "noul":
            return self.lo < answer.p_yes < self.hi
        if question.type == "choice":
            return answer.confidence is None or answer.confidence < self.c_min
        return False

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        first = await self.primary.decide(request)
        if self.escalation is None:
            return first
        unsure = {qid: q for qid, q in request.questions.items() if self.uncertain(q, first.answers[qid])}
        if not unsure:
            return first
        sub = DecisionRequest(str(self.escalation.describe()["model"]), request.state, unsure, request.seed)
        try:
            second = await self.escalation.decide(sub)
            merged = {**first.answers, **second.answers}
            check_answers(request, merged)
        except DecisionError:  # e.g. an all-zero reply: keep the primary's answer rather than fail the whole decision
            return EscalatedResponse(**vars(first), escalation_failed=tuple(sorted(unsure)))
        return dataclasses.replace(
            first, answers=merged, input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens, cost_usd=first.cost_usd + second.cost_usd,
            elapsed_s=first.elapsed_s + second.elapsed_s, cached=first.cached and second.cached,
            escalated=tuple(sorted(set(first.escalated) | set(unsure))))


# === v4.1 workers (goal 4.5, INTERFACES section 5) ============================================================
#
# One call per squad-round answers target, release and amount. Every reply carries an identity
# {served_model, revision, runtime, runtime_version}; a mismatch with the pinned identity is a DecisionError.
#
# Probability methods
#   system_one  d1-3B: softmax over the option tokens at the answer slot (the model's own readout).
#   logprobs    Qwen3.5 on vLLM, Bonsai on llama.cpp: see :func:`extract_label_probs`.
#   verbalised  DeepSeek V4.1 Flash via Nous: the model writes the probabilities in its JSON reply.

PROB_LOGPROBS, PROB_SYSTEM_ONE, PROB_VERBALISED = "logprobs", "system_one", "verbalised"
TOP_LOGPROBS = 20  # vLLM's default maximum; the target question has at most 8 labels
D1_MODEL = "LiquidAI/d1-3B"
D1_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
QWEN_9B = ("RedHatAI/Qwen3.5-9B-FP8-dynamic", "790f0576d2d77dd5322aa0603a470bd9e3a3d1f6")
QWEN_4B = ("RedHatAI/Qwen3.5-4B-FP8-dynamic", "397b7ba47a99b3221ebfc0cfc5a279118cb733ad")
NOUS_URL = "https://inference-api.nousresearch.com/v1"
NOUS_FW_MODEL = "deepseek/deepseek-v4.1-flash"
NOUS_FW_PRICE = (0.106, 0.513)  # $ per million tokens in / out (providers/ledger.py)
DEFAULT_THINKING_OFF: Mapping[str, Any] = {"reasoning": {"enabled": False}}  # live probe: 0 reasoning tokens; the
# {"thinking": ...} and chat_template_kwargs forms do NOT switch it off on the Nous portal
EMPTY_THINK = "<think>\n\n</think>\n\n"


@dataclasses.dataclass(frozen=True)
class Identity:
    """The pinned identity of a worker. ``runtime_version=None`` records the version without enforcing it."""

    served_model: str
    revision: str
    runtime: str
    runtime_version: Optional[str] = None

    @property
    def served_name(self) -> str:
        """The name the server is launched with (``--served-model-name`` / ``--alias``): the revision rides in it,
        because vLLM and llama.cpp cannot otherwise report which commit they loaded."""
        return self.served_model if self.revision == "none" else f"{self.served_model}@{self.revision}"

    def check(self, got: Mapping[str, Any]) -> Dict[str, str]:
        """Return ``got`` as a plain identity dict, or raise ``_Permanent`` naming the first mismatching field."""
        out = {k: str(got.get(k)) for k in ("served_model", "revision", "runtime", "runtime_version")}
        for key in ("served_model", "revision", "runtime"):
            if out[key] != getattr(self, key):
                raise _Permanent(f"identity mismatch: {key} is {out[key]!r}, pinned {getattr(self, key)!r}")
        if self.runtime_version is not None and out["runtime_version"] != self.runtime_version:
            raise _Permanent(f"identity mismatch: runtime_version is {out['runtime_version']!r}, "
                             f"pinned {self.runtime_version!r}")
        return out


def _split_served(name: str) -> Tuple[str, str]:
    model, sep, revision = name.rpartition("@")
    return (model, revision) if sep and model else (name, "none")


def _timing(elapsed_s: float, server_ms: int = 0, queue_ms: int = 0) -> Dict[str, int]:
    """``network_ms`` is wall time minus what the server reported (it includes unreported server time)."""
    wall = int(round(elapsed_s * 1000))
    return {"server_ms": int(server_ms), "network_ms": max(0, wall - int(server_ms) - int(queue_ms)),
            "queue_ms": int(queue_ms)}


def _label_probs_to_answers(request: DecisionRequest, probs: Mapping[str, Mapping[str, float]]) -> Dict[str, Answer]:
    """{qid: {label: p}} -> Answers. noul labels are yes/no; choice labels its options; score labels its options,
    keyed by option index ("0".."4"; the amount level is index + 1, INTERFACES section 8)."""
    out: Dict[str, Answer] = {}
    for qid, q in request.questions.items():
        p = probs[qid]
        if q.type == "noul":
            yes = min(1.0, max(0.0, p["yes"]))
            out[qid] = Answer("noul", yes, {"yes": yes, "no": 1.0 - yes})
        elif q.type == "choice":
            top = max(p[o] for o in q.options)
            out[qid] = Answer("choice", next(o for o in q.options if p[o] == top), {o: p[o] for o in q.options}, top)
        else:
            by_index = {str(i): p[o] for i, o in enumerate(q.options)}
            top = max(by_index.values())
            out[qid] = Answer("score", sum(int(k) * v for k, v in by_index.items()), by_index, top)
    return out


def _question_labels(q: Question) -> Tuple[str, ...]:
    return ("yes", "no") if q.type == "noul" else tuple(q.options)


def _short_ids(ids: Tuple[str, ...]) -> Dict[str, str]:
    """Shortest unique id prefix (>= 8 chars) -> full id; keeps the cite enum cheap in tokens."""
    for n in range(8, 65):
        short = {i[:n]: i for i in ids}
        if len(short) == len(set(ids)):
            return short
    return {i: i for i in ids}


def answer_schema(request: DecisionRequest, cite_ids: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """JSON schema of the single reply object: one enum string per question (in question order), plus ``cites``.

    Labels are strings, never numbers, so digits 1-5 follow a quote and tokenise as one token (L2: after a leading
    space they are two tokens on the Qwen3.5 and Bonsai tokenizers)."""
    props: Dict[str, Any] = {qid: {"type": "string", "enum": list(_question_labels(q))}
                             for qid, q in request.questions.items()}
    if cite_ids:
        props["cites"] = {"type": "array", "uniqueItems": True, "maxItems": min(12, len(cite_ids)),
                          "items": {"type": "string", "enum": list(_short_ids(cite_ids))}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_LABEL_SYSTEM = ("You answer typed questions about a state. Reply with ONE JSON object and nothing else. Each value "
                 "is exactly one of the labels listed for that question.")


def label_prompt(request: DecisionRequest, cite_ids: Tuple[str, ...] = ()) -> str:
    lines = [f"State: {request.state}", ""]
    for qid, q in request.questions.items():
        lines.append(f"{qid}: {q.instructions}")
        if q.type == "choice":
            lines.append("  labels: " + "; ".join(f"{o} = {q.criteria.get(o) or o}" for o in q.options))
        elif q.type == "noul":
            lines.append("  labels: yes; no")
        else:
            lines.append("  labels: " + "; ".join(q.options))
    if cite_ids:
        lines += ["cites: ids of the entries you relied on, from: " + ", ".join(_short_ids(cite_ids)), ""]
    return "\n".join(lines)


def extract_label_probs(request: DecisionRequest, content: str, logprob_content: Any
                        ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Per-label probabilities from the top logprobs at each answer position.

    Method (``prob_method="logprobs"``), per question ``qid`` with chosen label ``L``:

    1. ``content`` is the reply JSON; ``logprob_content`` is the server's per-token list (``token``, ``logprob``,
       ``top_logprobs``). The tokens must reassemble to ``content`` exactly.
    2. Find ``"<qid>": "`` in ``content``; the answer position is the token that starts at the next character. That
       token must be exactly ``L`` (a single-token label). A multi-token label is an error, not a guess.
    3. At that position, ``mass[label] = exp(logprob)`` for every offered label found in ``top_logprobs`` (the chosen
       token's own logprob is used if it is missing there).
    4. **Floor for absent labels.** A label missing from the top list is not zero, it is unseen. With
       ``residual = max(0, 1 - sum(found mass))`` and ``n_missing`` absent labels, each absent label gets
       ``min(residual / n_missing, min(found mass))``: it can never outrank a label that was actually seen. The
       question is then ``degraded`` (returned, logged with a warning, and set on the reply).
    5. Probabilities are the masses renormalised over the offered labels. Servers report logprobs from the raw
       distribution (before the JSON grammar masks illegal tokens), so renormalising removes the mass the grammar
       forbids.

    Returns ({qid: {label: p}}, {"labels": {qid: chosen}, "residual": {qid: float}, "degraded": [qid, ...]}).
    """
    try:
        chosen = json.loads(content)
    except ValueError as exc:
        raise DecisionError(f"logprobs: reply is not JSON ({exc})") from None
    if not isinstance(chosen, dict) or not set(request.questions) <= set(chosen):
        raise DecisionError("logprobs: reply keys do not match the question ids")
    if not isinstance(logprob_content, list) or not logprob_content:
        raise DecisionError("logprobs: the server returned no per-token logprobs")
    spans, pos = [], 0
    for entry in logprob_content:
        tok = entry.get("token") if isinstance(entry, dict) else None
        if not isinstance(tok, str):
            raise DecisionError("logprobs: malformed token entry")
        spans.append((pos, pos + len(tok)))
        pos += len(tok)
    if pos != len(content) or "".join(e["token"] for e in logprob_content) != content:
        raise DecisionError("logprobs: tokens do not reassemble to the reply text")
    out: Dict[str, Dict[str, float]] = {}
    residual: Dict[str, float] = {}
    degraded: List[str] = []
    for qid, q in request.questions.items():
        labels, label = _question_labels(q), chosen[qid]
        if label not in labels:
            raise DecisionError(f"logprobs: {qid}={label!r} is not one of {list(labels)}")
        match = re.search(r'"' + re.escape(qid) + r'"\s*:\s*"', content)
        if match is None:
            raise DecisionError(f"logprobs: no value for {qid!r} in the reply")
        start = match.end()
        idx = next((i for i, (a, _) in enumerate(spans) if a == start), None)
        if idx is None or spans[idx][1] - spans[idx][0] != len(label) or content[start:spans[idx][1]] != label:
            raise DecisionError(f"logprobs: label {label!r} for {qid!r} is not a single token")
        entry = logprob_content[idx]
        mass: Dict[str, float] = {}
        for cand in entry.get("top_logprobs") or ():
            tok, lp = cand.get("token"), cand.get("logprob")
            if tok in labels and isinstance(lp, (int, float)) and math.isfinite(lp):
                mass[tok] = max(mass.get(tok, 0.0), math.exp(lp))
        own = entry.get("logprob")
        if label not in mass and isinstance(own, (int, float)) and math.isfinite(own):
            mass[label] = math.exp(own)
        if not mass or sum(mass.values()) <= 0.0:
            raise DecisionError(f"logprobs: no label of {qid!r} appears in the top logprobs")
        found = sum(mass.values())
        residual[qid] = max(0.0, 1.0 - found)
        missing = [l for l in labels if l not in mass]
        if missing:
            floor = min(residual[qid] / len(missing), min(mass.values()) * (1 - 1e-6))  # strictly below: a floored label never wins a tie
            mass.update({l: floor for l in missing})
            degraded.append(qid)
            logger.warning("logprobs: %s: %d label(s) %s absent from the top %d; floored at %.3g", qid, len(missing),
                           missing, TOP_LOGPROBS, floor)
        total = sum(mass.values())
        if total <= 0.0:
            raise DecisionError(f"logprobs: no label of {qid!r} has positive mass")
        out[qid] = {l: mass[l] / total for l in labels}
    return out, {"labels": {qid: chosen[qid] for qid in request.questions}, "residual": residual,
                 "degraded": degraded}


class _WorkerHTTPBackend(_TypedHTTPBackend):
    """HTTP worker with a pinned :class:`Identity`. Subclasses fill in the endpoint, body and parse hooks."""

    path = "/v1/systemone"
    prob_method = ""
    role_timeout_s = 120.0

    def __init__(self, base_url: str, *, identity: Identity, model: Optional[str] = None, **kw: Any) -> None:
        kw.setdefault("timeout_s", self.role_timeout_s)
        super().__init__(base_url, model=model or identity.served_name, **kw)
        self.identity = identity
        self._runtime_version: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "identity": dataclasses.asdict(self.identity), "prob_method": self.prob_method}

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # hooks ------------------------------------------------------------------
    def _body(self, request: DecisionRequest) -> Dict[str, Any]:
        return request.payload()

    async def _fetch_runtime_version(self, client: httpx.AsyncClient) -> Optional[str]:
        return None  # the reply itself carries it (d1 shim, Nous)

    def _parse(self, request: DecisionRequest, data: Mapping[str, Any]) -> Tuple[Dict[str, Answer], Dict[str, Any]]:
        """-> (answers, {"identity": {...}, "tokens_in", "tokens_out", "cites", "server_ms", "queue_ms", "cost"})."""
        raise NotImplementedError

    # ------------------------------------------------------------------------
    async def _once(self, request: DecisionRequest) -> DecisionResponse:
        self.calls += 1
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=False, trust_env=False,
                                         transport=self._transport) as client:
                if self._runtime_version is None:
                    self._runtime_version = await self._fetch_runtime_version(client)
                reply = await client.post(f"{self.base_url}{self.path}", json=self._body(request),
                                          headers=self._headers())
        except httpx.HTTPError as exc:
            raise _Transient(f"transport error: {type(exc).__name__}") from None
        elapsed = time.monotonic() - start
        if reply.status_code == 429 or reply.status_code >= 500:
            raise _Transient(f"HTTP {reply.status_code}")
        if reply.status_code != 200:
            raise _Permanent(f"HTTP {reply.status_code}: {reply.text[:200]}")
        if len(reply.content) > MAX_RESPONSE_BYTES:
            raise _Permanent("response exceeds size limit")
        try:
            data = reply.json()
            answers, meta = self._parse(request, data)
        except _Permanent:
            raise
        except DecisionError as exc:
            raise _Permanent(str(exc)) from None
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            raise _Transient(f"malformed response: {type(exc).__name__}: {exc}") from None
        identity = self.identity.check(meta["identity"])  # mismatch -> _Permanent -> DecisionError
        try:
            check_answers(request, answers)
        except DecisionError as exc:
            raise _Transient(str(exc)) from None
        tokens_in, tokens_out = int(meta["tokens_in"]), int(meta["tokens_out"])
        cost = meta.get("cost")
        return DecisionResponse(
            answers=answers, served_model=identity["served_model"], request_hash=request.request_hash,
            input_tokens=tokens_in, output_tokens=tokens_out,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else cost_usd(tokens_in, tokens_out, self.price),
            elapsed_s=elapsed, identity=identity,
            timing=_timing(elapsed, meta.get("server_ms", 0), meta.get("queue_ms", 0)),
            cites=tuple(meta.get("cites", ())), prob_method=meta.get("prob_method", self.prob_method),
            degraded=bool(meta.get("degraded", False)), reasoning_tokens=int(meta.get("reasoning_tokens", 0)),
            alarm=meta.get("alarm") or {})


class D1Backend(_WorkerHTTPBackend):
    """Liquid d1-3B behind ``scripts/v41_d1_server.py`` (transformers, ``trust_remote_code``, pinned revision).

    The request body is :meth:`DecisionRequest.payload` (``images`` = base64 PNG patches for local look). The shim
    answers ``{"model", "identity", "answers", "usage", "timing"}``; ``answers`` is d1's own ``system_one`` output.
    """

    name = "d1"
    prob_method = PROB_SYSTEM_ONE

    def __init__(self, base_url: str, *, revision: str = D1_REVISION, runtime_version: Optional[str] = None,
                 model: str = D1_MODEL, api_key: Optional[str] = None, **kw: Any) -> None:
        super().__init__(base_url, identity=Identity(D1_MODEL, revision, "transformers", runtime_version),
                         model=model, api_key=api_key, **kw)

    def _parse(self, request, data):
        usage, timing = data.get("usage") or {}, data.get("timing") or {}
        return parse_wire_answers(request, data["answers"]), {
            "identity": data["identity"], "tokens_in": usage.get("input_tokens", 0),
            "tokens_out": usage.get("output_tokens", 0), "cites": (),
            "server_ms": int(timing.get("server_ms", 0)), "queue_ms": int(timing.get("queue_ms", 0))}


class _ChatLabelBackend(_WorkerHTTPBackend):
    """OpenAI-compatible /v1/chat/completions with a JSON-schema constraint; probabilities from top logprobs."""

    path = "/v1/chat/completions"
    prob_method = PROB_LOGPROBS
    max_tokens_cap = 240

    def _extra_body(self) -> Dict[str, Any]:
        return {}

    def _messages(self, request: DecisionRequest, cite_ids: Tuple[str, ...]) -> list:
        return [{"role": "system", "content": _LABEL_SYSTEM},
                {"role": "user", "content": label_prompt(request, cite_ids)}]

    def _body(self, request: DecisionRequest) -> Dict[str, Any]:
        cites = tuple(request.brief_ids)
        schema = answer_schema(request, cites)
        return {**self._extra_body(), "model": self.model, "messages": self._messages(request, cites),
                "temperature": 0.0, "seed": request.seed, "logprobs": True, "top_logprobs": TOP_LOGPROBS,
                "max_tokens": min(self.max_tokens_cap, 24 + 12 * len(request.questions) + (120 if cites else 0)),
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "decision", "strict": True, "schema": schema}}}

    def _parse(self, request, data):
        choice = data["choices"][0]
        if choice.get("finish_reason") not in ("stop", None):
            raise DecisionError(f"{self.name}: generation did not finish normally ({choice.get('finish_reason')})")
        content = choice["message"]["content"]
        logprobs = (choice.get("logprobs") or {}).get("content")
        probs, method, degraded = self._label_probs(request, content, logprobs)
        short = _short_ids(tuple(request.brief_ids))
        listed = json.loads(content).get("cites", [])
        cites = tuple(sorted({short[c] for c in listed if c in short})) if isinstance(listed, list) else ()
        served, revision = _split_served(str(data.get("model") or ""))
        usage, timings = data.get("usage") or {}, data.get("timings") or {}
        return _label_probs_to_answers(request, probs), {
            "identity": {"served_model": served, "revision": revision, "runtime": self.identity.runtime,
                         "runtime_version": self._runtime_version},
            "tokens_in": usage["prompt_tokens"], "tokens_out": usage["completion_tokens"], "cites": cites,
            "server_ms": int(timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)), "prob_method": method,
            "degraded": degraded}

    def _label_probs(self, request, content, logprobs):
        probs, info = extract_label_probs(request, content, logprobs)
        return probs, self.prob_method, bool(info["degraded"])


class VLLMLogprobBackend(_ChatLabelBackend):
    """Qwen3.5 on vLLM. Launch with ``--served-model-name <hf id>@<revision>`` (scripts/v41_pod_serve.sh): the reply's
    ``model`` field then carries model and revision, and ``GET /version`` proves the runtime and its version."""

    name = "vllm"

    def __init__(self, base_url: str, hf_id: str, revision: str, *, runtime_version: Optional[str] = None,
                 api_key: Optional[str] = None, **kw: Any) -> None:
        super().__init__(base_url, identity=Identity(hf_id, revision, "vllm", runtime_version), api_key=api_key, **kw)

    def _extra_body(self) -> Dict[str, Any]:
        return {"chat_template_kwargs": {"enable_thinking": False}}

    async def _fetch_runtime_version(self, client: httpx.AsyncClient) -> Optional[str]:
        reply = await client.get(f"{self.base_url}/version", headers=self._headers())
        if reply.status_code != 200:
            raise _Transient(f"HTTP {reply.status_code} from /version")
        version = reply.json().get("version")
        if not isinstance(version, str) or not version:
            raise _Permanent("identity mismatch: /version does not look like vLLM")
        return version


class LlamaCppBackend(_ChatLabelBackend):
    """Ternary Bonsai 8B on llama.cpp (PrismML ``prism`` branch, Metal). Launch with ``--alias <repo>@<revision>``.

    ``json_schema`` grammar; logprobs are read from the same fields as vLLM. If the build returns none and
    ``onehot_fallback`` is False (default) the call fails; with True the answer is one-hot on the chosen label and the
    response ``prob_method`` is ``"onehot"``, which must never feed calibration or conformal fitting.
    Thinking: ``enable_thinking=False`` goes to the template; if the template has no switch set ``prefill_empty_think``
    to end the prompt with an empty think block (a trailing assistant message, which llama-server continues).
    Verified on the real Bonsai 8B Q2_0_g64 template (2026-10-13, docs/v4.1/MAC-MODELS.md): it has no thinking switch
    but its generation prompt already ends with an empty think block, so leave ``prefill_empty_think`` False.
    """

    name = "llamacpp"

    def __init__(self, base_url: str, hf_id: str, revision: str, *, runtime_version: Optional[str] = None,
                 prefill_empty_think: bool = False, onehot_fallback: bool = False, api_key: Optional[str] = None,
                 **kw: Any) -> None:
        super().__init__(base_url, identity=Identity(hf_id, revision, "llama.cpp", runtime_version),
                         api_key=api_key, **kw)
        self.prefill_empty_think, self.onehot_fallback = prefill_empty_think, onehot_fallback

    def _extra_body(self) -> Dict[str, Any]:
        return {"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}

    def _messages(self, request, cite_ids):
        messages = super()._messages(request, cite_ids)
        return messages + [{"role": "assistant", "content": EMPTY_THINK}] if self.prefill_empty_think else messages

    async def _fetch_runtime_version(self, client: httpx.AsyncClient) -> Optional[str]:
        reply = await client.get(f"{self.base_url}/props", headers=self._headers())
        if reply.status_code != 200:
            raise _Transient(f"HTTP {reply.status_code} from /props")
        build = reply.json().get("build_info")
        if not isinstance(build, str) or not build:
            raise _Permanent("identity mismatch: /props does not look like llama.cpp")
        return build

    def _label_probs(self, request, content, logprobs):
        if logprobs or not self.onehot_fallback:
            return super()._label_probs(request, content, logprobs)
        chosen = json.loads(content)
        out = {}
        for qid, q in request.questions.items():
            labels = _question_labels(q)
            if chosen.get(qid) not in labels:
                raise DecisionError(f"llamacpp: {qid}={chosen.get(qid)!r} is not one of {list(labels)}")
            out[qid] = {l: float(l == chosen[qid]) for l in labels}
        return out, "onehot", False


class NousFWBackend(_WorkerHTTPBackend):
    """DeepSeek V4.1 Flash via the Nous portal (FW arm): the SLM prompt, thinking off, JSON output.

    Probabilities are verbalised (the model writes them), not logprobs; ``prob_method="verbalised"`` and the reply is
    parsed by :func:`parse_slm_object`. ``thinking_param`` is merged into the request body; the default
    ``{"reasoning": {"enabled": false}}`` is the form the live probe showed to work (0 reasoning tokens). Reasoning
    tokens from ``usage.completion_tokens_details`` are recorded; any above 0 sets an ``alarm`` payload on the reply.
    ``cites`` (optional list in the reply) is kept only for ids that are in ``request.brief_ids``.
    """

    name = "nous_fw"
    path = "/chat/completions"
    prob_method = PROB_VERBALISED

    def __init__(self, base_url: str = NOUS_URL, *, api_key: str, model: str = NOUS_FW_MODEL,
                 thinking_param: Optional[Mapping[str, Any]] = None, price: Tuple[float, float] = NOUS_FW_PRICE,
                 **kw: Any) -> None:
        if not api_key:
            raise ValueError("api_key is required (pass it in; this module never reads the environment)")
        super().__init__(base_url, identity=Identity(model, "none", "nous-portal", "none"), model=model,
                         api_key=api_key, price=price, **kw)
        self.thinking_param = dict(DEFAULT_THINKING_OFF if thinking_param is None else thinking_param)

    def _body(self, request: DecisionRequest) -> Dict[str, Any]:
        cites = tuple(request.brief_ids)
        prompt = _slm_prompt(request)
        if cites:
            prompt += ('\nAlso add "cites": a JSON list of the entry ids you relied on, chosen from: '
                       + ", ".join(_short_ids(cites)) + ".")
        return {**self.thinking_param, "model": self.model, "temperature": 0.0, "seed": request.seed,
                "max_tokens": slm_max_tokens(request) + (80 if cites else 0),
                "messages": [{"role": "system", "content": _SLM_SYSTEM}, {"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}

    def _parse(self, request, data):
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise DecisionError(f"{self.name}: generation did not finish normally ({choice.get('finish_reason')})")
        text = choice["message"]["content"].strip()
        fenced = _FENCE.match(text)
        try:
            raw = json.loads(fenced.group(1) if fenced else text)
        except ValueError as exc:
            raise DecisionError(f"{self.name}: reply is not JSON ({exc})") from None
        if not isinstance(raw, dict):
            raise DecisionError(f"{self.name}: reply is not a JSON object")
        short = _short_ids(tuple(request.brief_ids))
        listed = raw.pop("cites", [])
        cites = (tuple(sorted({short.get(c, c) for c in listed if isinstance(c, str)} & set(request.brief_ids)))
                 if isinstance(listed, list) else ())
        usage = data.get("usage") or {}
        reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        return parse_slm_object(request, raw), {
            "identity": {"served_model": str(data.get("model") or ""), "revision": "none", "runtime": "nous-portal",
                         "runtime_version": "none"},
            "tokens_in": usage["prompt_tokens"], "tokens_out": usage["completion_tokens"], "cites": cites,
            "cost": usage.get("cost"), "reasoning_tokens": reasoning,
            "alarm": ({"kind": "other", "value_ppm": reasoning,
                       "note": f"{reasoning} hidden reasoning tokens with thinking off"[:140]} if reasoning > 0 else {})}


# --- local-look patches ------------------------------------------------------------------------------------


def _png_gray8(pixels: Any) -> bytes:
    """Minimal 8-bit grayscale PNG (stdlib only)."""
    import struct
    import zlib

    h, w = pixels.shape

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + pixels[r].tobytes() for r in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def patch_to_images(patch: Any) -> Tuple[str, str]:
    """A (64, 64, 2) FLAIR+T1c patch (``perception.store.local_patch``) -> two base64 PNG strings, FLAIR first.

    Each channel is min-max scaled to 0..255 on its own (a constant channel becomes all zeros); deterministic."""
    import base64

    import numpy as np

    arr = np.asarray(patch, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError("patch must have shape (H, W, 2)")
    images = []
    for c in range(2):
        ch = arr[:, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        scaled = np.zeros(ch.shape, np.uint8) if hi <= lo else np.round((ch - lo) / (hi - lo) * 255.0).astype(np.uint8)
        images.append(base64.b64encode(_png_gray8(scaled)).decode("ascii"))
    return images[0], images[1]
