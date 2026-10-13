"""Production backends for v4.1 (S7): the pod endpoint and its secret from the pod state file, the Nous key read
in-process, Bonsai on a localhost URL, every paid call ledgered with its cost and every call behind a budget stop.

Secrets are read here and handed straight to the backend objects. They are never logged, printed, put in a config dict
or left in an exception message (the ledger writer scrubs them from error text).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Union

from antelligence.experiments import v41_budget
from antelligence.experiments.v41_harness import BONSAI, D1, FW, Q4, Q9, QUEEN_MODEL, Worker
from antelligence.providers.decision_backends import (D1_REVISION, NOUS_URL, CachedDecision, D1Backend, LedgeredDecision,
                                                      LlamaCppBackend, NousFWBackend, VLLMLogprobBackend)
from antelligence.providers.ledger import LedgerWriter, LedgeredChat
from antelligence.providers.redact import register_secret
from antelligence.providers.openai_compat import OpenAICompatProvider
from antelligence.providers.wrappers import Cached

REPO = Path(__file__).resolve().parents[2]
POD_STATE = REPO / "ledger/v4.1-pod-state.json"
USAGE_LEDGER = REPO / "ledger/v4.1-usage.jsonl"
POD_COSTS = REPO / "ledger/v4.1-pod-costs.jsonl"
CACHE_DIR = REPO / "cache/v4.1-replies"
HERMES_ENV = Path("~/.hermes/.env")
NOUS_ENV_FILE_VAR = "ANTELLIGENCE_NOUS_ENV_FILE"   # test/override path for the env file holding NOUS_API_KEY

# Pinned identities (scripts/v41_pod_serve.sh and config/v4_1/model_revisions.json hold the same values).
Q9_ID, Q9_REV = "RedHatAI/Qwen3.5-9B-FP8-dynamic", "790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"
Q4_ID, Q4_REV = "RedHatAI/Qwen3.5-4B-FP8-dynamic", "397b7ba47a99b3221ebfc0cfc5a279118cb733ad"
BONSAI_ID, BONSAI_REV = "prism-ml/Ternary-Bonsai-8B-gguf", "c2aefbeb4b24469cd11579c3384b990404c17a30"
POD_PORTS = {Q9: 8100, Q4: 8101, D1: 8102}
DEFAULT_BONSAI_URL = "http://127.0.0.1:8080"


class ProdConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class PodEndpoint:
    """The pod's three proxy URLs (no ``/v1`` suffix; the backends add their own path) and its API secret."""

    urls: Dict[str, str]
    secret: str = field(repr=False)
    pod_id: str = ""


def read_pod_endpoint(state_path: Union[str, Path] = POD_STATE) -> PodEndpoint:
    path = Path(state_path)
    if not path.exists():
        raise ProdConfigError(f"pod state file not found: {path.name}; start the pod first (scripts/v41_pod.py up)")
    state = json.loads(path.read_text())
    base, secret = str(state.get("base_url") or ""), state.get("api_secret")
    if not base or not secret:
        raise ProdConfigError("pod state file has no base_url / api_secret")
    root = base.rstrip("/").removesuffix("/v1")
    if not root.endswith(f"-{POD_PORTS[Q9]}.proxy.runpod.net"):
        raise ProdConfigError("pod state base_url is not the worker pod's :8100 proxy URL")
    stem = root[: -len(f"-{POD_PORTS[Q9]}.proxy.runpod.net")]
    return PodEndpoint({n: f"{stem}-{port}.proxy.runpod.net" for n, port in POD_PORTS.items()}, str(secret),
                       str(state.get("pod_id") or ""))


def read_nous_key(env_path: Optional[Union[str, Path]] = None) -> str:
    """The dedicated ``NOUS_API_KEY`` from the Hermes env file (``ANTELLIGENCE_NOUS_ENV_FILE`` overrides the path).

    Read in-process with ``dotenv_values`` and returned to the caller only: never put into ``os.environ``, logged or
    printed. Hermes's own login token file is never read: it expires hourly and refreshing it would rotate
    Hermes's login (decisions/2026-10-13-nous-dedicated-key.md)."""
    from dotenv import dotenv_values

    path = Path(env_path or os.environ.get(NOUS_ENV_FILE_VAR) or HERMES_ENV).expanduser()
    try:
        key = (dotenv_values(path).get("NOUS_API_KEY") or "").strip() if path.is_file() else ""
    except (OSError, UnicodeDecodeError):
        key = ""
    if not key:
        raise ProdConfigError("NOUS_API_KEY not available")
    return key


class BudgetGuard:
    """Refuses a paid call when spend reached the stop: Nous >= $7.50, RunPod >= $24.00 (``v41_budget.STOPS``)."""

    def __init__(self, provider: str, usage: Union[str, Path] = USAGE_LEDGER, pods: Union[str, Path] = POD_COSTS) -> None:
        if provider not in v41_budget.STOPS:
            raise KeyError(provider)
        self.provider, self.usage, self.pods = provider, Path(usage), Path(pods)

    def check(self) -> None:
        v41_budget.check_or_refuse(self.provider, usage=self.usage, pods=self.pods)


class GuardedDecision:
    """Budget stop in front of a decision backend (put under the cache, over the ledger)."""

    def __init__(self, inner: Any, guard: BudgetGuard) -> None:
        self.inner, self.guard = inner, guard

    def describe(self) -> Dict[str, Any]:
        return self.inner.describe()

    async def decide(self, request: Any) -> Any:
        self.guard.check()
        return await self.inner.decide(request)


class GuardedChat:
    def __init__(self, inner: Any, guard: BudgetGuard) -> None:
        self.inner, self.guard = inner, guard

    def describe(self) -> Dict[str, Any]:
        return self.inner.describe()

    async def complete(self, request: Any) -> Any:
        self.guard.check()
        return await self.inner.complete(request)


def reply_cache_dir(cache_dir: Optional[Path], name: str, revision: str, pinned: Optional[str] = None) -> Optional[Path]:
    """Worker reply-cache subdirectory ``<worker>@<revision>``, so replies never cross model revisions. A legacy
    ``<worker>`` directory (written before revisions were in the name, always under the pinned revision) is migrated by
    rename when the requested revision is the pinned one, and left alone otherwise."""
    if not cache_dir:
        return None
    new, old = Path(cache_dir) / f"{name}@{revision}", Path(cache_dir) / name
    if pinned is not None and revision == pinned and old.exists() and not new.exists():
        old.rename(new)
    return new


def _stack(backend: Any, guard: Optional[BudgetGuard], ledger: LedgerWriter, cache: Optional[Path]) -> Worker:
    """cache -> guard -> ledger -> backend: a cache hit makes no call, a refused call writes no row."""
    wrapped: Any = LedgeredDecision(backend, ledger, role="worker")
    if guard is not None:
        wrapped = GuardedDecision(wrapped, guard)
    if cache is not None:
        wrapped = CachedDecision(wrapped, cache)
    inner = backend
    return Worker(wrapped, str(inner.model))


def build_live_workers(*, pod_state: Union[str, Path] = POD_STATE, bonsai_url: str = DEFAULT_BONSAI_URL,
                       needed: Optional[Set[str]] = None, usage_ledger: Union[str, Path] = USAGE_LEDGER,
                       pod_costs: Union[str, Path] = POD_COSTS, cache_dir: Optional[Path] = CACHE_DIR,
                       nous_key: Callable[[], str] = read_nous_key, bonsai_id: str = BONSAI_ID,
                       bonsai_revision: str = BONSAI_REV, bonsai_prefill_empty_think: bool = False,
                       fw_url: str = NOUS_URL, transport: Any = None) -> Dict[str, Worker]:
    """D1 / Q4 / Q9 on the pod, Bonsai on localhost, FW on Nous. Only the workers in ``needed`` are built, so a Mac-only
    run never touches the pod state or the Nous key."""
    want = lambda n: needed is None or n in needed  # noqa: E731
    ledger = LedgerWriter(usage_ledger)
    pod_guard = BudgetGuard("runpod", usage_ledger, pod_costs)
    out: Dict[str, Worker] = {}
    if any(want(n) for n in (D1, Q4, Q9)):
        pod = read_pod_endpoint(pod_state)
        ledger.add_redaction(pod.secret)
        register_secret(pod.secret)
        sub = lambda n, rev: reply_cache_dir(cache_dir, n, rev, rev)  # noqa: E731  (the pinned revision for D1/Q9/Q4)
        if want(D1):
            out[D1] = _stack(D1Backend(pod.urls[D1], revision=D1_REVISION, api_key=pod.secret), pod_guard, ledger, sub(D1, D1_REVISION))
        if want(Q9):
            out[Q9] = _stack(VLLMLogprobBackend(pod.urls[Q9], Q9_ID, Q9_REV, api_key=pod.secret), pod_guard, ledger, sub(Q9, Q9_REV))
        if want(Q4):
            out[Q4] = _stack(VLLMLogprobBackend(pod.urls[Q4], Q4_ID, Q4_REV, api_key=pod.secret), pod_guard, ledger, sub(Q4, Q4_REV))
    if want(BONSAI):
        out[BONSAI] = _stack(LlamaCppBackend(bonsai_url, bonsai_id, bonsai_revision,
                                             prefill_empty_think=bonsai_prefill_empty_think), None, ledger,
                             reply_cache_dir(cache_dir, BONSAI, bonsai_revision, BONSAI_REV))
    if want(FW):
        key = nous_key()
        ledger.add_redaction(key)
        register_secret(key)
        out[FW] = _stack(NousFWBackend(fw_url, api_key=key, **({"transport": transport} if transport else {})), BudgetGuard("nous", usage_ledger, pod_costs), ledger,
                         reply_cache_dir(cache_dir, FW, "none", "none"))
    return out


def build_live_queen(*, usage_ledger: Union[str, Path] = USAGE_LEDGER, pod_costs: Union[str, Path] = POD_COSTS,
                     cache_dir: Optional[Path] = CACHE_DIR, nous_key: Callable[[], str] = read_nous_key,
                     url: str = NOUS_URL) -> Any:
    """The queen on Nous: cache -> budget stop -> ledger (cost field) -> provider."""
    key = nous_key()
    ledger = LedgerWriter(usage_ledger, redact=[key])
    register_secret(key)
    provider: Any = LedgeredChat(OpenAICompatProvider(url, api_key=key), ledger, role="queen",
                                 expected_model=QUEEN_MODEL, backend="nous", prefix=True)
    provider = GuardedChat(provider, BudgetGuard("nous", usage_ledger, pod_costs))
    return Cached(provider, Path(cache_dir) / "queen.jsonl") if cache_dir else provider
