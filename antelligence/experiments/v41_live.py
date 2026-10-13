"""Build the live pieces of a v4.1 harness from a JSON config: worker backends (wrapped in the reply cache and the
paid-call ledger), the queen provider, the case loader. Nothing is read from disk or the environment here except what
the caller passes in: API keys come from ``get_env`` (named by ``api_key_env`` in the config), so a test can pass a dict.

Config shape (all URLs and revisions are the operator's; model revisions default to ``config/v4_1/model_revisions.json``
ids pinned in ``decision_backends``)::

  {"cache_dir": "...", "ledger_path": "...",
   "d1": {"url": "http://127.0.0.1:8530"},
   "qwen3.5-4b": {"url": "...", "hf_id": "...", "revision": "..."},
   "qwen3.5-9b": {"url": "...", "hf_id": "...", "revision": "..."},
   "bonsai-8b": {"url": "...", "hf_id": "...", "revision": "...", "prefill_empty_think": false},
   "fw": {"url": "https://inference-api.nousresearch.com/v1", "api_key_env": "NOUS_API_KEY"},
   "queen": {"url": "https://inference-api.nousresearch.com/v1", "api_key_env": "NOUS_API_KEY"},
   "hardware": {"d1": "L40S", "bonsai-8b": "M4 Max"}}
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from antelligence.experiments.v41_harness import BONSAI, D1, FW, Q4, Q9, Worker
from antelligence.providers.decision_backends import (CachedDecision, D1Backend, LayaBackend, LlamaCppBackend,  # noqa: F401
                                                      LedgeredDecision, NousFWBackend, VLLMLogprobBackend)
from antelligence.providers.ledger import LedgerWriter
from antelligence.providers.openai_compat import OpenAICompatProvider
from antelligence.providers.wrappers import Cached

DEFAULT_LOADER = "antelligence.worlds.tumor_mri.store_adapter:harness_loader"


class LiveConfigError(ValueError):
    pass


def resolve_loader(spec: str = DEFAULT_LOADER) -> Callable[..., Any]:
    """``module:function`` -> the case loader. The default is L2c's adapter, which lands with that branch."""
    module, _, attr = spec.partition(":")
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise LiveConfigError(f"cannot import case loader {spec!r}: {exc}") from exc


def _key(section: Mapping[str, Any], get_env: Callable[[str], Optional[str]]) -> str:
    name = section.get("api_key_env")
    value = get_env(name) if name else None
    if not value:
        raise LiveConfigError(f"environment variable {name!r} is not set")
    return value


def build_workers(cfg: Mapping[str, Any], get_env: Callable[[str], Optional[str]], *, needed: Optional[set] = None
                  ) -> Dict[str, Worker]:
    """Live backends for the workers in ``needed`` (all configured if None), each behind cache and ledger."""
    cache_dir = Path(cfg["cache_dir"]) if cfg.get("cache_dir") else None
    ledger = LedgerWriter(cfg["ledger_path"]) if cfg.get("ledger_path") else None
    out: Dict[str, Worker] = {}
    for name in (D1, Q4, Q9, BONSAI, FW):
        if (needed is not None and name not in needed) or name not in cfg:
            continue
        sec = cfg[name]
        if name == D1:
            backend: Any = D1Backend(sec["url"], **({"revision": sec["revision"]} if "revision" in sec else {}))
        elif name in (Q4, Q9):
            backend = VLLMLogprobBackend(sec["url"], sec["hf_id"], sec["revision"])
        elif name == BONSAI:
            backend = LlamaCppBackend(sec["url"], sec["hf_id"], sec["revision"],
                                      prefill_empty_think=bool(sec.get("prefill_empty_think", False)))
        else:
            backend = NousFWBackend(sec.get("url") or "https://inference-api.nousresearch.com/v1", api_key=_key(sec, get_env))
        if ledger is not None:
            backend = LedgeredDecision(backend, ledger, role="worker")
        if cache_dir is not None:
            backend = CachedDecision(backend, cache_dir / name)
        out[name] = Worker(backend, _inner_model(backend))
    return out


def _inner_model(backend: Any) -> str:
    while not hasattr(backend, "model") and hasattr(backend, "inner"):
        backend = backend.inner
    return str(backend.model)


def build_queen_provider(cfg: Mapping[str, Any], get_env: Callable[[str], Optional[str]]) -> Any:
    sec = cfg["queen"]
    provider: Any = OpenAICompatProvider(sec["url"], api_key=_key(sec, get_env))
    if cfg.get("cache_dir"):
        provider = Cached(provider, Path(cfg["cache_dir"]) / "queen.jsonl")
    return provider


def load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
