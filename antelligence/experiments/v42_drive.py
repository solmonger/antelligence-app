"""v4.2 H1: shared engine of the tune / verdict / effect drivers (fake-backend safe, no network, no keys).

* ``run_task(task)``   one episode (or one message-effect run) in this process; top-level so a process pool can call it.
* ``run_pool(tasks)``  at most ``MAX_WORKERS`` (6) processes; ``workers=1`` runs inline (tests).
* ``guard_dev_cases``  every driver calls it before anything is loaded: evaluation / reserve / unknown ids exit with EvalRefused.

Cases and harnesses are cached per process. ``CASE_LOADER`` may be replaced in a test (inline mode only).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import yaml

import antelligence.experiments.v42_arms  # noqa: F401  (registers the control arms)
from antelligence.experiments.v42_harness import (CaseData, EvalRefused, V42Harness, assert_dev_case, bundle_of, load_dev_case,
                                                  resolve_arm)
from antelligence.kernel.colony.episode import ColonyConfig
from antelligence.worlds.tumor_mri import WorldConfig
from antelligence.worlds.tumor_mri.config import FactsConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "v4_2" / "world.yaml"
DEFAULT_CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
MAX_WORKERS = 6                                           # the Mac is shared

_CASES: Dict[str, CaseData] = {}
_HARNESS: Dict[str, V42Harness] = {}
CASE_LOADER: Optional[Callable[[str, Optional[Path]], CaseData]] = None


def guard_dev_cases(case_ids: Iterable[str]) -> None:
    for c in case_ids:
        assert_dev_case(c)


def load_configs(path: Path):
    doc = yaml.safe_load(Path(path).read_text())
    return WorldConfig.from_dict(doc["world"]), ColonyConfig.from_dict({**doc.get("colony", {}), "quiet_rule": False})


def facts_for(base: FactsConfig, has_channels: bool, *, f2: Optional[float] = None, f3: Optional[Sequence[float]] = None,
              f2_share_b: float = 0.2) -> FactsConfig:
    """F1 always on. ``f2`` = u_core (Source A when the case has TC / ET channels, else Source B with ``f2_share_b``);
    ``f3`` = (decay, share). With neither given, the config's own flags stand (the frozen config)."""
    f = replace(base, f1_simlook=True)
    if f2 is not None:
        f = replace(f, f2=True, u_core=float(f2), f2_source="A" if has_channels else "B", f2_share=0.0 if has_channels else f2_share_b)
    if f3 is not None:
        f = replace(f, f3=True, f3_decay=float(f3[0]), f3_share=float(f3[1]))
    return f


def _case(case_id: str, root: Optional[str]) -> CaseData:
    if case_id not in _CASES:
        _CASES[case_id] = (CASE_LOADER(case_id, Path(root) if root else None) if CASE_LOADER is not None
                           else load_dev_case(case_id, root=Path(root) if root else None))
    return _CASES[case_id]


def _harness(task: Mapping[str, Any], case: CaseData) -> V42Harness:
    wc, cc = load_configs(Path(task["config"]))
    if task.get("ticks"):
        wc = wc.with_(ticks=int(task["ticks"]))
    f = task.get("facts") or {}
    wc = wc.with_(facts=facts_for(wc.facts, case.scorer_channels is not None, f2=f.get("f2"), f3=f.get("f3")))
    key = json.dumps([str(task["config"]), task.get("ticks"), wc.facts.__dict__, bool(task.get("board", True)), case.case_id],
                     sort_keys=True, default=str)
    if key not in _HARNESS:
        if len(_HARNESS) > 24:
            _HARNESS.clear()
        _HARNESS[key] = V42Harness(wc, cc, with_board=bool(task.get("board", True)))
    return _HARNESS[key]


def _slim(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return {"arm": rec["arm"], "case_id": rec["case_id"], "seed": rec["seed"], "coverage": rec["metrics"]["coverage"],
            "off_target_share": rec["metrics"].get("off_target_share"), "timing_s": rec["timing_s"],
            "facts": rec.get("facts")}


def run_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """``kind``: ``episode`` (writes bundle / record under ``out`` when given; resumable) or ``effect`` (one run's removal replays)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    case = _case(task["case_id"], task.get("cases_root"))
    h = _harness(task, case)
    if task["kind"] == "episode":
        if task.get("out"):
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            try:
                import v42_run                                  # the Q1 CLI module: one writer for the on-disk layout
            finally:
                sys.path.pop(0)
            rec = v42_run.run_one(h, task["arm"], case, int(task["seed"]), Path(task["out"]))
            return {"skipped": True, **{k: rec[k] for k in ("arm", "case_id", "seed")}} if rec.get("skipped") else _slim(rec)
        return _slim(h.run_episode(task["arm"], case, int(task["seed"])))
    if task["kind"] == "effect":
        from antelligence.experiments.v42_metrics import effect_study
        key = (task["case_id"], int(task["seed"]))
        bundle = task["bundle"]

        def replay_fn(_key, b, drop):
            rec = h.run_episode(task["arm"], case, int(task["seed"]), replay=b, drop=drop)
            return SimpleNamespace(effect=rec.get("effect"), metrics=rec["metrics"])

        r = effect_study(task["arm"], {key: bundle}, replay_fn, n_runs=1, max_items=int(task.get("max_items", 30)))
        return {"arm": task["arm"], "case_id": key[0], "seed": key[1], "rows": r["rows"], "unapplied": r["unapplied"]}
    raise ValueError(f"unknown task kind {task['kind']!r}")


def run_pool(tasks: Sequence[Mapping[str, Any]], workers: int, on_result: Callable[[Mapping[str, Any], Dict[str, Any]], None]) -> None:
    """Run ``tasks``; ``on_result(task, result)`` is called in this process as each finishes (write checkpoints there)."""
    if workers > MAX_WORKERS:
        raise ValueError(f"at most {MAX_WORKERS} workers (the machine is shared); asked for {workers}")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1 or len(tasks) <= 1:
        for t in tasks:
            on_result(t, run_task(t))
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for f in as_completed(futs):
            on_result(futs[f], f.result())
