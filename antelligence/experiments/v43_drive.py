"""v4.3 drive layer: the engine shared by the verdict and effect drivers (fake-backend safe, no network, no keys).

A v4.3 copy of ``v42_drive``. It never imports a v4.1 or v4.2 SCRIPT (the on-disk writer is ``scripts/v43_run.py``'s ``run_one``).

* ``run_task(task)``   one episode (or one message-effect run) in this process; top level so a process pool can call it.
* ``run_pool(tasks)``  at most ``MAX_WORKERS`` (6) processes; ``workers=1`` runs inline (tests).
* ``guard_dev_cases``  every driver calls it before anything is loaded: evaluation / reserve / unknown ids exit with EvalRefused.
* ``uptake_level``     task override (``low`` | ``mid`` | ``high``) applied to ``facts.uptake_level`` of the config.
* the freeze guard     a SHARING ARM or CONTROL (anything but S, S-solo, S') on a development case is refused unless the freeze
  file matches the current code (``v43_freeze.require_frozen``). It is checked here, per task, so no driver can skip it.
* ``assert_out_allowed``  refuses an output folder inside any v4.1 / v4.2 evidence or config folder.

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
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from antelligence.experiments import v43_arms
from antelligence.experiments.v42_harness import CaseData, assert_dev_case, load_dev_case
from antelligence.experiments.v43_freeze import DEFAULT_FREEZE_FILE, require_frozen
from antelligence.experiments.v43_harness import V43Harness
from antelligence.worlds.tumor_mri.combined import DEFAULT_COEFFICIENTS_PATH
from antelligence.worlds.tumor_mri.config import UPTAKE_LEVEL_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "v4_3" / "world.yaml"
DEFAULT_CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
MAX_WORKERS = 6                                           # the Mac is shared
LEVELS = UPTAKE_LEVEL_NAMES
FORBIDDEN_OUT = (REPO_ROOT / "config" / "v4_1", REPO_ROOT / "config" / "v4_2", REPO_ROOT / "docs" / "research" / "v4.1-20261013",
                 REPO_ROOT / "docs" / "research" / "v4.2-20261014", Path("/Volumes/WD_BLACK/antelligence-v4.1"),
                 Path("/Volumes/WD_BLACK/antelligence-v4.2"))

_CASES: Dict[str, CaseData] = {}
_HARNESS: Dict[str, V43Harness] = {}
CASE_LOADER: Optional[Callable[[str, Optional[Path]], CaseData]] = None


class OutRefused(PermissionError):
    """The output folder is inside v4.1 / v4.2 evidence or configuration."""


def assert_out_allowed(out: Path) -> Path:
    """Return ``out`` resolved, or raise :class:`OutRefused` if it is (inside) a v4.1 / v4.2 evidence or config folder."""
    p = Path(out).resolve()
    for bad in FORBIDDEN_OUT:
        b = bad.resolve()
        if p == b or b in p.parents:
            raise OutRefused(f"refusing to write under {bad}: v4.3 output only")
    return p


def guard_dev_cases(case_ids: Iterable[str]) -> None:
    for c in case_ids:
        assert_dev_case(c)


def load_configs(path: Path):
    """(world config, colony config, ant rules) from a v4.3 config file (``scripts/v43_run.py``'s loader, one source)."""
    return _v43_run().load_configs(Path(path))


def _v43_run():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import v43_run                                       # the one writer of the on-disk layout
    finally:
        sys.path.pop(0)
    return v43_run


def _case(case_id: str, root: Optional[str]) -> CaseData:
    if case_id not in _CASES:
        _CASES[case_id] = (CASE_LOADER(case_id, Path(root) if root else None) if CASE_LOADER is not None
                           else load_dev_case(case_id, root=Path(root) if root else None))
    return _CASES[case_id]


def case_truth(case_id: str, root: Optional[Path] = None) -> np.ndarray:
    """Scorer-side tumour mask of a development case (for the honesty correlation in the results assembler)."""
    return _case(case_id, str(root) if root else None).crop.truth


def _harness(task: Mapping[str, Any], case: CaseData) -> V43Harness:
    v43_run = _v43_run()
    wc, cc, rules = v43_run.load_configs(Path(task["config"]))
    if task.get("ticks"):
        wc = wc.with_(ticks=int(task["ticks"]))
    level = task.get("uptake_level")
    if level is not None:
        if level not in LEVELS:
            raise ValueError(f"uptake_level must be one of {LEVELS}, got {level!r}")
        wc = wc.with_(facts=replace(wc.facts, uptake_level=level))
    coef = Path(task.get("coefficients") or DEFAULT_COEFFICIENTS_PATH)
    key = json.dumps([str(task["config"]), task.get("ticks"), wc.facts.__dict__, wc.drug_scale_k, bool(task.get("board", True)),
                      case.case_id, str(coef)], sort_keys=True, default=str)
    if key not in _HARNESS:
        if len(_HARNESS) > 24:
            _HARNESS.clear()
        _HARNESS[key] = V43Harness(wc, cc, rules, v43_run.load_estimate(rules, coef), with_board=bool(task.get("board", True)))
    return _HARNESS[key]


def _slim(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return {"arm": rec["arm"], "case_id": rec["case_id"], "seed": rec["seed"], "coverage": rec["metrics"]["coverage"],
            "off_target_share": rec["metrics"].get("off_target_share"), "timing_s": rec["timing_s"], "facts": rec.get("facts")}


def run_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    """``kind``: ``episode`` (writes bundle / record under ``out`` when given; resumable) or ``effect`` (one run's removal replays)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    v43_arms.register_all()                                  # idempotent; the global arm table may have been edited since import
    case = _case(task["case_id"], task.get("cases_root"))
    if case.split == "dev" and v43_arms.needs_freeze(task["arm"]):
        require_frozen(Path(task.get("freeze_file") or DEFAULT_FREEZE_FILE))
    h = _harness(task, case)
    if task["kind"] == "episode":
        if task.get("out"):
            rec = _v43_run().run_one(h, task["arm"], case, int(task["seed"]), Path(task["out"]))
            return {"skipped": True, **{k: rec[k] for k in ("arm", "case_id", "seed")}} if rec.get("skipped") else _slim(rec)
        return _slim(h.run_episode(task["arm"], case, int(task["seed"])))
    if task["kind"] == "effect":
        from antelligence.experiments.v43_metrics import effect_study
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
