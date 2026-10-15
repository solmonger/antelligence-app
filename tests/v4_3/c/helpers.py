"""Shared builders for the C tests: fabricated results, a phantom loaded under development-case ids, the script loader.

No development case is ever run here. ``inject_phantom`` makes the drive layer's case loader return the synthetic phantom (with
synthetic TC / ET channels) for any id, so a driver can be exercised end to end without any MRI data.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pytest

from antelligence.experiments import v43_drive as drive
from antelligence.experiments.v41_gate import load_checked_split
from tests.v4_3.b1.helpers import phantom_with_channels

REPO = Path(__file__).resolve().parents[3]


def script(name: str):
    """Import ``scripts/<name>.py`` as a module (library use; nothing runs on import)."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dev_ids() -> Dict[str, List[str]]:
    doc = load_checked_split(REPO)
    return {"dev_a": list(doc["dev_a"]), "dev_b": list(doc["dev_b"])}


@pytest.fixture
def phantom_loader(monkeypatch):
    """Drive-layer case loader that returns the phantom (with channels) under whatever development id is asked for."""
    base = phantom_with_channels()

    def loader(case_id: str, root: Any = None):
        return replace(base, case_id=case_id, split="dev")

    monkeypatch.setattr(drive, "CASE_LOADER", loader)
    monkeypatch.setattr(drive, "_CASES", {})
    monkeypatch.setattr(drive, "_HARNESS", {})
    return loader


def fab_runs(arms_to_cov: Mapping[str, Mapping[tuple, float]], splits: Mapping[str, str]) -> Dict[str, Any]:
    """A ``results.json`` body from ``{arm: {(case, seed): coverage}}`` and ``{case: "dev-A"|"dev-B"}``."""
    runs = [{"arm": a, "case_id": c, "seed": s, "split": splits[c], "coverage": float(v)}
            for a, d in arms_to_cov.items() for (c, s), v in d.items()]
    return {"meta": {"source_dir": "fabricated"}, "runs": runs}


def case_names(n_a: int = 30, n_b: int = 10) -> Dict[str, str]:
    return {**{f"A{i:02d}": "dev-A" for i in range(n_a)}, **{f"B{i:02d}": "dev-B" for i in range(n_b)}}


def keys(splits: Mapping[str, str], seeds: Sequence[int]) -> List[tuple]:
    return [(c, s) for c in sorted(splits) for s in seeds]
