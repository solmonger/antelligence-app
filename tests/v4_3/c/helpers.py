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


def write_test_freeze(path: Path, root: Path = REPO) -> Dict[str, Any]:
    """A freeze file for a TEST, written over the working tree of this repo into a temp path: exactly what ``write_freeze`` would
    record on a clean tree, without the ``dirty`` flag (the tree under test is, by definition, being edited). Never a real freeze."""
    import json
    from antelligence.experiments.v43_freeze import current
    doc = {**current(root), "uncommitted": [], "dirty": False}
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


SCRATCH_FILES = ("config/v4_3", "docs/research/v4.2-20261014/simlook/cells_v4_2.json",
                 "docs/research/v4.3-20261015/combined/coefficients.json", "config/v4_1/split.json")


def make_scratch_repo(dest: Path) -> Path:
    """A clean, committed scratch copy of everything the freeze covers (the code, config, three explicit files), in a temp dir.
    Tests edit it, never the real worktree; a freeze written over it is a real, clean ``write_freeze`` document."""
    import shutil
    import subprocess
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "antelligence", dest / "antelligence", ignore=ignore)
    (dest / "scripts").mkdir(parents=True)
    for p in sorted((REPO / "scripts").glob("v43_*.py")):
        shutil.copy2(p, dest / "scripts" / p.name)
    for rel in SCRATCH_FILES:
        src = REPO / rel
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest / rel, ignore=ignore)
        else:
            shutil.copy2(src, dest / rel)
    git = ["git", "-C", str(dest), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "scratch"]):
        subprocess.run([*git, *args], check=True, capture_output=True)
    return dest


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
