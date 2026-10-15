"""v4.2 H1: results assembler. Walks an output directory written by ``scripts/v42_run.py`` (or the verdict driver)::

    <out>/<arm>/<case>/seed<N>/{bundle.json, record.json}

and builds ``results.json`` in the schema documented in ``v42_report`` (``runs[]`` with arm, case_id, seed, split, coverage and the
``run_metrics`` fields). Development cases only: a case that is not in ``dev_a`` / ``dev_b`` of ``config/v4_1/split.json``
(read-only) - evaluation, reserve, unknown or synthetic - raises :class:`EvalRefused` before any file of it is parsed.
An episode counts only when its ``record.json`` says ``complete``; the others are listed in ``meta.incomplete``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from antelligence.experiments.v41_gate import load_checked_split
from antelligence.experiments.v42_harness import EvalRefused
from antelligence.experiments.v42_metrics import claim_accuracy_by_ant, run_metrics
from antelligence.worlds.tumor_mri.data import VOXEL_MM

REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED = re.compile(r"^seed(\d+)$")


def split_of(doc: Mapping[str, Any], case_id: str) -> str:
    """``"dev-A"`` / ``"dev-B"``; anything else is refused."""
    if case_id in doc["dev_a"]:
        return "dev-A"
    if case_id in doc["dev_b"]:
        return "dev-B"
    raise EvalRefused(f"case {case_id!r} is not a development case; the assembler takes development cases only")


def _episodes(out: Path):
    for arm_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for case_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in case_dir.iterdir() if p.is_dir() and _SEED.match(p.name)):
                yield arm_dir.name, case_dir.name, int(_SEED.match(seed_dir.name).group(1)), seed_dir  # type: ignore[union-attr]


def assemble(out_dir: Path, *, split_doc: Optional[Mapping[str, Any]] = None,
             truth_for: Optional[Callable[[str], Optional[np.ndarray]]] = None, voxel_mm: float = VOXEL_MM,
             meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``truth_for(case_id)`` returns the scorer-side tumour mask (for the honesty correlation), or None to skip it."""
    out = Path(out_dir)
    doc = split_doc if split_doc is not None else load_checked_split(REPO_ROOT)
    found = list(_episodes(out))
    for _, case, _, _ in found:                                    # refuse first, parse nothing
        split_of(doc, case)
    masks: Dict[str, Optional[np.ndarray]] = {}
    runs: List[Dict[str, Any]] = []
    incomplete: List[str] = []
    for arm, case, seed, d in found:
        try:
            rec = json.loads((d / "record.json").read_text())
        except (OSError, ValueError):
            rec = {}
        if not rec.get("complete"):
            incomplete.append(f"{arm}/{case}/seed{seed}")
            continue
        if rec.get("split") not in (None, "dev"):
            raise EvalRefused(f"{arm}/{case}/seed{seed}: record split {rec.get('split')!r}; development cases only")
        colony = None
        if arm != "S" and arm != "S'":
            colony = (json.loads((d / rec.get("bundle", "bundle.json")).read_text()).get("extra") or {}).get("colony")
        acc = None
        if colony is not None and truth_for is not None:
            if case not in masks:
                masks[case] = truth_for(case)
            m = masks[case]
            if m is not None:
                acc = claim_accuracy_by_ant(colony, lambda pos, m=m: bool(m[tuple(int(c) for c in pos)]))
        row = run_metrics(rec["metrics"], colony, accuracy=acc, voxel_mm=voxel_mm)
        row.update({"arm": rec["arm"], "case_id": case, "seed": seed, "split": split_of(doc, case),
                    "episode_s": rec.get("timing_s"), "n_regions": rec.get("n_regions")})
        if rec.get("control") is not None:
            row["control"] = rec["control"]
        runs.append(row)
    runs.sort(key=lambda r: (r["arm"], r["case_id"], r["seed"]))
    m = {"source_dir": str(out), "n_runs": len(runs), "incomplete": sorted(incomplete), **(dict(meta) if meta else {})}
    return {"meta": m, "runs": runs}


def write_results(out_dir: Path, path: Optional[Path] = None, **kw: Any) -> Dict[str, Any]:
    res = assemble(out_dir, **kw)
    target = Path(path) if path else Path(out_dir) / "results.json"
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    os.replace(tmp, target)
    return res
