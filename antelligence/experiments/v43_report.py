"""v4.3 report generator: ``results.json`` -> ``REPORT.md`` (tables only; the findings are hand-written by the lead).

The body is the v4.2 report (``v42_report.render``: run info, primary metrics, colony metrics, timings, quiet pair, message
effect, verdict statistics). v4.3 adds, each with a ``Source:`` line that names the file the numbers come from:

* the level and the freeze that the run records in ``meta``;
* probes per run and probes saved by others' uptake marks (per arm, mean per run);
* drug released on poor-uptake tissue, and the gap to S' split by cause (from the diagnose output ``diagnostics_numbers.json``);
* the effect rate of uptake marks, apart from the other marks and the messages;
* the realised mean u per sub-region and level (from ``uptake/realised_u.json``, B1's file).

``cite`` is the committed path that the results will live at (default: the results file itself); every table names it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from antelligence.experiments.v42_report import ARM_ORDER, _arms, _f, _mean, _table
from antelligence.experiments.v42_report import compute_stats, render as render_v42

REPO_ROOT = Path(__file__).resolve().parents[2]
REALISED_U = "docs/research/v4.3-20261015/uptake/realised_u.json"
PARTS = ("rim", "core", "edema", "wt")
OLD_HEAD = ["# v4.2 Phase 0 report (generated)", "",
            "Generated from `results.json` by `scripts/v42_report.py`. Tables only. No interpretation here.", ""]


def _src(*paths: Any) -> str:
    return "Source: " + ", ".join(f"`{p}`" for p in paths if p) + "."


def load_realised_u(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = Path(path) if path else REPO_ROOT / REALISED_U
    return json.loads(p.read_text()) if p.is_file() else None


def realised_u_section(doc: Optional[Mapping[str, Any]], cite: str = REALISED_U) -> List[str]:
    if not doc:
        return ["## Realised mean u per sub-region and level", "", f"Not available: `{cite}` was not found.", ""]
    levels = doc["levels"]
    rows = []
    for lvl in ("low", "mid", "high"):
        if lvl not in levels:
            continue
        ow = levels[lvl]["overall_voxel_weighted"]
        rows.append([lvl, levels[lvl].get("n_cases")] + [_f(ow[p]["mean_u"], 3) if p in ow else "n/a" for p in PARTS])
    return ["## Realised mean u per sub-region and level", "",
            "Voxel-weighted over the development cases, after clipping to [0.05, 1.0].", "",
            _table(["level", "cases", "rim", "core", "edema", "whole tumour"], rows), "", _src(cite), ""]


def probes_section(runs: Sequence[Mapping[str, Any]], arms: Sequence[str], cite: str) -> List[str]:
    col = [a for a in arms if any(r["arm"] == a and r.get("colony") for r in runs)]
    if not col:
        return []
    def m(a: str, k: str) -> Optional[float]:
        return _mean([r.get(k) for r in runs if r["arm"] == a and r.get("colony")])
    return ["## Probes (mean per run)", "",
            "A missing counter counts as 0 (the probe rule is a separate workstream; a run without it shows 0).", "",
            _table(["arm", "runs", "probes", "probes saved by others' uptake marks"],
                   [[a, sum(r["arm"] == a for r in runs), _f(m(a, "probes"), 2), _f(m(a, "probes_saved_by_marks"), 2)] for a in col]), "",
            _src(cite), ""]


def diagnostics_sections(diag: Optional[Mapping[str, Any]], cite: str) -> List[str]:
    if not diag:
        return []
    L: List[str] = []
    pool = diag.get("poor_tissue_release") or {}
    if pool:
        L += ["## Drug released on poor-uptake tissue (true u < 0.4), mean per run", "",
              _table(["arm", "runs", "total drug", "released (from marks)", "on poor tissue", "share of released", "false-alarm releases (>4 mm)"],
                     [[a, d.get("n_runs"), _f(d.get("total_drug"), 1), _f(d.get("released_from_marks"), 1), _f(d.get("poor_patch"), 1),
                       _f(d.get("poor_share_of_released")), _f(d.get("false_alarm"), 1)] for a, d in pool.items()]), "",
              _src(cite), ""]
    gap = diag.get("gap_to_s_prime_by_cause") or {}
    if gap:
        L += ["## Gap to S' by cause (coverage, mean per run)", "",
              _table(["arm", "runs", "gap S' - arm", "false-alarm releases", "poor-patch releases", "unreleased drug", "remainder"],
                     [[a, d.get("n_runs"), _f(d.get("gap"), 4), _f(d.get("false_alarm"), 4), _f(d.get("poor_patch"), 4),
                       _f(d.get("unreleased"), 4), _f(d.get("remainder"), 4)] for a, d in gap.items()]), "",
              next((d["assumption"] for d in gap.values() if d.get("assumption")), ""), "",
              "A cause shown as n/a is not available for that arm (S has no saved log); the remainder then holds it.", "", _src(cite), ""]
    return L


def effect_section(eff: Optional[Mapping[str, Any]], cite: str) -> List[str]:
    """``eff``: ``{arm: {"by_group": {"all": ..., "uptake_marks": ..., "other_marks": ..., "messages": ...}}}``."""
    if not eff:
        return []
    rows = []
    for arm, e in sorted(eff.items()):
        for g, s in (e.get("by_group") or {}).items():
            rows.append([arm, g, s.get("n_items"), _f(s.get("effect_rate")), _f(s.get("mean_delta"), 4), _f(s.get("spam_rate"))])
    return ["## Message effect by kind (removal replay)", "",
            _table(["arm", "group", "items", "effect rate", "mean delta", "spam rate"], rows), "", _src(cite), ""] if rows else []


def render(results: Mapping[str, Any], stats: Optional[Mapping[str, Any]] = None, *, diagnostics: Optional[Mapping[str, Any]] = None,
           effect: Optional[Mapping[str, Any]] = None, realised_u: Optional[Mapping[str, Any]] = None, cite: Optional[str] = None,
           cite_diagnostics: str = "diagnostics_numbers.json", cite_effect: str = "effect.json", cite_realised_u: str = REALISED_U) -> str:
    runs = results["runs"]
    meta = results.get("meta") or {}
    cite = cite or (str(Path(meta["source_dir"]) / "results.json") if meta.get("source_dir") else "results.json")
    body = render_v42(results, stats)
    old = "\n".join(OLD_HEAD)
    new = "\n".join(["# v4.3 Phase 0 report (generated)", "",
                     "Generated from `results.json` by `scripts/v43_report.py`. Tables only. No interpretation here.", ""])
    text = body.replace(old, new, 1) if body.startswith(old) else new + "\n" + body
    L: List[str] = [text.rstrip(), "", _src(cite), ""]
    arms = _arms(runs)
    L += probes_section(runs, arms, cite)
    L += diagnostics_sections(diagnostics, cite_diagnostics)
    L += effect_section(effect, cite_effect)
    L += realised_u_section(realised_u if realised_u is not None else load_realised_u(), cite_realised_u)
    return "\n".join(L).rstrip() + "\n"


def write_report(results_path: Path, out_path: Path, **kw: Any) -> str:
    results = json.loads(Path(results_path).read_text())
    kw.setdefault("cite", str(results_path))
    text = render(results, **kw)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text
