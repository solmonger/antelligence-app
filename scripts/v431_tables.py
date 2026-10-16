#!/usr/bin/env python
"""v4.31 Phase A: markdown tables for the G0 note, generated from results.json (no number is typed by hand).

    python scripts/v431_tables.py --results docs/research/v4.31-20261016/phaseA/stepD/results.json --out docs/research/v4.31-20261016/phaseA/stepD/tables.md

Every coverage figure is printed with its threshold and its +/-1-voxel band as `nominal [eroded .. dilated]`. Descriptive only; no bar, no verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEC_U = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
DEC_L = [0.1, 1.0, 5.0, 10.0, 100.0, 1000.0]      # 5.0 is the frozen D_t and is a grid point of the legacy curve


def _i(th: List[float], t: float) -> int:
    return min(range(len(th)), key=lambda j: abs(__import__("math").log10(th[j]) - __import__("math").log10(t)))


def fmt(nom: float, lo: float, hi: float, d: int = 2) -> str:
    return f"{nom:.{d}f} [{lo:.{d}f} .. {hi:.{d}f}]"


def _band_triplet(curves: Dict[str, Any], kind: str, idx: int):
    return curves["nominal"][kind]["mean"][idx], curves["eroded"][kind]["mean"][idx], curves["dilated"][kind]["mean"][idx]


def table_units(doc: Dict[str, Any], k: str, kind: str, title: str) -> str:
    u = doc["units"][k]
    th = doc["thresholds_units"]
    out = [f"#### {title}, clearance k = {k} ({u['physical']['k_clear_s']:g} /s, L = {u['physical']['derived']['L_mm']:.2f} mm)", "",
           "| arm | " + " | ".join(f"T = {t:g}" for t in DEC_U) + " |", "|---|" + "---|" * len(DEC_U)]
    S = u["S"]["curves"]
    out.append("| S | " + " | ".join(fmt(*_band_triplet(S, kind, _i(th, t))) for t in DEC_U) + " |")
    row = []
    for t in DEC_U:
        key = f"{t:g}"
        if key in u["Sp_plan"]:
            row.append(fmt(*_band_triplet(u["Sp_plan"][key]["curves"], kind, _i(th, t))))
        else:
            row.append("n/a")
    out.append("| S' planned for T | " + " | ".join(row) + " |")
    bound = u["bound"]["mean_over_cases"]
    out.append("| conservation bound (any placement) | " + " | ".join(f"{bound[_i(th, t)]:.2f}" for t in DEC_U) + " |")
    out.append("")
    return "\n".join(out)


def table_gap(doc: Dict[str, Any], k: str) -> str:
    u = doc["units"][k]
    th = doc["thresholds_units"]
    out = [f"#### S' (planned for T) minus S at T, plain coverage, k = {k}: the gap against the +/-1-voxel band", "",
           "| T | S | S' | gap (nominal) | band of the gap (eroded .. dilated) | band width of S' alone | gap larger than the band? |", "|---|---|---|---|---|---|---|"]
    for t in DEC_U:
        key = f"{t:g}"
        if key not in u["Sp_plan"]:
            continue
        j = _i(th, t)
        s = {b: u["S"]["curves"][b]["plain"]["mean"][j] for b in ("nominal", "eroded", "dilated")}
        p = {b: u["Sp_plan"][key]["curves"][b]["plain"]["mean"][j] for b in ("nominal", "eroded", "dilated")}
        gap = {b: p[b] - s[b] for b in s}
        width = abs(p["dilated"] - p["eroded"])
        out.append(f"| {t:g} | {s['nominal']:.2f} | {p['nominal']:.2f} | {gap['nominal']:+.2f} | {gap['eroded']:+.2f} .. {gap['dilated']:+.2f} | {width:.2f} | "
                   f"{'yes' if abs(gap['nominal']) > width else 'no'} |")
    out.append("")
    return "\n".join(out)


def table_legacy(doc: Dict[str, Any], kind: str) -> str:
    if "legacy" not in doc:
        return ""
    th = doc["thresholds_legacy"]
    out = [f"#### Legacy world (v4.3 constants, F1/F2/F3 off), {kind} coverage vs threshold in its OWN arbitrary dose units (D_t = 5)", "",
           "| arm | " + " | ".join(f"T = {t:g}" for t in DEC_L) + " |", "|---|" + "---|" * len(DEC_L)]
    for arm, name in (("S", "S"), ("Sp", "S' (plans against D_t = 5)")):
        c = doc["legacy"][arm]["curves"]
        out.append(f"| {name} | " + " | ".join(fmt(*_band_triplet(c, kind, _i(th, t))) for t in DEC_L) + " |")
    out.append("")
    return "\n".join(out)


def _t50(th: List[float], cov: List[float]) -> float:
    """Threshold (log-linear interpolation) where a nonincreasing coverage curve crosses 0.5; 0 if it starts below, inf if it never drops below."""
    import math
    if cov[0] < 0.5:
        return 0.0
    for j in range(1, len(th)):
        if cov[j] < 0.5 <= cov[j - 1]:
            f = (cov[j - 1] - 0.5) / (cov[j - 1] - cov[j])
            return 10.0 ** (math.log10(th[j - 1]) + f * (math.log10(th[j]) - math.log10(th[j - 1])))
    return float("inf")


def table_t50(doc: Dict[str, Any]) -> str:
    out = ["#### T50: the exposure level (umol*h/L) that half of the tumour reaches, plain coverage, nominal [eroded .. dilated]",
           "(a threshold-free summary of each curve; S' is read off its own diagonal, planned for T and scored at T, interpolated between decades)", "",
           "| k | S | S' (ceiling) | conservation bound |", "|---|---|---|---|"]
    th = doc["thresholds_units"]
    for k in ("low", "high"):
        if k not in doc["units"]:
            continue
        u = doc["units"][k]
        s_t = {b: _t50(th, u["S"]["curves"][b]["plain"]["mean"]) for b in ("nominal", "eroded", "dilated")}
        diag_th = [pt for pt in doc["plan_thresholds_units"] if f"{pt:g}" in u["Sp_diagonal_mean"]["nominal"]["plain"]]
        p_t = {b: _t50(diag_th, [u["Sp_diagonal_mean"][b]["plain"][f"{pt:g}"] for pt in diag_th]) for b in ("nominal", "eroded", "dilated")}
        b_t = _t50(th, u["bound"]["mean_over_cases"])
        out.append(f"| {k} | {s_t['nominal']:.3g} [{s_t['eroded']:.3g} .. {s_t['dilated']:.3g}] | {p_t['nominal']:.3g} [{p_t['eroded']:.3g} .. {p_t['dilated']:.3g}] | {b_t:.3g} |")
    out.append("")
    return "\n".join(out)


def table_scalars(doc: Dict[str, Any]) -> str:
    out = ["#### Off-target drug, exposure that left the crop, and grid honesty (mean over the 80 case-seeds)", "",
           "| world | arm | released | off-target (nominal [eroded .. dilated]) | exposure that left the crop | share of exposure inside the release voxel |",
           "|---|---|---|---|---|---|"]
    if "legacy" in doc:
        for arm, name in (("S", "S"), ("Sp", "S'")):
            a = doc["legacy"][arm]
            ot = a["off_target_share"]
            out.append(f"| legacy | {name} | {a['released_total_mean']:.0f} (arbitrary units) | share {fmt(ot['nominal'], ot['eroded'], ot['dilated'], 3)} | lost at the boundary {a['lost_boundary_share']:.3f} of released | n/a (stepper) |")
    for k in ("low", "high"):
        if k not in doc["units"]:
            continue
        u = doc["units"][k]
        a = u["S"]
        out.append(f"| units {k} k | S | {a['released_total_mean']:.1f} mg | {fmt(a['off_target_mg']['nominal'], a['off_target_mg']['eroded'], a['off_target_mg']['dilated'], 1)} mg "
                   f"(share {a['off_target_share']['nominal']:.2f}) | {a['lost_boundary_mg_exposure_left_the_crop']:.4f} mg | {a['release_voxel_share_of_exposure']:.2f} (max {a['release_voxel_share_max']:.2f}) |")
        for pt in doc["plan_thresholds_units"]:
            key = f"{pt:g}"
            if key in u["Sp_plan"]:
                a = u["Sp_plan"][key]
                out.append(f"| units {k} k | S' planned for {key} | {a['released_total_mean']:.1f} mg | {fmt(a['off_target_mg']['nominal'], a['off_target_mg']['eroded'], a['off_target_mg']['dilated'], 1)} mg "
                           f"(share {a['off_target_share']['nominal']:.2f}) | {a['lost_boundary_mg_exposure_left_the_crop']:.4f} mg | {a['release_voxel_share_of_exposure']:.2f} (max {a['release_voxel_share_max']:.2f}) |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    doc = json.loads(a.results.read_text())
    parts = ["<!-- generated by scripts/v431_tables.py from results.json; do not edit by hand -->", ""]
    for kind, title in (("plain", "Plain whole-tumour coverage"), ("w35", "Viability-weighted score (enhancing 1.0, label 2 at 0.3, oedema 0.35)"),
                        ("w50", "Viability-weighted score (oedema 0.5)")):
        for k in ("low", "high"):
            if k in doc["units"]:
                parts.append(table_units(doc, k, kind, title))
    for k in ("low", "high"):
        if k in doc["units"]:
            parts.append(table_gap(doc, k))
    parts.append(table_t50(doc))
    parts.append(table_legacy(doc, "plain"))
    parts.append(table_legacy(doc, "w35"))
    parts.append(table_scalars(doc))
    a.out.write_text("\n".join(parts))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
