#!/usr/bin/env python
"""v4.31 A.2: markdown tables for G0-approval-a2.md, generated from results_a2.json (no number typed by hand).

    python scripts/v431_tables_a2.py --results .../phaseA2/results_a2.json --out .../phaseA2/tables_a2.md

Every coverage figure carries its threshold, its scoring lattice and its +/-1-voxel band: `nominal [eroded .. dilated]`, on that lattice. The kill band (7-50) and the clinical level (262)
are MARKS on the curve, flagged every time, never bars. Descriptive only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v431_a2 as A2  # noqa: E402

ORDER = ("ref_0.5mm", "ref_1mm", "low_1mm", "high_1mm")
COLS = (1.0, 7.0, 10.0, 23.5, 50.0, 100.0, 262.0, 1000.0, 1.0e4, 1.0e5)
MARK = {7.0: "7 (kill: half)", 23.5: "23.5 (kill: 1 log)", 50.0: "50 (kill band top)", 262.0: "262 (clinical mean, cavity fluid)"}


def fmt(nom: float, lo: float, hi: float, d: int = 2) -> str:
    return f"{nom:.{d}f} [{lo:.{d}f} .. {hi:.{d}f}]"


def _i(th: List[float], t: float) -> int:
    return min(range(len(th)), key=lambda j: abs(math.log10(th[j]) - math.log10(t)))


def name(label: str, b: Dict[str, Any]) -> str:
    ph = b["physical"]
    role = "PRIMARY" if b["role"] == "primary" else "secondary"
    return f"{label} ({role}; k = {ph['k_clear_s']:g} /s, L = {ph['derived']['L_mm']:.3f} mm, scored on a {b['scoring_grid_mm']:g} mm lattice)"


def headline(d: Dict[str, Any]) -> str:
    out = ["#### Headline: coverage at the kill band and at the clinical level (plain whole-tumour coverage; nominal [eroded .. dilated])", "",
           A2.KILL_FLAG, "", A2.CLINICAL_FLAG, "", A2.POTENCY_STATEMENT, "", "| set | arm | " + " | ".join(MARK[t] for t in A2.KILL_THRESHOLDS + (262.0,)) + " |",
           "|---|---|" + "---|" * 4]
    for lab in ORDER:
        if lab not in d["sets"]:
            continue
        b = d["sets"][lab]
        for arm, key in (("S (script)", "S"), ("S' planned for T (oracle ceiling)", "Sp_planned_for_T")):
            cells = []
            for t in A2.KILL_THRESHOLDS + (262.0,):
                h = b["headline_coverage"].get(f"{t:g}")
                cells.append(fmt(h["nominal"][key], h["eroded"][key], h["dilated"][key]) if h else "n/a")
            out.append(f"| {lab} ({'PRIMARY' if b['role'] == 'primary' else 'secondary'}) | {arm} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def curves(d: Dict[str, Any], lab: str, kind: str, title: str) -> str:
    b = d["sets"][lab]
    th = d["thresholds_umol_h_L"]
    heads = [MARK.get(t, f"{t:g}") for t in COLS]
    out = [f"#### {title}: {name(lab, b)}", "", "| arm | " + " | ".join(f"T = {h}" for h in heads) + " |", "|---|" + "---|" * len(COLS)]
    S = b["S"]["curves"]
    out.append("| S | " + " | ".join(fmt(S["nominal"][kind]["mean"][_i(th, t)], S["eroded"][kind]["mean"][_i(th, t)], S["dilated"][kind]["mean"][_i(th, t)]) for t in COLS) + " |")
    cells = []
    for t in COLS:
        pl = b["Sp_plan"].get(f"{t:g}")
        cells.append(fmt(pl["curves"]["nominal"][kind]["mean"][_i(th, t)], pl["curves"]["eroded"][kind]["mean"][_i(th, t)], pl["curves"]["dilated"][kind]["mean"][_i(th, t)]) if pl else "n/a")
    out.append("| S' planned for T | " + " | ".join(cells) + " |")
    out.append("| conservation bound (any placement) | " + " | ".join(f"{b['bound']['mean_over_cases'][_i(th, t)]:.2f}" for t in COLS) + " |")
    out.append("")
    return "\n".join(out)


def gap_table(d: Dict[str, Any], lab: str) -> str:
    b = d["sets"][lab]
    out = [f"#### The corrected gap test, {name(lab, b)}", "", "S' (planned for T, scored at T) minus S at T, plain coverage. Does the gap keep its sign and its order across the three masks?", "",
           "| T | gap, nominal | gap, eroded | gap, dilated | keeps sign and order? | why |", "|---|---|---|---|---|---|"]
    for t in sorted(float(k) for k in b["gap_test"]):
        g = b["gap_test"][f"{t:g}"]
        m = g["by_mask"]
        out.append(f"| {MARK.get(t, f'{t:g}')} | {m['nominal']['gap']:+.3f} | {m['eroded']['gap']:+.3f} | {m['dilated']['gap']:+.3f} | **{'yes' if g['consistency']['holds'] else 'no'}** | {g['consistency']['reason']} |")
    out.append("")
    return "\n".join(out)


def geometry_table(d: Dict[str, Any]) -> str:
    out = ["#### Off-target scored geometrically (MEASURE ONLY: no penalty is added to any score)", "",
           "No maximum tolerated concentration, exposure or dose for normal brain parenchyma exists for any agent in any species (the absence is structural), so mg outside the tumour has nothing to score "
           "against. What exists is the PRECISE catheter protocol (Mueller 2011): release point at least 25 mm below the surface, at least 5 mm from the pia, at least 10 mm from the cavity wall or ependyma. "
           "**The pial measures use the brain mask. There are NO ventricle labels in the pipeline (MSD labels are background, edema, non-enhancing and enhancing; the mask includes CSF) and no resection cavity "
           "(the scans are pre-operative), so the ependymal numbers use an unvalidated intensity proxy and are a sensitivity, not a result.**", "",
           "| set | arm | releases within 25 mm of the surface (share of drug) | within 5 mm of the pia | within 10 mm of the ependyma (proxy) | exposure within 5 mm of the pia | exposure within 10 mm of the ependyma (proxy) | proxy ventricle volume, mL (min / median / max) |",
           "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "n/a" if x is None else f"{x:.2f}"       # noqa: E731
    for lab in ("ref_1mm", "low_1mm", "high_1mm"):
        if lab not in d["sets"]:
            continue
        b = d["sets"][lab]
        for arm, key in (("S", "S"), ("S' planned for 23.5", "Sp_plan23.5"), ("S' planned for 262", "Sp_plan262")):
            g = b["geometry_1mm"].get(key)
            if not g:
                continue
            r, e, v = g["rule_breaking_mass_share_mean"], g["exposure_share_mean"], g["ventricle_proxy_mL_min_median_max"]
            out.append(f"| {lab.split('_')[0]} | {arm} | {f(r['depth_lt_25mm'])} | {f(r['pia_lt_5mm'])} | {f(r['ependyma_lt_10mm_proxy'])} | {f(e['within_5mm_of_pia'])} | {f(e['within_10mm_of_ependyma_proxy'])} | {v[0]:.0f} / {v[1]:.0f} / {v[2]:.0f} |")
    out.append("")
    return "\n".join(out)


def scalars(d: Dict[str, Any]) -> str:
    out = ["#### Off-target in mg beside the geometry (Phase A's measure, on the world's 2 mm grid, with its +/-1-voxel band), exposure that left the crop, release-voxel share", "",
           "| set | arm | released (mg) | off-target mg, nominal [eroded .. dilated] | exposure that left the crop (mg) | release-voxel share, 2 mm world |", "|---|---|---|---|---|---|"]
    for lab in ("ref_0.5mm", "low_1mm", "high_1mm"):
        if lab not in d["sets"]:
            continue
        b = d["sets"][lab]
        rows = [("S", b["S"])] + [(f"S' planned for {k}", v) for k, v in b["Sp_plan"].items() if float(k) in (7.0, 23.5, 50.0, 262.0, 1000.0)]
        for arm, a in rows:
            ot = a["off_target_mg"]
            out.append(f"| {lab.split('_')[0]} | {arm} | {a['released_total_mean']:.1f} | {fmt(ot['nominal'], ot['eroded'], ot['dilated'], 1)} | {a['lost_boundary_mg_exposure_left_the_crop']:.4f} | {a['release_voxel_share_of_exposure']:.2f} |")
    out.append("")
    return "\n".join(out)


def conservation(d: Dict[str, Any]) -> str:
    out = ["#### The conservation bound, validated on every run's scored curves", "", "| set | checks | violations | closest any run comes to the exposure budget |", "|---|---|---|---|"]
    for lab in ORDER:
        if lab in d["sets"]:
            c = d["sets"][lab]["conservation_check_all_runs"]
            out.append(f"| {lab} | {c['checks']:,} | {c['violations']} | {100 * c['max_fraction_of_exposure_budget_needed']:.0f}% |")
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
    d = json.loads(a.results.read_text())
    parts = ["<!-- generated by scripts/v431_tables_a2.py from results_a2.json; do not edit by hand -->", "", headline(d)]
    for lab in ORDER:
        if lab in d["sets"]:
            parts.append(curves(d, lab, "plain", "Plain whole-tumour coverage"))
    for lab in ORDER:
        if lab in d["sets"]:
            parts.append(gap_table(d, lab))
    parts += [geometry_table(d), scalars(d), conservation(d)]
    for lab in ("ref_0.5mm",):
        if lab in d["sets"]:
            parts.append(curves(d, lab, "w35", "Viability-weighted score (enhancing 1.0, label 2 at 0.3, oedema 0.35)"))
    a.out.write_text("\n".join(parts))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
