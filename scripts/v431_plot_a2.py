#!/usr/bin/env python
"""v4.31 A.2: coverage-versus-threshold figure from results_a2.json, with the kill band (7-50) and the clinical level (262, range 55-651) marked. Aggregate curves only.

    python scripts/v431_plot_a2.py --results .../phaseA2/results_a2.json --out .../phaseA2/curves_a2.png

Solid = nominal truth mask; shaded = the +/-1-voxel erosion/dilation band on the scoring lattice. Blue S, orange S' (planned for T, scored at T), dashed grey the placement-free bound.
Orange-red band = kill band (derived by fitting one rat-glioma series); grey band + line = clinically delivered cavity-fluid exposure. Marks, never bars. Entry and motion are unphysical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from antelligence.experiments import v431_a2 as A2  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID, SURFACE, KILL, CLIN = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb", "#c23b22", "#6b6a66"
PANELS = (("ref_0.5mm", "PRIMARY k_ref = 2.0e-3 /s (L = 0.505 mm), scored at 0.5 mm"), ("ref_1mm", "same runs scored at 1 mm (not converged: shown for comparison)"),
          ("low_1mm", "secondary k = 1.2e-4 /s (L = 2.06 mm), 1 mm"), ("high_1mm", "secondary k = 6.0e-4 /s (L = 0.92 mm), 1 mm"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    d = json.loads(a.results.read_text())
    th = np.array(d["thresholds_umol_h_L"])
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.6), facecolor=SURFACE, sharey=True)
    for ax, (lab, title) in zip(axs.ravel(), PANELS):
        if lab not in d["sets"]:
            ax.set_visible(False)
            continue
        b = d["sets"][lab]
        S = {m: np.array(b["S"]["curves"][m]["plain"]["mean"]) for m in ("nominal", "eroded", "dilated")}
        dt = [pt for pt in d["plan_thresholds"] if f"{pt:g}" in b["Sp_plan"]]
        Pd = {m: np.array([np.interp(np.log10(pt), np.log10(th), b["Sp_plan"][f"{pt:g}"]["curves"][m]["plain"]["mean"]) for pt in dt]) for m in ("nominal", "eroded", "dilated")}
        ax.axvspan(*A2.KILL_BAND_UMOL_H_L, color=KILL, alpha=0.16, linewidth=0, zorder=1)
        ax.axvspan(*A2.CLINICAL_RANGE_UMOL_H_L, color=CLIN, alpha=0.10, linewidth=0, zorder=1)
        ax.axvline(A2.CLINICAL_MEAN_UMOL_H_L, color=CLIN, lw=1.2, ls=":", zorder=1)
        ax.plot(th, b["bound"]["mean_over_cases"], color=MUTED, lw=1.4, ls=(0, (5, 3)), label="any placement (bound)", zorder=2)
        for arr, color, label, xs in ((S, BLUE, "S (script)", th), (Pd, ORANGE, "S' planned for T", np.array(dt))):
            ax.fill_between(xs, arr["eroded"], arr["dilated"], color=color, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(xs, arr["nominal"], color=color, lw=2.0, label=label, zorder=3)
        if lab == "ref_0.5mm":
            ax.text(7.0, 1.0, "kill band 7-50\n(fitted, 1 rat study)", color=KILL, fontsize=8, ha="left", va="top")
            ax.text(262.0 * 1.15, 1.0, "clinical cavity-fluid\nexposure 262 (55-651)", color=CLIN, fontsize=8, ha="left", va="top")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xscale("log")
        ax.set_xlabel("exposure threshold T (umol*h/L)", color=MUTED)
        ax.set_facecolor(SURFACE)
        ax.set_ylim(-0.02, 1.03)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.legend(frameon=False, fontsize=9, loc="center right", labelcolor=INK)
    for ax in (axs[0, 0], axs[1, 0]):
        ax.set_ylabel("share of the tumour above T (plain coverage)", color=MUTED)
    fig.suptitle("Mean over 40 development cases x seeds 9 and 10. Shaded colour: +/-1-voxel erosion/dilation on the scoring lattice. Red band: kill band; grey: clinical level. Marks, not bars. "
                 "Entry and motion are unphysical.", fontsize=9.5, color=MUTED, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
