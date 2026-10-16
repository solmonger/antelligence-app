#!/usr/bin/env python
"""v4.31 Phase A: coverage-versus-threshold figure from results.json (aggregate curves only; no case data).

    python scripts/v431_plot.py --results docs/research/v4.31-20261016/phaseA/stepD/results.json --out docs/research/v4.31-20261016/phaseA/stepD/curves.png

Three panels with their own x axes (real units cannot share an axis with the legacy world's arbitrary dose units). Solid line = nominal truth mask,
shaded band = the +/-1-voxel erosion/dilation band. S: blue. S' (planned for T, scored at T = the attainable ceiling): orange. Dashed grey = the
placement-free conservation bound. Entry and motion are unphysical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"


def band(ax, x, c, color, label, lw=2.0, z=3):
    ax.fill_between(x, c["eroded"], c["dilated"], color=color, alpha=0.16, linewidth=0, zorder=z - 1)
    ax.plot(x, c["nominal"], color=color, lw=lw, zorder=z, label=label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    allowed = (REPO / "docs" / "research" / "v4.31-20261016").resolve()
    if allowed not in a.out.resolve().parents:
        raise SystemExit(f"refused: {a.out} is not under {allowed}")
    d = json.loads(a.results.read_text())
    th_u, th_l = np.array(d["thresholds_units"]), np.array(d["thresholds_legacy"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), facecolor=SURFACE, sharey=True)
    for ax, k, title in ((axes[0], "low", "Real units, k = 1.2e-4 /s (L = 2.06 mm)"), (axes[1], "high", "Real units, k = 6.0e-4 /s (L = 0.92 mm)")):
        u = d["units"][k]
        S = {b: np.array(u["S"]["curves"][b]["plain"]["mean"]) for b in ("nominal", "eroded", "dilated")}
        diag_t = [pt for pt in d["plan_thresholds_units"] if f"{pt:g}" in u["Sp_diagonal_mean"]["nominal"]["plain"]]
        P = {b: np.array([u["Sp_diagonal_mean"][b]["plain"][f"{pt:g}"] for pt in diag_t]) for b in ("nominal", "eroded", "dilated")}
        ax.plot(th_u, u["bound"]["mean_over_cases"], color=MUTED, lw=1.5, ls=(0, (5, 3)), label="any placement (bound)", zorder=2)
        band(ax, th_u, S, BLUE, "S (script)")
        band(ax, np.array(diag_t), P, ORANGE, "S' planned for T")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("exposure threshold T (umol*h/L)", color=MUTED)
        ax.set_xscale("log")
        ax.text(0.03, 0.04, f"S' at T=1e3: {P['nominal'][diag_t.index(1000.0)]:.2f}", transform=ax.transAxes, color=ORANGE, fontsize=9)
    ax = axes[2]
    for arm, color, name in (("S", BLUE, "S (script)"), ("Sp", ORANGE, "S' (plans for D_t = 5)")):
        c = d["legacy"][arm]["curves"]
        band(ax, th_l, {b: np.array(c[b]["plain"]["mean"]) for b in ("nominal", "eroded", "dilated")}, color, name)
    ax.axvline(5.0, color=MUTED, lw=1, ls=":")
    ax.text(5.6, 0.12, "D_t = 5", color=MUTED, fontsize=9)
    ax.set_title("Legacy world (v4.3 constants, F off), its own dose units", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("threshold T (arbitrary dose units)", color=MUTED)
    ax.set_xscale("log")
    axes[0].set_ylabel("share of the tumour above T (plain coverage)", color=MUTED)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_ylim(-0.02, 1.03)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    fig.suptitle("Mean over 40 development cases x seeds 9 and 10. Shaded: +/-1-voxel erosion/dilation of the truth mask. Entry and motion are unphysical here.",
                 fontsize=9.5, color=MUTED, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
