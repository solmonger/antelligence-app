"""v4.31 A.2 analysis constants and helpers. Kept apart from v431_phaseA.py (which the runner imports) so analysis can change while a batch is running.

Every number here comes from the A.2 goal (``GOAL-antelligence-v4.31-phaseA2.md``), which sources it. Nothing is tuned.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

# --- the two marked levels --------------------------------------------------------------------------------------------------------------------------------
# The curve stays primary. These are MARKS on it, never pass or fail bars.

KILL_BAND_UMOL_H_L = (7.0, 50.0)
KILL_HALF_UMOL_H_L = 7.0              # half-kill (50% survival)
KILL_ONE_LOG_UMOL_H_L = 23.5          # one log of kill (10% survival)
KILL_THRESHOLDS = (7.0, 23.5, 50.0)
KILL_FLAG = ("The kill band is DERIVED BY FITTING a single clonogenic series in C6 rat glioma (two 1 h exposures; Caballero Navarro 1992, Histol Histopathol 7:347-351) with "
             "-log10(survival) = 0.0426 x exposure (half-kill near 7, one log of kill at 23.5 umol*h/L). Single study, one rat line, and not cleanly proportional at the top dose. "
             "It is a band, never a line.")

CLINICAL_MEAN_UMOL_H_L = 262.0
CLINICAL_RANGE_UMOL_H_L = (55.0, 651.0)
CLINICAL_FLAG = ("The clinically delivered exposure is MEASURED in human resection-cavity FLUID by mass spectrometry (Ohnishi 2022, Brain Sci 12:567, n = 14-15): mean 262 umol*h/L, range 55-651. "
                 "It is cavity fluid, not tissue, so it is an upper reference for what the approved therapy produces where the wafer sits, not a tissue target and not a bar.")
POTENCY_STATEMENT = ("The clinically delivered exposure (262) is about ten times the one-log-kill level (23.5): potency is not the limitation; transport is.")

# --- the caveat every report carries from now on -------------------------------------------------------------------------------------------------------------
OUTCOME_CAVEAT = ("CAVEAT, carried in every report: in the only human study reporting per-patient cavity exposure, exposure did NOT predict the timing or the location of recurrence "
                  "(1849 vs 2457 ng*day/mL, p = 0.393; local vs non-local 2666 vs 1608, p = 0.185; n = 15). It is small and underpowered, so this is 'not shown', not 'shown false'. But we score "
                  "voxels by a quantity that has never been shown to predict a patient outcome, and that belongs in the limitations of anything we publish.")
CALIBRATION_STATEMENT = ("k_ref is CALIBRATED to an external published measurement (carmustine's own wafer penetration profiles); it is not independently measured, and no carmustine tissue "
                         "clearance constant has ever been measured directly. That is legitimate calibration and the opposite of what v4.1 did, which tuned a constant to widen our own result.")
MOTION_STATEMENT = ("Entry and motion are still unphysical: squads walk through tissue one voxel per tick and arrive carrying their whole dose. Nothing here is a claim about delivery in a patient.")

# --- the human resection-cavity series (A.2 section 4) --------------------------------------------------------------------------------------------------------
CARMUSTINE_MW = 214.05
K_CAVITY_PER_H = 0.010                     # every segment of the human series falls in 0.0086-0.0113 /h
K_CAVITY_RANGE_PER_H = (0.0086, 0.0113)


def ng_per_ml_to_umol_per_L(x: float) -> float:
    return x / CARMUSTINE_MW


# (time in hours, ng/mL, kind): 'point' = a measured value; 'lower' = at least this, through that time
HUMAN_SERIES = (("1 h", 1.0, 697.0, "point"), ("day 1", 24.0, 220.0, "point"), ("day 7", 168.0, 176.0, "lower"), ("day 14", 336.0, 14.9, "point"), ("day 30", 720.0, 0.28, "point"))


# --- the corrected band test ----------------------------------------------------------------------------------------------------------------------------------
def gap_consistency(gap_nominal: float, gap_eroded: float, gap_dilated: float, *, order: float = 3.0, floor: float = 0.005) -> Dict[str, object]:
    """Does a paired gap (oracle minus script) keep its sign and its order across the nominal, eroded and dilated masks?

    sign: the three gaps have the same sign (none within ``floor`` of zero counts as a sign change only if it really flips; a gap smaller than ``floor`` in all three is 'not resolved').
    order: the largest magnitude is at most ``order`` x the smallest (the same order of magnitude).
    The OLD column compared the gap with the band on the oracle's ABSOLUTE LEVEL; a paired difference does not inherit a level's uncertainty, so it answered 'no' almost everywhere."""
    g = [gap_nominal, gap_eroded, gap_dilated]
    mags = [abs(x) for x in g]
    if max(mags) < floor:
        return {"holds": False, "reason": "not resolved (all three gaps below the floor)", "same_sign": None, "ratio": None}
    same_sign = all(x > 0 for x in g) or all(x < 0 for x in g)
    lo = min(mags)
    ratio = (max(mags) / lo) if lo > 0 else math.inf
    ok_order = ratio <= order
    return {"holds": bool(same_sign and ok_order), "same_sign": bool(same_sign), "ratio": float(ratio), "within_order": bool(ok_order),
            "reason": "keeps its sign and its order" if same_sign and ok_order else ("changes sign" if not same_sign else f"changes order (max/min = {ratio:.1f} > {order:g})")}


OLD_COLUMN_NOTE = ("The old column 'gap larger than the band?' compared the oracle-minus-script gap with the +/-1-voxel band on the oracle's ABSOLUTE LEVEL. A paired difference does not inherit a "
                   "level's uncertainty (the same mask shift moves both arms), so that column answered 'no' almost everywhere and contradicted the narrative drawn from the next column. "
                   "It is replaced by: does the gap keep its sign and its order across the nominal, eroded and dilated masks?")
