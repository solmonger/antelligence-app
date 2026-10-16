"""v4.31 Phase A: the world in real units. Physical parameter block and the analytic exposure kernel.

Reference system: **carmustine (BCNU) in a Gliadel wafer, locally delivered** (grounding report, "The world the evidence
points to"). Nothing in :class:`PhysicalParams` may ever be tuned to a result.

**Declared substitution (also in every report): no measured effective diffusivity of BCNU in brain exists.** The small-molecule
value measured for free tetramethylammonium (TMA+) in brain extracellular space, corrected for tortuosity, stands in for it.

**What Phase A does not fix.** Entry and motion stay unphysical: squads still walk through tissue one voxel per tick and still
arrive carrying their whole dose. Nothing in this module says anything about delivery in a patient.

Units. Space is millimetres, time seconds, mass milligrams inside the kernel. Exposure (the time integral of concentration, which
is what the scorer accumulates as ``dose``) is reported in **micromole-hours per litre** (umol*h/L) for BCNU
(:attr:`PhysicalParams.exposure_to_umol_h_L` converts mg*s/mm^3).

The closed form (instantaneous point release of mass M, homogeneous medium, diffusivity D, first-order loss k, L = sqrt(D/k)):

    AUC(r) = M / (4 pi D r) * exp(-r / L)                                          (infinite horizon)
    AUC_T(r) = M / (8 pi D r) * [exp(-r/L) erfc(a - b) + exp(r/L) erfc(a + b)]     (horizon T; a = r/(2 sqrt(DT)), b = sqrt(kT))

The second line is the exact episode truncation (no correction term): an episode of ``ticks`` ticks integrates a release made
at tick t only over the ticks that are left. Both are verified against a direct numeric time integral in the tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, special

# ---------------------------------------------------------------------------------------------------------------------
# Constants (each with its source; none is ever tuned)
# ---------------------------------------------------------------------------------------------------------------------

D_STAR_CM2_S = 5.1e-6
D_STAR_SOURCE = (
    "Free TMA+ diffusivity 1.311e-5 cm2/s at 37 C (Vorisek & Sykova 1997, J Neurophysiol 78:912) divided by gray-matter "
    "tortuosity^2 with lambda ~ 1.55-1.60 (Hrabetova et al. 2018, J Neurosci 38:9355) gives D* ~ 5.1e-6 cm2/s. "
    "DECLARED SUBSTITUTION: no measured BCNU effective diffusivity in brain exists; the small-molecule TMA value stands in."
)
K_CLEAR_LOW_S = 1.2e-4
K_CLEAR_HIGH_S = 6.0e-4
K_CLEAR_SOURCES = {
    "low": (
        "Bork et al. 2024, PNAS 121(16):e2318444121: Cserr-derived brain efflux constant k_p = 0.432 /h = 1.2e-4 /s (rat). "
        "A model-fitted blood-brain efflux rate, not a direct measurement. LOW end of a genuine 5x literature conflict "
        "(grounding report, transport ledger)."
    ),
    "high": (
        "Bork et al. 2024, PNAS 121(16):e2318444121: the Groothuis replication of the efflux constant, about 5x faster, "
        "= 6.0e-4 /s, a difference the authors attribute to anaesthetic choice. HIGH end of a genuine 5x literature conflict; "
        "not rounding, so both ends are carried and every result is reported for both."
    ),
}
TICK_SECONDS_DEFAULT = 1800.0
TICK_SOURCE = (
    "Declared in config (v4.31 goal, Step A): a derived quantity, never implicit. Default 1800 s (30 min). The grounding "
    "report shows the tick is a trade-off, not a measurement: spread-per-tick and loss-per-tick cannot both be ~1 voxel / small."
)
TICKS_DEFAULT = 120
TICKS_SOURCE = "v4.1 world.yaml episode length (120 ticks); 60 h at the default 1800 s tick (v4.31 goal, Step A)."
VOXEL_MM = 2.0
VOXEL_SOURCE = "Perception store grid: store_adapter requires a 2.0 mm crop grid (v4.31 goal, Step A: 'from the perception store')."
BUDGET_CAP_MG = 61.6
BUDGET_SOURCE = (
    "Gliadel label / NICE TA121: at most 8 wafers x 7.7 mg BCNU = 61.6 mg, the clinical maximum (grounding report, delivery "
    "ledger). A per-case cap, never scaled by perceived volume and never tuned."
)
BCNU_MOLAR_MASS_G_MOL = 214.05
MOLAR_MASS_SOURCE = (
    "Carmustine C5H9Cl2N3O2, 214.05 g/mol (standard molecular formula and atomic masses; NOT from the grounding report). "
    "Only used to convert mg to umol."
)
PHYSICAL_FIELDS = ("d_star_cm2_s", "k_clear_s", "tick_seconds", "ticks", "voxel_mm", "budget_mg", "molar_mass_g_mol")

# The standing acceptance test: 0.4 mm <= L = sqrt(D/k) <= 2.0 mm (goal, Step A). The goal itself computes L = 2.06 mm for the LOW
# clearance end of the pair and says: "report the exact figure and either narrow k or widen the band with a written reason, rather
# than quietly nudging D". k is a literature pair that must be carried whole and D is declared, so the band is widened.
L_BAND_STATED_MM = (0.4, 2.0)
L_BAND_MM = (0.4, 2.1)
L_BAND_REASON = (
    "The stated band 0.4-2.0 mm is a rounded literature band. The cited efflux constant itself (Bork 2024, 1.2e-4 /s) with the declared "
    "D* gives L = 2.06 mm, and the grounding report quotes L 'approx 0.9-2.1 mm' for that pair. The upper edge is widened from 2.0 to "
    "2.1 mm, 5%, so the low end of the carried k pair is accepted; D and k are not nudged. Any parameter set beyond 2.1 mm "
    "or below 0.4 mm is still refused."
)

MM2_PER_CM2 = 100.0


class PhysicalParamsError(ValueError):
    """A physical field has no source, or a value is out of range, or L = sqrt(D/k) misses the acceptance band."""


def _src_map(sources: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in sources.items()))


@dataclass(frozen=True)
class PhysicalParams:
    """Frozen, hashable, JSON-plain physical block in SI-convertible units. Every physical field has a mandatory non-empty source.

    ``sources`` is a sorted tuple of ``(field, source)`` pairs (a dict would not be hashable).
    """

    d_star_cm2_s: float
    k_clear_s: float
    tick_seconds: float
    ticks: int
    voxel_mm: float
    budget_mg: float
    molar_mass_g_mol: float
    sources: Tuple[Tuple[str, str], ...]
    k_label: str = "custom"           # "low" | "high" | "custom"; reporting only

    def __post_init__(self) -> None:
        src = dict(self.sources)
        for name in PHYSICAL_FIELDS:
            if not str(src.get(name, "")).strip():
                raise PhysicalParamsError(f"physical field {name!r} has no source string (a field without a source is a construction error)")
        extra = set(src) - set(PHYSICAL_FIELDS)
        if extra:
            raise PhysicalParamsError(f"sources for unknown fields: {sorted(extra)}")
        for name in ("d_star_cm2_s", "k_clear_s", "tick_seconds", "voxel_mm", "budget_mg", "molar_mass_g_mol"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):
                raise PhysicalParamsError(f"{name} must be a positive finite number, got {v!r}")
        if not isinstance(self.ticks, int) or self.ticks < 1:
            raise PhysicalParamsError(f"ticks must be a positive int, got {self.ticks!r}")
        if self.budget_mg > BUDGET_CAP_MG + 1e-9:
            raise PhysicalParamsError(f"budget_mg {self.budget_mg} exceeds the clinical cap {BUDGET_CAP_MG} mg (8 x 7.7 mg)")
        object.__setattr__(self, "sources", _src_map(src))
        lo, hi = L_BAND_MM
        if not lo <= self.L_mm <= hi:
            raise PhysicalParamsError(
                f"acceptance test failed: L = sqrt(D/k) = {self.L_mm:.4f} mm is outside [{lo}, {hi}] mm. Do not nudge D; "
                "narrow k or widen the band with a written reason.")

    # --- construction ----------------------------------------------------------------------------------------------

    @classmethod
    def carmustine_wafer(cls, k: str = "low", *, tick_seconds: float = TICK_SECONDS_DEFAULT, ticks: int = TICKS_DEFAULT,
                         budget_mg: float = BUDGET_CAP_MG) -> "PhysicalParams":
        """The reference parameter set. ``k`` selects an end of the carried clearance pair: "low" 1.2e-4 /s, "high" 6.0e-4 /s."""
        if k not in ("low", "high"):
            raise PhysicalParamsError(f"k must be 'low' or 'high', got {k!r}")
        k_val = K_CLEAR_LOW_S if k == "low" else K_CLEAR_HIGH_S
        return cls(
            d_star_cm2_s=D_STAR_CM2_S, k_clear_s=k_val, tick_seconds=float(tick_seconds), ticks=int(ticks), voxel_mm=VOXEL_MM,
            budget_mg=float(budget_mg), molar_mass_g_mol=BCNU_MOLAR_MASS_G_MOL,
            sources=_src_map({"d_star_cm2_s": D_STAR_SOURCE, "k_clear_s": K_CLEAR_SOURCES[k], "tick_seconds": TICK_SOURCE,
                              "ticks": TICKS_SOURCE, "voxel_mm": VOXEL_SOURCE, "budget_mg": BUDGET_SOURCE,
                              "molar_mass_g_mol": MOLAR_MASS_SOURCE}),
            k_label=k)

    # --- derived quantities (properties; never hand-copied constants) -----------------------------------------------------

    @property
    def d_star_mm2_s(self) -> float:
        return self.d_star_cm2_s * MM2_PER_CM2

    @property
    def L_mm(self) -> float:
        """Steady-state penetration length sqrt(D*/k) in mm."""
        return math.sqrt(self.d_star_mm2_s / self.k_clear_s)

    @property
    def diffuse_sigma_mm(self) -> float:
        return math.sqrt(2.0 * self.d_star_mm2_s * self.tick_seconds)

    @property
    def diffuse_sigma_vox(self) -> float:
        return self.diffuse_sigma_mm / self.voxel_mm

    @property
    def decay(self) -> float:
        """Per-tick loss fraction 1 - exp(-k dt)."""
        return -math.expm1(-self.k_clear_s * self.tick_seconds)

    @property
    def episode_hours(self) -> float:
        return self.ticks * self.tick_seconds / 3600.0

    @property
    def exposure_to_umol_h_L(self) -> float:
        """Multiplier from mg*s/mm^3 to umol*h/L: 1e6 mm^3/L / 3600 s/h * 1000 umol/mmol / (MW g/mol = mg/mmol)."""
        return 1.0e6 / 3600.0 * 1000.0 / self.molar_mass_g_mol

    @property
    def voxel_volume_mm3(self) -> float:
        return self.voxel_mm ** 3

    def source(self, name: str) -> str:
        return dict(self.sources)[name]

    def with_tick(self, tick_seconds: float, ticks: Optional[int] = None) -> "PhysicalParams":
        return PhysicalParams(self.d_star_cm2_s, self.k_clear_s, float(tick_seconds), self.ticks if ticks is None else int(ticks),
                              self.voxel_mm, self.budget_mg, self.molar_mass_g_mol, self.sources, self.k_label)

    # --- serialisation -----------------------------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["sources"] = {k: v for k, v in self.sources}
        d["derived"] = {"d_star_mm2_s": self.d_star_mm2_s, "L_mm": self.L_mm, "diffuse_sigma_mm": self.diffuse_sigma_mm,
                        "diffuse_sigma_vox": self.diffuse_sigma_vox, "decay": self.decay, "episode_hours": self.episode_hours,
                        "exposure_to_umol_h_L": self.exposure_to_umol_h_L}
        return d

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "PhysicalParams":
        keys = ("d_star_cm2_s", "k_clear_s", "tick_seconds", "ticks", "voxel_mm", "budget_mg", "molar_mass_g_mol")
        return cls(**{k: d[k] for k in keys}, sources=_src_map(d["sources"]), k_label=str(d.get("k_label", "custom")))  # type: ignore[arg-type]


def tick_tradeoff(base: PhysicalParams, ticks_s: Sequence[float] = (600.0, 1800.0, 3922.0)) -> List[Dict[str, float]]:
    """Rows for the tick trade-off table (goal, Step A): per-tick spread against per-tick loss and the episode length, for both ends
    of the clearance pair. 3922 s is the tick that makes the per-tick spread exactly one 2-mm voxel at the declared D (the 'spread = 1
    voxel' tick). The L acceptance test is independent of the tick."""
    rows = []
    for ts in ticks_s:
        row: Dict[str, float] = {"tick_seconds": float(ts)}
        for lab, k in (("low", K_CLEAR_LOW_S), ("high", K_CLEAR_HIGH_S)):
            row[f"loss_per_tick_{lab}"] = -math.expm1(-k * ts)
        row["spread_mm"] = math.sqrt(2.0 * base.d_star_mm2_s * ts)
        row["spread_vox"] = row["spread_mm"] / base.voxel_mm
        row["episode_hours"] = base.ticks * ts / 3600.0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# Closed-form kernel
# ---------------------------------------------------------------------------------------------------------------------


def penetration_length_mm(D_mm2_s: float, k_s: float) -> float:
    return math.sqrt(D_mm2_s / k_s)


def point_exposure_inf(r_mm, D_mm2_s: float, k_s: float):
    """AUC per unit mass (mg*s/mm^3 per mg) at distance ``r`` from an instantaneous point release, infinite horizon."""
    r = np.asarray(r_mm, dtype=np.float64)
    L = penetration_length_mm(D_mm2_s, k_s)
    return np.exp(-r / L) / (4.0 * math.pi * D_mm2_s * r)


def point_exposure_T(r_mm, D_mm2_s: float, k_s: float, T_s: float):
    """Exact AUC over a finite horizon ``T_s`` per unit mass (mg*s/mm^3 per mg). Stable for all r > 0 (uses erfcx)."""
    r = np.asarray(r_mm, dtype=np.float64)
    if math.isinf(T_s):
        return point_exposure_inf(r, D_mm2_s, k_s)
    if T_s <= 0.0:
        return np.zeros_like(r)
    a = r / (2.0 * math.sqrt(D_mm2_s * T_s))
    b = math.sqrt(k_s * T_s)
    s = a + b
    d = a - b
    # exp(2ab) erfc(a+b) = exp(-(a^2+b^2)) erfcx(a+b);  exp(-2ab) erfc(a-b) = exp(-(a^2+b^2)) erfcx(a-b) for a>=b, else direct.
    t2 = np.exp(-(a * a + b * b)) * special.erfcx(s)
    t1 = np.where(d >= 0.0, np.exp(-(a * a + b * b)) * special.erfcx(np.maximum(d, 0.0)), np.exp(-2.0 * a * b) * special.erfc(d))
    return (t1 + t2) / (8.0 * math.pi * D_mm2_s * r)


def numeric_time_integral(r_mm: float, D_mm2_s: float, k_s: float, T_s: float = math.inf) -> float:
    """Direct numeric time integral of (4 pi D t)^-1.5 exp(-r^2/(4Dt)) exp(-kt) dt on [0, T] (reference for the closed form)."""
    def g(t: float) -> float:
        return (4.0 * math.pi * D_mm2_s * t) ** -1.5 * math.exp(-r_mm * r_mm / (4.0 * D_mm2_s * t) - k_s * t)
    tc = r_mm * r_mm / (4.0 * D_mm2_s) + 1.0 / k_s
    if math.isinf(T_s):
        parts = [(0.0, tc), (tc, 20.0 * tc), (20.0 * tc, math.inf)]
    else:
        parts = [(0.0, min(tc, T_s))]
        if T_s > tc:
            parts.append((tc, T_s))
    tot = 0.0
    for lo, hi in parts:
        v, _ = integrate.quad(g, lo, hi, epsabs=0.0, epsrel=1e-13, limit=500)
        tot += v
    return tot


def x10_distance_mm(L_mm: float, r0_mm: float = 0.5, frac: float = 0.1) -> float:
    """Distance at which a point release's exposure falls to ``frac`` of its value at ``r0_mm`` (exposure ~ exp(-r/L)/r, so (r0/r)
    exp(-(r-r0)/L) = frac). The profile check of the goal uses r0 = 0.5 mm and frac = 0.1."""
    from scipy.optimize import brentq
    return float(brentq(lambda r: (r0_mm / r) * math.exp(-(r - r0_mm) / L_mm) - frac, r0_mm * 1.0000001, 200.0 * L_mm))


def planar_x10_mm(L_mm: float, frac: float = 0.1) -> float:
    """Distance at which a PLANAR source's steady profile exp(-x/L) falls to ``frac`` of its surface value (x10 = 2.303 L): the geometry
    of the measured wafer profiles (grounding report). Reported beside the point-source figure, never in place of it."""
    return float(-L_mm * math.log(frac))


# --- exact average of 1/r over a box ------------------------------------------------------------------------------------


def _cube_corner_H(a, b, c):
    """H(a,b,c) = integral of 1/r over the box [0,a]x[0,b]x[0,c] (a,b,c >= 0), closed form (potential of a charged box)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    out = np.zeros(np.broadcast(a, b, c).shape, dtype=np.float64)
    ok = (a > 0) & (b > 0) & (c > 0)
    if not ok.any():
        return out
    A, B, C = (np.broadcast_to(x, out.shape)[ok] for x in (a, b, c))
    r = np.sqrt(A * A + B * B + C * C)
    F = (A * B * np.log(C + r) + B * C * np.log(A + r) + C * A * np.log(B + r)
         - 0.5 * A * A * np.arctan(B * C / (A * r)) - 0.5 * B * B * np.arctan(C * A / (B * r)) - 0.5 * C * C * np.arctan(A * B / (C * r)))
    out[ok] = (F - B * C * np.log(np.sqrt(B * B + C * C)) - A * C * np.log(np.sqrt(A * A + C * C))
               - A * B * np.log(np.sqrt(A * A + B * B)))
    return out


def box_inv_r_integral(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact integral of 1/r over axis-aligned boxes ``[lo, hi]`` (arrays (N,3), the source at the origin), by inclusion-exclusion over
    corners with the sign-odd extension of H (the integrand is even in every coordinate)."""
    lo = np.atleast_2d(np.asarray(lo, dtype=np.float64))
    hi = np.atleast_2d(np.asarray(hi, dtype=np.float64))
    tot = np.zeros(lo.shape[0], dtype=np.float64)
    for ex in (0, 1):
        for ey in (0, 1):
            for ez in (0, 1):
                x = hi[:, 0] if ex else lo[:, 0]
                y = hi[:, 1] if ey else lo[:, 1]
                z = hi[:, 2] if ez else lo[:, 2]
                sgn = (-1.0) ** (3 - ex - ey - ez) * np.sign(x) * np.sign(y) * np.sign(z)
                tot += sgn * _cube_corner_H(np.abs(x), np.abs(y), np.abs(z))
    return tot


CUBE_CENTRE_INV_R = 3.0 * math.log((math.sqrt(3.0) + 1.0) / (math.sqrt(3.0) - 1.0)) - math.pi / 2.0  # integral of 1/r over the unit cube about its centre


# --- voxel-averaged exposure --------------------------------------------------------------------------------------------

SOURCE_VOXEL_N_SUB = 24     # even, so no node sits on the singularity; the release voxel is the only one with a conical remainder
NEAR_VOX = 3                # within this Chebyshev distance (voxels) the 1/r singularity is removed in closed form; beyond it the kernel is sampled directly


def _gl_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w / 2.0           # nodes on [-1, 1], weights summing to 1 (an average)


def voxel_average_kernel(offsets_vox: np.ndarray, voxel_mm: float, D_mm2_s: float, k_s: float, T_s: float = math.inf,
                         n_sub: int = 3) -> np.ndarray:
    """Voxel-averaged exposure per unit mass (mg*s/mm^3 per mg) for a release at a voxel centre, at voxels offset by ``offsets_vox``
    (integer array (N,3)). Within ``NEAR_VOX`` voxels the 1/r singularity is removed exactly: K = 1/(4 pi D r) + R(r), the box average of 1/r is closed form
    and the bounded remainder R is averaged by an ``n_sub``^3 Gauss-Legendre sub-sampling of each voxel (the release voxel alone uses
    an even, finer count, :data:`SOURCE_VOXEL_N_SUB`, so no node sits on the singularity and its conical remainder converges)."""
    off = np.atleast_2d(np.asarray(offsets_vox, dtype=np.float64))
    s = float(voxel_mm)
    cheb = np.abs(off).max(axis=1)
    is_src = cheb == 0
    near = (cheb <= NEAR_VOX) & ~is_src            # voxels close enough that the 1/r variation across the voxel matters
    far = cheb > NEAR_VOX                          # smooth: a plain sub-sampled average, no cancellation (see below)
    n_src = max(SOURCE_VOXEL_N_SUB, int(n_sub) + (int(n_sub) % 2))      # one voxel, so refine it: its remainder is conical at r = 0
    out = np.zeros(off.shape[0], dtype=np.float64)
    if (is_src | near).any():
        sel_any = is_src | near
        lo, hi = (off[sel_any] - 0.5) * s, (off[sel_any] + 0.5) * s
        out[sel_any] = box_inv_r_integral(lo, hi) / s ** 3 / (4.0 * math.pi * D_mm2_s)
    for sel, n, subtract in ((near, int(n_sub), True), (is_src, n_src, True), (far, int(n_sub), False)):
        if not sel.any():
            continue
        x, w = _gl_nodes(n)
        g = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3) * (0.5 * s)          # (n^3, 3) offsets in mm
        wt = (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)
        pts = off[sel][:, None, :] * s + g[None, :, :]                                                   # (m, n^3, 3)
        r = np.sqrt((pts ** 2).sum(axis=-1))
        K = point_exposure_T(r, D_mm2_s, k_s, T_s)
        if subtract:
            out[sel] += ((K - 1.0 / (4.0 * math.pi * D_mm2_s * r)) * wt[None, :]).sum(axis=1)
        else:                                      # far field: K itself, never 1/r minus 1/r (that cancels to roundoff noise where K ~ 0)
            out[sel] = (K * wt[None, :]).sum(axis=1)
    return np.maximum(out, 0.0)                    # exposure cannot be negative; this only ever clips roundoff


def reference_voxel_average(offset_vox: Sequence[int], voxel_mm: float, D_mm2_s: float, k_s: float, T_s: float = math.inf,
                            n_ref: int = 24) -> float:
    """High-order reference (``n_ref``^3 sub-sampling) of one voxel's average, for the convergence check."""
    return float(voxel_average_kernel(np.asarray([offset_vox]), voxel_mm, D_mm2_s, k_s, T_s, n_ref)[0])


def release_voxel_share(voxel_mm: float, D_mm2_s: float, k_s: float, radius_vox: int, n_sub: int = 5) -> float:
    """Share of a point release's TOTAL exposure that falls inside the release voxel (infinite horizon, infinite medium). Total exposure
    integrated over all space is exactly M/k, so the share is (voxel exposure * voxel volume) / (1/k)."""
    own = voxel_average_kernel(np.zeros((1, 3)), voxel_mm, D_mm2_s, k_s, math.inf, n_sub)[0]
    return float(own * voxel_mm ** 3 * k_s)


# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------


def kernel_radius_vox(p: PhysicalParams, tol_umol_h_L: float = 0.1) -> int:
    """Smallest window radius (voxels) beyond which the whole 61.6 mg budget, released at one point, adds less than ``tol`` umol*h/L."""
    f = p.exposure_to_umol_h_L
    for w in range(1, 200):
        r = (w + 0.5) * p.voxel_mm
        if p.budget_mg * f * float(point_exposure_inf(r, p.d_star_mm2_s, p.k_clear_s)) < tol_umol_h_L:
            return w
    raise PhysicalParamsError("no window radius under 200 voxels meets the tolerance")


def saturation_ticks(p: PhysicalParams, rel: float = 1e-9) -> int:
    """Ticks after which an additional tick changes a release's exposure by less than ``rel`` (e^{-kT} < rel), capped at the episode."""
    n = int(math.ceil(-math.log(rel) / (p.k_clear_s * p.tick_seconds)))
    return max(1, min(n, p.ticks))


_TABLE_CACHE: Dict[tuple, "ExposureTables"] = {}


@dataclass
class ExposureTables:
    """Cumulative voxel-averaged exposure of a unit (1 mg) release, in umol*h/L, for 0..n_tab ticks after release.

    ``cum[n]`` is a ``(2W+1)^3`` float32 cube: exposure accumulated ``n`` ticks after a release at the cube centre (cum[0] = 0).
    Beyond ``n_tab`` ticks the exposure is the last table to within ``rel_tol`` (loss of the remaining mass is below 1e-9).
    """
    params: PhysicalParams
    radius_vox: int
    n_sub: int
    cum: np.ndarray            # (n_tab + 1, 2W+1, 2W+1, 2W+1) float32

    @property
    def n_tab(self) -> int:
        return self.cum.shape[0] - 1

    def cumulative(self, n_ticks: int) -> np.ndarray:
        return self.cum[min(max(int(n_ticks), 0), self.n_tab)]

    def increment(self, age_ticks: int) -> Optional[np.ndarray]:
        """Exposure added during tick number ``age_ticks`` (1-based) after a release, or None once saturated."""
        if age_ticks < 1 or age_ticks > self.n_tab:
            return None
        return self.cum[age_ticks] - self.cum[age_ticks - 1]


def _unique_offsets(w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unique sorted |offset| triples (a >= b >= c >= 0) with a <= w, and for every cube cell the index of its triple."""
    ax = np.abs(np.arange(-w, w + 1))
    A, B, C = np.meshgrid(ax, ax, ax, indexing="ij")
    srt = np.sort(np.stack([A, B, C], axis=-1), axis=-1)[..., ::-1]                    # descending
    key = (srt[..., 0] * (w + 1) + srt[..., 1]) * (w + 1) + srt[..., 2]
    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    c = uniq % (w + 1)
    b = (uniq // (w + 1)) % (w + 1)
    a = uniq // ((w + 1) ** 2)
    return np.stack([a, b, c], axis=1), inv.reshape(A.shape)


def build_tables(p: PhysicalParams, n_sub: int = 5, radius_vox: Optional[int] = None, tol_umol_h_L: float = 0.1) -> ExposureTables:
    """Voxel-averaged cumulative exposure tables for one parameter set. Cached in-process by (params, n_sub, radius)."""
    w = int(radius_vox) if radius_vox is not None else kernel_radius_vox(p, tol_umol_h_L)
    key = (p.sha(), int(n_sub), w)
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    n_tab = saturation_ticks(p)
    uo, inv = _unique_offsets(w)
    cube = 2 * w + 1
    cum = np.zeros((n_tab + 1, cube, cube, cube), dtype=np.float32)
    f = p.exposure_to_umol_h_L
    for n in range(1, n_tab + 1):
        vals = voxel_average_kernel(uo, p.voxel_mm, p.d_star_mm2_s, p.k_clear_s, n * p.tick_seconds, n_sub) * f
        cum[n] = vals[inv].astype(np.float32)
    out = ExposureTables(p, w, int(n_sub), cum)
    _TABLE_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Exact mass accounting (separable erf), for off-target mg
# ---------------------------------------------------------------------------------------------------------------------


def mass_fraction_cube(w: int, voxel_mm: float, D_mm2_s: float, T_s: float) -> np.ndarray:
    """Fraction of a point release's mass that is inside each voxel of the ``(2w+1)^3`` cube a time ``T_s`` after release, ignoring clearance
    (multiply by exp(-k T) for the mass still present). Exact: the Gaussian integrates over a box as a product of erf differences."""
    ax = np.arange(-w, w + 1, dtype=np.float64) * voxel_mm
    sd = 2.0 * math.sqrt(D_mm2_s * T_s)
    if T_s <= 0.0:
        f = np.zeros(2 * w + 1)
        f[w] = 1.0
    else:
        f = 0.5 * (special.erf((ax + 0.5 * voxel_mm) / sd) - special.erf((ax - 0.5 * voxel_mm) / sd))
    return f[:, None, None] * f[None, :, None] * f[None, None, :]
