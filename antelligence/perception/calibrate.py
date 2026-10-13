"""Calibration of the WT probability (temperature + isotonic) and the metrics around it.

Calibrators are fit on DEV cases only; eval cases only ever see ``apply``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import isotonic_regression, minimize_scalar

EPS = 1e-6
N_BINS = 15


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error, equal-width bins on [0,1] (p == 1 falls in the last bin)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    b = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins).astype(np.float64)
    sp = np.bincount(b, weights=p, minlength=n_bins)
    sy = np.bincount(b, weights=y, minlength=n_bins)
    ok = cnt > 0
    return float(np.sum(np.abs(sp[ok] - sy[ok])) / p.size)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice of two boolean arrays; two empty masks score 1.0."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    denom = int(pred.sum()) + int(truth.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, truth).sum() / denom)


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    z = _logit(np.asarray(p, dtype=np.float64).ravel())
    y = np.asarray(y, dtype=np.float64).ravel()
    res = minimize_scalar(lambda t: _nll(_sigmoid(z / t), y), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return _sigmoid(_logit(np.asarray(p, dtype=np.float64)) / t)


def fit_isotonic(p: np.ndarray, y: np.ndarray, n_bins: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic regression on quantile bins of ``p`` (deterministic). Returns knots (x, y)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    order = np.argsort(p, kind="stable")
    p, y = p[order], y[order]
    edges = np.linspace(0, p.size, min(n_bins, p.size) + 1).astype(np.int64)
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            xs.append(p[a:b].mean())
            ys.append(y[a:b].mean())
            ws.append(float(b - a))
    fit = isotonic_regression(np.asarray(ys), weights=np.asarray(ws), increasing=True).x
    return np.asarray(xs), np.asarray(fit)


@dataclass
class Calibrator:
    method: str = "none"  # "none" | "temperature" | "isotonic" (isotonic stacked on temperature)
    temperature: float = 1.0
    iso_x: list[float] = field(default_factory=list)
    iso_y: list[float] = field(default_factory=list)

    def apply(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=np.float64)
        if self.method == "none":
            return q
        q = apply_temperature(q, self.temperature)
        if self.method == "isotonic":
            q = np.interp(q, self.iso_x, self.iso_y)
        return q

    def to_dict(self) -> dict:
        return {"method": self.method, "temperature": self.temperature,
                "iso_x": list(self.iso_x), "iso_y": list(self.iso_y)}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(d["method"], float(d["temperature"]), list(d["iso_x"]), list(d["iso_y"]))


def fit_calibrator(p_dev: np.ndarray, y_dev: np.ndarray) -> tuple[Calibrator, dict]:
    """Fit temperature and isotonic on dev voxels; keep whichever has the lowest dev ECE."""
    p = np.asarray(p_dev, dtype=np.float64).ravel()
    y = np.asarray(y_dev, dtype=np.float64).ravel()
    t = fit_temperature(p, y)
    p_t = apply_temperature(p, t)
    ix, iy = fit_isotonic(p_t, y)
    cands = {
        "none": Calibrator("none"),
        "temperature": Calibrator("temperature", t),
        "isotonic": Calibrator("isotonic", t, ix.tolist(), iy.tolist()),
    }
    report = {name: {"ece": ece(c.apply(p), y), "brier": brier(c.apply(p), y)} for name, c in cands.items()}
    best = min(cands, key=lambda n: (report[n]["ece"], n))
    report["chosen"] = best
    report["n_voxels"] = int(p.size)
    return cands[best], report


def fit_select_heldout(p_a: np.ndarray, y_a: np.ndarray, p_b: np.ndarray, y_b: np.ndarray
                       ) -> tuple[Calibrator, dict]:
    """Held-out selection: fit candidates on dev-A, pick the lowest ECE on dev-B, refit the winner on A+B.

    Headline figures are the dev-B (held-out) ECE/Brier; dev-A numbers are in-sample and labelled so.
    """
    pa = np.asarray(p_a, dtype=np.float64).ravel()
    ya = np.asarray(y_a, dtype=np.float64).ravel()
    pb = np.asarray(p_b, dtype=np.float64).ravel()
    yb = np.asarray(y_b, dtype=np.float64).ravel()
    fitted_a, rep_a = fit_calibrator(pa, ya)
    t = fit_temperature(pa, ya)
    ix, iy = fit_isotonic(apply_temperature(pa, t), ya)
    cands = {"none": Calibrator("none"), "temperature": Calibrator("temperature", t),
             "isotonic": Calibrator("isotonic", t, ix.tolist(), iy.tolist())}
    held = {n: {"ece": ece(c.apply(pb), yb), "brier": brier(c.apply(pb), yb)} for n, c in cands.items()}
    best = min(cands, key=lambda n: (held[n]["ece"], n))
    p_ab, y_ab = np.concatenate([pa, pb]), np.concatenate([ya, yb])
    if best == "none":
        final = Calibrator("none")
    else:
        t2 = fit_temperature(p_ab, y_ab)
        final = Calibrator("temperature", t2)
        if best == "isotonic":
            x2, y2 = fit_isotonic(apply_temperature(p_ab, t2), y_ab)
            final = Calibrator("isotonic", t2, x2.tolist(), y2.tolist())
    report = {
        "chosen": best,
        "heldout_dev_b": held,
        "in_sample_dev_a": {k: v for k, v in rep_a.items() if k in cands},
        "refit_on": "dev_a+dev_b",
        "in_sample_dev_ab_final": {"ece": ece(final.apply(p_ab), y_ab), "brier": brier(final.apply(p_ab), y_ab)},
        "n_voxels_dev_a": int(pa.size), "n_voxels_dev_b": int(pb.size),
    }
    return final, report
