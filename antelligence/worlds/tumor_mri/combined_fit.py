"""Combined scan-and-look estimate ``c = P(tumour | scan p, look)`` (v4.3, workstream A1).

A logistic regression on four features: ``[1, logit(scan), logit(look), logit(scan) * logit(look)]``. Pure functions only;
the single I/O helpers are the JSON ones at the bottom. Nothing here touches MRI data, the case store or any colony code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA = "combined_v4_3"
FEATURES = ["1", "logit_scan", "logit_look", "logit_scan*logit_look"]
CLIP = (0.001, 0.999)
RIDGE = 1e-6
_COND_LIMIT = 1e12


def clip_logit(x: Any, lo: float = CLIP[0], hi: float = CLIP[1]) -> np.ndarray:
    """logit of ``x`` clipped to ``[lo, hi]``. Always finite, so p = 0 or 1 (or nan-free inputs at the edges) never gives inf."""
    p = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def design(scan_p: Any, look: Any, lo: float = CLIP[0], hi: float = CLIP[1]) -> np.ndarray:
    """Design matrix, one row per look: ``[1, logit(scan), logit(look), logit(scan) * logit(look)]``."""
    s = np.atleast_1d(clip_logit(scan_p, lo, hi))
    v = np.atleast_1d(clip_logit(look, lo, hi))
    if s.shape != v.shape:
        raise ValueError(f"scan_p and look must have the same shape, got {s.shape} and {v.shape}")
    return np.column_stack([np.ones_like(s), s, v, s * v])


def fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Maximum-likelihood logistic regression by Newton / IRLS. Returns ``(coef, info)``.

    A 1e-6 ridge is added to the Hessian only in an iteration where it is singular (non-finite, or condition number above
    1e12); ``info["ridge"]`` is True if that ever happened, so the caller can say so in its output.
    ``info`` also carries ``iterations`` and ``converged``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError("X must be (n, k) and y must be (n,)")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be 0/1")
    k = X.shape[1]
    beta = np.zeros(k)
    ridge_used = False
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        p = _sigmoid(X @ beta)
        w = p * (1.0 - p)
        grad = X.T @ (y - p)
        H = X.T @ (X * w[:, None])
        singular = (not np.all(np.isfinite(H))) or np.linalg.cond(H) > _COND_LIMIT
        if singular:
            H = H + RIDGE * np.eye(k)
            ridge_used = True
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    return beta, {"ridge": ridge_used, "iterations": it, "converged": converged}


def predict(coef: Sequence[float], scan_p: Any, look: Any, lo: float = CLIP[0], hi: float = CLIP[1]) -> np.ndarray:
    """``c`` for each (scan p, look) pair, in [0, 1]."""
    return _sigmoid(design(scan_p, look, lo, hi) @ np.asarray(coef, dtype=np.float64))


def ece(prob: Any, y: Any, bins: int = 10) -> Optional[float]:
    """Expected calibration error: equal-width bins on the predicted probability, weighted mean of |mean prediction - observed rate|.
    A probability of exactly 1.0 goes in the top bin. None for an empty input."""
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(prob) == 0:
        return None
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(prob[m].mean() - y[m].mean())
    return float(total / len(prob))


def auroc(score: Any, y: Any) -> Optional[float]:
    """P(score of a tumour > score of a non-tumour) + 0.5 P(tie), by ranks with tie averaging. None if a class is empty."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sv = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --- JSON helpers (the only I/O) -------------------------------------------------------------------------------------------

def coefficients_doc(coef: Sequence[float], *, split: str, n_looks: int, n_cases: int, ridge: bool, inputs_sha256: str) -> Dict[str, Any]:
    """The ``coefficients.json`` body, with exactly the v4.3 schema."""
    if len(coef) != len(FEATURES):
        raise ValueError(f"expected {len(FEATURES)} coefficients, got {len(coef)}")
    return {"schema": SCHEMA, "features": list(FEATURES), "coef": [float(c) for c in coef], "clip": list(CLIP),
            "fit": {"split": split, "n_looks": int(n_looks), "n_cases": int(n_cases), "ridge": bool(ridge),
                    "inputs_sha256": inputs_sha256}}


def save_json(doc: Mapping[str, Any], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def load_coefficients(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a ``coefficients.json``. Raises ValueError on a wrong schema."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA or doc.get("features") != FEATURES or len(doc.get("coef", [])) != len(FEATURES):
        raise ValueError(f"{path}: not a {SCHEMA} coefficients file")
    for key in ("clip", "fit"):
        if key not in doc:
            raise ValueError(f"{path}: missing {key!r}")
    return doc
