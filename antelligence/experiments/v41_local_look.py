"""Local-look calibration (goal 4.5, Phase 0): d1-3B answers "is the centre of this patch tumour?" from a 64x64
FLAIR + T1c patch. It is a sensor, identical in every arm. This module draws the calibration patches and scores them.

* **Sampler**: half the patch centres inside the truth tumour, half outside it (inside the brain), per case, seeded.
  It accepts **development cases only** (``config/v4_1/split.json``: ``dev_a`` + ``dev_b``); an evaluation or reserve
  case id raises :class:`SplitError`. The truth mask is used here for labels and nowhere at run time.
* **Scores**: ECE (equal-width bins) and Brier against the truth label at the centre.
* **Temperature refit**: when ECE > 0.10, fit one temperature T on the development patches (minimum negative
  log-likelihood of ``sigmoid(logit(p) / T)``) and report ECE/Brier before and after. The refit is in-sample on the
  development set (it is a one-number fit); the harness applies :func:`apply_temperature` to d1's look probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize_scalar

from antelligence.kernel.decision import DecisionRequest, Question
from antelligence.perception.store import local_patch
from antelligence.providers.decision_backends import patch_to_images
from antelligence.worlds.tumor_mri.tuning import SplitError, assert_dev_only, load_split  # noqa: F401  (re-exported)

ECE_THRESHOLD = 0.10
LOOK_QUESTION = "look"
LOOK_STATE = "Two axial MRI patches centred on a squad: FLAIR first, then T1c (contrast)."
LOOK_INSTRUCTIONS = "Is the centre of this patch tumour?"


FROZEN_TEMPERATURE_PATH = Path(__file__).resolve().parents[2] / "config" / "v4_1" / "look_temperature.json"
SEAL_KEY = "file_sha256"


class TemperatureFileError(ValueError):
    """The frozen look-temperature file is missing, corrupt, or its recorded sha256 does not match its content."""


def _seal_digest(doc: Mapping[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k != SEAL_KEY}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal_temperature_doc(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``doc`` plus ``file_sha256``: sha256 of the canonical JSON of every other key."""
    out = {k: v for k, v in doc.items() if k != SEAL_KEY}
    out[SEAL_KEY] = _seal_digest(out)
    return out


def load_frozen_temperature(path: Union[str, Path, None] = None) -> float:
    """The frozen look temperature, after verifying the file's recorded ``file_sha256`` against its content."""
    p = Path(path or FROZEN_TEMPERATURE_PATH)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        recorded, t = doc[SEAL_KEY], doc["temperature"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TemperatureFileError(f"{p}: unreadable or malformed ({type(exc).__name__}: {exc})") from exc
    if not isinstance(doc, dict) or recorded != _seal_digest(doc):
        raise TemperatureFileError(f"{p}: sha256 mismatch (file edited or corrupt)")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
        raise TemperatureFileError(f"{p}: temperature {t!r} is not a positive number")
    return float(t)


def dev_cases(split_path: Union[str, Path]) -> frozenset:
    """dev_a + dev_b of the committed split."""
    return frozenset(load_split(split_path)["dev"])


@dataclass(frozen=True)
class LookSample:
    case: str
    xyz: Tuple[int, int, int]
    label: int  # 1 = the centre voxel is truth tumour


def sample_centres(truth: np.ndarray, brain: np.ndarray, n: int, seed: int) -> List[Tuple[Tuple[int, int, int], int]]:
    """``n`` centres, half truth and half non-truth brain voxels (n odd: the extra one is outside). Without
    replacement; if a side has fewer voxels than asked, that side gives all it has."""
    rng = np.random.default_rng(seed)
    inside = np.argwhere(truth & brain)
    outside = np.argwhere(brain & ~truth)
    n_in = min(n // 2, len(inside))
    n_out = min(n - n // 2, len(outside))
    picks: List[Tuple[Tuple[int, int, int], int]] = []
    for pts, count, label in ((inside, n_in, 1), (outside, n_out, 0)):
        if count:
            idx = rng.choice(len(pts), size=count, replace=False)
            picks += [((int(pts[i][0]), int(pts[i][1]), int(pts[i][2])), label) for i in sorted(idx)]
    return picks


def sample_cases(arrays_by_case: Mapping[str, Mapping[str, np.ndarray]], dev: frozenset, n_per_case: int,
                 seed: int) -> List[LookSample]:
    """Balanced samples over dev cases. Refuses any case not in ``dev`` before touching its arrays."""
    assert_dev_only(arrays_by_case, {"dev": set(dev)})  # raises SplitError before any array is read
    out: List[LookSample] = []
    for case in sorted(arrays_by_case):
        a = arrays_by_case[case]
        case_seed = int.from_bytes(hashlib.sha256(f"{seed}|{case}".encode()).digest()[:4], "big")
        out += [LookSample(case, xyz, label) for xyz, label in sample_centres(a["truth"], a["brain_mask"], n_per_case, case_seed)]
    return out


def patches(samples: Sequence[LookSample], arrays_by_case: Mapping[str, Mapping[str, np.ndarray]]) -> Iterator[np.ndarray]:
    for s in samples:
        yield local_patch(s.case, s.xyz, arrays=dict(arrays_by_case[s.case]))


def look_request(model: str, patch: np.ndarray, seed: int = 0) -> DecisionRequest:
    """One noul question over the two patch images (sent to d1 only)."""
    flair, t1c = patch_to_images(patch)
    return DecisionRequest(model=model, state=LOOK_STATE, seed=seed, images=(flair, t1c),
                           questions={LOOK_QUESTION: Question("noul", LOOK_INSTRUCTIONS,
                                                              criteria={"true": "tumour", "false": "not tumour"})})


async def measure(backend: Any, model: str, patch_iter: Iterator[np.ndarray]) -> List[float]:
    """P(tumour) from the backend for each patch (sequential; the caller wraps the backend in cache + ledger)."""
    out: List[float] = []
    for patch in patch_iter:
        reply = await backend.decide(look_request(model, patch))
        out.append(reply.answers[LOOK_QUESTION].p_yes)
    return out


# --- scores -----------------------------------------------------------------------------------------------------


def _arrays(p: Sequence[float], y: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    pa, ya = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if pa.shape != ya.shape or pa.ndim != 1 or len(pa) == 0:
        raise ValueError("p and y must be equal-length non-empty 1-D sequences")
    if ((pa < 0) | (pa > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return pa, ya


def brier(p: Sequence[float], y: Sequence[int]) -> float:
    pa, ya = _arrays(p, y)
    return float(np.mean((pa - ya) ** 2))


def ece(p: Sequence[float], y: Sequence[int], bins: int = 10) -> float:
    """Expected calibration error: bin-weighted |mean(y) - mean(p)| over equal-width bins on [0, 1]."""
    pa, ya = _arrays(p, y)
    idx = np.minimum((pa * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(ya[sel].mean() - pa[sel].mean())
    return float(total)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(p, eps, 1 - eps)
    return np.log(q / (1 - q))


def apply_temperature(p: Union[float, Sequence[float]], temperature: float) -> Any:
    """sigmoid(logit(p) / T). T > 1 softens, T < 1 sharpens; T = 1 returns p (up to the 1e-6 clip)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scalar = np.isscalar(p)
    z = _logit(np.atleast_1d(np.asarray(p, dtype=np.float64))) / temperature
    out = 1.0 / (1.0 + np.exp(-z))
    return float(out[0]) if scalar else out


def fit_temperature(p: Sequence[float], y: Sequence[int]) -> float:
    """T minimising the negative log-likelihood of the temperature-scaled probabilities (search over log T)."""
    pa, ya = _arrays(p, y)
    z = _logit(pa)

    def nll(log_t: float) -> float:
        q = np.clip(1.0 / (1.0 + np.exp(-z / math.exp(log_t))), 1e-12, 1 - 1e-12)
        return float(-np.mean(ya * np.log(q) + (1 - ya) * np.log(1 - q)))

    res = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(res.x))


def calibration_report(p: Sequence[float], y: Sequence[int], threshold: float = ECE_THRESHOLD,
                       bins: int = 10) -> Dict[str, Any]:
    """ECE and Brier; if ECE > ``threshold`` also the refit temperature and the scores after it."""
    before_ece, before_brier = ece(p, y, bins), brier(p, y)
    report: Dict[str, Any] = {"n": len(p), "ece": before_ece, "brier": before_brier, "refit": before_ece > threshold,
                              "temperature": 1.0, "ece_after": before_ece, "brier_after": before_brier}
    if report["refit"]:
        t = fit_temperature(p, y)
        scaled = apply_temperature(list(p), t)
        report.update(temperature=t, ece_after=ece(scaled, y, bins), brier_after=brier(scaled, y))
    return report
