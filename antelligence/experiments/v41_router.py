"""kNN router (goal 4.5): which worker answers this squad-round. Frozen before G1, trained on development logs only.

* **Features** (standardised with the training mean/std; ten numbers): for each of target, release, amount, d1's top
  probability and entropy (nats); then region uncertainty (``local_unc``), payload left (``payload_fraction``),
  distance in mm to d1's chosen target region (0 for stay ``S`` and explore ``X``), and the round index.
* **Labels from S'**, evaluated at the swarm's own state: ``target`` = the S'-mapped candidate label
  (``sprime_candidate_label``, else ``X``), ``release`` = yes iff S' would release there, ``amount`` = the level whose
  fraction is nearest S's fraction.
* **Router label** = the cheapest model whose three answers (target, release, amount) match S' on the most questions.
  Ties go to the cheaper model (the order of ``models``, cheapest first).
* **kNN**: k = 15 (fewer if the training set is smaller), Euclidean distance on standardised features, neighbours
  ordered by (distance, index), plurality vote, vote ties to the cheaper model.
* The Bonsai arm gets its own router (``BONSAI_ROUTER_MODELS``), trained on its 10 development cases.

A frozen router is saved as JSON ``{"body", "sha256"}``; loading checks the hash.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from antelligence.kernel.canonical import canonical_json, content_hash
from antelligence.kernel.decision import Answer
from antelligence.worlds.tumor_mri.config import AMOUNT_FRACTIONS
from antelligence.worlds.tumor_mri.policies import MAP_RADIUS_MM, sprime_candidate_label
from antelligence.worlds.tumor_mri.questions import EXPLORE, STAY

K = 15
ROUTER_MODELS = ("d1", "qwen3.5-4b", "qwen3.5-9b")  # cheapest first
BONSAI_ROUTER_MODELS = ("d1", "bonsai-8b")
FEATURE_NAMES = ("target_top", "target_entropy", "release_top", "release_entropy", "amount_top", "amount_entropy",
                 "region_unc", "payload", "dist_target_mm", "round")
QUESTIONS = ("target", "release", "amount")


class RouterError(ValueError):
    pass


def entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def router_features(fields: Mapping[str, Any], d1_answers: Mapping[str, Answer]) -> List[float]:
    """The ten live features for one squad-round, from the observation ``fields`` and d1's answers."""
    row: List[float] = []
    for q in QUESTIONS:
        probs = list(d1_answers[q].probabilities.values())
        row += [max(probs), entropy(probs)]
    target = d1_answers["target"].value
    dist = 0.0
    if target not in (STAY, EXPLORE):
        dist = next((float(c["dist_mm"]) for c in fields.get("candidates", []) if c["label"] == target), 0.0)
    row += [float(fields["local_unc"]), float(fields["payload_fraction"]), dist, float(fields["round"])]
    return row


def nearest_level(fraction: float, fractions: Sequence[float] = AMOUNT_FRACTIONS) -> int:
    """Amount level (1-based) whose fraction is nearest; ties go to the smaller level."""
    return 1 + min(range(len(fractions)), key=lambda i: (abs(fractions[i] - fraction), i))


def sprime_labels(intent: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *, voxel_mm: float = 2.0,
                  radius_mm: float = MAP_RADIUS_MM, fractions: Sequence[float] = AMOUNT_FRACTIONS) -> Dict[str, str]:
    """S' decide-payload -> router labels. ``target_vox`` (S''s true goal) is mapped to a candidate label; without it
    the payload's own ``target`` is used. ``amount_fraction`` is mapped to the nearest level; else ``amount`` is the level."""
    if "target_vox" in intent:
        target = sprime_candidate_label(intent["target_vox"], candidates, voxel_mm, radius_mm)
    else:
        target = str(intent["target"])
    level = nearest_level(float(intent["amount_fraction"]), fractions) if "amount_fraction" in intent \
        else int(intent["amount"])
    return {"target": target, "release": "yes" if intent.get("release") else "no", "amount": str(level)}


def match_count(decision: Mapping[str, Any], labels: Mapping[str, str]) -> int:
    """How many of a model's three answers equal S''s labels. ``decision`` is the harness rule's output
    (``decide_from_answers``): target label, release 0/1, amount level."""
    return (int(decision["target"] == labels["target"])
            + int(("yes" if decision["release"] else "no") == labels["release"])
            + int(str(decision["amount"]) == labels["amount"]))


def best_model(decisions: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str],
               models: Sequence[str] = ROUTER_MODELS) -> str:
    """The cheapest model with the highest match count (``models`` is ordered cheapest first)."""
    best, best_score = None, -1
    for m in models:
        score = match_count(decisions[m], labels)
        if score > best_score:  # strict: an equal score keeps the cheaper model
            best, best_score = m, score
    assert best is not None
    return best


@dataclass(frozen=True)
class TrainRow:
    features: Sequence[float]
    label: str


def make_rows(samples: Sequence[Tuple[Sequence[float], Mapping[str, Mapping[str, Any]], Mapping[str, str]]],
              models: Sequence[str] = ROUTER_MODELS) -> List[TrainRow]:
    """(features, {model: decision}, S' labels) -> rows labelled with the cheapest best-matching model."""
    return [TrainRow(list(f), best_model(d, lab, models)) for f, d, lab in samples]


class KNNRouter:
    def __init__(self, models: Sequence[str], k: int, mean: np.ndarray, std: np.ndarray, x: np.ndarray,
                 y: Sequence[int]) -> None:
        self.models, self.k = tuple(models), int(k)
        self.mean, self.std, self.x, self.y = mean, std, x, np.asarray(y, dtype=int)

    @classmethod
    def fit(cls, rows: Sequence[TrainRow], models: Sequence[str] = ROUTER_MODELS, k: int = K) -> "KNNRouter":
        if not rows:
            raise RouterError("no training rows")
        if any(r.label not in models for r in rows):
            raise RouterError("a row is labelled with a model outside `models`")
        raw = np.asarray([r.features for r in rows], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(FEATURE_NAMES) or not np.isfinite(raw).all():
            raise RouterError(f"features must be finite with {len(FEATURE_NAMES)} columns")
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return cls(models, min(k, len(rows)), mean, std, (raw - mean) / std, [models.index(r.label) for r in rows])

    def predict(self, features: Sequence[float]) -> str:
        z = (np.asarray(features, dtype=np.float64) - self.mean) / self.std
        dist = np.sqrt(((self.x - z) ** 2).sum(axis=1))
        order = sorted(range(len(dist)), key=lambda i: (float(dist[i]), i))[: self.k]
        votes = np.zeros(len(self.models), dtype=int)
        for i in order:
            votes[self.y[i]] += 1
        return self.models[int(np.argmax(votes))]  # argmax keeps the first (cheapest) of tied models

    def body(self) -> Dict[str, Any]:
        return {"study": "v4.1", "kind": "knn-router", "models": list(self.models), "k": self.k,
                "features": list(FEATURE_NAMES), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "x": self.x.tolist(), "y": self.y.tolist()}

    @property
    def sha256(self) -> str:
        return content_hash(self.body())

    def save(self, path: Union[str, Path]) -> str:
        Path(path).write_text(canonical_json({"body": self.body(), "sha256": self.sha256}) + "\n", encoding="utf-8")
        return self.sha256

    @classmethod
    def load(cls, path: Union[str, Path], expect_sha256: Optional[str] = None) -> "KNNRouter":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        body = doc["body"]
        if content_hash(body) != doc["sha256"] or (expect_sha256 and doc["sha256"] != expect_sha256):
            raise RouterError(f"{path}: router does not match its sha256")
        if body["features"] != list(FEATURE_NAMES):
            raise RouterError("router was trained on different features")
        return cls(body["models"], body["k"], np.asarray(body["mean"]), np.asarray(body["std"]),
                   np.asarray(body["x"]), body["y"])


def fit_router(rows: Sequence[TrainRow], *, bonsai: bool = False, k: int = K) -> KNNRouter:
    """Main router (d1 / Qwen3.5-4B / 9B) or the separate Bonsai variant (d1 / Bonsai 8B)."""
    return KNNRouter.fit(rows, BONSAI_ROUTER_MODELS if bonsai else ROUTER_MODELS, k)
