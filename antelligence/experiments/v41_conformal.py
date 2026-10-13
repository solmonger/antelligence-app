"""Split-conformal deferral for d1-3B (goal 4.5), frozen before G1.

d1 answers target, release and amount in one call. For each question type we keep a threshold ``qhat`` fitted on
development logs from both arms:

* **Conformity label** = the deferral target's answer on the same state (Qwen3.5-9B; Bonsai 8B in the Bonsai arm,
  which gets its own thresholds). It is the target's argmax label for ``target`` and ``amount`` and ``yes`` iff
  P(yes) >= 0.5 for ``release`` (the harness decision rule).
* **Nonconformity** ``s = 1 - p_d1(label)``. ``qhat`` is the ceil((n+1)(1-alpha))/n empirical quantile of ``s``
  (infinite when that rank exceeds n, which puts every label in the set).
* **Prediction set** = labels ``l`` with ``1 - p_d1(l) <= qhat``. A question defers unless its set has exactly one
  label: more than one label is doubt, and an **empty set also defers**.
* **alpha per question** is chosen from a grid so that development deferral is closest to 20% (ties go to the smaller
  alpha). Deferral is not monotone in alpha (tiny alpha gives big sets, large alpha gives empty sets), so the grid is
  searched, not bisected.
* A decision defers when any of its three questions defers (``DeferralResult.deferred``). The frozen file also records
  the joint development rate, which is higher than 20% when the questions defer on different states.

Thresholds are frozen into JSON with a sha256 of their canonical body and never adapt during evaluation. A 10%
audit sample (deterministic hash of the decision key) has the deferral target answer even when d1 kept the decision;
it is logged only and used to report the coverage achieved, never acted on. A run's defer rate outside 5-40% yields an
``alarm`` entry payload (kind ``defer_rate``); nothing else changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from antelligence.kernel.canonical import canonical_json, content_hash
from antelligence.kernel.decision import Answer, DecisionRequest

QTYPES = ("target", "release", "amount")
DEFER_TARGET = 0.20
AUDIT_RATE = 0.10
ALARM_BAND = (0.05, 0.40)
ALPHA_GRID = tuple(i / 200 for i in range(1, 200))
EPS = 1e-12
TARGET_TOLERANCE = 0.02  # |dev deferral - 20%| allowed before a question is flagged off-target


class ConformalError(ValueError):
    """Bad input, or a frozen file that does not match its hash."""


@dataclass(frozen=True)
class DevRecord:
    """One development decision. ``probs[q]`` = d1's {label: p}; ``labels[q]`` = the deferral target's label."""

    probs: Mapping[str, Mapping[str, float]]
    labels: Mapping[str, str]


def label_distributions(request: DecisionRequest, answers: Mapping[str, Answer]) -> Dict[str, Dict[str, float]]:
    """Kernel Answers -> {question: {label: p}}. noul -> yes/no; score -> amount levels "1".."5" (index + 1)."""
    out: Dict[str, Dict[str, float]] = {}
    for qid, q in request.questions.items():
        a = answers[qid]
        if q.type == "noul":
            out[qid] = {"yes": a.probabilities["yes"], "no": a.probabilities["no"]}
        elif q.type == "choice":
            out[qid] = {o: a.probabilities[o] for o in q.options}
        else:
            out[qid] = {str(int(i) + 1): p for i, p in a.probabilities.items()}
    return out


def decision_labels(request: DecisionRequest, answers: Mapping[str, Answer]) -> Dict[str, str]:
    """The harness decision rule as labels: argmax (ties to the earlier label), release iff P(yes) >= 0.5."""
    from antelligence.worlds.tumor_mri.questions import decide_from_answers  # worlds may be imported from experiments

    d = decide_from_answers(answers, request)
    return {"target": d["target"], "release": "yes" if d["release"] else "no", "amount": str(d["amount"])}


def dev_record(request: DecisionRequest, d1_answers: Mapping[str, Answer],
               target_answers: Mapping[str, Answer]) -> DevRecord:
    return DevRecord(label_distributions(request, d1_answers), decision_labels(request, target_answers))


# --- fitting ----------------------------------------------------------------------------------------------------


def nonconformity(p: Mapping[str, float], label: str) -> float:
    if label not in p:
        raise ConformalError(f"label {label!r} is not among the offered labels {sorted(p)}")
    return 1.0 - float(p[label])


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score; +inf when that rank exceeds n."""
    n = len(scores)
    if n == 0:
        raise ConformalError("no calibration scores")
    if not 0.0 < alpha < 1.0:
        raise ConformalError("alpha must be in (0, 1)")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    return math.inf if rank > n else sorted(scores)[rank - 1]


def prediction_set(p: Mapping[str, float], qhat: float) -> Tuple[str, ...]:
    """Labels with 1 - p(l) <= qhat, in the order given (fixed label order, so ties stay deterministic)."""
    return tuple(l for l in p if 1.0 - float(p[l]) <= qhat + EPS)


def defers(label_set: Sequence[str]) -> bool:
    """Defer unless the set is exactly one label (more than one, or empty)."""
    return len(label_set) != 1


def choose_alpha(records: Sequence[DevRecord], qtype: str, target: float = DEFER_TARGET,
                 grid: Iterable[float] = ALPHA_GRID) -> Dict[str, Any]:
    """Alpha whose development deferral is closest to ``target``; returns alpha, qhat, n and the rate reached."""
    if not records:
        raise ConformalError("no development records")
    scores = [nonconformity(r.probs[qtype], r.labels[qtype]) for r in records]
    best: Optional[Tuple[float, float, float, float]] = None
    for alpha in grid:
        qhat = conformal_quantile(scores, alpha)
        rate = sum(defers(prediction_set(r.probs[qtype], qhat)) for r in records) / len(records)
        key = (abs(rate - target), alpha)
        if best is None or key < (best[0], best[1]):
            best = (key[0], alpha, qhat, rate)
    assert best is not None
    # on_target False means the closest alpha still misses (e.g. d1 and the deferral target disagree so often that
    # the minimum achievable deferral is above the target); the frozen file records it instead of hiding it.
    return {"alpha": best[1], "qhat": best[2], "n": len(records), "dev_defer_rate": best[3],
            "on_target": bool(abs(best[3] - target) <= TARGET_TOLERANCE)}


@dataclass(frozen=True)
class ConformalThresholds:
    """Frozen thresholds. ``arm``/``target_model`` say which deferral target labelled them."""

    arm: str
    target_model: str
    per_question: Mapping[str, Mapping[str, Any]]
    joint_dev_defer_rate: float
    defer_target: float = DEFER_TARGET

    def body(self) -> Dict[str, Any]:
        # inf is not JSON: it is stored as null and read back as inf
        pq = {q: {**v, "qhat": None if math.isinf(v["qhat"]) else v["qhat"]} for q, v in self.per_question.items()}
        return {"study": "v4.1", "kind": "conformal-thresholds", "arm": self.arm, "target_model": self.target_model,
                "defer_target": self.defer_target, "per_question": pq,
                "joint_dev_defer_rate": self.joint_dev_defer_rate}

    @property
    def sha256(self) -> str:
        return content_hash(self.body())

    def qhat(self, qtype: str) -> float:
        return float(self.per_question[qtype]["qhat"])

    def save(self, path: Union[str, Path]) -> str:
        Path(path).write_text(canonical_json({"body": self.body(), "sha256": self.sha256}) + "\n", encoding="utf-8")
        return self.sha256

    @classmethod
    def load(cls, path: Union[str, Path], expect_sha256: Optional[str] = None) -> "ConformalThresholds":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        body = doc["body"]
        if content_hash(body) != doc["sha256"] or (expect_sha256 and doc["sha256"] != expect_sha256):
            raise ConformalError(f"{path}: thresholds do not match their sha256")
        pq = {q: {**v, "qhat": math.inf if v["qhat"] is None else float(v["qhat"])}
              for q, v in body["per_question"].items()}
        return cls(body["arm"], body["target_model"], pq, body["joint_dev_defer_rate"], body["defer_target"])


def freeze_thresholds(records: Sequence[DevRecord], *, arm: str, target_model: str,
                      target: float = DEFER_TARGET) -> ConformalThresholds:
    """Fit all three question types on development records (from both SW-B and SW-noBoard runs) and freeze."""
    per_question = {q: choose_alpha(records, q, target) for q in QTYPES}
    draft = ConformalThresholds(arm, target_model, per_question, 0.0, target)
    joint = sum(ConformalDeferral(draft).decide(r.probs).deferred for r in records) / len(records)
    return ConformalThresholds(arm, target_model, per_question, joint, target)


# --- applying ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeferralResult:
    sets: Mapping[str, Tuple[str, ...]]
    deferred: bool

    @property
    def empty(self) -> bool:
        return any(len(s) == 0 for s in self.sets.values())


class ConformalDeferral:
    def __init__(self, thresholds: ConformalThresholds) -> None:
        self.thresholds = thresholds

    def decide(self, probs: Mapping[str, Mapping[str, float]]) -> DeferralResult:
        sets = {q: prediction_set(probs[q], self.thresholds.qhat(q)) for q in QTYPES}
        return DeferralResult(sets, any(defers(s) for s in sets.values()))

    def decide_answers(self, request: DecisionRequest, answers: Mapping[str, Answer]) -> DeferralResult:
        return self.decide(label_distributions(request, answers))


# --- audit sample and alarm -------------------------------------------------------------------------------------


def audit_selected(key: str, rate: float = AUDIT_RATE, salt: str = "v4.1|audit") -> bool:
    """Deterministic 10% sample: sha256(salt|key) read as a uniform number in [0, 1). The key is the decision's
    identity (e.g. ``f"{run_id}|{squad}|{round}"``), so a replay audits exactly the same decisions."""
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < rate


def audit_coverage(audits: Sequence[Tuple[Mapping[str, Tuple[str, ...]], Mapping[str, str]]]) -> Dict[str, float]:
    """Coverage achieved on audited decisions: share whose target label is inside d1's set, per question and all."""
    if not audits:
        return {}
    out = {q: sum(lab[q] in sets[q] for sets, lab in audits) / len(audits) for q in QTYPES}
    out["all"] = sum(all(lab[q] in sets[q] for q in QTYPES) for sets, lab in audits) / len(audits)
    return out


def defer_rate_alarm(deferred: int, total: int, band: Tuple[float, float] = ALARM_BAND) -> Optional[Dict[str, Any]]:
    """An ``alarm`` entry payload (INTERFACES section 4) when the run's defer rate leaves the band, else None."""
    if total <= 0:
        return None
    rate = deferred / total
    if band[0] <= rate <= band[1]:
        return None
    return {"kind": "defer_rate", "value_ppm": int(round(rate * 1_000_000)),
            "note": f"defer rate {rate:.1%} ({deferred}/{total}) outside {band[0]:.0%}-{band[1]:.0%}"[:140]}


class DeferralMonitor:
    """Counts a run's decisions; ``finish()`` returns the alarm payload or None. It changes nothing else."""

    def __init__(self, band: Tuple[float, float] = ALARM_BAND) -> None:
        self.band, self.total, self.deferred = band, 0, 0

    def record(self, deferred: bool) -> None:
        self.total += 1
        self.deferred += int(deferred)

    def finish(self) -> Optional[Dict[str, Any]]:
        return defer_rate_alarm(self.deferred, self.total, self.band)
