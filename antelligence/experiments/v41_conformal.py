"""Split-conformal deferral for d1-3B (goal 4.5, re-based on S' by decision 2026-10-13-conformity-label-sprime-resolved).

d1 answers target, release and amount in one call. For each question type we keep a threshold ``qhat`` fitted on
development logs from both arms:

* **Conformity label** = the **S'-resolved label** (S' is the oracle; development truth is allowed for calibration).
  ``target``: the region S' heads for, ``"r<id>"``, or ``explore``; ``release``: S' release yes/no; ``amount``: S' level.
  S' labels that are ``S`` or ``X`` are resolved the same way as a model's (see ``v41_resolve``). "The deferral target's
  answer" is no longer a label.
* **Resolved sets.** d1's probability of a resolved target is the SUM of its probabilities over every label that
  resolves to it (so ``S`` adds to the region it keeps, ``X`` to ``explore``), and prediction sets are over resolved
  regions, not letters. Release and amount are unchanged.
* **Nonconformity** ``s = 1 - p_d1(label)``. ``qhat`` is the ceil((n+1)(1-alpha))/n empirical quantile of ``s``
  (infinite when that rank exceeds n, which puts every label in the set).
* **Prediction set** = labels ``l`` with ``1 - p_d1(l) <= qhat``. A question defers unless its set has exactly one
  label: more than one label is doubt, and an **empty set also defers**.
* **Per-question deferral target.** Each question type defers to the candidate model (Qwen3.5-4B or 9B; Bonsai 8B in the
  Bonsai arm) with the best dev resolved match to S', but only if that match beats d1's by at least ``DEFER_MARGIN``
  (5 points, absolute). Otherwise the question does not defer: d1 answers and its set is only logged. The choice and the
  match numbers are frozen into the JSON (``deferral``) under the same sha256.
* **A decision defers when ANY of its deferring questions defers** (``DeferralResult.deferred``), so the quantity held at
  20% is the **joint** development defer rate over the deferring questions. ``freeze_thresholds`` runs one search over
  the deferring questions' alphas: coordinate descent on the alpha grid, started from each question's independent best
  alpha and from each question's least-deferring alpha, minimising |joint - 0.20| with ties to the smaller alpha sum.
  Deferral is not monotone in alpha (tiny alpha gives big sets, large alpha gives empty sets), so the grid is searched,
  not bisected. A non-deferring question keeps its independent best alpha (its sets are logged only).
  ``joint_on_target`` records whether the joint rate landed within 2 points of 20%.
* ``choose_alpha`` (one question alone) is kept for diagnostics and as the search's starting point.

Thresholds are frozen into JSON with a sha256 of their canonical body and never adapt during evaluation. A 10%
audit sample (deterministic hash of the decision key) has the deferral models answer even when d1 kept the decision;
it is logged only and used to report the coverage achieved, never acted on. A run's defer rate outside 5-40% yields an
``alarm`` entry payload (kind ``defer_rate``); nothing else changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from antelligence.kernel.canonical import canonical_json, content_hash
from antelligence.experiments.v41_resolve import resolve_answer_label, resolve_distribution
from antelligence.kernel.decision import Answer, DecisionRequest

QTYPES = ("target", "release", "amount")
DEFER_TARGET = 0.20
AUDIT_RATE = 0.10
ALARM_BAND = (0.05, 0.40)
ALPHA_GRID = tuple(i / 200 for i in range(1, 200))
DEFER_MARGIN = 0.05  # a model must beat d1's resolved match to S' by this much (absolute) for a question to defer
D1_NAME = "d1"
EPS = 1e-12
TARGET_TOLERANCE = 0.02  # |dev deferral - 20%| allowed before a question is flagged off-target


class ConformalError(ValueError):
    """Bad input, or a frozen file that does not match its hash."""


@dataclass(frozen=True)
class DevRecord:
    """One development decision. ``probs[q]`` = d1's {resolved label: p}; ``labels[q]`` = S''s resolved label."""

    probs: Mapping[str, Mapping[str, float]]
    labels: Mapping[str, str]


def label_distributions(request: DecisionRequest, answers: Mapping[str, Answer],
                        target_map: Optional[Mapping[str, str]] = None) -> Dict[str, Dict[str, float]]:
    """Kernel Answers -> {question: {label: p}}. noul -> yes/no; score -> amount levels "1".."5" (index + 1).
    With ``target_map`` (label -> resolved key) the target distribution is over resolved regions (sums per key)."""
    out: Dict[str, Dict[str, float]] = {}
    for qid, q in request.questions.items():
        a = answers[qid]
        if q.type == "noul":
            out[qid] = {"yes": a.probabilities["yes"], "no": a.probabilities["no"]}
        elif q.type == "choice":
            out[qid] = resolve_distribution({o: a.probabilities[o] for o in q.options},
                                            target_map if qid == "target" else None)
        else:
            out[qid] = {str(int(i) + 1): p for i, p in a.probabilities.items()}
    return out


def decision_labels(request: DecisionRequest, answers: Mapping[str, Answer],
                    target_map: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The harness decision rule as labels: argmax (ties to the earlier label), release iff P(yes) >= 0.5. With
    ``target_map`` the target is its resolved key (what the answer refers to), else the letter."""
    from antelligence.worlds.tumor_mri.questions import decide_from_answers  # worlds may be imported from experiments

    d = decide_from_answers(answers, request)
    return {"target": resolve_answer_label("target", d["target"], target_map),
            "release": "yes" if d["release"] else "no", "amount": str(d["amount"])}


def dev_record(request: DecisionRequest, d1_answers: Mapping[str, Answer], sprime_resolved: Mapping[str, str],
               target_map: Optional[Mapping[str, str]] = None) -> DevRecord:
    return DevRecord(label_distributions(request, d1_answers, target_map), dict(sprime_resolved))


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
    joint_on_target: bool = False
    # per question: {"model": name or None, "defers": bool, "match": {model: rate}, "d1_match": rate, "margin": 0.05, "n": n}
    # empty = a pre-S' file: every question defers to ``target_model``
    deferral: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def body(self) -> Dict[str, Any]:
        # inf is not JSON: it is stored as null and read back as inf
        pq = {q: {**v, "qhat": None if math.isinf(v["qhat"]) else v["qhat"]} for q, v in self.per_question.items()}
        body = {"study": "v4.1", "kind": "conformal-thresholds", "arm": self.arm, "target_model": self.target_model,
                "defer_target": self.defer_target, "per_question": pq,
                "joint_dev_defer_rate": self.joint_dev_defer_rate, "joint_on_target": self.joint_on_target}
        if self.deferral:
            body["conformity_label"] = "sprime-resolved"
            body["deferral"] = {q: dict(v) for q, v in self.deferral.items()}
        return body

    def model_for(self, qtype: str) -> Optional[str]:
        """The model that answers ``qtype`` when it defers, or None if that question never defers."""
        if not self.deferral:
            return self.target_model
        info = self.deferral.get(qtype) or {}
        return info.get("model") if info.get("defers") else None

    def deferring(self) -> Tuple[str, ...]:
        return tuple(q for q in QTYPES if self.model_for(q) is not None)

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
        return cls(body["arm"], body["target_model"], pq, body["joint_dev_defer_rate"], body["defer_target"],
                   body["joint_on_target"], {q: dict(v) for q, v in (body.get("deferral") or {}).items()})


def _deferral_table(records: Sequence[DevRecord], qtype: str, grid: Sequence[float]) -> Tuple[List[float], np.ndarray]:
    """qhat and the per-record defer flags for every alpha on the grid (rows = alphas)."""
    scores = [nonconformity(r.probs[qtype], r.labels[qtype]) for r in records]
    qhats = [conformal_quantile(scores, a) for a in grid]
    flags = np.array([[defers(prediction_set(r.probs[qtype], q)) for r in records] for q in qhats], dtype=bool)
    return qhats, flags


def choose_deferral_models(resolved: Sequence[Mapping[str, Mapping[str, str]]], sprime: Sequence[Mapping[str, str]],
                           candidates: Sequence[str], *, d1: str = D1_NAME, margin: float = DEFER_MARGIN
                           ) -> Dict[str, Dict[str, Any]]:
    """Per question type: the candidate with the best dev match to S' (resolved labels), and whether it may defer.

    ``resolved[i][model][q]`` is model's resolved label for question ``q`` on decision ``i`` (``d1`` included);
    ``sprime[i][q]`` is S''s resolved label. The best candidate wins ties by order (``candidates`` is cheapest first). A
    question defers only if ``best - d1 >= margin`` (absolute); otherwise d1 answers and the sets are only logged."""
    n = len(sprime)
    if n == 0 or n != len(resolved):
        raise ConformalError("need one resolved answer set per S' label")
    out: Dict[str, Dict[str, Any]] = {}
    for q in QTYPES:
        match = {m: sum(r[m][q] == sp[q] for r, sp in zip(resolved, sprime)) / n for m in (d1, *candidates)}
        best = max(candidates, key=lambda m: (match[m], -list(candidates).index(m)))
        out[q] = {"model": best, "defers": bool(match[best] - match[d1] >= margin - 1e-12),
                  "match": {m: match[m] for m in (d1, *candidates)}, "d1_match": match[d1], "margin": margin, "n": n}
    return out


def freeze_thresholds(records: Sequence[DevRecord], *, arm: str, target_model: str, target: float = DEFER_TARGET,
                      grid: Sequence[float] = ALPHA_GRID,
                      deferral: Optional[Mapping[str, Mapping[str, Any]]] = None) -> ConformalThresholds:
    """Fit all three question types on development records (from both SW-B and SW-noBoard runs) and freeze.

    One joint search over the deferring questions' alphas so that the rate at which ANY of them defers is closest to
    ``target``. ``deferral`` (from :func:`choose_deferral_models`) says which questions defer; without it all three do."""
    if not records:
        raise ConformalError("no development records")
    grid = tuple(grid)
    tables = {q: _deferral_table(records, q, grid) for q in QTYPES}
    deferring = tuple(q for q in QTYPES if deferral is None or deferral[q]["defers"])
    if not deferring:
        best = {q: grid.index(choose_alpha(records, q, target, grid)["alpha"]) for q in QTYPES}
        return _thresholds(records, tables, best, grid, deferring, arm, target_model, target, deferral)

    def joint(idx: Mapping[str, int]) -> float:
        return float(np.logical_or.reduce([tables[q][1][idx[q]] for q in deferring]).mean())

    def key(idx: Mapping[str, int]) -> Tuple[float, float]:
        return (round(abs(joint(idx) - target), 12), sum(grid[idx[q]] for q in QTYPES))

    independent = {q: grid.index(choose_alpha(records, q, target, grid)["alpha"]) for q in QTYPES}
    starts = [dict(independent)]
    starts += [{q: (int(np.argmin(tables[q][1].mean(axis=1))) if q in deferring else independent[q]) for q in QTYPES}]
    best: Optional[Dict[str, int]] = None
    for start in starts:
        idx = dict(start)
        for _ in range(20):  # coordinate descent to a fixed point
            changed = False
            for q in deferring:
                cand = min(range(len(grid)), key=lambda i: key({**idx, q: i}))
                if key({**idx, q: cand}) < key(idx):
                    idx[q], changed = cand, True
            if not changed:
                break
        if best is None or key(idx) < key(best):
            best = idx
    assert best is not None
    return _thresholds(records, tables, best, grid, deferring, arm, target_model, target, deferral)


def _thresholds(records: Sequence[DevRecord], tables: Mapping[str, Tuple[List[float], np.ndarray]],
                best: Mapping[str, int], grid: Sequence[float], deferring: Sequence[str], arm: str, target_model: str,
                target: float, deferral: Optional[Mapping[str, Mapping[str, Any]]]) -> ConformalThresholds:
    per_question = {q: {"alpha": grid[best[q]], "qhat": tables[q][0][best[q]], "n": len(records),
                        "dev_defer_rate": float(tables[q][1][best[q]].mean()),
                        "on_target": bool(tables[q][1][best[q]].mean() <= target + TARGET_TOLERANCE)}
                    for q in QTYPES}
    rate = float(np.logical_or.reduce([tables[q][1][best[q]] for q in deferring]).mean()) if deferring else 0.0
    return ConformalThresholds(arm, target_model, per_question, rate, target,
                               joint_on_target=bool(deferring and abs(rate - target) <= TARGET_TOLERANCE),
                               deferral={q: dict(v) for q, v in (deferral or {}).items()})


# --- applying ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeferralResult:
    sets: Mapping[str, Tuple[str, ...]]
    deferred: bool
    deferred_questions: Tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return any(len(s) == 0 for s in self.sets.values())


class ConformalDeferral:
    def __init__(self, thresholds: ConformalThresholds) -> None:
        self.thresholds = thresholds

    def decide(self, probs: Mapping[str, Mapping[str, float]]) -> DeferralResult:
        """``probs`` are d1's RESOLVED distributions (``label_distributions(..., target_map)``). Sets are computed for
        every question (and logged); only the questions that have a deferral model can defer the decision."""
        sets = {q: prediction_set(probs[q], self.thresholds.qhat(q)) for q in QTYPES}
        dq = tuple(q for q in self.thresholds.deferring() if defers(sets[q]))
        return DeferralResult(sets, bool(dq), dq)

    def decide_answers(self, request: DecisionRequest, answers: Mapping[str, Answer],
                       target_map: Optional[Mapping[str, str]] = None) -> DeferralResult:
        return self.decide(label_distributions(request, answers, target_map))


# --- audit sample and alarm -------------------------------------------------------------------------------------


def audit_selected(key: str, rate: float = AUDIT_RATE, salt: str = "v4.1|audit") -> bool:
    """Deterministic 10% sample: sha256(salt|key) read as a uniform number in [0, 1). The key is the decision's
    identity (e.g. ``f"{run_id}|{squad}|{round}"``), so a replay audits exactly the same decisions."""
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < rate


def audit_coverage(audits: Sequence[Tuple[Mapping[str, Tuple[str, ...]], Mapping[str, str]]]) -> Dict[str, float]:
    """Coverage achieved on audited decisions: share whose deferral-model label is inside d1's set, per question and all.
    An audit holds only the questions that have a deferral model; a question absent from every audit is not reported."""
    if not audits:
        return {}
    out: Dict[str, float] = {}
    for q in QTYPES:
        have = [(sets, lab) for sets, lab in audits if q in lab]
        if have:
            out[q] = sum(lab[q] in sets[q] for sets, lab in have) / len(have)
    out["all"] = sum(all(lab[q] in sets[q] for q in lab) for sets, lab in audits) / len(audits)
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
