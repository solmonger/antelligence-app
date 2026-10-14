"""v4.2 M1 metrics (goal section 8): per-run primary and colony metrics, honesty, timings, message-effect study.

Inputs are plain dicts: ``metrics`` = ``world.metrics()``, ``colony`` = ``ColonyResult.colony`` (bundle ``extra.colony``).
Nothing here reads a world, a key or a model.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from antelligence.kernel.colony import rng
from antelligence.kernel.colony.marks import MarkField, MarkParams


# --- timings ---------------------------------------------------------------------------------------------------------

class Timings:
    """Wall-clock recorder for the C0 finding: ``look_ms`` (per look call) and ``queen_ms`` (per queen step)."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self.samples: Dict[str, List[float]] = {"look_ms": [], "queen_ms": []}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        t0 = self.clock()
        try:
            yield
        finally:
            self.samples.setdefault(name, []).append((self.clock() - t0) * 1000.0)

    def wrap(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        def timed(*a: Any, **k: Any) -> Any:
            with self.time(name):
                return fn(*a, **k)
        return timed

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.samples.items():
            out[k] = {"n": len(v), "total": float(sum(v)), "mean": float(np.mean(v)) if v else 0.0,
                      "max": float(max(v)) if v else 0.0}
        return out


# --- marks: strong and disputed --------------------------------------------------------------------------------------

def mark_quality(colony: Mapping[str, Any], voxel_mm: float = 1.0) -> Dict[str, Any]:
    """Mark counts by kind, plus the share that were *strong* / *disputed* when they first became visible (tick + 1).

    Replays the deposit log through a fresh :class:`MarkField` with the run's own half-lives and radii.
    """
    p = colony.get("params", {})
    hl = p.get("half_life")
    mp = MarkParams(voxel_mm, p.get("read_radius_mm", 6.0), p.get("confirm_radius_mm", 4.0)) if hl is None else \
        MarkParams(voxel_mm, p.get("read_radius_mm", 6.0), p.get("confirm_radius_mm", 4.0), hl)
    field = MarkField("m1-metrics", mp)
    recs = sorted(colony.get("marks", []), key=lambda r: r["tick"])
    by_kind: Dict[str, int] = {}
    strong = disputed = 0
    i = 0
    while i < len(recs):
        t = recs[i]["tick"]
        group = []
        while i < len(recs) and recs[i]["tick"] == t:
            r = recs[i]
            field.deposit(r["ant"], r["mark_kind"], r["pos_vox"], r["tick"], amount=r["amount_milli"] / 1000.0,
                          reading_ppm=r["reading_ppm"])
            group.append(r)
            i += 1
        for r in group:
            by_kind[r["mark_kind"]] = by_kind.get(r["mark_kind"], 0) + 1
            if field.aggregate(r["mark_kind"], r["pos_vox"], t + 1).strong:
                strong += 1
            if field.disputed_at(r["pos_vox"], t + 1):
                disputed += 1
    total = sum(by_kind.values())
    return {"by_kind": dict(sorted(by_kind.items())), "total": total,
            "share_strong": strong / total if total else 0.0, "share_disputed": disputed / total if total else 0.0}


# --- honesty ---------------------------------------------------------------------------------------------------------

def claim_accuracy_by_ant(colony: Mapping[str, Any], truth_label: Callable[[Sequence[int]], bool]
                          ) -> Dict[str, Dict[str, int]]:
    """Scorer-checked claim accuracy per ant (development cases only). A claim is a ``tumour_here`` / ``not_tumour`` mark or
    a yes/no ``is_tumour`` answer; ``need_more`` and ``unknown`` answers are not checkable. Returns ``{ant_id: {correct, checked}}``."""
    out: Dict[str, Dict[str, int]] = {}

    def add(ant: int, said_tumour: bool, pos: Sequence[int]) -> None:
        d = out.setdefault(f"ant-{int(ant):02d}", {"correct": 0, "checked": 0})
        d["checked"] += 1
        d["correct"] += int(bool(truth_label(pos)) == said_tumour)

    for r in colony.get("marks", []):
        if r["mark_kind"] in ("tumour_here", "not_tumour"):
            add(r["ant"], r["mark_kind"] == "tumour_here", r["pos_vox"])
    asks = {m["msg_id"]: m for m in colony.get("messages", []) if m["type"] == "ask"}
    for m in colony.get("messages", []):
        if m["type"] != "answer" or m.get("answer") not in ("yes", "no"):
            continue
        ask = asks.get(m.get("reply_to"))
        if ask is None or ask["question"] != "is_tumour" or ask["about"]["pos_vox"] is None:
            continue
        add(m["from_ant"], m["answer"] == "yes", ask["about"]["pos_vox"])
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def honesty(colony: Mapping[str, Any], accuracy: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    """Correlation, over ants, of the in-run confirmation rate (confirmed / (confirmed + contradicted), from ``cards_after``)
    with the scorer-checked accuracy of the ant's claims. Ants with nothing resolved or nothing checked are left out;
    fewer than 3 usable ants or no spread -> ``None``."""
    xs, ys = [], []
    for ant, card in sorted((colony.get("cards_after") or {}).items()):
        c = card["in_run"]
        resolved = c["confirmed"] + c["contradicted"]
        acc = accuracy.get(ant)
        if resolved == 0 or not acc or acc["checked"] == 0:
            continue
        xs.append(c["confirmed"] / resolved)
        ys.append(acc["correct"] / acc["checked"])
    return {"correlation": pearson(xs, ys), "n_ants": len(xs)}


# --- per run ---------------------------------------------------------------------------------------------------------

def run_metrics(metrics: Mapping[str, Any], colony: Optional[Mapping[str, Any]] = None, *,
                accuracy: Optional[Mapping[str, Mapping[str, int]]] = None, timings: Optional[Timings] = None,
                voxel_mm: float = 1.0) -> Dict[str, Any]:
    """One run's flat record (goal section 8). ``drug_on_non_tumour`` = released drug that is not on target
    (``off_target_share * released_total``; includes drug that left the crop, as ``off_target_share`` does)."""
    released = float(metrics.get("released_total", 0.0))
    off = float(metrics.get("off_target_share", 0.0))
    out: Dict[str, Any] = {
        "coverage": float(metrics["coverage"]),
        "off_target_share": off,
        "time_to_50_coverage": metrics.get("time_to_50_coverage"),
        "coverage_at_queen": {str(k): float(v) for k, v in dict(metrics.get("coverage_at_queen", {})).items()},
        "drug_on_non_tumour": float(metrics["drug_on_non_tumour"]) if "drug_on_non_tumour" in metrics else off * released,
    }
    if timings is not None:
        out["timings"] = timings.summary()
    if colony is not None:
        c = colony.get("counters", {})
        q = mark_quality(colony, voxel_mm)
        out["colony"] = {
            "marks_by_kind": q["by_kind"], "marks_total": q["total"], "share_strong": q["share_strong"],
            "share_disputed": q["share_disputed"],
            "asks_sent": c.get("asks_sent", 0), "asks_answered": c.get("asks_answered", 0),
            "asks_lapsed": c.get("asks_lapsed", 0), "asks_refused": c.get("asks_refused", 0),
            "answers_refused": c.get("answers_refused", 0), "marks_refused": c.get("marks_refused", 0),
            "quorum_events": c.get("quorum_events", 0),
            "turns_skipped": int(sum((colony.get("turns_skipped") or {}).values())),
            "honesty": honesty(colony, accuracy) if accuracy is not None else {"correlation": None, "n_ants": 0},
        }
    return out


# --- message-effect study --------------------------------------------------------------------------------------------

def _items(colony: Mapping[str, Any], arm: str) -> List[Dict[str, Any]]:
    items = [{"kind": "mark", "id": i} for i in range(len(colony.get("marks", [])))]
    if arm == "S-talk":
        items += [{"kind": "message", "id": m["msg_id"]} for m in colony.get("messages", [])]
    return items


def sample_runs(arm: str, keys: Sequence[tuple], n_runs: int = 10) -> List[tuple]:
    """Seeded run sample: each (case, seed) draws its first uniform from ``effect_sample``; the ``n_runs`` lowest win."""
    ranked = sorted(keys, key=lambda k: (float(rng("effect_sample", arm, k[0], k[1]).random()), str(k)))
    return sorted(ranked[:n_runs], key=str)


def sample_items(arm: str, case_id: str, seed: int, colony: Mapping[str, Any], max_items: int = 30) -> List[Dict[str, Any]]:
    """Up to ``max_items`` marks/messages of one run, seeded by ``effect_sample`` (arm, case, seed)."""
    items = _items(colony, arm)
    g = rng("effect_sample", arm, case_id, seed)
    g.random()                                      # the draw sample_runs used
    if len(items) > max_items:
        idx = sorted(g.choice(len(items), size=max_items, replace=False).tolist())
        items = [items[i] for i in idx]
    return items


def effect_study(arm: str, baselines: Mapping[tuple, Mapping[str, Any]],
                 replay_fn: Callable[[tuple, Mapping[str, Any], Dict[str, int]], Any], *, n_runs: int = 10,
                 max_items: int = 30) -> Dict[str, Any]:
    """Removal-replay study. ``baselines``: ``{(case, seed): colony bundle}`` (``metrics`` inside). ``replay_fn(key, bundle,
    drop)`` runs ``run_colony_episode(replay=bundle, drop=drop)`` and returns the ``ColonyResult``. Items whose removal the
    replay did not apply are excluded and counted in ``unapplied``."""
    runs = sample_runs(arm, list(baselines), n_runs)
    rows, unapplied = [], 0
    for key in runs:
        bundle = baselines[key]
        base_cov = float(bundle["metrics"]["coverage"])
        for item in sample_items(arm, key[0], key[1], bundle, max_items):
            res = replay_fn(key, bundle, {item["kind"]: int(item["id"])})
            eff = res.effect
            if eff is None or not eff.get("applied", False):
                unapplied += 1
                continue
            rows.append({"case_id": key[0], "seed": key[1], "item": item, "changed": bool(eff["changed"]),
                         "no_ant_changed": eff.get("first_divergence") is None,
                         "coverage_delta": float(res.metrics["coverage"]) - base_cov})
    return {"arm": arm, "runs": len(runs), "unapplied": unapplied, "rows": rows, **effect_summary(rows)}


def effect_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Effect rate (receiver's next action changed), mean / median end-of-run coverage delta (replay minus baseline),
    spam rate (share of items after which no ant's action changed)."""
    n = len(rows)
    if n == 0:
        return {"n_items": 0, "effect_rate": None, "mean_delta": None, "median_delta": None, "mean_abs_delta": None,
                "spam_rate": None}
    d = np.array([r["coverage_delta"] for r in rows], float)
    return {"n_items": n, "effect_rate": sum(r["changed"] for r in rows) / n, "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)), "mean_abs_delta": float(np.abs(d).mean()),
            "spam_rate": sum(r["no_ant_changed"] for r in rows) / n}
