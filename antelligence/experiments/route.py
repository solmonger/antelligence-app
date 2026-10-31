"""Routing: send each kind of question to the setup that does best on it, using answers that are already saved.

Why: on the first fresh-question run (docs/research/fresh-20261031/RESULTS.md) no single setup won everywhere. The
reader was best on word problems and assignment puzzles, Jev alone on syllogisms, and the ants with notes beat the
reader on syllogisms too. So the cheap question is whether a router that knows the kind of question beats the best
single setup. It makes no model call: every arm's answer to every question is already on disk.

Picking the best arm per kind on the same questions you then score it on flatters it, so this module only ever
reports routing three ways, and says which:

* ``in_sample``: picked and scored on the same questions. Optimistic, for reference only.
* ``split_half``: each kind's questions split in two at random, routes picked on one half and scored on the other,
  averaged over many splits. A fair estimate from one set.
* ``held_out``: picked on one whole set (the routes are frozen to a file first), scored on a brand-new set. This is
  the real test.

The router knows each question's kind. On the fresh sets the kind is given by construction; a real system would have
to tell kinds apart, which this doesn't test.

Pure functions only: tables in, numbers out.
"""

from __future__ import annotations

import statistics as st
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import fair_test as ft

# When two arms tie on the picking questions, the earlier one here wins: free and cheap before paid, and the plain
# reader first because it is one call. Arms not listed come after these, in name order.
PREFER = ("reader", "notes8", "both8", "alone8", "d1-3b:dm1", "d1-3b:read", "d1-3b:vote",
          "jev:dm1", "jev:read", "jev:vote")

# Frozen reading of the held-out test (routed minus the reader alone, on the new set).
LINES = {"helps": 0.05, "no_gain": 0.05}


Table = Dict[str, Dict[str, Any]]   # question id -> {"group": kind, "right": {arm: 0.0 or 1.0}}


def table(swarm: Sequence[dm.SwarmRow], dms: Mapping[str, Mapping[str, Mapping[str, Any]]],
          key: Mapping[str, str]) -> Table:
    """Every arm's right/wrong on every scored exam question. Arms: the swarm's own (reader, alone8, notes8, both8),
    and for each decision model ``<name>:dm1`` (alone), ``<name>:read`` (reading the ants) and ``<name>:vote`` (one
    more voter), exactly as :func:`dm_addon.score` scores them."""
    out: Table = {}
    for r in swarm:
        if r.kind != "exam" or r.id not in key:
            continue
        truth = key[r.id]
        right = {arm: ft.score_item("exam", ans, truth) for arm, ans in r.baselines.items()}
        ants = [a.answer for a in r.ants]
        for name, answers in dms.items():
            got = answers.get(r.id)
            if got is None:
                continue
            right[f"{name}:dm1"] = ft.score_item("exam", dm.dm_value("exam", got.get("dm1")), truth)
            right[f"{name}:vote"] = ft.score_item("exam", dm.vote_with_dm("exam", ants, got.get("dm1")), truth)
            if got.get("read") is not None:
                right[f"{name}:read"] = ft.score_item("exam", dm.dm_value("exam", got.get("read")), truth)
        out[r.id] = {"group": r.group, "right": right}
    return out


def common_arms(t: Table, allowed: Optional[Sequence[str]] = None) -> List[str]:
    """Arms every question has (so a route never points at a missing answer), in preference order."""
    if not t:
        return []
    arms = set.intersection(*(set(v["right"]) for v in t.values()))
    if allowed is not None:
        arms &= set(allowed)
    return order(arms)


def order(arms) -> List[str]:
    rank = {a: i for i, a in enumerate(PREFER)}
    return sorted(arms, key=lambda a: (rank.get(a, len(PREFER)), a))


def groups(t: Table) -> List[str]:
    return sorted({v["group"] for v in t.values()})


def best_arm(t: Table, ids: Sequence[str], arms: Sequence[str]) -> Tuple[str, float]:
    """The arm with the highest share right on ``ids`` (ties to the earlier arm in ``arms``)."""
    best, best_score = arms[0], -1.0
    for arm in arms:
        s = st.mean(t[i]["right"][arm] for i in ids) if ids else 0.0
        if s > best_score + 1e-12:
            best, best_score = arm, s
    return best, best_score


def pick_routes(t: Table, arms: Sequence[str]) -> Dict[str, Any]:
    """Per kind, the best arm on all of ``t``; plus the single best arm overall (the rival to beat)."""
    routes = {}
    for g in groups(t):
        ids = [i for i, v in t.items() if v["group"] == g]
        arm, s = best_arm(t, ids, arms)
        routes[g] = {"arm": arm, "score": round(s, 4), "n": len(ids)}
    single, s = best_arm(t, list(t), arms)
    return {"routes": routes, "single": {"arm": single, "score": round(s, 4)}, "arms": list(arms)}


def apply(t: Table, routes: Mapping[str, Mapping[str, Any]], fallback: str) -> Dict[str, float]:
    """Each question answered by its kind's routed arm (a kind with no route uses ``fallback``)."""
    return {i: v["right"][routes.get(v["group"], {}).get("arm", fallback)] for i, v in t.items()}


def split_half(t: Table, arms: Sequence[str], reps: int = 200, seed: int = 20261031) -> Dict[str, Any]:
    """Cross-checking on one set: split each kind's questions in two at random, pick the routes (and the single best
    arm) on one half and score them on the other, both ways round, and average over ``reps`` random splits. Leave-one-out
    is not used: with near-tied arms, dropping one question flips the pick against that very question, which skews it.
    ``spread`` is the middle 95% of the per-split gains (how much the answer moves with the split), not a confidence
    range."""
    import random

    rng = random.Random(seed)
    by_group: Dict[str, List[str]] = {}
    for i, v in t.items():
        by_group.setdefault(v["group"], []).append(i)
    reader_mean = st.mean(v["right"]["reader"] for v in t.values())
    routed_means, single_means = [], []
    for _ in range(reps):
        halves: Tuple[List[str], List[str]] = ([], [])
        for ids in by_group.values():
            ids = list(ids)
            rng.shuffle(ids)
            halves[0].extend(ids[: len(ids) // 2])
            halves[1].extend(ids[len(ids) // 2:])
        routed, single = [], []
        for pick_on, score_on in ((halves[0], halves[1]), (halves[1], halves[0])):
            sub = {i: t[i] for i in pick_on}
            p = pick_routes(sub, arms)
            for i in score_on:
                arm = p["routes"].get(t[i]["group"], {}).get("arm", p["single"]["arm"])
                routed.append(t[i]["right"][arm])
                single.append(t[i]["right"][p["single"]["arm"]])
        routed_means.append(st.mean(routed))
        single_means.append(st.mean(single))

    def spread(xs: Sequence[float]) -> List[float]:
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]

    return {"reps": reps, "routed": round(st.mean(routed_means), 4), "single": round(st.mean(single_means), 4),
            "reader": round(reader_mean, 4),
            "routed_vs_reader": {"gain": round(st.mean(routed_means) - reader_mean, 4),
                                 "spread": spread([m - reader_mean for m in routed_means])},
            "routed_vs_single": {"gain": round(st.mean(r - s for r, s in zip(routed_means, single_means)), 4),
                                 "spread": spread([r - s for r, s in zip(routed_means, single_means)])}}


def _boot(a: Mapping[str, float], b: Mapping[str, float]) -> Dict[str, Any]:
    ids = [i for i in a if i in b]
    gain, lo, hi = ft.paired_boot([a[i] - b[i] for i in ids])
    return {"gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(ids)}


def mean(xs: Mapping[str, float]) -> float:
    return round(st.mean(xs.values()), 4) if xs else 0.0


def verdict(vs_reader: Mapping[str, Any]) -> str:
    if vs_reader["gain"] >= LINES["helps"] and vs_reader["ci95"][0] > 0:
        return "helps: routing by kind beats the reader alone on new questions"
    if vs_reader["ci95"][1] < LINES["no_gain"]:
        return "no gain: routing by kind is not clearly better than the reader alone on new questions"
    return "unclear: routing may help, but the range is too wide to call"


def study(t: Table, arms: Sequence[str]) -> Dict[str, Any]:
    """In-sample and leave-one-out routing on one set, against the reader and the best single arm."""
    picked = pick_routes(t, arms)
    reader = {i: v["right"]["reader"] for i, v in t.items()}
    ins = apply(t, picked["routes"], picked["single"]["arm"])
    single_ins = {i: v["right"][picked["single"]["arm"]] for i, v in t.items()}
    return {"n": len(t), "arms": list(arms), "picked": picked,
            "by_kind": {g: {arm: round(st.mean(v["right"][arm] for v in t.values() if v["group"] == g), 4)
                            for arm in arms} for g in groups(t)},
            "reader": mean(reader),
            "in_sample": {"routed": mean(ins), "single": mean(single_ins),
                          "routed_vs_reader": _boot(ins, reader), "routed_vs_single": _boot(ins, single_ins)},
            "split_half": split_half(t, arms)}


def held_out(frozen: Mapping[str, Any], t: Table) -> Dict[str, Any]:
    """The frozen routes (picked on another set) scored on ``t``, against the reader and the frozen single best arm."""
    routes, single = frozen["routes"], frozen["single"]["arm"]
    missing = sorted(({r["arm"] for r in routes.values()} | {single}) - set(common_arms(t)))
    if missing:
        raise ValueError(f"the new set has no answers for {missing}: run those arms on it first")
    reader = {i: v["right"]["reader"] for i, v in t.items()}
    routed = apply(t, routes, single)
    single_ans = {i: v["right"][single] for i, v in t.items()}
    vs_reader = _boot(routed, reader)
    by_kind = {}
    for g in groups(t):
        ids = [i for i, v in t.items() if v["group"] == g]
        arm = routes.get(g, {}).get("arm", single)
        by_kind[g] = {"arm": arm, "routed": round(st.mean(routed[i] for i in ids), 4),
                      "reader": round(st.mean(reader[i] for i in ids), 4), "n": len(ids)}
    return {"n": len(t), "routed": mean(routed), "reader": mean(reader), "single": {"arm": single, "score": mean(single_ans)},
            "routed_vs_reader": vs_reader, "routed_vs_single": _boot(routed, single_ans), "by_kind": by_kind,
            "verdict": verdict(vs_reader)}


__all__ = ["LINES", "PREFER", "apply", "best_arm", "common_arms", "groups", "held_out", "pick_routes", "split_half",
           "study", "table", "verdict"]
