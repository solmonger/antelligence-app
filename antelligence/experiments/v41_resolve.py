"""Resolve answer labels to what they refer to (decision 2026-10-13-conformity-label-sprime-resolved, rule a).

``S`` means "keep the current target", so the letter alone says nothing about where the squad goes. Before an answer is
compared with S' (or used as a conformity label) its ``target`` label is resolved to a *resolved key*:

* a region label ``A``..``F`` -> ``"r<region_id>"`` (the region it is offered as, for this squad, in this round);
* ``S`` -> the squad's current target: ``"r<region_id>"``, ``"explore"`` if it is exploring, or ``"hold"`` when it has
  no target (a squad that has never been given one stands still; S then means "stay put" and matches nothing but
  another hold);
* ``X`` -> ``"explore"`` (no region).

Release (``yes``/``no``) and amount (``1``..``5``) labels are their own resolution. The same function resolves S''s
target label, so a model matches S' iff both point at the same region (or both explore, or both hold).

Exact state comes from the world (``target_region`` in the observation, and the candidate list of the round). Older
calibration logs did not record it; ``derive_resolutions`` rebuilds it from the decision rows and the board's route
entries as far as that is possible *exactly* (see its docstring) and flags what it cannot.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

EXPLORE_KEY = "explore"
HOLD_KEY = "hold"
STAY_LABEL, EXPLORE_LABEL = "S", "X"
CANDIDATE_LABELS = ("A", "B", "C", "D", "E", "F")


class ResolveError(ValueError):
    pass


def region_key(region_id: int) -> str:
    return f"r{int(region_id)}"


def current_key(target_label: Optional[str], target_region: Optional[int]) -> str:
    """The squad's current target as a resolved key, from the world's ``target`` / ``target_region`` observation."""
    if target_label == EXPLORE_LABEL:
        return EXPLORE_KEY
    if target_region is not None:
        return region_key(target_region)
    return HOLD_KEY


def resolve_label(label: str, label_regions: Mapping[str, str], current: Optional[str]) -> str:
    """The single resolution function. ``label_regions`` maps offered region labels to region keys; ``current`` is the
    squad's current-target key (``None`` is treated as no target)."""
    if label == STAY_LABEL:
        return current or HOLD_KEY
    if label == EXPLORE_LABEL:
        return EXPLORE_KEY
    try:
        return label_regions[label]
    except KeyError:
        raise ResolveError(f"label {label!r} is not offered (offered: {sorted(label_regions)})") from None


def target_map(labels: Iterable[str], label_regions: Mapping[str, str], current: Optional[str]) -> Dict[str, str]:
    """{target label: resolved key} for every offered label, in the given (fixed) label order."""
    return {l: resolve_label(l, label_regions, current) for l in labels}


def state_from_fields(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolution state of one squad-round from the observation: offered labels, their regions, the current target."""
    label_regions = {c["label"]: region_key(c["region_id"]) for c in fields.get("candidates", [])}
    current = current_key(fields.get("target"), fields.get("target_region"))
    return {"labels": list(label_regions) + [STAY_LABEL, EXPLORE_LABEL], "label_regions": label_regions,
            "current": current}


def resolution_record(fields: Mapping[str, Any], sprime: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """What the harness logs per decision: the current target and the label -> key map (exact), and S''s resolved target."""
    st = state_from_fields(fields)
    tmap = target_map(st["labels"], st["label_regions"], st["current"])
    out: Dict[str, Any] = {"source": "exact", "current": st["current"], "target_map": tmap}
    if sprime:
        out["sprime_target"] = resolve_label(sprime["target"], st["label_regions"], st["current"])
    return out


def resolve_distribution(probs: Mapping[str, float], tmap: Optional[Mapping[str, str]]) -> Dict[str, float]:
    """Sum a label distribution over the labels that resolve to the same key. The key order is the first occurrence in
    ``probs`` (fixed label order), so ties between keys stay deterministic. ``None``/empty map = identity."""
    if not tmap:
        return {k: float(v) for k, v in probs.items()}
    out: Dict[str, float] = {}
    for label, p in probs.items():
        k = tmap.get(label, label)
        out[k] = out.get(k, 0.0) + float(p)
    return out


def resolve_answer_label(question: str, label: str, tmap: Optional[Mapping[str, str]]) -> str:
    return tmap.get(label, label) if (question == "target" and tmap) else label


# --- legacy logs: derive the state from rows + the board's route entries -------------------------------------------------


def load_routes(run_dir: Path) -> Dict[int, List[Tuple[int, List[int]]]]:
    """{squad: [(tick, region_ids)]} from the board export next to ``decisions.jsonl`` (``<run_id>.jsonl``); {} if absent."""
    out: Dict[int, List[Tuple[int, List[int]]]] = defaultdict(list)
    for f in sorted(Path(run_dir).glob("*.jsonl")):
        if f.name == "decisions.jsonl":
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("type") == "route":
                p = e["payload"]
                out[int(p["squad"])].append((int(e["tick"]), [int(r) for r in p["region_ids"]]))
    return {s: sorted(v) for s, v in out.items()}


def derive_resolutions(rows: Sequence[Mapping[str, Any]], routes: Mapping[int, Sequence[Tuple[int, Sequence[int]]]]
                       ) -> Dict[int, Dict[str, Any]]:
    """Resolution records for legacy decision rows, keyed by the row's index in ``rows``.

    What is exact. The world offers a squad ``route region ids first, then the queen's other candidates``, so after a
    route entry of ``k`` regions the labels ``A``..(k-th) are exactly those regions; and a squad with no target adopts
    ``route[0]`` (``assign_route``). The squad's own previous decisions say what it chose (``S`` keeps, ``X`` explores,
    a label picks that label's region). Everything is replayed in (squad, round) order from those facts.

    What is not. The other labels (the queen's extra candidates) are not logged. They are keyed ``e<tick>:<label>``,
    meaning "label L as offered from the route step at <tick>". Inside one step labels compare exactly. A target carried
    over from an EARLIER step (``stale``) can still be compared with this step's labels only when it is a route region
    (equal ids) or explore/hold; against the queen's extra candidates it is unknown whether the queen re-offered the same
    region. Only an ``S`` answer (and the current-target rule) ever depends on that, and ``match_to_sprime`` reports
    those comparisons as a lower bound (unknown = no match) and an upper bound (unknown = match). Rows from squads without route entries get no record (the caller falls back to the raw labels)."""
    by_squad: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_squad[int(r["squad"])].append(i)
    out: Dict[int, Dict[str, Any]] = {}
    for squad, idxs in by_squad.items():
        steps = list(routes.get(squad, []))
        if not steps:
            continue
        idxs.sort(key=lambda i: (rows[i]["tick"], i))
        cur, cur_step, applied = HOLD_KEY, -1, -1
        for i in idxs:
            row = rows[i]
            tick = int(row["tick"])
            # route steps up to this decision's tick have been applied (queen acts before the round at the same tick)
            while applied + 1 < len(steps) and steps[applied + 1][0] <= tick:
                applied += 1
                if cur == HOLD_KEY and steps[applied][1]:
                    cur, cur_step = region_key(steps[applied][1][0]), steps[applied][0]
            if applied < 0:
                continue
            step_tick, ids = steps[applied][0], list(steps[applied][1])
            options = _offered_labels(row)
            label_regions = {l: (region_key(ids[k]) if k < len(ids) else f"e{step_tick}:{l}")
                             for k, l in enumerate(CANDIDATE_LABELS) if l in options}
            tmap = target_map(list(label_regions) + [STAY_LABEL, EXPLORE_LABEL], label_regions, cur)
            rec: Dict[str, Any] = {"source": "derived", "current": cur, "target_map": tmap,
                                   "stale": bool(cur not in (HOLD_KEY, EXPLORE_KEY) and cur_step != step_tick)}
            sp = row.get("sprime")
            if sp:
                rec["sprime_target"] = resolve_label(sp["target"], label_regions, cur)
            out[i] = rec
            chosen = str(row.get("target"))
            if chosen == EXPLORE_LABEL:
                cur = EXPLORE_KEY
            elif chosen != STAY_LABEL:
                cur, cur_step = tmap.get(chosen, cur), step_tick
    return out


def _offered_labels(row: Mapping[str, Any]) -> List[str]:
    """Candidate labels the row's counterfactual d1 distribution carries (A..F present), else all six."""
    cf = (row.get("counterfactual") or {})
    for v in cf.values():
        probs = (v.get("probs") or {}).get("target")
        if probs:
            return [l for l in probs if l in CANDIDATE_LABELS]
    return list(CANDIDATE_LABELS)


def compare_keys(a: str, b: str) -> Optional[bool]:
    """Do two resolved keys point at the same place, when ``a`` may come from an earlier route step? ``None`` = unknown.

    ``r<id>``, ``explore`` and ``hold`` are exact and compare by equality. A derived key ``e<tick>:<label>`` (a queen
    extra candidate) against a region key or a key from another step is unknown: the queen may have re-offered the same
    region. Two keys of the same step compare by label."""
    if a == b:
        return True
    ea, eb = a.startswith("e") and ":" in a, b.startswith("e") and ":" in b
    if not ea and not eb:
        return False
    if ea and eb:
        return False if a.split(":")[0] == b.split(":")[0] else None
    other = b if ea else a
    return False if other in (EXPLORE_KEY, HOLD_KEY) else None


# --- the metric ------------------------------------------------------------------------------------------------------


def match_to_sprime(samples: Sequence[Any]) -> Dict[str, Any]:
    """Raw and resolved match rates to S' per model per question (rule a).

    ``samples`` are ``v41_fit.Sample`` objects: ``decisions[model]`` holds the model's {target, release, amount};
    ``sprime`` the raw S' labels; ``target_map`` / ``sprime_resolved`` the resolution (empty = raw, reported as such).
    Raw compares label to label. Resolved compares the resolved target key; release and amount are identical in both.
    ``resolved`` counts only comparisons that are known to match; ``resolved_upper`` also counts the ones that cannot be
    told (derived keys, see ``compare_keys``), so for logs the harness resolved exactly the two are equal.
    Returns ``{"n", "resolution_sources", "stale_current", "models": {model: {question: {raw, resolved, resolved_upper, n}}}}``."""
    models: Dict[str, Dict[str, Dict[str, float]]] = {}
    usable = [s for s in samples if s.sprime]
    names = sorted({m for s in usable for m in s.decisions})
    for m in names:
        per: Dict[str, Dict[str, float]] = {}
        use = [s for s in usable if m in s.decisions]
        for q in ("target", "release", "amount"):
            raw = res = unknown = 0
            for s in use:
                lab = _labels(s.decisions[m])
                raw += int(lab[q] == s.sprime[q])
                if q == "target":
                    mine = s.target_map.get(lab[q], lab[q]) if s.target_map else lab[q]
                    theirs = s.sprime_resolved.get("target", s.sprime[q]) if s.sprime_resolved else s.sprime[q]
                    # only a carried-over current target (an S answer on a stale row) can be of an earlier route step
                    same = compare_keys(mine, theirs) if (lab[q] == STAY_LABEL and s.stale) else (mine == theirs)
                    res += int(same is True)
                    unknown += int(same is None)
                else:
                    res += int(lab[q] == s.sprime[q])
            n = len(use)
            nan = float("nan")
            per[q] = {"raw": raw / n if n else nan, "resolved": res / n if n else nan,
                      "resolved_upper": (res + unknown) / n if n else nan, "n": n}
        models[m] = per
    sources: Dict[str, int] = defaultdict(int)
    for s in usable:
        sources[s.resolution] += 1
    return {"n": len(usable), "resolution_sources": dict(sources),
            "stale_current": sum(1 for s in usable if s.stale), "models": models}


def _labels(d: Mapping[str, Any]) -> Dict[str, str]:
    return {"target": str(d["target"]), "release": "yes" if d["release"] else "no", "amount": str(int(d["amount"]))}


def format_match(report: Mapping[str, Any]) -> str:
    lines = [f"decisions with S' labels: {report['n']}  resolution: {report['resolution_sources']}  "
             f"(current target from an earlier route step: {report['stale_current']})",
             f"{'model':<14}{'question':<9}{'raw':>8}{'resolved':>10}{'upper':>8}{'n':>7}"]
    for m, per in report["models"].items():
        for q, v in per.items():
            lines.append(f"{m:<14}{q:<9}{v['raw']:>8.3f}{v['resolved']:>10.3f}{v['resolved_upper']:>8.3f}{v['n']:>7d}")
    return "\n".join(lines)
