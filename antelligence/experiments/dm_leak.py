"""Has the decision model seen these exam questions before? A cheap probe of the add-on's 81.7% (2026-10-31).

Jev alone got 81.7% of the 240 public AGIEval questions right, and 75% on the logic puzzles, far above the local
reader. That could be memory rather than reasoning. Two arms ask the same model the same questions with one thing
changed, and every answer is mapped back to the original letters before scoring:

- ``relabel``: the full question, but the letters are moved so that every option gets a different letter than it
  had (a seeded derangement). A model that remembers "the answer is C" now picks a wrong option. A model that reasons
  is not affected.
- ``blind``: the options alone, with no passage and no question, letters moved the same way. Nobody can reason to an
  answer from the options alone, so a score far above chance means the options were remembered (or give the answer
  away by how they are written; the AQuA-RAT maths options are the likeliest to).

The verdict is frozen here, before any call, in :func:`verdict`. Plan: ``docs/plans/2026-10-31-dm-leak-probe.md``.
"""

from __future__ import annotations

import hashlib
import random
import statistics as st
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import fair_test as ft
from antelligence.kernel.decision import Question

ARMS = ("relabel", "blind")
INSTRUCTIONS = {
    "relabel": dm.DM_INSTRUCTIONS[("exam", "dm1")],          # word for word what dm1 asked
    "blind": "Which option is most likely the correct answer?",
}
# Frozen reading of the result. "drop" is dm1 minus relabel; "above_chance" is blind minus the chance level.
LINES = {"relabel_drop": 0.10, "blind_above_chance": 0.25, "clean_drop": 0.05, "clean_above_chance": 0.10}


def derangement(letters: Sequence[str], qid: str) -> Dict[str, str]:
    """original letter -> new letter, seeded by the question, with no option keeping its letter."""
    letters = list(letters)
    if len(letters) < 2:
        raise ValueError(f"{qid}: need two options to move letters")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"leak:{qid}".encode()).digest()[:8], "big"))
    while True:
        new = letters[:]
        rng.shuffle(new)
        if all(a != b for a, b in zip(letters, new)):
            return dict(zip(letters, new))


def probe_state(raw: Mapping[str, Any], arm: str, moved: Mapping[str, str]) -> str:
    """The state for one arm, with each option under its new letter and listed in new-letter order."""
    texts = dm.option_texts(raw)
    listing = "Options:\n" + "\n".join(f"({moved[o]}){texts[o]}" for o in sorted(texts, key=lambda o: moved[o]))
    if arm == "blind":
        return listing
    parts = [str(raw["passage"]).strip()] if raw.get("passage") else []
    parts += [str(raw["question"]).strip(), listing]
    return "\n\n".join(parts)


def probe_request(model: str, raw: Mapping[str, Any], arm: str) -> Tuple[dm.LongRequest, Dict[str, str]]:
    """(request, new letter -> original letter)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    qid = str(raw["id"])
    texts = dm.option_texts(raw)
    moved = derangement(sorted(texts), qid)
    options = {moved[o]: texts[o] for o in sorted(texts, key=lambda o: moved[o])}
    q = Question("choice", INSTRUCTIONS[arm], tuple(options), options)
    req = dm.LongRequest.build(model, probe_state(raw, arm, moved), {"answer": q}, dm.call_seed(qid, f"leak-{arm}"))
    return req, {new: old for old, new in moved.items()}


def back(answer: Optional[Mapping[str, Any]], back_map: Mapping[str, str]) -> Optional[str]:
    """The model's pick as an original letter, or None if the call failed."""
    return None if answer is None else back_map.get(answer["choice"])


def chance(raw: Mapping[str, Any]) -> float:
    return 1.0 / len(raw["options"])


def _boot(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    gain, lo, hi = ft.paired_boot([x - y for x, y in zip(a, b)])
    return {"gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def score(raws: Sequence[Mapping[str, Any]], picks: Mapping[str, Mapping[str, Optional[str]]],
          key: Mapping[str, str], dm1: Optional[Mapping[str, Optional[str]]] = None) -> Dict[str, Any]:
    """``picks``: arm -> question id -> original letter (or None). ``dm1``: question id -> the add-on's own dm1 letter.
    Only questions every given arm answered are compared, so all arms are scored on the same questions."""
    arms = [a for a in ARMS if a in picks]
    ids = [str(r["id"]) for r in raws if all(str(r["id"]) in picks[a] for a in arms)
           and (dm1 is None or str(r["id"]) in dm1)]
    by_id = {str(r["id"]): r for r in raws}
    right = {a: [ft.score_item("exam", picks[a][i], key[i]) for i in ids] for a in arms}
    if dm1 is not None:
        right["dm1"] = [ft.score_item("exam", dm1[i], key[i]) for i in ids]
    luck = [chance(by_id[i]) for i in ids]
    out: Dict[str, Any] = {"n": len(ids), "chance": round(st.mean(luck), 4) if ids else None,
                           "usable": {a: round(sum(picks[a][i] is not None for i in ids) / len(ids), 4) if ids else 0.0
                                      for a in arms},
                           "means": {a: round(st.mean(v), 4) if v else None for a, v in right.items()}}
    if "blind" in right and ids:
        out["blind_above_chance"] = _boot(right["blind"], luck)
    if "dm1" in right and "relabel" in right and ids:
        out["relabel_drop"] = _boot(right["dm1"], right["relabel"])
    groups: Dict[str, Dict[str, Any]] = {}
    for gname, members in dm.EXAM_GROUPS.items():
        idx = [n for n, i in enumerate(ids) if by_id[i].get("task") in members]
        if idx:
            groups[gname] = {"n": len(idx), "chance": round(st.mean(luck[n] for n in idx), 4),
                             **{a: round(st.mean(v[n] for n in idx), 4) for a, v in right.items()}}
    out["by_group"] = groups
    out["verdict"] = verdict(out)
    return out


def verdict(result: Mapping[str, Any]) -> str:
    """The frozen reading, in plain words."""
    usable = result.get("usable") or {}
    if not usable or min(usable.values()) < dm.MIN_USABLE:
        return f"invalid: under {dm.MIN_USABLE:.0%} of answers were usable"
    signs = []
    drop, above = result.get("relabel_drop"), result.get("blind_above_chance")
    if drop and drop["gain"] >= LINES["relabel_drop"] and drop["ci95"][0] > 0:
        signs.append("it remembers answer letters (moving the letters cost it "
                     f"{drop['gain']:.0%})")
    if above and above["gain"] >= LINES["blind_above_chance"] and above["ci95"][0] > 0:
        signs.append(f"the options alone give it the answer ({above['gain']:.0%} above chance)")
    if signs:
        return "likely memorised: " + "; ".join(signs)
    clean_drop = drop is None or drop["ci95"][1] < LINES["clean_drop"]
    clean_blind = above is None or above["ci95"][1] < LINES["clean_above_chance"]
    if clean_drop and clean_blind:
        return "no sign of memory on this probe"
    return "unclear: some sign of memory, under the frozen lines"


def estimate(raws: Sequence[Mapping[str, Any]], model: str, price: Tuple[float, float]) -> Dict[str, Any]:
    import json

    chars = sum(len(json.dumps(probe_request(model, r, a)[0].payload())) for r in raws for a in ARMS)
    tokens = chars // 4
    return {"calls": len(raws) * len(ARMS), "input_tokens": tokens, "usd": round(tokens * price[0] / 1_000_000, 4)}


def run(runner: dm.DMRunner, raws: Sequence[Mapping[str, Any]], model: str, concurrency: int = 1
        ) -> Dict[str, Dict[str, Optional[str]]]:
    """Both arms for every question; arm -> question id -> original letter."""
    import asyncio

    picks: Dict[str, Dict[str, Optional[str]]] = {a: {} for a in ARMS}

    async def go() -> None:
        sem = asyncio.Semaphore(concurrency)

        async def one(raw: Mapping[str, Any]) -> None:
            async with sem:
                for arm in ARMS:
                    req, back_map = probe_request(model, raw, arm)
                    picks[arm][str(raw["id"])] = back(dm.to_answer("exam", await runner.ask(req)), back_map)

        await asyncio.gather(*(one(r) for r in raws))

    asyncio.run(go())
    return picks


def read_dm1(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[str]]:
    """question id -> the add-on's dm1 letter, from its ``dm_<name>_<source>.jsonl`` file."""
    return {str(r["id"]): (r.get("dm1") or {}).get("choice") for r in rows if r.get("kind") == "exam"}


def exam_raws(rows: Sequence[Mapping[str, Any]], ids: Optional[Sequence[str]] = None) -> List[Mapping[str, Any]]:
    keep = None if ids is None else set(ids)
    return [r for r in rows if keep is None or str(r["id"]) in keep]
