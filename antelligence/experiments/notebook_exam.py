"""Field-notebook ants on exam questions: read who said what, count each source once, then answer.

Why: the 2026-10-31 dream run's "field notebook" (docs/research/swarm-scale-20261029/NOTEBOOK.md, draft PR 24) gave
each ant in the scripted rumour world a notebook of what every other source had said, kept once per source. It beat
the best setup by about 8 to 10 points there. On the exam swarm, the reader's notes lift the ants by about 10 points
(docs/research/fresh-20261031/RESULTS.md), but a reader who is wrong can pull every ant with it. The question here:
if each ant also reads what its teammates found working *alone*, one line per teammate, does it stop following the
reader's mistakes?

This is a second round on top of a finished shared-read run, so the first round's answers are reused, not asked
again. Per question, each of the 8 ants (4 models x 2 ways of looking) answers again in two new arms:

* ``nb``: the full question, plus the notebook.
* ``nb_notes``: the full question, the reader's notes, plus the notebook (``both`` plus the notebook).

The notebook lists every *other* model once, with both of its first-round tries working alone (answer and a short
reason), then the ant's own two tries alone. A model that tried twice is still one source, which is the notebook's
"count each source once". The reader's own answer is never shown.

Read against the first round on the same questions: ``nb`` against ``alone``, ``nb_notes`` against ``both``, both
against the reader alone, and how often ants pick the reader's letter on questions the reader gets wrong.

Everything here is pure, like ``shared_read``: calls go through ``shared_read.Runner``.
"""

from __future__ import annotations

import asyncio
import random
import statistics as st
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import fair_test as ft
from antelligence.experiments import shared_read as sr

ARMS = ("nb", "nb_notes")
REASON_WORDS = 30
INTRO = ("Field notebook. Each teammate below worked on this question on its own, without notes. Each teammate is "
         "listed once, however many tries it made.")
ADVICE = ("Each teammate is one source. Agreement only counts if the reasons hold up, and a mistake that several "
          "make is still a mistake. Work the question yourself, then answer.")
# Frozen reading (docs/plans/2026-10-31-night-run.md): pooled over every fresh set the night runs.
LINES = {"helps": 0.03, "copy_drop": 0.05}
TRIES = ("first try", "second try")


def short(reason: str, words: int = REASON_WORDS) -> str:
    w = " ".join(str(reason or "").split()).split(" ")
    text = " ".join(w[:words])
    return (text + " ...") if len(w) > words else text


def first_round(items: Sequence[sr.Item], rows: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Mapping[str, Any]],
                roster: Sequence[Tuple[sr.ModelSpec, int]], parser: int) -> Dict[str, List[Tuple[Optional[str], str]]]:
    """Per question, each ant's first-round ``alone`` answer (from the saved rows) and its reason (from the reply
    cache, found by rebuilding the exact request). A reason is used only when the cached reply reads as the same
    answer, so a cache from another run can't slip in."""
    out: Dict[str, List[Tuple[Optional[str], str]]] = {}
    for item in items:
        row = rows.get(item.id)
        if row is None:
            continue
        got = []
        for col, (spec, lens) in enumerate(roster):
            ans = row["alone"][col]
            req = sr.make_request(spec, sr.ant_messages(item, "alone", lens, spec), item.id, f"alone:{spec.tag}", lens,
                                  sr.ANT_MAX_TOKENS)
            hit = cache.get(req.request_hash)
            reason = ""
            if hit is not None and not hit.get("error"):
                a, r = sr.parse_answer(hit.get("content") or "", parser)
                if a == ans:
                    reason = r
            got.append((ans, reason))
        out[item.id] = got
    return out


def _tries(entries: Sequence[Tuple[Optional[str], str]]) -> str:
    parts = []
    for name, (ans, reason) in zip(TRIES, entries):
        if ans is None:
            parts.append(f"{name}: no answer")
        else:
            why = short(reason)
            parts.append(f"{name} {ans}" + (f' ("{why}")' if why else ""))
    return "; ".join(parts)


def notebook(first: Sequence[Tuple[Optional[str], str]], roster: Sequence[Tuple[sr.ModelSpec, int]],
             me: sr.ModelSpec) -> str:
    """The notebook one ant reads: every other model once (its tries alone), then the ant's own tries."""
    by_model: Dict[str, List[Tuple[Optional[str], str]]] = {}
    family: Dict[str, str] = {}
    for (spec, _), entry in zip(roster, first):
        by_model.setdefault(spec.tag, []).append(entry)
        family[spec.tag] = spec.family
    lines = [INTRO]
    for k, (tag, entries) in enumerate((t, e) for t, e in by_model.items() if t != me.tag):
        lines.append(f"- Teammate {k + 1} ({family[tag]}): {_tries(entries)}")
    lines.append(f"Your own work alone: {_tries(by_model.get(me.tag, []))}")
    lines.append(ADVICE)
    return "\n".join(lines)


def messages(item: sr.Item, arm: str, lens: int, spec: sr.ModelSpec, book: str, notes: Optional[str] = None
             ) -> Tuple[Dict[str, str], ...]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if arm == "nb_notes" and not notes:
        raise ValueError("nb_notes needs the reader's notes")
    body = item.full
    if arm == "nb_notes":
        body += f"\n\nNotes from a reader who read this question:\n{notes}"
    tail = f"How to think about it: {ft.LENSES['exam'][lens]}\n\n{ft._format('exam')}"
    return ({"role": "system", "content": ft.SYSTEM},
            {"role": "user", "content": sr._user(spec, f"{body}\n\n{book}\n\n{tail}")})


def request(item: sr.Item, arm: str, lens: int, spec: sr.ModelSpec, book: str, notes: Optional[str] = None):
    return sr.make_request(spec, messages(item, arm, lens, spec, book, notes), item.id, f"{arm}:{spec.tag}", lens,
                           sr.ANT_MAX_TOKENS)


async def run(runner: sr.Runner, items: Sequence[sr.Item], rows: Mapping[str, Mapping[str, Any]],
              first: Mapping[str, Sequence[Tuple[Optional[str], str]]], ants: Sequence[sr.ModelSpec],
              lenses: Sequence[int] = sr.LENS_IDS, on_model: Optional[Callable[[sr.ModelSpec], None]] = None,
              after_model: Optional[Callable[[sr.ModelSpec], None]] = None, parser: int = sr.PARSER
              ) -> Dict[str, Dict[str, List[Optional[str]]]]:
    """One ant model at a time, every ant answers every question in both notebook arms. A question with no notes
    gets no ``nb_notes`` answers (None), which the usable-share check counts."""
    roster = sr.ant_roster(ants, lenses)
    usable = partial(sr._usable_answer, parser=parser)
    out = {i.id: {arm: [None] * len(roster) for arm in ARMS} for i in items if i.id in first}
    for spec in ants:
        if on_model:
            on_model(spec)
        jobs = []
        for col, (s, lens) in enumerate(roster):
            if s is not spec:
                continue
            for item in items:
                if item.id not in first:
                    continue
                book = notebook(first[item.id], roster, spec)
                notes = (rows[item.id].get("reader_notes") or "") if item.id in rows else ""
                for arm in ARMS:
                    if arm == "nb_notes" and not notes:
                        continue
                    jobs.append((item.id, arm, col, request(item, arm, lens, spec, book, notes or None)))

        async def one(job):
            qid, arm, col, req = job
            out[qid][arm][col] = sr._answer(await runner.ask(req, usable), parser)

        await asyncio.gather(*(one(j) for j in jobs))
        if after_model:
            after_model(spec)
    return out


def merge(rows: Sequence[Mapping[str, Any]], nb: Mapping[str, Mapping[str, List[Optional[str]]]]
          ) -> List[Dict[str, Any]]:
    """The first-round rows with the two notebook arms added (rows without notebook answers are left out)."""
    return [{**r, **nb[r["id"]]} for r in rows if r["id"] in nb]


# ---------------------------------------------------------------------- scoring

ALL_ARMS = ("alone", "notes", "both", "nb", "nb_notes")


def _right(a: Optional[str], t: str) -> float:
    return float(a is not None and a == t)


def copy_rate(answers: Sequence[Optional[str]], reader: Optional[str]) -> Optional[float]:
    """Share of an item's usable ant answers that pick the reader's letter (None if none are usable)."""
    usable = [a for a in answers if a is not None]
    return sum(a == reader for a in usable) / len(usable) if usable else None


def _boot(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Any]:
    g, lo, hi = ft.paired_boot([a - b for a, b in zip(xs, ys)])
    return {"gain": round(g, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(xs)}


def _alike_boot(rs: Sequence[Mapping[str, Any]], truths: Sequence[str], a: str, b: str, reps: int = 2000,
                seed: int = 20261031) -> Dict[str, Any]:
    rng = random.Random(seed)

    def rise(idx: Sequence[int]) -> float:
        t = [truths[i] for i in idx]
        return sr.error_alikeness([rs[i][a] for i in idx], t) - sr.error_alikeness([rs[i][b] for i in idx], t)

    point = rise(range(len(rs)))
    draws = sorted(rise([rng.randrange(len(rs)) for _ in rs]) for _ in range(reps))
    return {"rise": round(point, 4), "ci95": [round(draws[int(0.025 * reps)], 4), round(draws[int(0.975 * reps) - 1], 4)]}


def usable_share(rs: Sequence[Mapping[str, Any]]) -> float:
    vals = [a for r in rs for arm in ARMS for a in r[arm]]
    return sum(a is not None for a in vals) / len(vals) if vals else 0.0


def verdict(out: Mapping[str, Any]) -> Dict[str, str]:
    if out["usable_share"] < sr.MIN_USABLE:
        return {"invalid": f"only {out['usable_share']:.0%} of the notebook answers were usable "
                           f"(at least {sr.MIN_USABLE:.0%} needed)"}
    h = out["nb_notes_vs_both"]
    c = out["copy_drop_vs_both"]
    helps = ("yes" if h["gain"] >= LINES["helps"] and h["ci95"][0] > 0 else
             "no, it hurts" if h["ci95"][1] < 0 else "no clear change")
    stops = ("yes" if c and c["gain"] >= LINES["copy_drop"] and c["ci95"][0] > 0 else
             "no, they copy more" if c and c["ci95"][1] < 0 else "no clear change")
    beats = out["best_vs_reader"]
    return {"notebook_helps": helps, "stops_copying_reader": stops,
            "notebook_beats_reader": "yes" if beats["ci95"][0] > 0 else "no"}


def score(rs: Sequence[Mapping[str, Any]], key: Mapping[str, str], *, boot_reps: int = 2000) -> Dict[str, Any]:
    """Every arm (first round and notebook), the comparisons, how often ants copy a wrong reader, and the verdict.
    ``rs`` are :func:`merge`'d rows; ids must be unique (prefix them with the set name when pooling sets)."""
    rs = [r for r in rs if r["id"] in key]
    if not rs:
        raise ValueError("no scored rows")
    truths = [key[r["id"]] for r in rs]
    vote = {arm: [_right(ft.vote(r[arm]), t) for r, t in zip(rs, truths)] for arm in ALL_ARMS}
    reader = [_right(r["reader"], t) for r, t in zip(rs, truths)]
    arms = {}
    for arm in ALL_ARMS:
        lists = [r[arm] for r in rs]
        arms[arm] = {"vote": round(st.mean(vote[arm]), 4),
                     "ant_mean": round(st.mean(_right(a, t) for lst, t in zip(lists, truths) for a in lst), 4),
                     "error_alikeness": round(sr.error_alikeness(lists, truths), 4),
                     "same_wrong": round(sr.same_wrong(lists, truths), 4)}
    wrong_reader = [k for k, (r, t) in enumerate(zip(rs, truths)) if r["reader"] is not None and r["reader"] != t]
    copy: Dict[str, List[Optional[float]]] = {arm: [copy_rate(rs[k][arm], rs[k]["reader"]) for k in wrong_reader]
                                              for arm in ALL_ARMS}
    for arm in ALL_ARMS:
        vals = [v for v in copy[arm] if v is not None]
        arms[arm]["copies_wrong_reader"] = round(st.mean(vals), 4) if vals else None

    def drop(a: str, b: str) -> Optional[Dict[str, Any]]:
        pairs = [(x, y) for x, y in zip(copy[a], copy[b]) if x is not None and y is not None]
        return _boot([x for x, _ in pairs], [y for _, y in pairs]) if pairs else None

    best = max(ARMS, key=lambda a: st.mean(vote[a]))
    out: Dict[str, Any] = {
        "n": len(rs), "usable_share": round(usable_share(rs), 4), "reader_alone": round(st.mean(reader), 4),
        "reader_wrong_items": len(wrong_reader), "arms": arms,
        "nb_notes_vs_both": _boot(vote["nb_notes"], vote["both"]),
        "nb_vs_alone": _boot(vote["nb"], vote["alone"]),
        "nb_notes_vs_reader": _boot(vote["nb_notes"], reader),
        "nb_vs_reader": _boot(vote["nb"], reader),
        "best_arm": best, "best_vs_reader": _boot(vote[best], reader),
        # positive = the notebook arm copies the wrong reader less than the first-round arm
        "copy_drop_vs_both": drop("both", "nb_notes"), "copy_drop_vs_notes": drop("notes", "nb_notes"),
        "alikeness_rise_nb_notes_vs_both": _alike_boot(rs, truths, "nb_notes", "both", reps=boot_reps),
    }
    groups: Dict[str, List[int]] = {}
    for k, r in enumerate(rs):
        groups.setdefault(r.get("group") or "", []).append(k)
    out["by_task"] = {g: {"n": len(idx), "reader": round(st.mean(reader[k] for k in idx), 4),
                          **{arm: round(st.mean(vote[arm][k] for k in idx), 4) for arm in ALL_ARMS}}
                      for g, idx in sorted(groups.items())}
    out["verdict"] = verdict(out)
    return out


def plan(n_items: int, n_ants: int) -> Dict[str, int]:
    return {"items": n_items, "calls": n_items * n_ants * len(ARMS)}


__all__ = ["ARMS", "LINES", "copy_rate", "first_round", "merge", "messages", "notebook", "plan", "request", "run",
           "score", "short"]
