"""Dev puzzles for the harness, with a syllogism maker that is harder to shortcut.

Why this exists: the fresh-question syllogism maker (:func:`antelligence.experiments.fresh_questions.make_syllogism`)
picks its wrong options at random from the statements that don't follow. The right answer then looks different from
the wrong ones in ways that need no logic: it is "Some ... are not ..." 56% of the time against 35% for a wrong one,
and its two words sit in the premises in telltale places. A lookup over such surface features, learned on 1,000
puzzles and tried on 1,000 others, picks the right option 63% of the time (chance is 20%), with no logic at all.
So a high score on those puzzles may be pattern matching. See :func:`shortcut_report`.

:func:`make_syllogism` here builds the same kind of puzzle, from the same premises and checks, but picks wrong options
that look like the right one: the same form ("All", "No", "Some", "Some ... not") and the same place for their words
in the premises, as far as the premises allow. ``fresh_questions.py`` is left exactly as it was, because the fresh
test's preregistration pins its hash.
"""

from __future__ import annotations

import collections
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import fresh_questions as fq

Statement = Tuple[str, int, int]


def surface(s: Statement, premises: Sequence[Statement]) -> Tuple[Any, ...]:
    """What a reader could see about an option without doing any logic: its form, whether its two words are paired
    in a premise, and how often each word is a premise's subject or predicate."""
    form, x, y = s
    subj = collections.Counter(p[1] for p in premises)
    pred = collections.Counter(p[2] for p in premises)
    paired = frozenset((x, y)) in {frozenset(p[1:]) for p in premises}
    return form, paired, min(subj[x], 2), min(pred[x], 2), min(subj[y], 2), min(pred[y], 2)


def _distance(a: Tuple[Any, ...], b: Tuple[Any, ...]) -> int:
    return 4 * (a[0] != b[0]) + 2 * (a[1] != b[1]) + sum(abs(u - v) for u, v in zip(a[2:], b[2:]))


def reworded_text(st: Statement, words: Sequence[str]) -> str:
    """The same four statement forms, written another way, so a model can't lean on the fresh-question wording."""
    form, x, y = st
    a, b = words[x], words[y]
    return {"all": f"Each one of the {a} is one of the {b}.", "no": f"Not a single one of the {a} is one of the {b}.",
            "some": f"At least one of the {a} is also one of the {b}.",
            "some_not": f"At least one of the {a} is not one of the {b}."}[form]


REWORDED_PASSAGE = "Here is everything that is known about a box of objects:\n"
REWORDED_QUESTION = "Going only by these facts, which one of the following has to be the case?"


def make_reworded_syllogism(rng: random.Random) -> fq.Made:
    """A balanced syllogism (see :func:`make_syllogism`) in different words: other phrasings of "all", "no", "some"
    and "some ... not", another opening line and another question."""
    m = make_syllogism(rng)
    words = m.spec["words"]
    prem = [tuple(p) for p in m.spec["premises"]]
    sts = [tuple(x) for x in m.spec["statements"]]
    passage = REWORDED_PASSAGE + "\n".join(reworded_text(p, words) for p in prem)
    return fq.Made("syllogism", passage, REWORDED_QUESTION, tuple(reworded_text(x, words) for x in sts), 0,
                   {**m.spec, "maker": "balanced", "wording": "reworded"})


def check_syllogism(q: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    """Re-solve a syllogism from its spec by brute force, in either wording: exactly the labelled option follows
    from the premises with and without assuming every kind exists."""
    text = reworded_text if spec.get("wording") == "reworded" else fq.syl_text
    prem = [tuple(p) for p in spec["premises"]]
    by_text = {text(tuple(x), spec["words"]): tuple(x) for x in spec["statements"]}
    right = [i for i, o in enumerate(q["options"])
             if fq.syl_follows(by_text[o[3:]], prem, False) and fq.syl_follows(by_text[o[3:]], prem, True)]
    return len(right) == 1 and fq.LETTERS[right[0]] == spec["label"]


def make_syllogism(rng: random.Random, max_distance: int = 1) -> fq.Made:
    """A syllogism whose four wrong options each match the right one's form and pairing and differ in at most
    ``max_distance`` word-place counts. Wrong options follow on neither reading (with or without assuming every
    kind exists); at most one contradicts the premises, as before."""
    statements = [(f, x, y) for f in fq.FORMS for x in range(fq.N_KINDS) for y in range(fq.N_KINDS)
                  if x != y and (f in ("all", "some_not") or x < y)]
    for _ in range(20000):
        premises = rng.sample(statements, rng.randint(3, 4))
        if not (fq.syl_sat(premises, False) and fq.syl_sat(premises, True)):
            continue

        def both(s: Statement) -> bool:
            return fq.syl_follows(s, premises, False) and fq.syl_follows(s, premises, True)

        def either(s: Statement) -> bool:
            return fq.syl_follows(s, premises, False) or fq.syl_follows(s, premises, True)

        right = [s for s in statements if s not in premises and both(s)
                 and not any(fq.syl_follows(s, [p], False) or fq.syl_follows(s, [p], True) for p in premises)]
        if not right:
            continue
        r = rng.choice(right)
        want = surface(r, premises)
        neither = [s for s in statements if s not in premises and not either(s)]
        close = [s for s in neither if _distance(surface(s, premises), want) <= max_distance]
        consistent = [s for s in close if fq.syl_sat(list(premises) + [s], True)]
        clash = [s for s in close if s not in consistent]
        if len(consistent) < 3 or len(consistent) + len(clash) < 4:
            continue
        wrong = rng.sample(consistent, 3)
        rest = [s for s in consistent + clash if s not in wrong]
        wrong.append(rng.choice(rest))
        words = rng.sample(fq.NONCE, fq.N_KINDS)
        passage = "In a certain collection of things, all of the following are true:\n" + "\n".join(
            fq.syl_text(p, words) for p in premises)
        return fq.Made("syllogism", passage, "If the statements above are true, which one of the following must "
                       "also be true?", tuple(fq.syl_text(s, words) for s in [r, *wrong]), 0,
                       {"words": words, "premises": [list(p) for p in premises],
                        "statements": [list(s) for s in [r, *wrong]], "maker": "balanced"})
    raise RuntimeError("could not make a balanced syllogism")


MAKERS: Dict[str, Callable[[random.Random], fq.Made]] = {**fq.MAKERS, "syllogism": make_syllogism}


def make_set(seed: int, per_task: int = 40, tasks: Sequence[str] = fq.TASKS,
             makers: Optional[Mapping[str, Callable[[random.Random], fq.Made]]] = None
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, Any]]]:
    """Like :func:`fresh_questions.make_set` (same ids, lettering and specs, so ``fresh_questions.check`` re-solves
    every puzzle), with the balanced syllogism maker."""
    makers = dict(makers or MAKERS)
    rng = random.Random(seed)
    qs: List[Dict[str, Any]] = []
    ans: List[Dict[str, str]] = []
    specs: List[Dict[str, Any]] = []
    seen = set()
    counts = {t: 0 for t in tasks}
    while min(counts.values()) < per_task:
        for task in tasks:
            if counts[task] >= per_task:
                continue
            made = makers[task](rng)
            if (made.passage, made.question) in seen:
                continue
            seen.add((made.passage, made.question))
            counts[task] += 1
            qid = f"fresh-{task}-{counts[task]:03d}"
            opts, right = fq.letter(made, rng)
            qs.append({"id": qid, "task": task, "passage": made.passage, "question": made.question, "options": opts})
            ans.append({"id": qid, "label": right})
            specs.append({"id": qid, "task": task, "label": right, "options": opts, **made.spec})
    return qs, ans, specs


# ---------------------------------------------------------------- the shortcut test

def _items(maker: Callable[[random.Random], fq.Made], n: int, seed: int
           ) -> List[Tuple[List[Statement], List[Statement]]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        m = maker(rng)
        out.append(([tuple(s) for s in m.spec["statements"]], [tuple(p) for p in m.spec["premises"]]))
    return out


def _accuracy(items: Sequence[Tuple[List[Statement], List[Statement]]],
              score: Callable[[Statement, List[Statement]], float]) -> float:
    """Share of puzzles where the top-scored option is the right one (always the first in the spec); ties split."""
    total = 0.0
    for sts, prem in items:
        vals = [score(s, prem) for s in sts]
        top = [i for i, v in enumerate(vals) if v == max(vals)]
        total += (0 in top) / len(top)
    return total / len(items)


def shortcut_report(maker: Callable[[random.Random], fq.Made], n: int = 1000, seed: int = 7) -> Dict[str, float]:
    """How well no-logic shortcuts pick the right option: learned on ``n`` puzzles, scored on ``n`` others.

    * ``form_prior``: the options alone, by how often each form is the right answer;
    * ``surface_lookup``: a lookup over :func:`surface` plus which forms the premises use;
    * ``chance``: one in five.
    """
    train, test = _items(maker, n, seed), _items(maker, n, seed + 1)
    right_f: collections.Counter = collections.Counter()
    all_f: collections.Counter = collections.Counter()
    hit: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()

    def key(s: Statement, prem: Sequence[Statement]) -> Tuple[Any, ...]:
        return surface(s, prem) + (tuple(sorted(p[0] for p in prem)),)

    for sts, prem in train:
        for i, s in enumerate(sts):
            all_f[s[0]] += 1
            right_f[s[0]] += i == 0
            k = key(s, prem)
            seen[k] += 1
            hit[k] += i == 0
    return {"chance": 0.2,
            "form_prior": round(_accuracy(test, lambda s, p: right_f[s[0]] / max(1, all_f[s[0]])), 3),
            "surface_lookup": round(_accuracy(test, lambda s, p: (hit[key(s, p)] + 0.2) / (seen[key(s, p)] + 1)), 3)}
