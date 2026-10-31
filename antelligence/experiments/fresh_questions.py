"""Exam questions no model can have seen: made by code at run time, with known answers (2026-10-31).

Jev got 81.7% of the public AGIEval questions right (docs/research/dm-addon-20261030/RESULTS.md). To tell reasoning
from memory, this module writes new questions of the same four kinds, with new names and numbers from a seed drawn
when the run starts, so none of them exists anywhere before the run. Every question has five options and exactly one
right answer, and that is checked by brute force over every case before the question is kept.

Kinds, matching AGIEval's split into logic and puzzles:

- ``order`` (puzzle, like LSAT analytical reasoning): six things in a row under four to six rules; "which could be
  the order" or "which must be true".
- ``assign`` (puzzle): four people, each with one colour and one day, under clues that leave one solution.
- ``arith`` (puzzle, like AQuA-RAT): a short word problem with an exact whole-number answer; the wrong options are
  the usual slips.
- ``syllogism`` (logic, like LSAT logical reasoning and LogiQA): statements about made-up kinds of thing, "which must
  also be true". An option is right only if it follows whether or not every kind is assumed to exist, so the answer
  never depends on that reading.

Questions are written in the fair test's exam format (``id``, ``task``, ``passage``, ``question``, ``options`` as
"(A)..."), so the shared-read harness and the decision-model add-on read them unchanged. Answers go to a separate file.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

LETTERS = "ABCDE"
TASKS = ("order", "assign", "arith", "syllogism")
GROUPS = {"logic": ("syllogism",), "puzzles": ("order", "assign", "arith")}
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
# capital letters that can't be mistaken for option letters A to E
THING_LETTERS = "FGHJKLMNPRSTVWXZ"
NAMES = ("Arlo", "Bex", "Cato", "Dara", "Emrys", "Fen", "Gil", "Hollis", "Ines", "Jory", "Kit", "Lio", "Mace", "Noor",
         "Odin", "Pell", "Quinn", "Rue", "Sabe", "Tove", "Uma", "Vale", "Wren", "Yael", "Zev")
COLOURS = ("red", "blue", "green", "yellow", "white", "black", "orange", "grey")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday")
NONCE = ("blickets", "daxes", "wugs", "feps", "toves", "zorbs", "quels", "miks", "nalks", "rindles", "gorps", "veems",
         "tulks", "spibs", "cranes", "hools", "jinks", "plovs", "snerds", "yarks")


@dataclass(frozen=True)
class Made:
    """One question before lettering: the texts of its five options and which one is right."""

    task: str
    passage: str
    question: str
    options: Tuple[str, ...]
    right: int
    spec: Dict[str, Any]          # what the question was built from, for checking it again


# ----------------------------------------------------------------------- order

ORDER_SCENES = (
    "A radio host plays exactly six songs, {things}, one at a time and each exactly once.",
    "Exactly six runners, {things}, finish a race one after another, with no ties.",
    "A museum holds exactly six talks, {things}, on six consecutive days, one talk a day.",
    "A chef serves exactly six dishes, {things}, one after another, each exactly once.",
    "Exactly six speakers, {things}, give their speeches one at a time, each exactly once.",
)


def order_holds(rule: Tuple[Any, ...], pos: Dict[str, int]) -> bool:
    kind = rule[0]
    if kind == "before":
        return pos[rule[1]] < pos[rule[2]]
    if kind == "next":
        return pos[rule[2]] == pos[rule[1]] + 1
    if kind == "apart":
        return abs(pos[rule[1]] - pos[rule[2]]) != 1
    if kind == "at_one":
        return pos[rule[1]] in rule[2]
    if kind == "not_at":
        return pos[rule[1]] != rule[2]
    raise ValueError(kind)


def order_text(rule: Tuple[Any, ...]) -> str:
    kind = rule[0]
    if kind == "before":
        return f"{rule[1]} comes at some time before {rule[2]}."
    if kind == "next":
        return f"{rule[1]} comes immediately before {rule[2]}."
    if kind == "apart":
        return f"{rule[1]} and {rule[2]} do not come one right after the other."
    if kind == "at_one":
        a, b = sorted(rule[2])
        return f"{rule[1]} comes either {ORDINAL[a]} or {ORDINAL[b]}."
    if kind == "not_at":
        return f"{rule[1]} does not come {ORDINAL[rule[2]]}."
    raise ValueError(kind)


def _order_rule(rng: random.Random, things: Sequence[str]) -> Tuple[Any, ...]:
    kind = rng.choice(("before", "before", "next", "apart", "at_one", "not_at"))
    x, y = rng.sample(list(things), 2)
    if kind in ("before", "next", "apart"):
        return (kind, x, y)
    if kind == "at_one":
        return (kind, x, frozenset(rng.sample(range(1, 7), 2)))
    return (kind, x, rng.randint(1, 6))


def order_solutions(things: Sequence[str], rules: Sequence[Tuple[Any, ...]]) -> List[Tuple[str, ...]]:
    out = []
    for perm in itertools.permutations(things):
        pos = {t: i + 1 for i, t in enumerate(perm)}
        if all(order_holds(r, pos) for r in rules):
            out.append(perm)
    return out


def _statement_holds(st: Tuple[Any, ...], perm: Sequence[str]) -> bool:
    pos = {t: i + 1 for i, t in enumerate(perm)}
    if st[0] == "is_at":
        return pos[st[1]] == st[2]
    return order_holds(st, pos)


def _statement_text(st: Tuple[Any, ...]) -> str:
    if st[0] == "is_at":
        return f"{st[1]} comes {ORDINAL[st[2]]}."
    return order_text(st)


def make_order(rng: random.Random) -> Made:
    things = sorted(rng.sample(THING_LETTERS, 6))
    for _ in range(500):
        rules: List[Tuple[Any, ...]] = []
        while len(rules) < rng.randint(4, 6):
            r = _order_rule(rng, things)
            if r not in rules:
                rules.append(r)
        sols = order_solutions(things, rules)
        if not 3 <= len(sols) <= 30:
            continue
        scene = rng.choice(ORDER_SCENES).format(things=", ".join(things[:-1]) + f", and {things[-1]}")
        passage = scene + " The order must meet these conditions:\n" + "\n".join(order_text(r) for r in rules)
        spec = {"things": things, "rules": [list(r) if not isinstance(r[2], frozenset) else [r[0], r[1], sorted(r[2])]
                                            for r in rules]}
        if rng.random() < 0.5:
            made = _order_could(rng, things, rules, sols, passage, spec)
        else:
            made = _order_must(rng, things, rules, sols, passage, spec)
        if made is not None:
            return made
    raise RuntimeError("could not make an order puzzle")


def _order_could(rng, things, rules, sols, passage, spec) -> Optional[Made]:
    right = rng.choice(sols)
    one_off = []
    for perm in itertools.permutations(things):
        pos = {t: i + 1 for i, t in enumerate(perm)}
        if sum(not order_holds(r, pos) for r in rules) == 1:
            one_off.append(perm)
    if len(one_off) < 4:
        return None
    wrong = rng.sample(one_off, 4)
    opts = [", ".join(p) for p in [right, *wrong]]
    return Made("order", passage, "Which one of the following could be the order, from first to sixth?",
                tuple(opts), 0, {**spec, "type": "could"})


def _order_must(rng, things, rules, sols, passage, spec) -> Optional[Made]:
    cands: List[Tuple[Any, ...]] = [("is_at", t, p) for t in things for p in range(1, 7)]
    cands += [(k, x, y) for k in ("before", "next") for x in things for y in things if x != y]
    stated = {tuple(r) for r in rules}
    must = [c for c in cands if c not in stated and all(_statement_holds(c, s) for s in sols)]
    # a right answer that only restates one rule's direct consequence is fine; one that is a rule is not
    could_not_must = [c for c in cands if c not in stated
                      and any(_statement_holds(c, s) for s in sols) and not all(_statement_holds(c, s) for s in sols)]
    if not must or len(could_not_must) < 4:
        return None
    right = rng.choice(must)
    wrong = rng.sample(could_not_must, 4)
    opts = [_statement_text(c) for c in [right, *wrong]]
    if len(set(opts)) < 5:
        return None
    return Made("order", passage, "Which one of the following must be true?", tuple(opts), 0,
                {**spec, "type": "must", "statements": [list(c) for c in [right, *wrong]]})


# ---------------------------------------------------------------------- assign

def assign_holds(clue: Tuple[Any, ...], colour: Dict[str, str], day: Dict[str, int]) -> bool:
    kind = clue[0]
    by_colour = {c: p for p, c in colour.items()}
    if kind == "has":
        return colour[clue[1]] == clue[2]
    if kind == "not_has":
        return colour[clue[1]] != clue[2]
    if kind == "on":
        return day[clue[1]] == clue[2]
    if kind == "not_on":
        return day[clue[1]] != clue[2]
    if kind == "colour_on":
        return clue[1] in by_colour and day[by_colour[clue[1]]] == clue[2]
    if kind == "colour_not_on":
        return clue[1] not in by_colour or day[by_colour[clue[1]]] != clue[2]
    if kind == "earlier":
        return day[clue[1]] < day[clue[2]]
    if kind == "after_colour":
        return clue[2] in by_colour and day[clue[1]] == day[by_colour[clue[2]]] + 1
    raise ValueError(kind)


def assign_text(clue: Tuple[Any, ...]) -> str:
    kind = clue[0]
    d = lambda i: DAYS[i]  # noqa: E731
    if kind == "has":
        return f"{clue[1]} wears {clue[2]}."
    if kind == "not_has":
        return f"{clue[1]} does not wear {clue[2]}."
    if kind == "on":
        return f"{clue[1]} goes on {d(clue[2])}."
    if kind == "not_on":
        return f"{clue[1]} does not go on {d(clue[2])}."
    if kind == "colour_on":
        return f"The person wearing {clue[1]} goes on {d(clue[2])}."
    if kind == "colour_not_on":
        return f"The person wearing {clue[1]} does not go on {d(clue[2])}."
    if kind == "earlier":
        return f"{clue[1]} goes on an earlier day than {clue[2]}."
    if kind == "after_colour":
        return f"{clue[1]} goes on the day right after the person wearing {clue[2]}."
    raise ValueError(kind)


def assign_solutions(people: Sequence[str], colours: Sequence[str], clues: Sequence[Tuple[Any, ...]]
                     ) -> List[Tuple[Dict[str, str], Dict[str, int]]]:
    out = []
    for cs in itertools.permutations(colours):
        colour = dict(zip(people, cs))
        for ds in itertools.permutations(range(4)):
            day = dict(zip(people, ds))
            if all(assign_holds(c, colour, day) for c in clues):
                out.append((colour, day))
    return out


def _assign_clue(rng: random.Random, people, colours, colour, day) -> Tuple[Any, ...]:
    """A clue true of the hidden solution."""
    p, q = rng.sample(list(people), 2)
    c = rng.choice(list(colours))
    by_colour = {v: k for k, v in colour.items()}
    kind = rng.choice(("not_has", "not_on", "colour_on", "colour_not_on", "earlier", "after_colour", "not_has"))
    if kind == "not_has":
        return (kind, p, rng.choice([x for x in colours if x != colour[p]]))
    if kind == "not_on":
        return (kind, p, rng.choice([i for i in range(4) if i != day[p]]))
    if kind == "colour_on":
        return (kind, c, day[by_colour[c]])
    if kind == "colour_not_on":
        return (kind, c, rng.choice([i for i in range(4) if i != day[by_colour[c]]]))
    if kind == "earlier":
        a, b = sorted((p, q), key=lambda x: day[x])
        return (kind, a, b)
    later = [x for x in people if day[x] == day[by_colour[c]] + 1]
    if later:
        return ("after_colour", later[0], c)
    return ("not_on", p, rng.choice([i for i in range(4) if i != day[p]]))


def make_assign(rng: random.Random) -> Made:
    for _ in range(200):
        people = sorted(rng.sample(NAMES, 4))
        colours = sorted(rng.sample(COLOURS, 4))
        colour = dict(zip(people, rng.sample(colours, 4)))
        day = dict(zip(people, rng.sample(range(4), 4)))
        clues: List[Tuple[Any, ...]] = []
        sols = assign_solutions(people, colours, clues)
        while len(sols) > 1 and len(clues) < 12:
            c = _assign_clue(rng, people, colours, colour, day)
            if c in clues:
                continue
            new = [s for s in sols if assign_holds(c, *s)]
            if len(new) < len(sols):
                clues.append(c)
                sols = new
        if len(sols) != 1 or len(clues) < 5:
            continue
        true_s = [("has", p, colour[p]) for p in people] + [("on", p, day[p]) for p in people]
        true_s = [s for s in true_s if s not in clues]
        false_s = [("has", p, c) for p in people for c in colours if c != colour[p]]
        false_s += [("on", p, i) for p in people for i in range(4) if i != day[p]]
        if not true_s or len(false_s) < 4:
            continue
        right = rng.choice(true_s)
        wrong = rng.sample(false_s, 4)
        passage = (f"Four people, {', '.join(people[:-1])}, and {people[-1]}, each wear exactly one of four colours, "
                   f"{', '.join(colours[:-1])}, and {colours[-1]}, with no two wearing the same colour. Each goes on "
                   "exactly one of four days, Monday, Tuesday, Wednesday, and Thursday, with no two on the same day. "
                   "The following are true:\n" + "\n".join(assign_text(c) for c in clues))
        return Made("assign", passage, "Which one of the following must be true?",
                    tuple(assign_text(s) for s in [right, *wrong]), 0,
                    {"people": people, "colours": colours, "clues": [list(c) for c in clues],
                     "statements": [list(s) for s in [right, *wrong]]})
    raise RuntimeError("could not make an assignment puzzle")


# ----------------------------------------------------------------------- arith

def _distinct(right: int, slips: Sequence[int], rng: random.Random) -> Optional[Tuple[int, ...]]:
    """The right answer and four different, positive wrong ones (slips first, then near misses)."""
    wrong: List[int] = []
    for s in list(slips) + [right + d for d in rng.sample([-3, -2, -1, 1, 2, 3, 4, 5, -4, 6], 10)]:
        if s > 0 and s != right and s not in wrong:
            wrong.append(int(s))
        if len(wrong) == 4:
            return (right, *wrong)
    return None


def _arith_work(rng):
    pairs = [(a, b) for a in range(3, 61) for b in range(a + 1, 91) if (a * b) % (a + b) == 0]
    a, b = rng.choice(pairs)
    t = a * b // (a + b)
    what = rng.choice(("pipe", "printer", "crew"))
    text = {"pipe": f"Pipe P alone fills a tank in {a} minutes. Pipe Q alone fills it in {b} minutes.",
            "printer": f"Printer P alone prints a batch in {a} minutes. Printer Q alone prints it in {b} minutes.",
            "crew": f"Crew P alone paints a fence in {a} hours. Crew Q alone paints it in {b} hours."}[what]
    unit = "hours" if what == "crew" else "minutes"
    q = f"Working together at the same rates, how many {unit} do P and Q take?"
    return text, q, t, [(a + b) // 2, b - a, t + 2 if (a + b) // 2 == t + 1 else t + 1]


def _arith_percent(rng):
    for _ in range(1000):
        p = rng.randrange(200, 3001, 100)
        x, y = rng.choice(range(10, 60, 5)), rng.choice(range(10, 60, 5))
        final = Fraction(p) * (100 + x) * (100 - y) / 10000
        if final.denominator == 1 and x != y:
            slip = p * (100 + x - y) // 100 if (p * (100 + x - y)) % 100 == 0 else p
            return (f"A price of {p} dollars is raised by {x} percent, and the new price is then cut by {y} percent.",
                    "What is the final price, in dollars?", int(final), [slip, p, int(final) + (p * x) // 100])
    raise RuntimeError


def _arith_speed(rng):
    pairs = [(u, v) for u in range(20, 91, 2) for v in range(u + 4, 121, 2) if (2 * u * v) % (u + v) == 0]
    u, v = rng.choice(pairs)
    d = rng.randrange(60, 301, 10)
    avg = 2 * u * v // (u + v)
    return (f"A driver goes {d} km from one town to another at {u} km per hour and comes back the same way at {v} km "
            "per hour.", "What is the average speed for the whole trip, in km per hour?", avg, [(u + v) // 2, v - u + avg // 2, avg + 2])


def _arith_mixture(rng):
    for _ in range(1000):
        x, y = rng.randrange(2, 21), rng.randrange(2, 21)
        a, b = rng.randrange(5, 41, 5), rng.randrange(45, 91, 5)
        c = Fraction(a * x + b * y, x + y)
        if c.denominator == 1 and x != y:
            return (f"{x} litres of a solution that is {a} percent salt are mixed with {y} litres of a solution that is "
                    f"{b} percent salt.", "What percent salt is the mixture?", int(c), [(a + b) // 2, b - a, int(c) + 5])
    raise RuntimeError


def _arith_ages(rng):
    b = rng.randrange(4, 30)
    d = rng.randrange(2, 25)
    n = rng.randrange(2, 15)
    s = 2 * b + d
    return (f"The ages of two sisters add up to {s} years. The older one is {d} years older than the younger one.",
            f"How old will the older sister be in {n} years?", b + d + n, [b + n, b + d, s - b + n + d])


def _arith_remainder(rng):
    for _ in range(1000):
        m1, m2 = rng.sample(range(3, 13), 2)
        if math.gcd(m1, m2) != 1:
            continue
        r1, r2 = rng.randrange(1, m1), rng.randrange(1, m2)
        n = next(k for k in range(1, m1 * m2 + 1) if k % m1 == r1 and k % m2 == r2)
        only1 = next(k for k in range(1, 400) if k % m1 == r1 and k % m2 != r2)
        only2 = next(k for k in range(1, 400) if k % m2 == r2 and k % m1 != r1)
        return (f"A number leaves a remainder of {r1} when divided by {m1}, and a remainder of {r2} when divided by "
                f"{m2}.", "What is the smallest positive number that does this?", n, [only1, only2, n + m1 * m2])
    raise RuntimeError


def _arith_compound(rng):
    for _ in range(1000):
        p = rng.randrange(1000, 20001, 500)
        r = rng.choice((4, 5, 8, 10, 12, 15, 20))
        total = Fraction(p) * (100 + r) ** 2 / 10000
        if total.denominator == 1:
            simple = p + 2 * p * r // 100
            return (f"{p} dollars is saved at {r} percent interest a year, compounded once a year.",
                    "How many dollars are there after 2 years?", int(total), [simple, p + p * r // 100, int(total) + r])
    raise RuntimeError


ARITH = (_arith_work, _arith_percent, _arith_speed, _arith_mixture, _arith_ages, _arith_remainder, _arith_compound)


def make_arith(rng: random.Random) -> Made:
    for _ in range(100):
        f = rng.choice(ARITH)
        passage, question, right, slips = f(rng)
        opts = _distinct(right, slips, rng)
        if opts is None:
            continue
        return Made("arith", passage, question, tuple(str(o) for o in opts), 0, {"template": f.__name__, "answer": right})
    raise RuntimeError("could not make a word problem")


# ------------------------------------------------------------------- syllogism
#
# A model is the set of "types" of thing that exist, a type being which of the four kinds a thing belongs to. With
# four kinds there are 16 types and 2**16 models. A statement follows when it holds in every model where the premises
# hold: once with no kind assumed to exist ("without import") and once with every kind assumed to exist.

FORMS = ("all", "no", "some", "some_not")
N_KINDS = 4
N_TYPES = 1 << N_KINDS


def _mask(pred: Callable[[int], bool]) -> int:
    return sum(1 << t for t in range(N_TYPES) if pred(t))


def _holds_mask(st: Tuple[str, int, int]) -> Tuple[str, int]:
    """(test, types mask): the statement holds in model M when ``M & mask`` is zero ('none') or non-zero ('any')."""
    form, x, y = st
    has = lambda t, k: bool(t >> k & 1)  # noqa: E731
    if form == "all":
        return "none", _mask(lambda t: has(t, x) and not has(t, y))
    if form == "no":
        return "none", _mask(lambda t: has(t, x) and has(t, y))
    if form == "some":
        return "any", _mask(lambda t: has(t, x) and has(t, y))
    if form == "some_not":
        return "any", _mask(lambda t: has(t, x) and not has(t, y))
    raise ValueError(form)


def syl_holds(st: Tuple[str, int, int], model: int) -> bool:
    test, m = _holds_mask(st)
    return (model & m == 0) if test == "none" else (model & m != 0)


_KIND_MASKS = [_mask(lambda t, k=k: bool(t >> k & 1)) for k in range(N_KINDS)]


def syl_models(premises: Sequence[Tuple[str, int, int]], import_: bool) -> List[int]:
    tests = [_holds_mask(p) for p in premises]
    out = []
    for model in range(1 << N_TYPES):
        if import_ and any(model & km == 0 for km in _KIND_MASKS):
            continue
        if all((model & m == 0) if t == "none" else (model & m != 0) for t, m in tests):
            out.append(model)
    return out


def syl_text(st: Tuple[str, int, int], words: Sequence[str]) -> str:
    form, x, y = st
    a, b = words[x], words[y]
    return {"all": f"All {a} are {b}.", "no": f"No {a} are {b}.", "some": f"Some {a} are {b}.",
            "some_not": f"Some {a} are not {b}."}[form]


FULL = (1 << N_TYPES) - 1


def syl_sat(statements: Sequence[Tuple[str, int, int]], import_: bool) -> bool:
    """Can all of these be true at once? Take every type that no "all"/"no" statement rules out: if that model fails
    a "some" statement, so does every smaller one. So this is exact, and as quick as a few bit operations."""
    forbidden = 0
    needs = []
    for st in statements:
        test, m = _holds_mask(st)
        if test == "none":
            forbidden |= m
        else:
            needs.append(m)
    allowed = FULL & ~forbidden
    if import_:
        needs += _KIND_MASKS
    return all(m & allowed for m in needs)


def _negate(st: Tuple[str, int, int]) -> Tuple[str, int, int]:
    form, x, y = st
    return ({"all": "some_not", "some_not": "all", "no": "some", "some": "no"}[form], x, y)


def syl_follows(st: Tuple[str, int, int], premises: Sequence[Tuple[str, int, int]], import_: bool) -> bool:
    return syl_sat(premises, import_) and not syl_sat(list(premises) + [_negate(st)], import_)


def make_syllogism(rng: random.Random) -> Made:
    # "No X are Y" and "Some X are Y" say the same as with X and Y swapped, so each is kept once
    statements = [(f, x, y) for f in FORMS for x in range(N_KINDS) for y in range(N_KINDS)
                  if x != y and (f in ("all", "some_not") or x < y)]
    for _ in range(2000):
        premises = rng.sample(statements, rng.randint(3, 4))
        if not (syl_sat(premises, False) and syl_sat(premises, True)):
            continue
        both = lambda s: syl_follows(s, premises, False) and syl_follows(s, premises, True)  # noqa: E731
        either = lambda s: syl_follows(s, premises, False) or syl_follows(s, premises, True)  # noqa: E731
        # right: follows on both readings, isn't a premise, and needs two premises together
        right = [s for s in statements if s not in premises and both(s)
                 and not any(syl_follows(s, [p], False) or syl_follows(s, [p], True) for p in premises)]
        # wrong: follows on neither reading; most of them could still be true
        neither = [s for s in statements if s not in premises and not either(s)]
        consistent = [s for s in neither if syl_sat(list(premises) + [s], True)]
        if not right or len(consistent) < 3 or len(neither) < 4:
            continue
        wrong = rng.sample(consistent, 3)
        wrong.append(rng.choice([s for s in neither if s not in wrong]))
        words = rng.sample(NONCE, N_KINDS)
        r = rng.choice(right)
        passage = "In a certain collection of things, all of the following are true:\n" + "\n".join(
            syl_text(p, words) for p in premises)
        return Made("syllogism", passage, "If the statements above are true, which one of the following must also be "
                    "true?", tuple(syl_text(s, words) for s in [r, *wrong]), 0,
                    {"words": words, "premises": [list(p) for p in premises],
                     "statements": [list(s) for s in [r, *wrong]]})
    raise RuntimeError("could not make a syllogism")


MAKERS: Dict[str, Callable[[random.Random], Made]] = {
    "order": make_order, "assign": make_assign, "arith": make_arith, "syllogism": make_syllogism}


# ---------------------------------------------------------------------- output

def letter(made: Made, rng: random.Random) -> Tuple[List[str], str]:
    """The options in a random order, labelled "(A)" to "(E)", and the right letter."""
    order = list(range(len(made.options)))
    rng.shuffle(order)
    opts = [f"({LETTERS[i]}){made.options[j]}" for i, j in enumerate(order)]
    return opts, LETTERS[order.index(made.right)]


def make_set(seed: int, per_task: int = 40, tasks: Sequence[str] = TASKS
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, Any]]]:
    """(questions, answers, specs). Questions alternate between the tasks; no two have the same text."""
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
            made = MAKERS[task](rng)
            if (made.passage, made.question) in seen:
                continue
            seen.add((made.passage, made.question))
            counts[task] += 1
            qid = f"fresh-{task}-{counts[task]:03d}"
            opts, right = letter(made, rng)
            qs.append({"id": qid, "task": task, "passage": made.passage, "question": made.question, "options": opts})
            ans.append({"id": qid, "label": right})
            specs.append({"id": qid, "task": task, "label": right, "options": opts, **made.spec})
    return qs, ans, specs


# ----------------------------------------------------------------- re-checking

def check(q: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    """Solve a question again from its spec by brute force: exactly the labelled option is right."""
    texts = [o[3:] for o in q["options"]]
    right = [i for i, t in enumerate(texts) if _is_right(spec, t)]
    return len(right) == 1 and LETTERS[right[0]] == spec["label"]


def _is_right(spec: Mapping[str, Any], text: str) -> bool:
    task = spec["task"]
    if task == "arith":
        return int(text) == spec["answer"]
    if task == "order":
        rules = [tuple(r[:2]) + ((frozenset(r[2]),) if r[0] == "at_one" else (r[2],)) for r in spec["rules"]]
        sols = order_solutions(spec["things"], rules)
        if spec["type"] == "could":
            return tuple(text.split(", ")) in sols
        st = next(tuple(s) for s in spec["statements"] if _statement_text(tuple(s)) == text)
        return all(_statement_holds(st, s) for s in sols)
    if task == "assign":
        clues = [tuple(c) for c in spec["clues"]]
        sols = assign_solutions(spec["people"], spec["colours"], clues)
        st = next(tuple(s) for s in spec["statements"] if assign_text(tuple(s)) == text)
        return all(assign_holds(st, *s) for s in sols)
    if task == "syllogism":
        prem = [tuple(p) for p in spec["premises"]]
        st = next(tuple(s) for s in spec["statements"] if syl_text(tuple(s), spec["words"]) == text)
        # brute force over every model, not the quick test the maker used
        return all(all(syl_holds(st, m) for m in syl_models(prem, imp)) for imp in (False, True))
    raise ValueError(task)

