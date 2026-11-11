"""The fifth switch: shorter fixed instructions for the ants, found by a small GEPA-style search.

GEPA (Agrawal et al., 2025, "Reflective Prompt Evolution Can Outperform Reinforcement Learning"; Hermes Agent's
self-evolution repo uses it) improves a prompt by (1) running it on a few training questions, (2) showing a model the
prompt with where it went wrong and asking for a better one ("reflective mutation"), and (3) keeping every prompt
that no other prompt beats on all counts (a Pareto set), mutating the best of them next.

Here the counts are accuracy and tokens, and only the ants' fixed text changes: the system line and the reply form
(how long a reason to write). The question, the way-of-looking line (each ant's lens) and the reply's JSON shape stay,
so answers are read the same way. The search runs on training questions only; the pick is the shortest prompt on the
Pareto set whose accuracy is within ``tolerance`` of the original's there. Whether it holds is then checked once on
the test set, every ant, against the original prompt (the script does that).

Everything runs on the local models: it costs nothing but time.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import fair_test as ft
from antelligence.experiments import shared_read as sr
from antelligence.harness.borrowed import custom_ants
from antelligence.harness.core import Task
from antelligence.harness.recipes import Ctx
from antelligence.harness.tools import LiveBox, ant_tool
from antelligence.providers.base import ChatRequest


@dataclass(frozen=True)
class Prompt:
    system: str
    fmt: str
    origin: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return self.system, self.fmt


ORIGINAL = Prompt(ft.SYSTEM, ft._format("exam"), "original")
SEEDS = (
    Prompt("Answer on your own. Reply with one JSON object only.",
           'Reply as JSON: {"reason": "<at most 25 words>", "answer": "<one letter>"}', "seed: 25-word reason"),
    Prompt("Answer on your own. Reply with one JSON object only.",
           'Reply as JSON: {"reason": "<at most 12 words>", "answer": "<one letter>"}', "seed: 12-word reason"),
    Prompt("Answer on your own. Reply with one JSON object only.",
           'Reply as JSON: {"answer": "<one letter>"}', "seed: no reason"),
)
MAX_LEN = 300


def valid(p: Prompt) -> bool:
    """A prompt is usable if it is short, still asks for JSON with an "answer" key holding one letter, and names no
    option letter (so it can't steer the answer)."""
    f = p.fmt
    return (0 < len(p.system) <= MAX_LEN and 0 < len(f) <= MAX_LEN and '"answer"' in f and "letter" in f.lower()
            and "json" in f.lower() and not re.search(r'"answer"\s*:\s*"[A-E]"', f))


@dataclass
class Score:
    prompt: Prompt
    accuracy: float
    tokens: float                 # per ant call, in plus out
    tokens_out: float
    n: int
    misses: List[Tuple[str, str, str]]   # (question start, what was answered, the right letter)

    def to_json(self) -> Dict[str, Any]:
        return {"system": self.prompt.system, "format": self.prompt.fmt, "origin": self.prompt.origin,
                "accuracy": round(self.accuracy, 4), "tokens": round(self.tokens, 1),
                "tokens_out": round(self.tokens_out, 1), "n": self.n}


MakeBox = Callable[[Optional[Callable[..., Any]]], LiveBox]


async def score_many(make_box: MakeBox, prompts: Sequence[Prompt], tasks: Sequence[Task], key: Mapping[str, str],
                     ants: Sequence[int]) -> Dict[Tuple[str, str], Score]:
    """Each prompt's score over ``ants`` (each alone) on ``tasks``. One ant model at a time, every prompt on it, so a
    one-model-at-a-time server loads each model once per call of this, not once per prompt."""
    boxes = {p.key: make_box(None if p.key == ORIGINAL.key else custom_ants(p.system, p.fmt)) for p in prompts}
    tally: Dict[Tuple[str, str], List[Any]] = {p.key: [0, 0, 0, 0, []] for p in prompts}   # right, n, in, out, misses
    for i in ants:
        for p in prompts:
            got = await asyncio.gather(*(Ctx(boxes[p.key], t).call(ant_tool(i, "alone")) for t in tasks))
            row = tally[p.key]
            for t, s in zip(tasks, got):
                row[1] += 1
                row[2] += s.tokens_in
                row[3] += s.tokens_out
                if s.answer == key.get(t.id):
                    row[0] += 1
                elif len(row[4]) < 12:
                    row[4].append((" ".join(t.stem.split())[:220], s.answer or "nothing usable", key.get(t.id, "?")))
    return {p.key: Score(p, r / max(1, n), (tin + tout) / max(1, n), tout / max(1, n), n, misses)
            for p in prompts for r, n, tin, tout, misses in [tally[p.key]]}


async def score(make_box: MakeBox, p: Prompt, tasks: Sequence[Task], key: Mapping[str, str], ants: Sequence[int]
                ) -> Score:
    return (await score_many(make_box, [p], tasks, key, ants))[p.key]


REFLECT_SYSTEM = "You improve the instructions given to small AI models that answer multiple-choice questions."
REFLECT_ASK = """Small models are shown a question, then one line on how to think about it, then these instructions.

SYSTEM LINE: {system}
REPLY FORM: {fmt}

With these instructions they were right on {acc} of the answers and used {tokens:.0f} tokens per answer ({out:.0f} of \
them written). Some answers they got wrong:
{misses}

Write new instructions that use fewer tokens without losing accuracy: a shorter system line, and a reply form that \
asks for a shorter reason, or none if a reason does not help. The reply form must still ask for one JSON object with \
an "answer" key holding one option letter. Do not mention any particular option letter. Reply with only this JSON:
{{"system": "<the new system line>", "format": "<the new reply form>"}}"""
_OBJ = re.compile(r"\{.*\}", re.S)


def parse_prompt(text: str, origin: str) -> Optional[Prompt]:
    text = sr.strip_think(text or "")
    for m in reversed(_OBJ.findall(text)):
        try:
            d = json.loads(m)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("system"), str) and isinstance(d.get("format"), str):
            p = Prompt(" ".join(d["system"].split()), " ".join(d["format"].split()), origin)
            return p if valid(p) else None
    return None


async def reflect(runner: sr.Runner, reader: sr.ModelSpec, parent: Score, round_no: int, j: int) -> Optional[Prompt]:
    misses = "\n".join(f"- {q} ... they answered {a}; the right answer was {r}" for q, a, r in parent.misses[:4]) \
        or "- (none in this sample)"
    ask = REFLECT_ASK.format(system=parent.prompt.system, fmt=parent.prompt.fmt, acc=f"{100 * parent.accuracy:.0f}%",
                             tokens=parent.tokens, out=parent.tokens_out, misses=misses)
    req = ChatRequest(model=reader.tag, messages=({"role": "system", "content": REFLECT_SYSTEM},
                                                  {"role": "user", "content": sr._user(reader, ask)}),
                      max_tokens=500, temperature=0.8, seed=sr.seed(f"promptopt:{parent.prompt.key}", "reflect",
                                                                    round_no * 10 + j),
                      extra_body=reader.extra_body)
    row = await runner.ask(req, lambda text: parse_prompt(text, "") is not None)
    return parse_prompt(row.get("content") or "", f"reflection {round_no}.{j} on: {parent.prompt.origin}")


def frontier(scores: Sequence[Score]) -> List[Score]:
    """Prompts no other prompt beats on both accuracy (higher) and tokens (lower)."""
    out = []
    for s in scores:
        if not any(o.accuracy >= s.accuracy and o.tokens <= s.tokens and (o.accuracy, o.tokens) != (s.accuracy, s.tokens)
                   for o in scores):
            out.append(s)
    return sorted(out, key=lambda s: s.tokens)


def value(s: Score, base: Score, w_tokens: float) -> float:
    return s.accuracy - w_tokens * s.tokens / max(1.0, base.tokens)


async def search(make_box: MakeBox, runner: sr.Runner, reader: sr.ModelSpec, tasks: Sequence[Task],
                 key: Mapping[str, str], ants: Sequence[int], *, rounds: int = 3, per_round: int = 2,
                 w_tokens: float = 0.05, tolerance: float = 0.01, log: Callable[[str], None] = lambda s: None
                 ) -> Dict[str, Any]:
    seen: Dict[Tuple[str, str], Score] = {}

    async def add(ps: Sequence[Optional[Prompt]]) -> None:
        new: List[Prompt] = []
        for p in ps:
            if p is not None and p.key not in seen and p.key not in {q.key for q in new}:
                new.append(p)
        if not new:
            return
        got = await score_many(make_box, new, tasks, key, ants)
        for p in new:
            s = seen[p.key] = got[p.key]
            log(f"  [{p.origin}] {100 * s.accuracy:.1f}% right, {s.tokens:.0f} tokens an answer ({s.tokens_out:.0f} "
                f"written)")
    await add([ORIGINAL, *SEEDS])
    base = seen[ORIGINAL.key]
    for r in range(1, rounds + 1):
        front = frontier(list(seen.values()))
        parent = max(front, key=lambda s: value(s, base, w_tokens))
        kids = [await reflect(runner, reader, parent, r, j) for j in range(per_round)]   # the reader's turns first,
        await add(kids)                                                                 # then the ants'
    front = frontier(list(seen.values()))
    ok = [s for s in front if s.accuracy >= base.accuracy - tolerance]
    pick = min(ok, key=lambda s: (s.tokens, -s.accuracy)) if ok else base
    return {"tried": [s.to_json() for s in seen.values()], "frontier": [s.to_json() for s in front],
            "original": base.to_json(), "pick": pick.to_json(), "tolerance": tolerance,
            "n_questions": len(tasks), "ants": list(ants)}


def prompt_of(d: Mapping[str, Any]) -> Prompt:
    return Prompt(str(d["system"]), str(d["format"]), str(d.get("origin") or ""))
