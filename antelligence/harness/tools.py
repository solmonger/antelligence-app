"""The toolbox: every worker behind one call, ``run(tool, task, ctx) -> Step``.

Tool names (plain strings, so logs stay readable):

* ``reader``: the bigger local model answers alone, once.
* ``notes``: the reader writes short notes on the question (no answer). Other tools can read them.
* ``ant{i}:{arm}``: small model number ``i`` of the roster (a model and a way of looking) answers, ``alone``, from
  the reader's ``notes`` (without the passage), or with ``both``.
* ``code``: a model writes a program that checks each option against the question's rules; the program runs in the
  sandbox; when it fails, or passes no option or more than one, the model is shown what went wrong and tries again
  (twice at most). Its answer is the one option that passed, or nothing.
* ``dm:{name}:{arm}``: a decision model (``jev``, ``d1-3b``) gives a probability per option: ``dm1`` reading the
  question alone, ``read`` reading the question and every ant's answer.

The prompts and request shapes are exactly the shared-read and decision-model tests' (same seeds, same hashes), so a
live run can reuse their caches and the importer can price their saved work call by call.

Two toolboxes share this contract: :class:`LiveBox` calls models; :class:`ReplayBox` reads steps saved earlier, for
scoring recipes at no cost.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import fair_test as ft
from antelligence.experiments import shared_read as sr
from antelligence.harness import sandbox
from antelligence.harness.core import Step, Task
from antelligence.harness.resilience import Patience
from antelligence.providers.base import ChatRequest

ARMS = sr.ARMS                                   # alone, notes, both


class Missing(KeyError):
    """The toolbox has no such tool, or (replay) no saved step for this task."""


@dataclass(frozen=True)
class AntInfo:
    index: int
    tag: str
    family: str
    lens: int
    weight: float                 # rough size in GB


def ant_tool(i: int, arm: str) -> str:
    return f"ant{i}:{arm}"


def parse_ant(tool: str) -> Optional[Tuple[int, str]]:
    m = re.fullmatch(r"ant(\d+):(alone|notes|both)", tool)
    return (int(m.group(1)), m.group(2)) if m else None


def deps(tool: str, n_ants: int) -> Tuple[str, ...]:
    """What a tool needs done first (its cost is counted once, by whoever asks first)."""
    a = parse_ant(tool)
    if a and a[1] != "alone":
        return ("notes",)
    if tool.startswith("dm:") and tool.endswith(":read"):
        return tuple(ant_tool(i, "alone") for i in range(n_ants))
    return ()


class Box(Protocol):
    roster: Tuple[AntInfo, ...]

    def has(self, tool: str) -> bool: ...

    async def run(self, tool: str, task: Task, ctx: Mapping[str, Step]) -> Step: ...


# ------------------------------------------------------------------- code tool

CODE_SYSTEM = "You write short, correct Python programs that check puzzles and questions by trying every case."
CODE_ASK = (
    "Write one short Python program (under 60 lines, no comments, no explanation) that decides, for each option, "
    "whether that option correctly answers the question as asked, given every rule and fact above. Check every "
    "possible case instead of guessing. A good shape:\n"
    "1. List every possible case: every arrangement or assignment (itertools.permutations, itertools.product), or, "
    "for statements about groups (all, some, no), every way the groups can overlap.\n"
    "2. Keep only the cases where every rule and fact holds.\n"
    "3. Test each option the way the question asks: 'must be true' means true in every kept case, 'could be true' "
    "means true in at least one, 'cannot be true' or 'must be false' means true in none. For a number, work it out "
    "with fractions.Fraction (not float) and compare it with the number in each option.\n"
    'The program must print exactly one line: a JSON object from each option letter to true or false, like '
    '{"A": false, "B": true, "C": false, "D": false, "E": false}. Use only these standard modules: itertools, math, '
    "fractions, collections, functools, re, json. No input, files or network. Reply with the program in one "
    "```python block and nothing else.")
CODE_MAX_TOKENS = 2000
CODE_TEMPERATURE = 0.2
CODE_RETRIES = 2
CODE_MAX_CUTS = 2          # give up after this many cut-off programs
_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
_JSON_LINE = re.compile(r"\{[^{}]*\}")


def extract_code(reply: str) -> str:
    text = sr.strip_think(reply)
    blocks = _BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip() if ("print" in text and "\n" in text) else ""


def read_passes(stdout: str, letters: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """The options the program marked true, from the last JSON object it printed; None if it printed none."""
    for m in reversed(_JSON_LINE.findall(stdout or "")):
        try:
            obj = json.loads(m)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj and all(str(k).strip("() ").upper() in letters for k in obj):
            return tuple(sorted(str(k).strip("() ").upper() for k, v in obj.items() if v is True))
    return None


CODE_CUT = ("was cut off at the length limit before it ended. Write a much shorter program: under 60 lines, no "
            "comments, no explanation")
CUT_REPLY = "(my reply ran too long and was cut off)"


def cut_off(row: Mapping[str, Any]) -> bool:
    """Did the model run out of room? (A provider that raises on it leaves an error row; one that doesn't, a reply
    whose finish reason is 'length'.)"""
    return row.get("error") == "ProviderTruncated" or row.get("finish_reason") == "length"


def code_problem(run: sandbox.Result, passes: Optional[Tuple[str, ...]], code: str) -> str:
    """What to tell the model when its program didn't settle the question ('' when it did)."""
    if not code:
        return "did not come back as a program in a ```python block"
    if not run.ok:
        return run.error
    if passes is None:
        return "printed no JSON object from option letters to true or false"
    if len(passes) != 1:
        return (f"marked {len(passes)} options as correct ({', '.join(passes) or 'none'}), but exactly one option is "
                "correct, so a rule or the question's ask is coded wrong")
    return ""


def code_messages(task: Task, history: Sequence[Tuple[str, str]], spec: sr.ModelSpec) -> Tuple[Dict[str, str], ...]:
    """The first ask, then for each failed try: the model's reply and what went wrong."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": CODE_SYSTEM},
                                  {"role": "user", "content": sr._user(spec, f"{task.full}\n\n{CODE_ASK}")}]
    for reply, problem in history:
        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": sr._user(
            spec, f"Your program {problem}. Fix it and reply with the whole corrected program only.")})
    return tuple(msgs)


# ---------------------------------------------------------------- answer reading

_ANSWER_VALUE = re.compile(r'"answer"\s*:\s*"?([^"\n}]*)', re.I)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_LETTER_VALUE = re.compile(r"(?:option\s*|choice\s*)?\(?([A-Ea-e])\)?[.)]?", re.I)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w./%-]+", " ", text.lower()).split()).rstrip(".").strip()


def _number(text: str) -> Optional[Fraction]:
    """The one number in ``text`` (commas and a leading $ allowed), or None when there are none or several."""
    found = _NUMBER.findall(text.replace(",", ""))
    if len(found) != 1:
        return None
    try:
        return Fraction(found[0])
    except (ValueError, ZeroDivisionError):
        return None


def read_answer(text: str, task: Task, parser: int = 2) -> Optional[str]:
    """The option letter a reply picks, or None. First the shared-read parser (a letter in the asked JSON form);
    then, when the "answer" value isn't a letter, the one option whose text it matches, word for word or by its
    number (a small model often answers "15" when option C reads "15 hours"). Never a guess: "None of the above",
    or a value two options share, stays None."""
    letter, _ = sr.parse_answer(text or "", parser)
    if letter in task.letters:
        return letter
    values = _ANSWER_VALUE.findall(sr.strip_think(text or ""))
    if not values:
        return None
    value = values[-1].strip().rstrip(".").strip()
    m = _LETTER_VALUE.fullmatch(value)
    if m and m.group(1).upper() in task.letters:
        return m.group(1).upper()
    if not value:
        return None
    hits = [x for x in task.letters if _norm(task.option_text(x)) == _norm(value)]
    if len(hits) == 1:
        return hits[0]
    num = _number(value)
    if num is not None and len(value) <= 40:
        hits = [x for x in task.letters if _number(task.option_text(x)) == num]
        if len(hits) == 1:
            return hits[0]
    return None


REPAIR_ASK = ("You must choose exactly one of the options: exactly one is correct, even if none looks perfect. Reply "
              'with only this JSON and nothing else: {{"answer": "<one letter: {letters}>"}}')
REPAIR_MAX_TOKENS = 30


def repair_messages(messages: Sequence[Mapping[str, str]], reply: str, task: Task, spec: sr.ModelSpec
                    ) -> Tuple[Dict[str, str], ...]:
    """The original ask, the model's own (possibly cut-off) reply, then one short ask for just the letter."""
    said = sr.strip_think(reply or "").strip()
    said = ("..." + said[-1500:]) if len(said) > 1500 else (said or CUT_REPLY)
    return tuple(dict(m) for m in messages) + (
        {"role": "assistant", "content": said},
        {"role": "user", "content": sr._user(spec, REPAIR_ASK.format(letters=", ".join(task.letters)))})


# -------------------------------------------------------------------- live box

class LiveBox:
    """Calls real (or fake) models through an OpenAI-shaped provider, with the shared-read caching runner, so a
    stopped run resumes and a second run of the same question costs nothing.

    ``dms`` maps a decision model's name to ``(DMSpec, DMRunner)``; leave it empty to keep them off (Jev costs money).
    """

    def __init__(self, runner: sr.Runner, reader: sr.ModelSpec, ants: Sequence[sr.ModelSpec], *,
                 lenses: Sequence[int] = sr.LENS_IDS, parser: int = 2, coder: Optional[sr.ModelSpec] = None,
                 code: bool = True, dms: Optional[Mapping[str, Tuple[dm.DMSpec, dm.DMRunner]]] = None,
                 patience: Optional[Patience] = None, sandbox_python: Optional[str] = None,
                 repair: bool = True) -> None:
        self.runner, self.reader, self.parser = runner, reader, parser
        self.coder = coder or reader
        self.code_on = code
        self.dms = dict(dms or {})
        self.patience = patience or Patience()
        self.sandbox_python = sandbox_python
        self.repair = repair                         # one short re-ask for the letter when a reply has none
        self.code_trace: Dict[str, List[str]] = {}   # task id -> what happened on each code try, for diagnose
        self.specs = [s for s, _ in sr.ant_roster(ants, lenses)]
        self.roster = tuple(AntInfo(i, s.tag, s.family, lens, s.gb)
                            for i, (s, lens) in enumerate(sr.ant_roster(ants, lenses)))

    def has(self, tool: str) -> bool:
        if tool in ("reader", "notes"):
            return True
        if tool == "code":
            return self.code_on
        a = parse_ant(tool)
        if a:
            return a[0] < len(self.roster)
        if tool.startswith("dm:"):
            parts = tool.split(":")
            return len(parts) == 3 and parts[1] in self.dms and parts[2] in ("dm1", "read")
        return False

    async def _ask(self, what: str, req: ChatRequest, usable) -> Dict[str, Any]:
        return await self.patience.run(what, lambda: self.runner.ask(req, usable))

    @staticmethod
    def _bill(row: Mapping[str, Any]) -> Tuple[int, int, float]:
        return int(row.get("prompt_tokens") or 0), int(row.get("completion_tokens") or 0), float(row.get("elapsed_s") or 0)

    async def run(self, tool: str, task: Task, ctx: Mapping[str, Step]) -> Step:
        if not self.has(tool):
            raise Missing(tool)
        item = sr.Item(task.id, task.kind, task.passage, task.stem, task.options)
        if tool == "reader":
            return await self._answer(tool, task, self.reader, sr.make_request(
                self.reader, sr.reader_answer_messages(item, self.reader), task.id, "reader", 0, ft.MAX_TOKENS))
        if tool == "notes":
            row = await self._ask(f"{task.id} notes", sr.make_request(
                self.reader, sr.notes_messages(item, self.reader), task.id, "notes", 0, sr.READER_MAX_TOKENS),
                sr._usable_notes)
            ti, to, s = self._bill(row)
            notes = sr.clean_notes(row["content"]) if not row.get("error") else ""
            return Step(tool, None, ti, to, s, weight=self.reader.gb, payload={"notes": notes, "leak": sr.leaks(notes)})
        a = parse_ant(tool)
        if a:
            i, arm = a
            spec, info = self.specs[i], self.roster[i]
            notes = (ctx.get("notes").payload.get("notes") if ctx.get("notes") else "") or ""
            if arm != "alone" and not notes:
                return Step(tool, None, calls=0, weight=spec.gb, note="no notes to read")
            req = sr.make_request(spec, sr.ant_messages(item, arm, info.lens, spec, notes or None), task.id,
                                  f"{arm}:{spec.tag}", info.lens, sr.ANT_MAX_TOKENS)
            return await self._answer(tool, task, spec, req)
        if tool == "code":
            return await self._code(task)
        _, name, arm = tool.split(":")
        return await self._dm(name, arm, task, ctx)

    async def _answer(self, tool: str, task: Task, spec: sr.ModelSpec, req: ChatRequest) -> Step:
        """One answer call; when its reply names no option (cut off, "None", the wrong form), one short re-ask for
        just the letter, with the model's own reply in front of it. Both calls are billed."""
        usable = lambda text: read_answer(text, task, self.parser) is not None  # noqa: E731
        row = await self._ask(f"{task.id} {tool}", req, usable)
        ti, to, s = self._bill(row)
        content = row.get("content") or ""
        got = read_answer(content, task, self.parser) if row.get("error") in (None, "Unusable") else None
        if got is not None or not self.repair:
            note = "cut off" if got is None and cut_off(row) else ""
            return Step(tool, got, ti, to, s, weight=spec.gb, note=note)
        why = "cut off" if cut_off(row) else ("call failed" if row.get("error") not in (None, "Unusable") else
                                              "no option named")
        fix = ChatRequest(model=spec.tag, messages=repair_messages(req.messages, content, task, spec),
                          max_tokens=REPAIR_MAX_TOKENS, temperature=0.0,
                          seed=sr.seed(task.id, f"repair:{tool}", 0), extra_body=spec.extra_body)
        row2 = await self._ask(f"{task.id} {tool} re-ask", fix, usable)
        a, b, c = self._bill(row2)
        got = read_answer(row2.get("content") or "", task, self.parser) if row2.get("error") in (None, "Unusable") \
            else None
        note = f"{why}; {'fixed by' if got else 'still blank after'} a short re-ask for the letter"
        return Step(tool, got, ti + a, to + b, s + c, calls=2, weight=spec.gb, note=note)

    async def _code(self, task: Task) -> Step:
        history: List[Tuple[str, str]] = []
        ti = to = 0
        secs = 0.0
        passes: Optional[Tuple[str, ...]] = None
        problem = "never ran"
        tries = cuts = 0
        for attempt in range(CODE_RETRIES + 1):
            tries = attempt + 1
            req = ChatRequest(model=self.coder.tag, messages=code_messages(task, history, self.coder),
                              max_tokens=CODE_MAX_TOKENS, temperature=CODE_TEMPERATURE,
                              seed=sr.seed(task.id, "code", attempt), extra_body=self.coder.extra_body)
            row = await self._ask(f"{task.id} code {attempt + 1}", req, lambda text: True)
            a, b, s = self._bill(row)
            ti, to, secs = ti + a, to + b, secs + s
            reply = row.get("content") or ""
            if cut_off(row):
                cuts += 1
                flat = " ".join(sr.strip_think(reply).split())
                seen = f" (starts: {flat[:50]!r} ... ends: {flat[-70:]!r})" if flat else ""
                self.code_trace.setdefault(task.id, []).append(f"cut off at {CODE_MAX_TOKENS} tokens{seen}")
                history.append((CUT_REPLY, CODE_CUT))
                problem = CODE_CUT
                if cuts >= CODE_MAX_CUTS:            # it ran out of room twice: a third try costs the same again
                    break
                continue
            if row.get("error") not in (None, "Unusable"):
                problem = f"did not come back (the call failed: {row.get('error')})"
                self.code_trace.setdefault(task.id, []).append(problem)
                history.append((reply or "(no reply came back)", problem))
                continue
            code = extract_code(reply)
            result = (await asyncio.to_thread(sandbox.run, code, python=self.sandbox_python) if code
                      else sandbox.Result(False, "", "", 0.0))
            secs += result.seconds
            passes = read_passes(result.stdout, task.letters) if result.ok else None
            problem = code_problem(result, passes, code)
            self.code_trace.setdefault(task.id, []).append(problem or f"settled on {passes[0]}")
            if not problem:
                return Step("code", passes[0], ti, to, secs, calls=attempt + 1, weight=self.coder.gb, sure=1.0,
                            passes=passes, note=f"settled on try {attempt + 1}")
            history.append((reply, problem))
        return Step("code", None, ti, to, secs, calls=tries, weight=self.coder.gb, sure=0.0,
                    passes=passes, note=f"not settled: {problem[:160]}")

    async def _dm(self, name: str, arm: str, task: Task, ctx: Mapping[str, Step]) -> Step:
        spec, runner = self.dms[name]
        q = ft.Question("exam", task.id, task.kind, task.full)
        options = {l: task.option_text(l) for l in task.letters}
        ants = tuple(dm.Ant(ctx[ant_tool(i, "alone")].answer) for i in range(len(self.roster))) if arm == "read" else ()
        req = dm.dm_request(spec.model, q, arm, options, ants)
        row = await self.patience.run(f"{task.id} {name}:{arm}", lambda: runner.ask(req))
        ans = dm.to_answer("exam", row)
        return Step(f"dm:{name}:{arm}", dm.dm_value("exam", ans), int(row.get("input_tokens") or 0), 0,
                    float(row.get("elapsed_s") or 0), float(row.get("cost_usd") or 0), weight=0.0,
                    sure=dm.confidence("exam", ans) if ans else 0.0)


# ------------------------------------------------------------------ replay box

class ReplayBox:
    """Saved steps, read back. ``steps[task_id][tool] -> Step``. Costs nothing; a recipe that needs a step that was
    never saved raises :class:`Missing`, so it is only ever scored on tasks where all of its work is on file."""

    def __init__(self, steps: Mapping[str, Mapping[str, Step]], roster: Sequence[AntInfo]) -> None:
        self.steps = steps
        self.roster = tuple(roster)
        self._tools = {t for per in steps.values() for t in per}

    def has(self, tool: str) -> bool:
        return tool in self._tools

    async def run(self, tool: str, task: Task, ctx: Mapping[str, Step]) -> Step:
        try:
            return self.steps[task.id][tool]
        except KeyError:
            raise Missing(f"{task.id}:{tool}") from None


# ---------------------------------------------------------------- step files

def save_steps(path: Path, steps: Mapping[str, Mapping[str, Step]], roster: Sequence[AntInfo],
               meta: Optional[Mapping[str, Any]] = None) -> None:
    """One JSON file: the roster, any notes about where the steps came from, and every step."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    obj = {"roster": [vars(a) for a in roster], "meta": dict(meta or {}),
           "steps": {tid: {t: s.to_json() for t, s in per.items()} for tid, per in steps.items()}}
    Path(path).write_text(json.dumps(obj, sort_keys=True) + "\n")


def load_steps(path: Path) -> Tuple[Dict[str, Dict[str, Step]], Tuple[AntInfo, ...], Dict[str, Any]]:
    obj = json.loads(Path(path).read_text())
    roster = tuple(AntInfo(**a) for a in obj["roster"])
    steps = {tid: {t: Step.from_json(s) for t, s in per.items()} for tid, per in obj["steps"].items()}
    return steps, roster, dict(obj.get("meta") or {})


def trim_loads(steps: Mapping[str, Mapping[str, Step]], factor: float = 10.0, min_s: float = 30.0
               ) -> Tuple[Dict[str, Dict[str, Step]], int]:
    """Count a step that took more than ``factor`` times its tool's median time (and over ``min_s``) at the median:
    on a one-model-at-a-time server that step was almost always waiting for the model to load. Returns the steps and
    how many were trimmed, so a report can say so."""
    import statistics
    from dataclasses import replace

    times: Dict[str, List[float]] = {}
    for per in steps.values():
        for tool, s in per.items():
            times.setdefault(tool, []).append(s.latency_s)
    med = {t: statistics.median(v) for t, v in times.items()}
    out: Dict[str, Dict[str, Step]] = {}
    n = 0
    for tid, per in steps.items():
        out[tid] = {}
        for tool, s in per.items():
            if med[tool] and s.latency_s > max(factor * med[tool], min_s):
                n += 1
                s = replace(s, latency_s=med[tool], note=(s.note + "; " if s.note else "")
                            + f"load-time outlier counted at the median ({s.latency_s:.0f}s)")
            out[tid][tool] = s
    return out, n
