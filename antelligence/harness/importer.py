"""Turn work already done into harness steps, at no cost: the fresh-question and night runs' saved answers.

Each saved set (``questions.jsonl``, ``answers.jsonl``, ``rows.jsonl``, ``cache.jsonl``, and where present
``dm_<name>.jsonl`` with ``cache_<name>.jsonl``) already holds every tool's answer to every question: the reader
alone, the reader's notes, and 8 ants (4 small models x 2 ways of looking) alone, from the notes and with both. The
caches also hold every call's tokens and seconds. They are matched to the answers by rebuilding each request exactly
as the run made it (same prompt, same seed, so the same hash), which is why the model file must be the run's own.

What can't be matched is said, not hidden: the import report gives the share of steps whose bill was found
(``priced``), and fills any unmatched step with the typical bill of the same model and role. A call that took more than
ten times its model's typical time (almost always the stand-in loading a model from disk) is counted at the typical
time, and the report says how many.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from antelligence.experiments import dm_addon as dm
from antelligence.experiments import fair_test as ft
from antelligence.experiments import shared_read as sr
from antelligence.harness.core import Step, Task
from antelligence.harness.resilience import read_text
from antelligence.harness.tools import ARMS, AntInfo, ant_tool, read_answer

LOAD_FACTOR = 10.0          # a call this many times slower than its model's median is treated as a model load
LOAD_MIN_S = 30.0
DM_NAMES = ("jev", "d1-3b")


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def _cache(path: Path) -> Dict[str, Dict[str, Any]]:
    """Request hash -> the call's row: the last usable one, else the last one (an unusable reply was still paid)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in _jsonl(path):
        h = row.get("request_hash")
        if h and (not row.get("error") or h not in out or out[h].get("error")):
            out[h] = row
    return out


def roster_of(run: Path, cfg: sr.Config) -> Tuple[List[Tuple[sr.ModelSpec, int]], Tuple[AntInfo, ...]]:
    """The run's ants in column order, from its meta.json when it has one (else the model file's)."""
    pairs = sr.ant_roster(cfg.ants)
    meta = run / "meta.json"
    if meta.exists():
        listed = json.loads(read_text(meta)).get("ants") or []
        if listed:
            tags = {s.tag: s for s in cfg.ants}
            missing = [t for t, _, _ in listed if t not in tags]
            if missing:
                raise ValueError(f"{run}: ants {missing} are not in the model file; use the run's own model file")
            pairs = [(tags[t], int(lens)) for t, _, lens in listed]
    return pairs, tuple(AntInfo(i, s.tag, s.family, lens, s.gb) for i, (s, lens) in enumerate(pairs))


class _Pricer:
    """Bills for steps, with the fill-in and load-time rules from the module note."""

    def __init__(self) -> None:
        self.found = self.missing = self.loads = self.cut = 0
        self.raw: List[Tuple[str, str, str, Optional[Dict[str, Any]], Dict[str, Any]]] = []

    def add(self, task_id: str, tool: str, group: str, row: Optional[Dict[str, Any]], step: Dict[str, Any]) -> None:
        self.raw.append((task_id, tool, group, row, step))

    def steps(self) -> Dict[str, Dict[str, Step]]:
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for _, _, g, row, _ in self.raw:
            if row is not None and not row.get("error"):
                by_group.setdefault(g, []).append(row)
        med = {g: {"tin": st.median(int(r.get("prompt_tokens") or r.get("input_tokens") or 0) for r in rs),
                   "tout": st.median(int(r.get("completion_tokens") or 0) for r in rs),
                   "s": st.median(float(r.get("elapsed_s") or 0) for r in rs),
                   "usd": st.median(float(r.get("cost_usd") or 0) for r in rs)} for g, rs in by_group.items()}
        out: Dict[str, Dict[str, Step]] = {}
        for tid, tool, g, row, step in self.raw:
            m = med.get(g, {"tin": 0, "tout": 0, "s": 0.0, "usd": 0.0})
            if row is None:
                self.missing += 1
                tin, tout, secs, usd, note = int(m["tin"]), int(m["tout"]), float(m["s"]), float(m["usd"]), "unpriced"
            elif row.get("error") == "ProviderTruncated" and not row.get("completion_tokens") and step.get("max_tokens"):
                # a reply the provider threw away for running out of room: it kept no counts, but it ran to its
                # length limit, so bill that many tokens out, the usual prompt, and the time that many tokens take
                self.found += 1
                self.cut += 1
                tin, tout = int(m["tin"]), int(step["max_tokens"])
                rate = (float(m["s"]) / m["tout"]) if m["tout"] else 0.0
                secs, usd, note = rate * tout, float(m["usd"]), "cut off: billed at its length limit"
            else:
                self.found += 1
                tin = int(row.get("prompt_tokens") or row.get("input_tokens") or 0)
                tout = int(row.get("completion_tokens") or 0)
                secs, usd, note = float(row.get("elapsed_s") or 0), float(row.get("cost_usd") or 0), ""
                if m["s"] and secs > max(LOAD_FACTOR * m["s"], LOAD_MIN_S):
                    self.loads += 1
                    secs, note = float(m["s"]), f"load-time outlier counted at the median ({row.get('elapsed_s')}s)"
            out.setdefault(tid, {})[tool] = Step(tool, step["answer"], tin, tout, round(secs, 4), round(usd, 8),
                                                 1, step["weight"], step.get("sure"), None, note, step.get("payload") or {})
        return out


def import_run(run: Path, cfg: sr.Config, prefix: str) -> Tuple[List[Task], Dict[str, str], Dict[str, Dict[str, Step]],
                                                                 Tuple[AntInfo, ...], Dict[str, Any]]:
    """(tasks, key, steps, roster, report) for one saved set. Task ids get ``prefix:`` so sets can be pooled."""
    run = Path(run)
    raws = {str(r["id"]): r for r in _jsonl(run / "questions.jsonl")}
    key = {f"{prefix}:{r['id']}": str(r["label"]) for r in _jsonl(run / "answers.jsonl")}
    rows = {str(r["id"]): r for r in _jsonl(run / "rows.jsonl")}
    cache = _cache(run / "cache.jsonl")
    pairs, roster = roster_of(run, cfg)
    pricer = _Pricer()
    reparsed = 0
    tasks: List[Task] = []
    reader = cfg.reader
    for qid, raw in raws.items():
        if qid not in rows:
            continue
        r = rows[qid]
        task = Task.from_raw(raw)
        tid = f"{prefix}:{qid}"
        tasks.append(Task(tid, task.passage, task.stem, task.options, task.kind))
        item = sr.Item(qid, task.kind, task.passage, task.stem, task.options)
        notes = r.get("reader_notes") or ""

        def look(req) -> Optional[Dict[str, Any]]:
            return cache.get(req.request_hash)

        def answer(recorded: Optional[str], row: Optional[Mapping[str, Any]]) -> Optional[str]:
            """The run's own answer; when it had none, the harness's reader of the same saved reply (it maps an
            answer given as the option's text, like "15", to that option's letter)."""
            nonlocal reparsed
            if recorded is not None or not row or row.get("error") not in (None, "Unusable"):
                return recorded
            got = read_answer(row.get("content") or "", task, cfg.parser)
            reparsed += got is not None
            return got

        req = sr.make_request(reader, sr.reader_answer_messages(item, reader), qid, "reader", 0, ft.MAX_TOKENS)
        pricer.add(tid, "reader", f"{reader.tag}:reader", look(req),
                   {"answer": answer(r.get("reader"), look(req)), "weight": reader.gb, "max_tokens": ft.MAX_TOKENS})
        req = sr.make_request(reader, sr.notes_messages(item, reader), qid, "notes", 0, sr.READER_MAX_TOKENS)
        pricer.add(tid, "notes", f"{reader.tag}:notes", look(req),
                   {"answer": None, "weight": reader.gb, "max_tokens": sr.READER_MAX_TOKENS,
                    "payload": {"notes": notes, "leak": bool(r.get("leak"))}})
        for col, (spec, lens) in enumerate(pairs):
            for arm in ARMS:
                answers = r.get(arm) or []
                if arm != "alone" and not notes:
                    continue
                req = sr.make_request(spec, sr.ant_messages(item, arm, lens, spec, notes or None), qid,
                                      f"{arm}:{spec.tag}", lens, sr.ANT_MAX_TOKENS)
                pricer.add(tid, ant_tool(col, arm), f"{spec.tag}:{arm}", look(req),
                           {"answer": answer(answers[col] if col < len(answers) else None, look(req)),
                            "weight": spec.gb, "max_tokens": sr.ANT_MAX_TOKENS})
    dm_report = _import_dms(run, raws, rows, prefix, pricer)
    steps = pricer.steps()
    total = pricer.found + pricer.missing
    report = {"run": str(run), "set": prefix, "tasks": len(tasks), "steps": total,
              "priced": round(pricer.found / total, 4) if total else 0.0, "load_outliers": pricer.loads,
              "reparsed": reparsed, "cut_off_billed": pricer.cut,
              "roster": [[a.tag, a.lens] for a in roster], "decision_models": dm_report}
    return tasks, key, steps, roster, report


def _import_dms(run: Path, raws: Mapping[str, Any], rows: Mapping[str, Any], prefix: str, pricer: _Pricer
                ) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    qs = {q.id: q for q in ft.load_questions(run / "questions.jsonl", "exam")}
    swarm = {s.id: s for s in dm.export_shared_read(list(rows.values()))}
    for name in DM_NAMES:
        path = run / f"dm_{name}.jsonl"
        if not path.exists():
            continue
        spec = dm.DMS[name]
        cache = _cache(run / f"cache_{name}.jsonl")
        n = 0
        for row in _jsonl(path):
            qid = str(row["id"])
            if qid not in qs or qid not in swarm:
                continue
            opts = dm.option_texts(raws[qid])
            for arm in ("dm1", "read"):
                if arm not in row:
                    continue
                req = dm.dm_request(spec.model, qs[qid], arm, opts, swarm[qid].ants if arm == "read" else ())
                ans = row[arm]
                pricer.add(f"{prefix}:{qid}", f"dm:{name}:{arm}", f"{name}:{arm}", cache.get(req.request_hash),
                           {"answer": dm.dm_value("exam", ans), "weight": 0.0 if spec.hosted else 6.0,
                            "sure": dm.confidence("exam", ans) if ans else 0.0})
                n += 1
        report[name] = n
    return report


def import_sets(runs: Mapping[str, Path], cfg: sr.Config):
    """Several sets pooled: (tasks, key, steps, roster, reports). Every set must have the same roster."""
    tasks: List[Task] = []
    key: Dict[str, str] = {}
    steps: Dict[str, Dict[str, Step]] = {}
    roster: Optional[Tuple[AntInfo, ...]] = None
    reports = []
    for name, run in runs.items():
        t, k, s, r, rep = import_run(run, cfg, name)
        if roster is not None and [(a.tag, a.lens) for a in r] != [(a.tag, a.lens) for a in roster]:
            raise ValueError(f"set {name} has a different roster from the others")
        roster = r
        tasks += t
        key.update(k)
        steps.update(s)
        reports.append(rep)
    return tasks, key, steps, roster or (), reports


def set_of(task_id: str) -> str:
    return task_id.split(":", 1)[0] if ":" in task_id else ""
