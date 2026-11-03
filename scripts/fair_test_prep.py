"""Build the fair test's exam set from AGIEval (English multiple choice), with answers in a separate file. $0.

Takes four AGIEval tasks from different fields (LSAT analytical reasoning, LSAT logical reasoning, LogiQA, AQuA-RAT
maths), samples ``--per-task`` questions from each with a fixed seed, and writes:

* ``exam_questions.jsonl`` -- id, task, passage, question, options. Nothing else: no label, answer or worked
  solution, so the harness cannot show the answer to the model even by mistake (``load_questions`` refuses any row
  carrying an answer field).
* ``exam_answers.jsonl`` -- id, label. Read only by ``scripts/fair_test.py score``, after every call is done.
* ``exam_manifest.json`` -- where each file came from and its SHA-256, for the preregistration.

The source files are read from ``--raw-dir`` when given (one ``<task>.jsonl`` per task, as published), otherwise
downloaded from the AGIEval repository on GitHub.

Usage::

    uv run python scripts/fair_test_prep.py --out docs/research/fair-test-20261029
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TASKS = ("lsat-ar", "lsat-lr", "logiqa-en", "aqua-rat")
SOURCE = "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1_1/{task}.jsonl"
SEED = 20261029
KEEP = ("passage", "question", "options")


def fetch(task: str, raw_dir: Optional[Path]) -> bytes:
    if raw_dir is not None:
        return (raw_dir / f"{task}.jsonl").read_bytes()
    with urllib.request.urlopen(SOURCE.format(task=task), timeout=60) as resp:  # noqa: S310 (fixed https URL)
        return resp.read()


def parse(raw: bytes, task: str) -> List[Dict[str, Any]]:
    rows = []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        label = str(r.get("label") or "").strip().upper()
        options = r.get("options") or []
        if len(label) != 1 or not label.isalpha() or not options or ord(label) - ord("A") >= len(options):
            continue  # free-response or malformed rows are not used
        rows.append({"id": f"{task}-{i:04d}", "task": task, "passage": r.get("passage"),
                     "question": r["question"], "options": options, "label": label})
    return rows


def build(per_task: int, raw_dir: Optional[Path], seed: int = SEED) -> Dict[str, Any]:
    questions, answers, sources = [], [], {}
    for task in TASKS:
        raw = fetch(task, raw_dir)
        sources[task] = {"url": SOURCE.format(task=task), "sha256": hashlib.sha256(raw).hexdigest()}
        rows = parse(raw, task)
        if len(rows) < per_task:
            raise ValueError(f"{task} has only {len(rows)} usable rows")
        picked = sorted(random.Random(f"{seed}:{task}").sample(rows, per_task), key=lambda r: r["id"])
        sources[task]["usable_rows"] = len(rows)
        for r in picked:
            questions.append({"id": r["id"], "task": task, **{k: r[k] for k in KEEP}})
            answers.append({"id": r["id"], "label": r["label"]})
    return {"questions": questions, "answers": answers, "sources": sources}


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=60)
    ap.add_argument("--raw-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    built = build(a.per_task, a.raw_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"benchmark": "AGIEval v1.1 (English multiple choice)", "tasks": list(TASKS), "per_task": a.per_task,
                "seed": SEED, "sources": built["sources"],
                "exam_questions.jsonl": write_jsonl(a.out / "exam_questions.jsonl", built["questions"]),
                "exam_answers.jsonl": write_jsonl(a.out / "exam_answers.jsonl", built["answers"])}
    (a.out / "exam_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{len(built['questions'])} exam questions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
