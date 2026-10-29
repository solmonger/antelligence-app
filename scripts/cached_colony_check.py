"""Free check of two "smarter swarm" ideas on the colony run's saved answers. No model call, $0.

Ideas checked (``docs/plans/2026-10-29-swarm-at-scale.md``, section 3):

* **Earned trust**: each model's vote counts by its track record (log-odds of its accuracy on the
  *development* split), so a weak member stops dragging the vote down.
* **Grow only for hard problems**: ask one model once; ask more only when that answer fails the run's own source
  check (``verification.status != "verified"``).

Data: the Oct 10 colony run's cells (branch ``hermes/track1-colony-experiment-20261010``,
``docs/research/colony-v2-20261010/cells/<model>-{development,evaluation}.jsonl``): Qwen3.8-27B, Phi-4-mini,
Gemma 4 E4B and Haiku; 50 finance and 50 medical items; 4 samples per item. Each item is replayed 4 times, each time
with a different sample as the "first" answer, so every policy sees every sample once. Rotations of one item are not
independent: read the numbers as a lead, not a result.

Usage::

    mkdir -p /tmp/cells && for m in qwen haiku phi gemma; do for s in development evaluation; do
      git show origin/hermes/track1-colony-experiment-20261010:docs/research/colony-v2-20261010/cells/$m-$s.jsonl \
        > /tmp/cells/$m-$s.jsonl; done; done
    uv run python scripts/cached_colony_check.py /tmp/cells --out docs/research/swarm-scale-20261029/cached_colony.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

MODELS = ("qwen", "haiku", "phi", "gemma")
DOMAINS = ("finance", "medical")
SAMPLES = (1, 2, 3, 4)
Key = Tuple[str, str, str, int]  # model, domain, task_id, sample


def load(cells: Path, split: str) -> Dict[Key, dict]:
    out: Dict[Key, dict] = {}
    for m in MODELS:
        for line in (cells / f"{m}-{split}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(m, r["domain"], r["task_id"], int(r["sample"]))] = r
    return out


def verified(r: Mapping) -> bool:
    return (r.get("verification") or {}).get("status") == "verified"


def norm(answer) -> Optional[str]:
    return None if answer is None else str(answer).strip().lower()


def trust_weights(dev: Mapping[Key, dict]) -> Dict[Tuple[str, str], float]:
    """Log-odds of each model's development accuracy (Laplace-smoothed); 0 when no better than a coin."""
    w = {}
    for m in MODELS:
        for d in DOMAINS:
            rows = [r for k, r in dev.items() if k[0] == m and k[1] == d]
            p = (sum(bool(r["correct"]) for r in rows) + 1) / (len(rows) + 2)
            w[(m, d)] = math.log(p / (1 - p)) if p > 0.5 else 0.0
    return w


def vote(rows: Sequence[Mapping], weight: Callable[[Mapping], float] = lambda r: 1.0) -> bool:
    """Weighted plurality over normalised answers; correct iff the winning answer was marked correct. Ties go to
    the answer seen first (the order of ``rows``), so a policy's first answer wins a tie."""
    score: Dict[str, float] = collections.OrderedDict()
    truth: Dict[str, bool] = {}
    for r in rows:
        a = norm(r["answer"])
        if a is None:
            continue
        score[a] = score.get(a, 0.0) + weight(r)
        truth.setdefault(a, bool(r["correct"]))
    if not score:
        return False
    best = max(score.values())
    return truth[next(a for a, s in score.items() if s == best)]


def evaluate(ev: Mapping[Key, dict], w: Mapping[Tuple[str, str], float]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for d in DOMAINS:
        items = sorted({k[2] for k in ev if k[1] == d})
        tally: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0])

        def add(name: str, ok: bool, calls: int) -> None:
            tally[name][0] += ok
            tally[name][1] += calls

        for t in items:
            for k in SAMPLES:
                order = [k] + [j for j in SAMPLES if j != k]
                q = [ev[("qwen", d, t, j)] for j in order]
                mix = [ev[(m, d, t, k)] for m in ("qwen", "phi", "gemma")]
                haiku = ev[("haiku", d, t, k)]
                add("best member once (qwen)", bool(q[0]["correct"]), 1)
                add("best member x3, vote", vote(q[:3]), 3)
                add("mixed colony, equal votes", vote(mix), 3)
                add("mixed colony, earned trust", vote(mix, lambda r: w[(r["model"], d)]), 3)
                if verified(q[0]):
                    add("grow on hard: +2 qwen", bool(q[0]["correct"]), 1)
                    add("grow on hard: ask haiku", bool(q[0]["correct"]), 1)
                else:
                    add("grow on hard: +2 qwen", vote(q[:3]), 3)
                    add("grow on hard: ask haiku", bool(haiku["correct"]), 2)
                add("haiku once", bool(haiku["correct"]), 1)
        n = len(items) * len(SAMPLES)
        out[d] = {name: {"accuracy": round(c / n, 4), "calls_per_item": round(calls / n, 3)}
                  for name, (c, calls) in tally.items()}
        out[d]["_n_replays"] = {"items": len(items), "replays": n}
    return out


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cells", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    w = trust_weights(load(a.cells, "development"))
    res = evaluate(load(a.cells, "evaluation"), w)
    for d, table in res.items():
        print(d)
        for name, v in table.items():
            if not name.startswith("_"):
                print(f"  {name:32s} accuracy {v['accuracy']:.3f}  calls/item {v['calls_per_item']:.2f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "cached_colony_check", "trust_weights":
                                     {f"{m}/{d}": round(x, 4) for (m, d), x in w.items()},
                                     "results": res}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
