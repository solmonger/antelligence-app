"""Free check: call a stronger model only where cheap ants disagree, and count it as one vote or as the verdict. $0.

Question (``docs/plans/2026-10-29-swarm-at-scale.md``, section 0.3): a deep thinker is expensive, so the swarm should
ask it only where the cheap ants split. When it is asked, should its answer *decide* (a verdict, which makes it a
teacher) or be *one more vote*, weighed by its track record exactly like every other ant (earned trust)?

Data: the Oct 10 colony run's saved answers (same cells and loading as ``scripts/cached_colony_check.py``). No saved
run used a long-reasoning model, so Haiku (the strongest saved model, 0.88 against Qwen's 0.77 on finance) stands in
for the "deep" model. Every answer was given alone, so this is think-alone throughout. Two cheap crowds:

* ``qwen x3``: three samples of Qwen3.8-27B (one model, errors that line up);
* ``mixed``: one sample each of Qwen3.8-27B, Phi-4-mini and Gemma 4 E4B (three models).

Policies per crowd: cheap vote alone; deep once; deep as a verdict where the crowd split; deep as one trust-weighted
vote where the crowd split; deep as one trust-weighted vote everywhere. Trust weights are the log-odds of each model's
accuracy on the development split (as in idea 1). Each item is replayed 4 times, rotating which sample is used, so
rotations of one item are not independent: read the numbers as a lead.

Usage (fill the cells dir as ``scripts/cached_colony_check.py`` says)::

    uv run python scripts/deep_on_split_check.py /tmp/cells --out docs/research/swarm-scale-20261029/deep_on_split.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cached_colony_check import DOMAINS, SAMPLES, Key, load, norm, trust_weights, vote  # noqa: E402

DEEP = "haiku"


def unanimous(rows: Sequence[Mapping]) -> bool:
    answers = {norm(r["answer"]) for r in rows}
    return len(answers) == 1 and None not in answers


def evaluate(ev: Mapping[Key, dict], w: Mapping[Tuple[str, str], float]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for d in DOMAINS:
        items = sorted({k[2] for k in ev if k[1] == d})
        tally: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0])  # right, cheap calls, deep calls
        split_right: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])     # right on split items, split items

        def add(name: str, ok: bool, cheap: int, deep: int, split: bool) -> None:
            tally[name][0] += ok
            tally[name][1] += cheap
            tally[name][2] += deep
            if split:
                split_right[name][0] += ok
                split_right[name][1] += 1

        weight = lambda r: w[(r["model"], d)]  # noqa: E731
        for t in items:
            for k in SAMPLES:
                order = [k] + [j for j in SAMPLES if j != k]
                crowds = {"qwen x3": [ev[("qwen", d, t, j)] for j in order[:3]],
                          "mixed": [ev[(m, d, t, k)] for m in ("qwen", "phi", "gemma")]}
                deep = ev[(DEEP, d, t, k)]
                add("deep once", bool(deep["correct"]), 0, 1, False)
                for cname, crowd in crowds.items():
                    split = not unanimous(crowd)
                    add(f"{cname}: cheap vote", vote(crowd), 3, 0, split)
                    add(f"{cname}: cheap vote, earned trust", vote(crowd, weight), 3, 0, split)
                    if split:
                        add(f"{cname}: deep decides where split", bool(deep["correct"]), 3, 1, True)
                        add(f"{cname}: deep is one vote where split", vote(crowd + [deep], weight), 3, 1, True)
                    else:
                        add(f"{cname}: deep decides where split", vote(crowd), 3, 0, False)
                        add(f"{cname}: deep is one vote where split", vote(crowd), 3, 0, False)
                    add(f"{cname}: deep is one vote everywhere", vote(crowd + [deep], weight), 3, 1, split)
        n = len(items) * len(SAMPLES)
        table: Dict[str, Dict[str, float]] = {}
        for name, (right, cheap, deep_calls) in tally.items():
            s_right, s_n = split_right.get(name, [0, 0])
            table[name] = {"accuracy": round(right / n, 4), "cheap_calls_per_item": round(cheap / n, 3),
                           "deep_calls_per_item": round(deep_calls / n, 3),
                           "split_items": s_n, "accuracy_on_split_items": round(s_right / s_n, 4) if s_n else None}
        table["_n_replays"] = {"items": len(items), "replays": n}
        out[d] = table
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
                acc_split = "" if v["accuracy_on_split_items"] is None else f"{v['accuracy_on_split_items']:.3f}"
                print(f"  {name:40s} acc {v['accuracy']:.3f}  deep calls/item {v['deep_calls_per_item']:.2f}  "
                      f"split items {v['split_items']:>3} acc there {acc_split}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "deep_on_split_check", "deep_stand_in": DEEP,
                                     "trust_weights": {f"{m}/{d}": round(x, 4) for (m, d), x in w.items()},
                                     "results": res}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
