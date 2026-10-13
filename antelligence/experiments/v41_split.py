"""v4.1 case split: deterministic, hash-ordered, written once (brief S5).

Case ids are sorted by ``sha256("v4.1|" + case_id)``; the first 40 are dev (dev-A 30, dev-B 10), the next
80 are eval, and the rest are reserve. ``compute_split`` is pure, and the committed ``config/v4_1/split.json``
must be reproducible from its own id union.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

STUDY = "v4.1"
PREFIX = "v4.1|"
N_DEV_A, N_DEV_B, N_EVAL = 30, 10, 80
RULE = ("sort MSD Task01 training case ids by sha256('v4.1|'+case_id); first 40 dev (dev-A 30, dev-B 10), "
        "next 80 eval, rest reserve")
SOURCE = "MSD Task01_BrainTumour dataset.json (CC BY-SA 4.0), tar md5 240a19d752f0d9e9101544901065d872"


def case_key(case_id: str) -> str:
    return hashlib.sha256((PREFIX + case_id).encode()).hexdigest()


def ids_from_dataset(dataset: Dict[str, Any]) -> List[str]:
    """Basenames (without .nii.gz) of the MSD dataset.json ``training`` images."""
    out = []
    for item in dataset["training"]:
        name = Path(item["image"]).name
        out.append(name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name)
    return out


def compute_split(case_ids: Iterable[str]) -> Dict[str, Any]:
    ids = list(case_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    if len(ids) < N_DEV_A + N_DEV_B + N_EVAL:
        raise ValueError("too few cases for the dev/eval split")
    ordered = sorted(ids, key=case_key)
    a, b = N_DEV_A, N_DEV_A + N_DEV_B
    e = b + N_EVAL
    return {"study": STUDY, "rule": RULE, "source": SOURCE, "dev_a": ordered[:a], "dev_b": ordered[a:b],
            "eval": ordered[b:e], "reserve": ordered[e:]}


def split_ids(doc: Dict[str, Any]) -> List[str]:
    return [c for k in ("dev_a", "dev_b", "eval", "reserve") for c in doc[k]]


def dump(doc: Dict[str, Any]) -> str:
    """The exact on-disk format of config/v4_1/split.json (indent 1, no trailing newline)."""
    return json.dumps(doc, indent=1)


def load_split(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def which_split(doc: Dict[str, Any], case: str) -> str:
    """'dev', 'eval', 'reserve' or raises KeyError."""
    if case in doc["dev_a"] or case in doc["dev_b"]:
        return "dev"
    if case in doc["eval"]:
        return "eval"
    if case in doc["reserve"]:
        return "reserve"
    raise KeyError(f"case {case!r} is not in the split")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-json", required=True, help="MSD Task01 dataset.json")
    ap.add_argument("--out", default="config/v4_1/split.json")
    args = ap.parse_args(argv)
    doc = compute_split(ids_from_dataset(json.loads(Path(args.dataset_json).read_text())))
    out = Path(args.out)
    if out.exists() and out.read_text() != dump(doc):
        raise SystemExit(f"{out} exists and differs; the split is write-once")
    out.write_text(dump(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
