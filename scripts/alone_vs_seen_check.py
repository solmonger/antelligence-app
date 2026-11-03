"""Free check of "think alone, then compare" on saved W2 answers: blind teams against teams that saw each other.

W2 (branch ``hermes/m0-w2-benchmark-20261010``, ``docs/research/slm-vs-frontier-20261008/cells/<model>.jsonl``) ran the
same three-agent team on the same Q&A items and seeds under several protocols. Two pairs isolate "seeing each other":

* ``independent_vote`` (three blind samples, strict majority) against ``signal_board`` (the same three agents post
  typed claims and revise twice after reading each other's). Same full evidence for everyone.
* ``evidence_isolated`` (three agents with different evidence shards, never told what the others found) against
  ``evidence_exchange`` (the same shards, then each agent sees the others' findings and cited passages and revises).

Neither pair has a checker or a queen in the loop: the team answer is a plain strict majority.

For each model and pair, matched on (task, seed): team accuracy; how often the three final answers are unanimous, and
unanimous *and wrong*; per-agent accuracy (against a reference answer taken from any cell scored correct on that task);
and paired help/harm (blind right and seen wrong, or the reverse). No model call, $0.

Usage::

    mkdir -p /tmp/w2 && for m in qwen38-27b-q3k claude-haiku-5.5 claude-sonnet-5.5; do
      git show origin/hermes/m0-w2-benchmark-20261010:docs/research/slm-vs-frontier-20261008/cells/$m.jsonl \
        > /tmp/w2/cells-$m.jsonl; done
    uv run python scripts/alone_vs_seen_check.py /tmp/w2 --out docs/research/swarm-scale-20261029/alone_vs_seen.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MODELS = ("qwen38-27b-q3k", "claude-haiku-5.5", "claude-sonnet-5.5")
PAIRS = (("independent_vote", "signal_board"), ("evidence_isolated", "evidence_exchange"))


def norm(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    s = re.sub(r"[\s,$%]", "", str(answer).strip().lower())
    return s or None


def load(cells_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for m in MODELS:
        path = cells_dir / f"cells-{m}.jsonl"
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def gold_answers(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Reference answer per task: the normalised answer of any research_qa cell the benchmark scored correct."""
    gold: Dict[str, str] = {}
    for r in rows:
        if r.get("world") == "research_qa" and r.get("correct") is True and norm(r.get("answer")):
            gold.setdefault(r["task_id"], norm(r["answer"]))
    return gold


def _finals(r: Mapping[str, Any]) -> List[Optional[str]]:
    return [norm(a) for a in (r.get("final_answers") or [])]


def compare_pair(rows: Sequence[Mapping[str, Any]], model: str, blind: str, seen: str,
                 gold: Mapping[str, str], domain: Optional[str] = None) -> Dict[str, Any]:
    def index(arm: str) -> Dict[Tuple[str, Any], Mapping[str, Any]]:
        return {(r["task_id"], r["seed"]): r for r in rows
                if r.get("model_key") == model and r.get("arm") == arm and r.get("world") == "research_qa"
                and (domain is None or r.get("domain") == domain)}

    b, s = index(blind), index(seen)
    keys = sorted(set(b) & set(s), key=str)
    out: Dict[str, Any] = {"model": model, "blind_arm": blind, "seen_arm": seen, "domain": domain or "all", "pairs": len(keys)}
    if not keys:
        return out
    stats = {arm: {"team_right": 0, "completed": 0, "unanimous": 0, "unanimous_wrong": 0, "agent_right": 0,
                   "agents": 0, "both_wrong_pairs": 0, "wrong_agent_pairs": 0} for arm in ("blind", "seen")}
    both_completed = [k for k in keys if b[k].get("status") == "completed" and s[k].get("status") == "completed"]
    help_, harm = 0, 0
    for k in keys:
        for arm, r in (("blind", b[k]), ("seen", s[k])):
            # Strict, as W2 scored it: an invalid, errored or abstained cell is a wrong team answer.
            stats[arm]["team_right"] += bool(r.get("correct"))
            stats[arm]["completed"] += r.get("status") == "completed"
        rb, rs = bool(b[k].get("correct")), bool(s[k].get("correct"))
        help_ += (not rb) and rs
        harm += rb and not rs
    # How the answers relate is read only where both cells completed, so format failures do not count as agreement.
    out["seen_helped_completed"] = sum((not b[k].get("correct")) and bool(s[k].get("correct")) for k in both_completed)
    out["seen_hurt_completed"] = sum(bool(b[k].get("correct")) and not s[k].get("correct") for k in both_completed)
    for k in both_completed:
        for arm, r in (("blind", b[k]), ("seen", s[k])):
            st = stats[arm]
            finals = _finals(r)
            unanimous = len(finals) == 3 and finals[0] is not None and len(set(finals)) == 1
            st["unanimous"] += unanimous
            st["unanimous_wrong"] += unanimous and not r.get("correct")
            g = gold.get(k[0])
            if g is not None and finals:
                oks = [f == g for f in finals]
                st["agent_right"] += sum(oks)
                st["agents"] += len(oks)
                for x, y in itertools.combinations(oks, 2):
                    st["wrong_agent_pairs"] += (not x) + (not y)
                    st["both_wrong_pairs"] += (not x) and (not y)
    n, nc = len(keys), len(both_completed)
    out["pairs_both_completed"] = nc
    for arm, st in stats.items():
        out[arm] = {"team_accuracy": round(st["team_right"] / n, 4),
                    "completed": round(st["completed"] / n, 4),
                    "unanimous": round(st["unanimous"] / nc, 4) if nc else None,
                    "unanimous_and_wrong": round(st["unanimous_wrong"] / nc, 4) if nc else None,
                    "agent_accuracy": round(st["agent_right"] / st["agents"], 4) if st["agents"] else None,
                    # When one agent is wrong, how often is a teammate wrong too: 1.0 means every error is shared,
                    # lower means errors are more independent.
                    "wrong_together": round(2 * st["both_wrong_pairs"] / st["wrong_agent_pairs"], 4)
                    if st["wrong_agent_pairs"] else None}
    out["seen_helped"], out["seen_hurt"] = help_, harm
    return out


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cells_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = load(a.cells_dir)
    gold = gold_answers(rows)
    results = []
    for model in MODELS:
        for blind, seen in PAIRS:
            for domain in (None, "finance", "medical"):
                res = compare_pair(rows, model, blind, seen, gold, domain)
                if res["pairs"]:
                    results.append(res)
    print(f"{'model':>18} {'pair':>36} {'dom':>8} {'n':>4} {'nc':>4} | {'team acc':>13} | {'agent acc':>13} | "
          f"{'unanimous+wrong':>15} | {'wrong together':>15} | seen helped/hurt")
    for r in results:
        b, s = r["blind"], r["seen"]
        fmt = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
        print(f"{r['model']:>18} {(r['blind_arm'] + ' vs ' + r['seen_arm'])[:36]:>36} {r['domain']:>8} {r['pairs']:>4} {r['pairs_both_completed']:>4} | "
              f"{fmt(b['team_accuracy'])}/{fmt(s['team_accuracy'])} | {fmt(b['agent_accuracy'])}/{fmt(s['agent_accuracy'])} | "
              f"{fmt(b['unanimous_and_wrong'])}/{fmt(s['unanimous_and_wrong'])} | {fmt(b['wrong_together'])}/{fmt(s['wrong_together'])} | "
              f"{r['seen_helped']}/{r['seen_hurt']} (both completed: {r.get('seen_helped_completed', 0)}/{r.get('seen_hurt_completed', 0)})")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "alone_vs_seen_check", "gold_tasks": len(gold), "results": results},
                                    indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
