#!/usr/bin/env python
"""v4.2 Phase 1 prep (P1, goal section 11 items 1-4). No model call, no key, no pod. Fake backends only.

    python scripts/v42_phase1_prep.py download-tokenizers   # the ONLY step that touches the network (public files, no token)
    python scripts/v42_phase1_prep.py token-counts          # scripted dev episodes -> rendered states -> token counts + limit
    python scripts/v42_phase1_prep.py projection            # scripted records + C0 + v4.1 latencies -> PROJECTION.md

Run with HF_HUB_OFFLINE=1 and PYTHONPATH=<worktree> for everything except ``download-tokenizers``, which sets
``HF_HUB_OFFLINE=0`` and ``HF_HUB_DISABLE_IMPLICIT_TOKEN=1`` itself and passes ``token=False`` on every call. No stored token is
read; a download that needs one is skipped and logged.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "docs" / "research" / "v4.2-20261014" / "phase1"
TOKENIZER_CACHE = Path("/Volumes/WD_BLACK/cache/hf-tokenizers")
VERDICT = Path("/Volumes/WD_BLACK/antelligence-v4.2/runs/verdict1")
V41_RUNS = Path("/Volumes/WD_BLACK/antelligence-v4.1/runs")                  # read-only
THROUGHPUT = REPO / "docs" / "research" / "v4.2-20261014" / "throughput"
CASES_ROOT = Path("/Volumes/WD_BLACK/antelligence-v4.1/data_v4_1/cases")
CONFIG = REPO / "config" / "v4_2" / "world.yaml"

# Repo ids and revisions are the ones v4.1 pinned (providers/decision_backends.py; config/v4_1/model_revisions.json).
# The two Qwen3.5 ids are the FP8 repackagings v4.1 served; they carry the same tokenizer files as the base models.
TOKENIZERS: Dict[str, Tuple[str, str]] = {
    "qwen3.5-4b": ("RedHatAI/Qwen3.5-4B-FP8-dynamic", "397b7ba47a99b3221ebfc0cfc5a279118cb733ad"),
    "qwen3.5-9b": ("RedHatAI/Qwen3.5-9B-FP8-dynamic", "790f0576d2d77dd5322aa0603a470bd9e3a3d1f6"),
    "d1-3b": ("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432"),
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


# --- tokenizers --------------------------------------------------------------------------------------------------------

def tokenizer_path(name: str, cache_dir: Path = TOKENIZER_CACHE) -> Optional[str]:
    """Local path of ``tokenizer.json`` for ``name`` from the offline cache; None if it is not there."""
    repo, rev = TOKENIZERS[name]
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, "tokenizer.json", revision=rev, cache_dir=str(cache_dir), local_files_only=True, token=False)
    except Exception:
        return None


def load_tokenizers(names: Sequence[str] = tuple(TOKENIZERS)) -> Dict[str, Any]:
    from tokenizers import Tokenizer
    out = {}
    for n in names:
        p = tokenizer_path(n)
        if p:
            out[n] = Tokenizer.from_file(p)
    return out


def n_tokens(tok: Any, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def cmd_download() -> int:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import hf_hub_download
    TOKENIZER_CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log: Dict[str, Any] = {"cache_dir": str(TOKENIZER_CACHE), "token": "never used: token=False on every call, "
                           "HF_HUB_DISABLE_IMPLICIT_TOKEN=1", "results": {}}
    for name, (repo, rev) in TOKENIZERS.items():
        res: Dict[str, Any] = {"repo": repo, "revision": rev, "files": {}}
        for fn in TOKENIZER_FILES:
            try:
                p = hf_hub_download(repo, fn, revision=rev, cache_dir=str(TOKENIZER_CACHE), token=False)
                res["files"][fn] = {"status": "ok", "path": p}
            except Exception as exc:                                 # a 401/403 means it needs a token: skip and log
                res["files"][fn] = {"status": "skipped", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        res["usable"] = res["files"]["tokenizer.json"]["status"] == "ok"
        log["results"][name] = res
        print(name, "usable" if res["usable"] else "SKIPPED", file=sys.stderr)
    (OUT / "tokenizer_download.json").write_text(json.dumps(log, indent=1, sort_keys=True) + "\n")
    return 0


# --- rendered states from real scripted runs ----------------------------------------------------------------------------

def _pctile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(xs: Sequence[float]) -> Dict[str, float]:
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 2), "p50": round(_pctile(xs, 0.5), 1), "p95": round(_pctile(xs, 0.95), 1),
            "p99": round(_pctile(xs, 0.99), 1), "max": max(xs)}


def capture_states(arm: str, case_id: str, seed: int, sink: List[Dict[str, Any]]) -> None:
    """Run one scripted episode (no model) and append, per decision, the rendered colony state for a model ant."""
    import antelligence.experiments.v42_drive as D
    import antelligence.experiments.v42_harness as H
    from antelligence.worlds.tumor_mri.colony_state import with_options
    from antelligence.worlds.tumor_mri.questions import build_questions, render_state
    from antelligence.worlds.tumor_mri.colony_state import Q_MARK

    def ctrl(world, case, sd, wc, cc):
        def wrap(policy_for):
            def f(aid):
                p = policy_for(aid)

                class W:
                    def decide(self, view):
                        o = dict(view.observation)
                        o = with_options(o, world.regions, world.voxel_mm)              # the model ant's own A-F options
                        sink.append({"arm": arm, "case": case_id, "seed": seed, "tick": view.tick, "ant": aid, "fields": o})
                        return p.decide(view)

                    def describe(self):
                        return p.describe()
                return W()
            return f
        return {"wrap_policy_for": wrap}

    saved = H.ARM_CONTROLS.get(arm)
    H.register_controls(arm, ctrl)
    try:
        D.run_task({"kind": "episode", "arm": arm, "case_id": case_id, "seed": seed, "config": str(CONFIG), "board": False,
                    "cases_root": str(CASES_ROOT)})
    finally:
        if saved is None:
            H.ARM_CONTROLS.pop(arm, None)
        else:
            H.ARM_CONTROLS[arm] = saved


def cmd_token_counts(cases: Optional[Sequence[str]] = None, seeds: Sequence[int] = (3, 4)) -> int:
    import json as _json
    from antelligence.experiments.v42_drive import guard_dev_cases
    from antelligence.kernel.decision import MAX_STATE_CHARS
    from antelligence.worlds.tumor_mri.colony_state import MARK_LABELS, NOTEBOOK_CHARS
    from antelligence.worlds.tumor_mri.questions import ALL_LABELS, build_questions, render_state

    toks = load_tokenizers()
    missing = [n for n in TOKENIZERS if n not in toks]
    cases = list(cases) if cases else sorted(p.name for p in (VERDICT / "S-talk").iterdir() if p.is_dir())
    guard_dev_cases(cases)                                                    # dev cases only; evaluation ids exit here
    NATURAL = 100_000                                                         # no dropping: the natural length
    res: Dict[str, Any] = {"cases": len(cases), "seeds": list(seeds), "missing_tokenizers": missing, "arms": {}, "tokenizers": {
        n: {"repo": TOKENIZERS[n][0], "revision": TOKENIZERS[n][1]} for n in toks}}
    all_chars: Dict[str, List[int]] = {}
    for arm in ("S-solo", "S-marks", "S-talk"):
        rows: List[Dict[str, Any]] = []
        for c in cases:
            for s in seeds:
                capture_states(arm, c, s, rows)
        chars = [len(render_state(r["fields"], max_chars=NATURAL)) for r in rows]
        base = [len(render_state({k: v for k, v in r["fields"].items() if k in _V41_KEYS}, max_chars=NATURAL)) for r in rows]
        qs_chars = len(_json.dumps({q: x.wire() for q, x in build_questions(["A", "B", "C", "D", "E", "F"], with_mark=True).items()}))
        a: Dict[str, Any] = {"states": len(rows), "chars": dist(chars), "v41_part_chars": dist(base),
                             "share_over_1200_chars": round(sum(c > MAX_STATE_CHARS for c in chars) / len(chars), 4),
                             "tokens": {}}
        for n, t in toks.items():
            tk = [n_tokens(t, render_state(r["fields"], max_chars=NATURAL)) for r in rows]
            a["tokens"][n] = dist(tk)
        res["arms"][arm] = a
        all_chars[arm] = chars
        print(arm, a["chars"], file=sys.stderr)
    # notebook worst case: 300 characters of lessons
    nb = ("When two ants disagree, trust the one with readings in this zone. " * 6)[:NOTEBOOK_CHARS]
    res["notebook_300_chars_tokens"] = {n: n_tokens(t, nb) for n, t in toks.items()}
    # single-token checks: bare, after the JSON prefix, leading space
    prefix = '{"mark": "'
    labs = list(ALL_LABELS) + list(MARK_LABELS)
    chk: Dict[str, Any] = {}
    for n, t in toks.items():
        base_n = n_tokens(t, prefix)
        rows_ = {}
        for lab in MARK_LABELS:
            rows_[lab] = {"bare": n_tokens(t, lab), "after_json_quote": n_tokens(t, prefix + lab) - base_n,
                          "leading_space": n_tokens(t, " " + lab)}
        chk[n] = {"mark_labels": rows_, "all_single": all(v["bare"] == 1 and v["after_json_quote"] == 1 and v["leading_space"] == 1
                                                         for v in rows_.values()),
                  "all_labels_bare_single": all(n_tokens(t, l) == 1 for l in labs)}
    res["single_token_check"] = chk
    # proposal from the richest arm (S-talk): natural-length p95 + the 300-char notebook (zero in the verdict runs: notebooks frozen empty)
    tk = res["arms"]["S-talk"]
    p95c, maxc = tk["chars"]["p95"], tk["chars"]["max"]
    prop = int(math.ceil((p95c + NOTEBOOK_CHARS) / 100.0) * 100)
    res["proposal"] = {"basis": "S-talk natural length p95 + 300 notebook characters, rounded up to 100",
                       "chars": prop, "v41_chars": MAX_STATE_CHARS,
                       "share_of_S_talk_states_over_proposal": round(sum(c > prop for c in all_chars["S-talk"]) / len(all_chars["S-talk"]), 4)}
    res["proposal"]["tokens_p95_plus_notebook"] = {n: round(tk["tokens"][n]["p95"] + res["notebook_300_chars_tokens"][n]) for n in toks}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "token_counts.json").write_text(_json.dumps(res, indent=1, sort_keys=True) + "\n")
    (OUT / "TOKENS.md").write_text(render_tokens_md(res))
    print(_json.dumps(res["proposal"], indent=1))
    return 0


def render_tokens_md(res: Mapping[str, Any]) -> str:
    L: List[str] = []
    w = L.append
    w("# Rendered state size and the proposed limit (P1, goal 11 items 1-2)")
    w("")
    w(f"Generated by `scripts/v42_phase1_prep.py token-counts` from scripted dev episodes (verdict1 dev cases, {res['cases']} cases x seeds "
      f"{res['seeds']}): every decision of every ant is rendered as a model ant would see it (own A-F options by the engine rule, colony sentences, no brief) "
      "and counted. No model was called. Notebooks are empty in scripted runs, so the 300-character notebook is added separately.")
    w("")
    w("Tokenizers (public files only, `token=False`): " + "; ".join(f"{n} = `{v['repo']}@{v['revision'][:8]}`" for n, v in res["tokenizers"].items())
      + ". The two Qwen3.5 repositories share one tokenizer, so their counts are equal.")
    if res["missing_tokenizers"]:
        w(f"**Missing tokenizers: {res['missing_tokenizers']}** (skipped and logged in `tokenizer_download.json`).")
    w("")
    w("| Arm | states | chars p50 / p95 / max | over 1,200 chars | " + " | ".join(f"{n} tokens p50 / p95 / max" for n in res["tokenizers"]) + " |")
    w("|---|---|---|---|" + "---|" * len(res["tokenizers"]))
    for arm, a in res["arms"].items():
        c = a["chars"]
        w(f"| {arm} | {a['states']:,} | {c['p50']:.0f} / {c['p95']:.0f} / {c['max']} | {100*a['share_over_1200_chars']:.2f}% | "
          + " | ".join(f"{a['tokens'][n]['p50']:.0f} / {a['tokens'][n]['p95']:.0f} / {a['tokens'][n]['max']}" for n in res["tokenizers"]) + " |")
    w("")
    w(f"The v4.1 part alone (task, position, scan, regions) is about {res['arms']['S-talk']['v41_part_chars']['p50']:.0f} characters. "
      f"A 300-character notebook is {'/'.join(str(v) for v in res['notebook_300_chars_tokens'].values())} tokens.")
    pr = res["proposal"]
    w("")
    w("## Proposed state limit")
    w("")
    w(f"**{pr['chars']:,} characters** (v4.1: {pr['v41_chars']:,}, set for Laya, which is dropped). Basis: {pr['basis']}. "
      f"{100*pr['share_of_S_talk_states_over_proposal']:.2f}% of the S-talk states are over it even before the notebook; "
      f"p95 plus a full notebook is {pr['tokens_p95_plus_notebook']} tokens. Over the limit the renderer drops whole sentences, lowest priority first "
      "(notebook, colony note, readings, zones, marks, messages), never mid-sentence.")
    w("")
    w("Changing `kernel/decision.py` `MAX_STATE_CHARS` (it also bounds `DecisionRequest.state`) is not done here: it needs a lead-signed note. "
      "`render_state(..., max_chars=N)` already takes the limit, and the default stays 1,200 so every v4.1 request hash is unchanged.")
    w("")
    w("## Mark labels")
    w("")
    w("`mark` = `N` (none), `T` (tumour here), `R` (not tumour, ruled out), `M` (needs more drug). Tokens of each label bare, after the JSON quote the "
      "backends force (`{\"mark\": \"`), and with a leading space:")
    w("")
    w("| Tokenizer | all 12 checks = 1 token | all A-F, S, X, 1-5, yes, no, N, T, R, M bare = 1 token |")
    w("|---|---|---|")
    for n, c in res["single_token_check"].items():
        w(f"| {n} | {c['all_single']} | {c['all_labels_bare_single']} |")
    w("")
    return "\n".join(L) + "\n"


_V41_KEYS = ("local_p_tumour", "local_unc", "local_dose", "local_look", "round", "rounds", "tick", "position_vox", "goal_dist_mm",
             "target_source", "payload_fraction", "local_dose_rel", "last_release", "candidates")


# --- projection --------------------------------------------------------------------------------------------------------

def load_arm_stats(arm: str, root: Path = VERDICT) -> Dict[str, float]:
    """Mean and p95 per episode of the scripted decision turns, asks, answers and quiet-rule skips (verdict1 bundles)."""
    rows = []
    for f in sorted(glob.glob(str(root / arm / "*" / "*" / "bundle.json"))):
        b = json.load(open(f))["extra"]["colony"]
        kinds = [a["kind"] for a in b["action_log"]]
        c = b["counters"]
        rows.append({"decisions": kinds.count("decision"), "skipped": kinds.count("skipped"), "answer_turns": kinds.count("answer"),
                     "asks_sent": c["asks_sent"], "asks_answered": c["asks_answered"], "asks_refused": c["asks_refused"],
                     "answers_refused": c["answers_refused"]})
    n = len(rows)
    if not n:
        raise FileNotFoundError(f"no bundles for {arm} under {root}")
    out: Dict[str, float] = {"episodes": n}
    for k in rows[0]:
        xs = [r[k] for r in rows]
        out[k] = sum(xs) / n
        out[k + "_p95"] = _pctile(xs, 0.95)
    return out


# --- projection model -------------------------------------------------------------------------------------------------

POD_RATE_PER_H = 1.09          # RunPod $/h: the v4 pilot's pod (docs/research/v4-pilot-20261011/results.json, "1 h at $1.09/h")
LLM_QUEEN_USD = 0.004          # Nous, queen per run, 4 calls (v4 pilot preregistration section on budget)
ENGINE_S = 1.7                 # CPU seconds per scripted episode (C0 profile.json)
DECISION_ROUNDS = 24           # T=120, decision every 5 ticks (record.json: rounds 24)
N_ANTS = 8
QUEEN_CALLS = 4                # tick 0 and every 30 ticks
ASK_CAP = 96                   # budget: 1 ask per ant per 2 decision rounds = 8 x 12
EPISODES_PER_ARM = 80          # 40 dev cases x 2 seeds, as in verdict1 (an assumption: the Phase 1 plan is not fixed)
CONCURRENCY = 16               # C0: one process, 16 episodes
MODELS = ("d1", "qwen3.5-4b", "qwen3.5-9b")
MODEL_LABEL = {"d1": "d1-3B", "qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B", "frontier": "frontier (Nous)"}
BIGGER = {"d1": "qwen3.5-4b", "qwen3.5-4b": "qwen3.5-9b", "qwen3.5-9b": "frontier"}
Dist = List[Tuple[float, float]]                        # (value, weight)


def point(x: float) -> Dist:
    return [(float(x), 1.0)]


def from_quantiles(q: Sequence[float]) -> Dist:
    """101 recorded quantiles (0, 0.01 ... 1.0) as an equally weighted empirical distribution."""
    return [(float(v), 1.0 / len(q)) for v in q]


def mixture(parts: Sequence[Tuple[float, Dist]]) -> Dist:
    return [(v, w * pw) for pw, d in parts for v, w in d]


def convolve(a: Dist, b: Dist) -> Dist:
    return [(x + y, wx * wy) for x, wx in a for y, wy in b]


def cdf_at(d: Dist, x: float, _cache: Dict[int, Tuple[List[float], List[float]]] = {}) -> float:
    key = id(d)
    if key not in _cache or _cache[key][2] is not d:        # type: ignore[misc]
        s = sorted(d)
        vals, cum, tot = [v for v, _ in s], [], 0.0
        for _, w in s:
            tot += w
            cum.append(tot)
        _cache[key] = (vals, cum, d)                          # type: ignore[assignment]
    vals, cum, _ = _cache[key]                                # type: ignore[misc]
    import bisect
    i = bisect.bisect_right(vals, x)
    return 0.0 if i == 0 else min(1.0, cum[i - 1] / cum[-1])


def expected_max(dists: Sequence[Dist], active: float = 1.0) -> float:
    """E[max] of independent draws, one per entry of ``dists``; each is present with probability ``active`` (else 0 s).
    Exact for the empirical distributions: E = sum_j x_j (G(x_j) - G(x_j-)), G = product of the CDFs."""
    if not dists:
        return 0.0
    xs = sorted({0.0} | {v for d in dists for v, _ in d})
    prev, e = 0.0, 0.0
    for x in xs:
        g = 1.0
        for d in dists:
            g *= (1.0 - active) + active * cdf_at(d, x)
        e += x * (g - prev)
        prev = g
    return e


def load_latencies(path: Path = THROUGHPUT / "latency_dist.json") -> Dict[str, Dist]:
    d = json.load(open(path))["models"]
    return {k: from_quantiles(v["q"]) for k, v in d.items()}


def load_c0_efficiency(path: Path = THROUGHPUT / "results" / "concurrency-s1.0.json") -> Dict[str, float]:
    d = json.load(open(path))["inprocess"]
    return {"wall_1": d["1"]["wall_s"], "wall_16": d["16"]["wall_s"], "efficiency": d["1"]["wall_s"] / d["16"]["wall_s"],
            "episodes_per_hour_16": d["16"]["episodes_per_hour"]}


def load_round_ratio(path: Path = THROUGHPUT / "attribution.json") -> float:
    """Slowest call in a round / mean call, from v4.1 step 7 (calib1): the 9B critical path per round over the 9B mean call.
    Calls started together on one server finish together far more than independent draws would, so this is near 1."""
    a = json.load(open(path))["sets"]["calib1"]
    crit = a["critical_by_model_s_per_episode"]["qwen3.5-9b"] / DECISION_ROUNDS
    return crit / a["per_model_live_only"]["qwen3.5-9b"]["mean_s"]


def load_frontier(root: Path = V41_RUNS) -> Dict[str, float]:
    """The one live frontier episode in v4.1's plumbing run (FW arm, DeepSeek via Nous): cost, calls and wall."""
    for line in open(root / "plumbing3" / "cells.jsonl"):
        c = json.loads(line)
        if c["arm"] == "FW" and c["cost_usd"] > 0:
            calls = DECISION_ROUNDS * N_ANTS + c["queen_calls"]
            return {"cost_usd": c["cost_usd"], "calls": calls, "usd_per_call": c["cost_usd"] / calls, "wall_s": c["wall_s"],
                    "round_s": c["wall_s"] / DECISION_ROUNDS, "tokens_in": c["tokens_in"], "tokens_out": c["tokens_out"]}
    raise FileNotFoundError("no live FW cell in v4.1 plumbing3")


def pod_hours(n_episodes: int, wall_s: float, efficiency: float, concurrency: int = CONCURRENCY) -> float:
    """Batches of ``concurrency`` episodes run together; a batch takes wall/efficiency (C0: straggler loss)."""
    return math.ceil(n_episodes / concurrency) * (wall_s / efficiency) / 3600.0


class Inputs:
    def __init__(self, stats: Mapping[str, Mapping[str, float]], lat: Mapping[str, Dist], c0: Mapping[str, float],
                 frontier: Mapping[str, float], round_ratio: float = 1.0) -> None:
        self.stats, self.lat, self.c0, self.frontier, self.round_ratio = stats, lat, c0, frontier, round_ratio
        d, q = stats["S-marks"]["decisions"], stats["S-marks-quiet"]
        self.skip_share = q["skipped"] / (q["skipped"] + q["decisions"])
        talk = stats["S-talk"]
        self.unsure = talk["asks_sent"] / talk["decisions"]      # proxy: how often an ant is unsure


def mean_of(inp: "Inputs", model: str) -> float:
    return inp.frontier["round_s"] if model == "frontier" else sum(v * w for v, w in inp.lat[model])


def dist_for(inp: Inputs, model: str) -> Dist:
    return point(inp.frontier["round_s"]) if model == "frontier" else inp.lat[model]


# Arm table. ``stats`` = scripted arm whose decision/ask/answer counts stand in; ``queen``: none | scripted | llm.
ARMS: Tuple[Dict[str, Any], ...] = (
    {"name": "M-solo", "stats": "S-solo", "queen": "scripted"},
    {"name": "M-marks", "stats": "S-marks", "queen": "scripted"},
    {"name": "M-talk (scripted queen)", "stats": "S-talk", "queen": "scripted"},
    {"name": "M-escalate", "stats": "S-solo", "queen": "scripted", "escalate": True},
    {"name": "M-talk, no queen", "stats": "S-talk", "queen": "none"},
    {"name": "M-talk, LLM queen", "stats": "S-talk", "queen": "llm"},
)


def answer_phase_share(answers: float) -> float:
    """Share of the 24 rounds that have an answer phase: 1 - exp(-answers / 24), i.e. answers spread over rounds at random.
    (Scripted asks cluster, which would lower it; this is the simple upper-leaning choice.)"""
    return 1.0 - math.exp(-answers / DECISION_ROUNDS)


def project_arm(inp: Inputs, arm: Mapping[str, Any], composition: Sequence[str], quiet: bool, *, ask_scale: float = 1.0,
                ask_total: Optional[float] = None, mode: str = "batched") -> Dict[str, Any]:
    """Calls, asks, answers, wall, pod-hours and Nous dollars for ONE episode of ``arm`` with ants of ``composition``
    (8 model names). ``ask_total`` overrides the asks per episode (the sensitivity rows)."""
    st = inp.stats[arm["stats"]]
    keep = (1.0 - inp.skip_share) if quiet else 1.0
    n = len(composition)
    decisions = st["decisions"] * keep if not quiet else st["decisions"] * keep
    asks = (st["asks_sent"] if ask_total is None else ask_total) * ask_scale * keep
    answers = (st["asks_answered"] * (asks / st["asks_sent"]) if st["asks_sent"] else 0.0) if ask_total is None else asks * (st["asks_answered"] / st["asks_sent"])
    share = {m: composition.count(m) / n for m in set(composition)}
    calls = {m: decisions * s + answers * s for m, s in share.items()}
    esc = 0.0
    if arm.get("escalate"):
        esc = decisions * inp.unsure                                  # one extra call of the bigger model per unsure turn
        for m, s in share.items():
            big = BIGGER[m]
            calls[big] = calls.get(big, 0.0) + esc * s
    queen_calls = QUEEN_CALLS if arm["queen"] == "llm" else 0
    if queen_calls:
        calls["frontier"] = calls.get("frontier", 0.0) + queen_calls
    # time ------------------------------------------------------------------------------------------------------------
    # batched (headline): a round takes round_ratio x the slowest model's mean call; independent (upper bound): the exact
    # expected maximum of independent draws from the recorded distributions.
    active = keep
    cnt = {m: composition.count(m) for m in set(composition)}
    if mode == "batched":
        t_m = {}
        for m, c in cnt.items():
            t = mean_of(inp, m)
            if arm.get("escalate"):
                t += mean_of(inp, BIGGER[m]) * (1.0 - (1.0 - inp.unsure) ** c)
            t_m[m] = t
        round_s = inp.round_ratio * max(t_m.values())
        ans_s = DECISION_ROUNDS * answer_phase_share(answers) * inp.round_ratio * max(mean_of(inp, m) for m in cnt) if answers > 0 else 0.0
    elif mode == "independent":
        if arm.get("escalate"):
            ant = [mixture([(1 - inp.unsure, dist_for(inp, m)), (inp.unsure, convolve(dist_for(inp, m), dist_for(inp, BIGGER[m])))])
                   for m in composition]
        else:
            ant = [dist_for(inp, m) for m in composition]
        round_s = expected_max(ant, active)
        per_round_answers = answers / DECISION_ROUNDS
        k = max(1, int(round(per_round_answers)))
        ans_s = 0.0
        if answers > 0:
            mix = [composition[i % n] for i in range(k)]
            ans_s = DECISION_ROUNDS * answer_phase_share(answers) * expected_max([dist_for(inp, m) for m in mix])
    else:
        raise ValueError(mode)
    queen_s = queen_calls * inp.frontier["round_s"]
    wall = DECISION_ROUNDS * round_s + ans_s + queen_s + ENGINE_S
    uses_pod = any(m != "frontier" for m in calls)
    pod_models = {m for m in calls if m != "frontier"}
    usd_nous = calls.get("frontier", 0.0) * inp.frontier["usd_per_call"] if not queen_calls else \
        (calls.get("frontier", 0.0) - queen_calls) * inp.frontier["usd_per_call"] + LLM_QUEEN_USD
    ph = pod_hours(EPISODES_PER_ARM, wall, inp.c0["efficiency"]) if uses_pod else 0.0
    return {"mode": mode, "decisions": decisions, "asks": asks, "answers": answers, "escalations": esc, "queen_calls": queen_calls,
            "calls": calls, "calls_total": sum(calls.values()), "round_s": round_s, "answer_s": ans_s, "queen_s": queen_s,
            "wall_s": wall, "pod_hours": ph, "pod_usd": ph * POD_RATE_PER_H, "nous_usd": usd_nous * EPISODES_PER_ARM,
            "nous_usd_per_episode": usd_nous, "pod_models": sorted(pod_models)}


def project_frontier_reference(inp: Inputs, quiet: bool) -> Dict[str, Any]:
    """Frontier reference colony (M-talk shape, every ant on the frontier model, scripted queen)."""
    st, keep = inp.stats["S-talk"], ((1.0 - inp.skip_share) if quiet else 1.0)
    calls = (st["decisions"] + st["asks_answered"]) * keep
    r = inp.frontier["round_s"]
    wall = DECISION_ROUNDS * r + DECISION_ROUNDS * answer_phase_share(st["asks_answered"] * keep) * r + ENGINE_S
    usd = calls * inp.frontier["usd_per_call"]
    return {"decisions": st["decisions"] * keep, "asks": st["asks_sent"] * keep, "answers": st["asks_answered"] * keep, "escalations": 0.0,
            "queen_calls": 0, "calls": {"frontier": calls}, "calls_total": calls, "round_s": r, "answer_s": wall - DECISION_ROUNDS * r - ENGINE_S,
            "queen_s": 0.0, "wall_s": wall, "pod_hours": 0.0, "pod_usd": 0.0, "nous_usd": usd * EPISODES_PER_ARM, "nous_usd_per_episode": usd,
            "pod_models": []}


MIXED = ["d1"] * 3 + ["qwen3.5-4b"] * 3 + ["qwen3.5-9b"] * 2


def fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def cmd_projection() -> int:
    stats = {a: load_arm_stats(a) for a in ("S-solo", "S-marks", "S-talk", "S-marks-quiet")}
    inp = Inputs(stats, load_latencies(), load_c0_efficiency(), load_frontier(), load_round_ratio())
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for quiet in (False, True):
            for base in MODELS:
                r = project_arm(inp, arm, [base] * N_ANTS, quiet)
                rows.append({"arm": arm["name"], "quiet": quiet, "colony": base, **r})
    for quiet in (False, True):
        rows.append({"arm": "Frontier reference (M-talk shape)", "quiet": quiet, "colony": "frontier", **project_frontier_reference(inp, quiet)})
        rows.append({"arm": "Mixed colony 3 d1 + 3 Qwen3.5-4B + 2 Qwen3.5-9B (M-talk)", "quiet": quiet, "colony": "mixed",
                     **project_arm(inp, ARMS[2], MIXED, quiet)})
    bound = []
    for arm in (ARMS[0], ARMS[2]):
        for base in MODELS:
            bound.append({"arm": arm["name"], "colony": base, **project_arm(inp, arm, [base] * N_ANTS, False, mode="independent")})
    sens = []
    for label, tot in (("asks x0.5 of scripted", stats["S-talk"]["asks_sent"] * 0.5), ("scripted asks", stats["S-talk"]["asks_sent"]),
                       (f"ask budget saturated ({ASK_CAP}, the ceiling)", float(ASK_CAP))):
        r = project_arm(inp, ARMS[2], ["qwen3.5-9b"] * N_ANTS, False, ask_total=tot)
        sens.append({"label": label, **r})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "projection.json").write_text(json.dumps({"rows": rows, "sensitivity": sens, "independent_bound": bound, "inputs": {
        "stats": stats, "skip_share": inp.skip_share, "unsure": inp.unsure, "c0": inp.c0, "frontier": inp.frontier,
        "pod_rate_per_h": POD_RATE_PER_H, "episodes_per_arm": EPISODES_PER_ARM}}, indent=1, sort_keys=True, default=float) + "\n")
    (OUT / "PROJECTION.md").write_text(render_projection_md(inp, rows, sens, bound))
    print(f"wrote {OUT/'PROJECTION.md'}", file=sys.stderr)
    return 0


def render_projection_md(inp: Inputs, rows: Sequence[Mapping[str, Any]], sens: Sequence[Mapping[str, Any]],
                         bound: Sequence[Mapping[str, Any]]) -> str:
    st, c0, fr = inp.stats, inp.c0, inp.frontier
    L: List[str] = []
    w = L.append
    w("# Phase 1 projection (P1, goal 11 item 4)")
    w("")
    w("Generated by `scripts/v42_phase1_prep.py projection`. No model was called. Every input below is read from a file on disk; every "
      "assumption is labelled **[A#]**. The numbers are projections from scripted ants, not predictions of what model ants will do.")
    w("")
    w("## 1. Inputs, with source")
    w("")
    w("| Input | Value | Source |")
    w("|---|---|---|")
    for arm in ("S-solo", "S-marks", "S-talk", "S-marks-quiet"):
        s = st[arm]
        w(f"| scripted `{arm}`, per episode ({int(s['episodes'])} runs) | decision turns {fmt(s['decisions'],2)}, skipped {fmt(s['skipped'],2)}, "
          f"asks sent {fmt(s['asks_sent'],2)}, answered {fmt(s['asks_answered'],2)}, refused {fmt(s['asks_refused'],2)} | "
          f"`/Volumes/WD_BLACK/antelligence-v4.2/runs/verdict1/{arm}/*/*/bundle.json` (`action_log`, `counters`) |")
    w(f"| quiet-rule skip share | {fmt(100*inp.skip_share,2)}% = {fmt(st['S-marks-quiet']['skipped'],2)} / ({fmt(st['S-marks-quiet']['skipped'],2)} + "
      f"{fmt(st['S-marks-quiet']['decisions'],2)}) | `S-marks-quiet` runs, same source |")
    w(f"| unsure share (proxy) | {fmt(inp.unsure,3)} = asks sent {fmt(st['S-talk']['asks_sent'],2)} / decision turns {fmt(st['S-talk']['decisions'],2)} | `S-talk` runs |")
    for m in MODELS:
        d = inp.lat[m]
        mean = sum(v * wt for v, wt in d)
        s = sorted(v for v, _ in d)
        w(f"| per-call latency, {MODEL_LABEL[m]} | mean {fmt(mean,2)} s, p50 {fmt(s[50],2)} s, p95 {fmt(s[95],2)} s | "
          f"`docs/research/v4.2-20261014/throughput/latency_dist.json` (v4.1 calib1 live, non-cached calls; derived from "
          f"`/Volumes/WD_BLACK/antelligence-v4.1/runs/calib1/runs/*/decisions.jsonl`) |")
    w(f"| C0 concurrency | 1 episode {fmt(c0['wall_1'],0)} s; 16 together {fmt(c0['wall_16'],0)} s; efficiency {fmt(c0['efficiency'],3)} "
      f"({fmt(c0['episodes_per_hour_16'],1)} episodes/h at 16) | `throughput/results/concurrency-s1.0.json` |")
    w(f"| round ratio | {fmt(inp.round_ratio,3)} = (9B critical path {fmt(inp.round_ratio * DECISION_ROUNDS * 13.819,1)} s / 24 rounds) / 9B mean call 13.82 s | "
      "`throughput/attribution.json`, set `calib1` (`critical_by_model_s_per_episode`, `per_model_live_only`) |")
    w(f"| C0 engine time | {ENGINE_S} s CPU per scripted episode | `throughput/analysis.md` |")
    w(f"| frontier episode (Nous, DeepSeek V4.1 Flash) | ${fr['cost_usd']:.5f} for {int(fr['calls'])} calls "
      f"({DECISION_ROUNDS * N_ANTS} decisions + {QUEEN_CALLS} queen), wall {fmt(fr['wall_s'],1)} s, {int(fr['tokens_in']):,} in / {int(fr['tokens_out']):,} out tokens "
      f"| `/Volumes/WD_BLACK/antelligence-v4.1/runs/plumbing3/cells.jsonl` (FW cell) |")
    w(f"| pod rate | ${POD_RATE_PER_H}/h | v4 pilot `results.json` ('1 h at $1.09/h'); **[A1]** the Phase 1 pod costs the same |")
    w(f"| LLM queen | ${LLM_QUEEN_USD} per episode (4 calls) | v4 pilot preregistration budget section |")
    w("")
    w("## 2. Assumptions")
    w("")
    for t in (
        "**[A1]** Pod: one pod serves all three models at the rate above. v4.1 also ran d1 on the pod (hardware L40S in `decisions.jsonl`).",
        f"**[A2]** Plan size: {EPISODES_PER_ARM} episodes per arm (40 dev cases x 2 seeds, as in verdict1). The Phase 1 plan is not fixed; "
        "scale linearly (pod-hours scale in steps of 16 episodes).",
        "**[A3]** One model call per ant per decision turn; the four questions (target, release, amount, mark) share the call, as in v4.1. "
        "The `mark` question adds about one output token. A skipped turn (quiet rule) makes no call.",
        "**[A4]** An *ask* costs no call: the trigger is computed from the asker's own answer. An *answer* is one `noul` call to the receiver "
        "(only if it has a reading within 6 mm; otherwise `unknown` and no call, which this projection ignores: it counts every answered ask as a call, "
        "an upper bound).",
        "**[A5]** Model ants ask and answer as often as the scripted ants (`S-talk`: asks sent / answered above). This is the weakest assumption: "
        "the real rate depends on the thresholds fitted in Phase 1. Section 5 gives 0.5x, 2x and the budget ceiling.",
        "**[A6]** Quiet rule: model ants skip the same 4.3% of turns the scripted ants skip in `S-marks-quiet`; asks and answers fall in proportion.",
        "**[A7]** Time model. The calls of one round are started together on one server and finish close together: in v4.1 step 7 the slowest 9B call of a "
        "round was only 1.057 x the mean 9B call (round ratio above), far below what 8 independent draws would give. Headline: round time = ratio x the mean call of "
        "the slowest model in the colony (assumes the same ratio holds for d1 and 4B, which step 7 cannot show, and for a pod less loaded than in step 7: Phase 1 makes "
        "1 call per ant turn, step 7 made 3). Section 4b gives the opposite bound, the exact expected maximum of independent draws. "
        "Prompts are longer (state about 280 against 140 tokens); prefill time is assumed negligible.",
        "**[A8]** An answer call takes as long as a decision call of the same model; the answers of one round are batched (one round-ratio wait). A round has an "
        "answer phase with probability 1 - exp(-answers / 24) (answers spread over rounds at random).",
        "**[A9]** Escalation (M-escalate): an unsure turn (probability = the unsure share above) makes a second call to the next bigger model, in "
        "sequence after the first (d1 to 4B, 4B to 9B, 9B to frontier); no sharing. A round has one such second call with probability 1 - (1 - u)^c for c ants on that model.",
        "**[A10]** A frontier call costs what the one live v4.1 frontier episode cost per call and a frontier round takes that episode's "
        f"wall / {DECISION_ROUNDS} = {fmt(fr['round_s'],1)} s. A queen call takes one frontier round. The prompts in the colony are shorter than FW's "
        "(about 1,130 tokens per call there), so the dollars are probably high.",
        "**[A11]** Episodes run together 16 at a time in one process (C0); a batch takes wall / efficiency, and the pod bills for the whole batch.",
        "**[A12]** The scripted queen and the engine cost the C0 engine time only; the LLM queen costs 4 frontier calls (wall) and the pilot's $0.004.",
        "**[A13]** Tokenizer counts: state p50/p95/max are in `token_counts.json`; they do not enter this table.",
    ):
        w(f"- {t}")
    w("")
    w("## 3. Model calls, asks and answers per episode")
    w("")
    w("Counts do not depend on the model colony (M-escalate and the queen rows add calls to other models). Shown for a Qwen3.5-9B colony.")
    w("")
    w("| Arm | quiet | decisions | asks | answers | escalation calls | queen calls | total calls |")
    w("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["colony"] in ("qwen3.5-9b", "frontier", "mixed"):
            w(f"| {r['arm']} | {'on' if r['quiet'] else 'off'} | {fmt(r['decisions'])} | {fmt(r['asks'])} | {fmt(r['answers'])} | "
              f"{fmt(r['escalations'])} | {r['queen_calls']} | {fmt(r['calls_total'])} |")
    w("")
    w("Arithmetic (M-talk, quiet off): decisions = 24 rounds x 8 ants = 192; asks = 63.52 (scripted); answers = 63.36 (each answered ask is one `noul` "
      "call); total = 192 + 63.36 = 255.36. With the quiet rule: x (1 - 0.0430) -> decisions 183.75, asks 60.8, answers 60.6, total 244.4. "
      "M-escalate: 192 + 0.331 x 192 = 255.5 calls, 63.5 of them to the bigger model.")
    w("")
    w("## 4. Wall time, pod-hours, dollars")
    w("")
    w(f"Per episode: wall time (s). Per {EPISODES_PER_ARM} episodes: pod-hours, pod dollars, Nous dollars. "
      "Each cell is `wall s / pod-h / pod $ / Nous $`. Column = the model every ant runs on.")
    w("")
    w("| Arm | quiet | d1-3B | Qwen3.5-4B | Qwen3.5-9B |")
    w("|---|---|---|---|---|")
    by: Dict[Tuple[str, bool], Dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["arm"], r["quiet"]), {})[r["colony"]] = r
    cell = lambda r: f"{fmt(r['wall_s'],0)} / {fmt(r['pod_hours'],2)} / {fmt(r['pod_usd'],2)} / {fmt(r['nous_usd'],2)}"
    for (arm, quiet), d in by.items():
        if all(m in d for m in MODELS):
            w(f"| {arm} | {'on' if quiet else 'off'} | " + " | ".join(cell(d[m]) for m in MODELS) + " |")
    w("")
    w("| Arm | quiet | wall s / pod-h / pod $ / Nous $ |")
    w("|---|---|---|")
    for (arm, quiet), d in by.items():
        for k in ("frontier", "mixed"):
            if k in d:
                w(f"| {arm} | {'on' if quiet else 'off'} | {cell(d[k])} |")
    w("")
    ex = next(r for r in rows if r["arm"] == "M-talk (scripted queen)" and r["colony"] == "qwen3.5-9b" and not r["quiet"])
    w("### Worked example: M-talk, Qwen3.5-9B colony, quiet off")
    w("")
    m9 = mean_of(inp, "qwen3.5-9b")
    w(f"1. Decision round = round ratio x mean 9B call = {fmt(inp.round_ratio,3)} x {fmt(m9,2)} s = {fmt(ex['round_s'],2)} s.")
    w(f"2. 24 rounds x {fmt(ex['round_s'],2)} s = {fmt(24*ex['round_s'],1)} s.")
    w(f"3. Answer phase: {fmt(ex['answers'],1)} answers / 24 rounds = {fmt(ex['answers']/24,2)} per round, so a round has one with probability "
      f"1 - exp(-{fmt(ex['answers']/24,2)}) = {fmt(answer_phase_share(ex['answers']),3)}; 24 x {fmt(answer_phase_share(ex['answers']),3)} x {fmt(inp.round_ratio,3)} x {fmt(m9,2)} s = {fmt(ex['answer_s'],1)} s.")
    w(f"4. Engine {ENGINE_S} s. Episode wall = {fmt(24*ex['round_s'],1)} + {fmt(ex['answer_s'],1)} + {ENGINE_S} = {fmt(ex['wall_s'],1)} s.")
    w(f"5. {EPISODES_PER_ARM} episodes at 16 together = {math.ceil(EPISODES_PER_ARM/16)} batches, each {fmt(ex['wall_s'],1)} / {fmt(inp.c0['efficiency'],3)} = "
      f"{fmt(ex['wall_s']/inp.c0['efficiency'],1)} s; pod-hours = {math.ceil(EPISODES_PER_ARM/16)} x {fmt(ex['wall_s']/inp.c0['efficiency'],1)} / 3600 = {fmt(ex['pod_hours'],3)} h; "
      f"x ${POD_RATE_PER_H} = ${fmt(ex['pod_usd'],2)}.")
    solo = next(r for r in rows if r["arm"] == "M-solo" and r["colony"] == "qwen3.5-9b" and not r["quiet"])
    w("")
    w(f"Check against v4.1: M-solo in a 9B colony projects {fmt(solo['wall_s'],0)} s per episode. v4.1 step 7 measured a mean wall of 479 s per episode, of "
      "which 379 s was model wait on the critical path (all three models called each turn) and 101 s was unlogged (d1 looks, engine); "
      "v4.2 has no d1 look calls, so the projection should sit near the 379 s, not near 479 s.")
    w("")
    w("### 4b. Upper bound: calls in a round independent")
    w("")
    w("Same arithmetic with the exact expected maximum of 8 independent draws from the recorded latency distribution (answers: the expected maximum of the "
      "answers in a round). Not expected to be reached; it shows how much the batching assumption [A7] is worth.")
    w("")
    w("| Arm (quiet off) | d1-3B | Qwen3.5-4B | Qwen3.5-9B |")
    w("|---|---|---|---|")
    for an in ("M-solo", "M-talk (scripted queen)"):
        cells = []
        for m in MODELS:
            r = next(x for x in bound if x["arm"] == an and x["colony"] == m)
            h = next(x for x in rows if x["arm"] == an and x["colony"] == m and not x["quiet"])
            cells.append(f"{fmt(r['wall_s'],0)} s / {fmt(r['pod_hours'],2)} h (headline {fmt(h['wall_s'],0)} s / {fmt(h['pod_hours'],2)} h)")
        w(f"| {an} | " + " | ".join(cells) + " |")
    w("")
    w("## 5. Sensitivity to the ask rate [A5]")
    w("")
    w("M-talk, Qwen3.5-9B colony, quiet off.")
    w("")
    w("| Ask rate | asks | answers | total calls | wall s | pod-h per 80 episodes |")
    w("|---|---|---|---|---|---|")
    for s in sens:
        w(f"| {s['label']} | {fmt(s['asks'])} | {fmt(s['answers'])} | {fmt(s['calls_total'])} | {fmt(s['wall_s'],0)} | {fmt(s['pod_hours'],2)} |")
    w("")
    w("## 6. What to take from it")
    w("")
    base = lambda a, m: by[(a, False)][m]
    w(f"- Wall time is set by the slowest model in each round, so it moves with the model (9B round {fmt(base('M-solo','qwen3.5-9b')['round_s'],1)} s, "
      f"4B {fmt(base('M-solo','qwen3.5-4b')['round_s'],1)} s, d1 {fmt(base('M-solo','d1')['round_s'],1)} s) and barely with the number of calls. "
      "The quiet rule cuts calls by about 4% and wall time by a small fraction of that (a round still waits for its slowest call).")
    w("- Talking adds the answer turns, which are sequential with the decision rounds: that is where M-talk pays in wall time, not in dollars.")
    w("- Nous dollars are zero everywhere except the LLM queen, M-escalate in a 9B colony (escalates to frontier), and the frontier reference.")
    w("- A round waits for its slowest call, so the quiet rule (which removes calls, not rounds) barely moves wall time; it moves calls, tokens and dollars.")
    w("- The pod bills in batches of 16: 80 episodes is 5 batches, so a 17th episode costs a whole extra batch.")
    w("")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("download-tokenizers", "token-counts", "projection"))
    a = ap.parse_args(argv)
    if a.cmd == "download-tokenizers":
        return cmd_download()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if a.cmd == "token-counts":
        return cmd_token_counts()
    return cmd_projection()


if __name__ == "__main__":
    raise SystemExit(main())
