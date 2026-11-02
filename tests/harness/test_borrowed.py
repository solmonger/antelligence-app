"""Tests for the five borrowed switches (stop rule, confidence gate, router, cache-aware count, prompt search), the
answer-odds plumbing under them, and the script that scores them. $0, no network."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path

import httpx
import pytest

from antelligence.experiments import fair_test as ft
from antelligence.experiments import fresh_questions as fq
from antelligence.experiments import shared_read as sr
from antelligence.harness import borrowed as bw
from antelligence.harness import learner as ln
from antelligence.harness import promptopt as po
from antelligence.harness import scorecard as sc
from antelligence.harness.core import Step, Task
from antelligence.harness.fake_odds import OddsFakeProvider, fake_logprobs
from antelligence.harness.planner import sweep_batch
from antelligence.harness.recipes import Ctx
from antelligence.harness.tools import AntInfo, LiveBox, OddsRunner, ReplayBox, ant_tool
from antelligence.providers import logprobs as lp
from antelligence.providers.base import ChatRequest
from antelligence.providers.openai_compat import OpenAICompatProvider

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "docs/research/fresh-20261031/models.json"
CFG = sr.load_config(MODELS)
TASK = Task("t1", "", "Which one?", ("(A) one", "(B) two", "(C) three", "(D) four"), "exam")
ROSTER = tuple(AntInfo(i, f"m{i}", "f", 0, float(i + 1)) for i in range(4))     # ant0 is the cheapest


def _features_script():
    spec = importlib.util.spec_from_file_location("_t_harness_features", REPO / "scripts/harness_features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def dev(seed: int = 3, per_task: int = 2):
    qs, ans, _ = fq.make_set(seed, per_task)
    return [Task.from_raw(q) for q in qs], {a["id"]: a["label"] for a in ans}


def replay(answers, sure=None, reader="A", code=None):
    """A ReplayBox for one task: ants' answers (and sureness) in order, the reader's answer, a code check's passes."""
    per = {ant_tool(i, "alone"): Step(ant_tool(i, "alone"), a, 100, 10, 1.0, weight=ROSTER[i].weight,
                                      sure=(sure[i] if sure else None)) for i, a in enumerate(answers)}
    per["reader"] = Step("reader", reader, 500, 50, 5.0, weight=10.0)
    if code is not None:
        per["code"] = Step("code", code[0], 900, 300, 9.0, passes=code[1])
    return ReplayBox({TASK.id: per}, ROSTER)


async def run(recipe, box):
    ctx = Ctx(box, TASK)
    got = await recipe.run(ctx)
    return got, [s.tool for s in ctx.steps]


# ------------------------------------------------------------------ 1. stop rule

def test_beta_conf_matches_adaptive_consistency_values():
    assert bw.beta_conf(2, 0) == pytest.approx(0.875)
    assert bw.beta_conf(3, 0) == pytest.approx(0.9375)
    assert bw.beta_conf(3, 1) == pytest.approx(0.8125)
    assert bw.beta_conf(4, 0) == pytest.approx(31 / 32)
    assert bw.beta_conf(1, 1) == 0.0
    assert bw.beta_conf(1, 3) == pytest.approx(1 - bw.beta_conf(3, 1))
    # it is the regularized incomplete beta I_0.5(v2 + 1, v1 + 1), checked by numeric integration
    for v1, v2 in ((5, 1), (6, 2), (3, 2)):
        n, a, b = 20000, v2 + 1, v1 + 1
        xs = [(k + 0.5) / n for k in range(n)]
        f = [x ** (a - 1) * (1 - x) ** (b - 1) for x in xs]
        whole = sum(f) / n
        part = sum(v for x, v in zip(xs, f) if x < 0.5) / n
        assert bw.beta_conf(v1, v2) == pytest.approx(part / whole, abs=1e-3)


def test_seq_beta_stops_as_soon_as_it_is_sure():
    r = bw.SeqBeta("ac", tuple(ant_tool(i, "alone") for i in range(4)), "alone", 0.85, None, 4, (0, 1, 2, 3))
    got, asked = asyncio.run(run(r, replay(["B", "B", "C", "C"])))
    assert got == "B" and asked == ["ant0:alone", "ant1:alone"]        # 2-0 is 0.875 >= 0.85
    r9 = bw.SeqBeta("ac", r.tools, "alone", 0.9, None, 4, (0, 1, 2, 3))
    got, asked = asyncio.run(run(r9, replay(["B", "B", "B", "C"])))
    assert got == "B" and len(asked) == 3                              # 3-0 is 0.9375


def test_seq_beta_hands_up_when_it_cannot_get_sure():
    tools = tuple(ant_tool(i, "alone") for i in range(4)) + ("reader",)
    r = bw.SeqBeta("ac>reader", tools, "alone", 0.95, "reader", 4, (0, 1, 2, 3))
    got, asked = asyncio.run(run(r, replay(["A", "B", "C", "D"], reader="C")))
    # after 2 split answers even 2 more agreeing (3-1, 0.81) can't reach 0.95, so it stops asking ants
    assert got == "C" and asked == ["ant0:alone", "ant1:alone", "reader"]


def test_escalation_to_the_code_checker_votes_inside_its_passes():
    tools = tuple(ant_tool(i, "alone") for i in range(4)) + ("code",)
    r = bw.SeqBeta("ac>code", tools, "alone", 0.95, "code", 4, (0, 1, 2, 3))
    got, _ = asyncio.run(run(r, replay(["A", "B", "C", "D"], code=(None, ("B", "C")))))
    assert got in ("B", "C")


def test_stop_recipes_are_named_and_fit():
    rs = bw.stop_recipes(ROSTER)
    names = {r.name for r in rs}
    assert "ac:alone:c90" in names and "ac:notes:c80>reader" in names and "ac:both:c95>code" in names
    assert len(rs) == 3 * 3 * 3


# ------------------------------------------------------------------ 2. confidence gate

def test_gate_takes_a_sure_cheap_answer_and_escalates_an_unsure_one():
    g = bw.Gate("g", (ant_tool(0, "alone"), "reader"), "alone", 0.8, 1, "reader", (0, 1, 2, 3))
    got, asked = asyncio.run(run(g, replay(["B", "C", "C", "C"], sure=[0.9, 0.5, 0.5, 0.5], reader="A")))
    assert got == "B" and asked == ["ant0:alone"]
    got, asked = asyncio.run(run(g, replay(["B", "C", "C", "C"], sure=[0.6, 0.5, 0.5, 0.5], reader="A")))
    assert got == "A" and asked == ["ant0:alone", "reader"]


def test_gate_with_more_tries_and_no_escalation_keeps_the_surest():
    g = bw.Gate("g", tuple(ant_tool(i, "alone") for i in range(3)), "alone", 0.95, 3, None, (0, 1, 2, 3))
    got, asked = asyncio.run(run(g, replay(["B", "C", "D", "A"], sure=[0.6, 0.9, 0.7, 0.99])))
    assert got == "C" and len(asked) == 3
    # an unknown sureness never passes the gate
    got, asked = asyncio.run(run(g, replay(["B", "C", "D", "A"])))
    assert asked == ["ant0:alone", "ant1:alone", "ant2:alone"]


def test_conf_vote_weighs_votes_by_sureness():
    ants = tuple(ant_tool(i, "alone") for i in range(4))
    v = bw.ConfVote("cv", ants, "alone", 0.6, None, 2, (0, 1, 2, 3))
    # B, C, C by count; but B was very sure and the C's not: 0.99 vs 0.2 + 0.2
    got, asked = asyncio.run(run(v, replay(["B", "C", "C", "C"], sure=[0.99, 0.2, 0.2, 0.2])))
    assert got == "B" and len(asked) == 2             # after 2 answers B holds 0.99 of 1.19 = 83% of the weight
    v2 = bw.ConfVote("cv", ants, "alone", 0.9, None, 2, (0, 1, 2, 3))
    got, asked = asyncio.run(run(v2, replay(["B", "B", "C", "C"], sure=[0.9, 0.9, 0.5, 0.5])))
    assert got == "B" and len(asked) == 2              # 2 answers, all for B: stop


def test_conf_vote_escalates_when_it_cannot_reach_its_share():
    tools = tuple(ant_tool(i, "alone") for i in range(4)) + ("reader",)
    v = bw.ConfVote("cv>reader", tools, "alone", 0.95, "reader", 2, (0, 1, 2, 3))
    got, asked = asyncio.run(run(v, replay(["A", "B", "C", "D"], sure=[1.0, 1.0, 1.0, 1.0], reader="D")))
    assert got == "D" and asked[-1] == "reader" and len(asked) < 5


def test_gate_recipes_and_has_sure():
    names = {r.name for r in bw.gate_recipes(ROSTER)}
    assert "gate:alone:p85:t2>reader" in names and "cvote:alone:c90" in names and "cvote:alone:c95>reader" in names
    assert bw.has_sure(replay(["A"] * 4, sure=[0.5] * 4).steps)
    assert not bw.has_sure(replay(["A"] * 4).steps)


# ------------------------------------------------------------------ answer odds plumbing

MLX_LIKE = {"content": [
    {"token": '{"', "logprob": -0.01, "top_logprobs": [{"token": '{"', "logprob": -0.01}, {"token": "{", "logprob": -5}]},
    {"token": "answer", "logprob": -0.1, "top_logprobs": [{"token": "answer", "logprob": -0.1}]},
    {"token": 'Ġ"', "logprob": -0.05, "top_logprobs": [{"token": 'Ġ"', "logprob": -0.05}]},
    # the answer slot, alternatives in no order and the entry not the sampled one, as mlx_lm.server sends them
    {"token": "D", "logprob": math.log(0.05), "top_logprobs": [
        {"token": "D", "logprob": math.log(0.05)}, {"token": "B", "logprob": math.log(0.7)},
        {"token": "A", "logprob": math.log(0.2)}, {"token": "ĠC", "logprob": math.log(0.05)}]},
    {"token": '"}', "logprob": -0.001, "top_logprobs": [{"token": '"}', "logprob": -0.001}]}]}


def test_summarize_reads_the_answer_slot_from_unsorted_alternatives():
    s = lp.summarize(MLX_LIKE)
    assert s["n"] == 5 and s["text"] is True
    assert set(s["slot"]) == {"A", "B", "C", "D"}
    assert lp.letter_sure(s, "B") == pytest.approx(0.7, abs=1e-6)
    assert lp.letter_sure(s, "E") == 0.0                 # slot found, answer not in its top letters
    assert lp.letter_sure(s, None) is None and lp.letter_sure(None, "B") is None
    assert s["conf"] > 0 and s["low_group"] <= s["conf"] + 1e-9


def test_summarize_older_mlx_format_has_no_letter_odds():
    old = {"token_logprobs": [-0.1, -0.2], "tokens": [5, 6], "top_logprobs": [{"5": -0.1, "9": -3}, {"6": -0.2}]}
    s = lp.summarize(old)
    assert s["n"] == 2 and s["text"] is False and s["slot"] is None
    assert lp.letter_sure(s, "A") is None
    assert lp.summarize({}) is None and lp.summarize(None) is None


def test_norm_token_and_cached_tokens():
    assert lp.norm_token('Ġ"C') == "C" and lp.norm_token("▁(B)") == "B"
    assert lp.cached_tokens({"prompt_tokens_details": {"cached_tokens": 12}}) == 12
    assert lp.cached_tokens({"prompt_tokens": 3}) is None and lp.cached_tokens(None) is None
    assert lp.cached_tokens({"prompt_tokens_details": {"cached_tokens": True}}) is None


def test_openai_compat_keeps_cached_tokens_and_an_odds_summary():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": '{"answer": "B"}'},
                                                                    "finish_reason": "stop", "logprobs": MLX_LIKE}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": 5,
                                                   "prompt_tokens_details": {"cached_tokens": 40}}})
    prov = OpenAICompatProvider("https://example.invalid/v1", api_key="k-test", require_model_match=False,
                                transport=httpx.MockTransport(handler))
    req = ChatRequest(model="m", messages=({"role": "user", "content": "hi"},),
                      extra_body={"logprobs": True, "top_logprobs": 8})
    resp = asyncio.run(prov.complete(req))
    assert seen["logprobs"] is True and seen["top_logprobs"] == 8
    assert resp.extra["cached_tokens"] == 40
    assert lp.letter_sure(resp.extra["logprobs"], "B") == pytest.approx(0.7, abs=1e-6)


def test_fake_logprobs_round_trip():
    raw = fake_logprobs('{"reason": "x", "answer": "C"}', "C", 0.83, "ABCD")
    s = lp.summarize(raw, "ABCD")
    assert lp.letter_sure(s, "C") == pytest.approx(0.83, abs=1e-6)


def _odds_box(tmp_path, tasks, key, logprobs=8, name="c.jsonl", ant_messages=None):
    skill = {CFG.reader.tag: 0.8, **{s.tag: 0.5 for s in CFG.ants}}
    prov = OddsFakeProvider(tasks, key, skill, {s.tag: s.gb for s in [CFG.reader, *CFG.ants]})
    runner = OddsRunner(prov, tmp_path / name, concurrency=2, check_names=False, max_fail_streak=10 ** 6)
    return LiveBox(runner, CFG.reader, CFG.ants, parser=CFG.parser, code=False, logprobs=logprobs,
                   ant_messages=ant_messages), prov


def test_livebox_asks_for_odds_and_keeps_sureness_and_cache_counts(tmp_path):
    tasks, key = dev()
    box, prov = _odds_box(tmp_path, tasks, key)
    assert box.ant_spec(0).extra_body["top_logprobs"] == 8
    steps = asyncio.run(sweep_batch(box, tasks[:3], [ant_tool(i, "alone") for i in range(len(box.roster))],
                                    order="question"))
    ants = [s for per in steps.values() for s in per.values()]
    assert ants and all(s.sure is not None and 0 <= s.sure <= 1 for s in ants if s.answer)
    assert all("lp" in s.payload and isinstance(s.payload["cached_in"], int) for s in ants)
    # the second way of looking at a question, same model, was mostly read from the cache
    by = {(a.tag, a.lens): a.index for a in box.roster}
    tag = box.roster[0].tag
    lenses = sorted(a.lens for a in box.roster if a.tag == tag)
    second = steps[tasks[0].id][ant_tool(by[(tag, lenses[1])], "alone")]
    assert second.payload["cached_in"] > 0.5 * second.tokens_in
    off, _ = _odds_box(tmp_path, tasks, key, logprobs=0, name="d.jsonl")
    assert off.ant_spec(0) is off.specs[0]


def test_sweep_batch_question_order_keeps_a_questions_calls_together(tmp_path):
    tasks, key = dev()
    box, prov = _odds_box(tmp_path, tasks, key)
    order = []
    real = prov.complete

    async def spy(req):
        order.append(req.messages[-1]["content"][:400])
        return await real(req)
    prov.complete = spy
    tools = [ant_tool(i, "alone") for i in range(len(box.roster))]
    asyncio.run(sweep_batch(box, tasks[:4], tools, order="question", workers=1))
    per_model = len([a for a in box.roster if a.tag == box.roster[0].tag])
    first = order[:per_model]
    assert len(set(q.split("How to think")[0] for q in first)) == 1     # one question's ways of looking, together
    with pytest.raises(ValueError):
        asyncio.run(sweep_batch(box, tasks[:1], tools, order="random"))


# ------------------------------------------------------------------ 4. cache-aware count

def test_fresh_in_counts_a_shared_start_once(tmp_path):
    tasks, key = dev()
    box, _ = _odds_box(tmp_path, tasks, key)
    steps = asyncio.run(sweep_batch(box, tasks[:2], [ant_tool(i, "alone") for i in range(len(box.roster))]))
    prompts = bw.Prompts(box.roster, CFG.reader, {s.tag: s for s in CFG.ants})
    rate = bw.token_rates(tasks[:2], steps, prompts)
    assert rate and all(0.1 < r < 1.0 for r in rate.values())
    t = tasks[0]
    tag = box.roster[0].tag
    same = [a.index for a in box.roster if a.tag == tag]
    seq = [steps[t.id][ant_tool(i, "alone")] for i in same]
    fresh = bw.fresh_in(seq, t, prompts, rate)
    assert fresh[0] == seq[0].tokens_in                   # the first ask reads everything
    assert fresh[1] < 0.5 * seq[1].tokens_in              # the second shares the question
    other = [a.index for a in box.roster if a.tag != tag][0]
    two = [steps[t.id][ant_tool(same[0], "alone")], steps[t.id][ant_tool(other, "alone")]]
    assert bw.fresh_in(two, t, prompts, rate)[1] == two[1].tokens_in      # another model shares nothing
    # the server's own count is used only when asked for (a sweep's real order), and then it wins
    measured = bw.fresh_in(seq, t, prompts, rate, measured=True)
    assert measured[1] == seq[1].tokens_in - seq[1].payload["cached_in"]


def test_reask_gets_its_first_prompt_back():
    tasks, key = dev()
    t = tasks[0]
    prompts = bw.Prompts(ROSTER)
    once = Step(ant_tool(0, "alone"), "A", 300, 10, 1.0)
    twice = Step(ant_tool(0, "alone"), "A", 600, 20, 2.0, calls=2)
    rate = {"m0": 0.25}
    assert bw.fresh_in([once], t, prompts, rate) == [300]
    assert bw.fresh_in([twice], t, prompts, rate)[0] < 600


# ------------------------------------------------------------------ 3. router

def test_knn_router_follows_the_most_similar_questions():
    mk = lambda i, words, kind: Task(f"q{i}", "", words, ("(A) x", "(B) y"), kind)  # noqa: E731
    train = [mk(i, "apple banana cherry fruit salad", "exam") for i in range(6)] + \
            [mk(10 + i, "engine piston cylinder motor oil", "exam") for i in range(6)]
    o = lambda t, r, right, tok: sc.Outcome(t, "exam", "exam", r, "A", right, tok, float(tok), 1.0, 0.0, 1)  # noqa
    table = {"reader": {}, "cheap": {}}
    for t in train:
        fruit = t.stem.startswith("apple")
        table["reader"][t.id] = o(t.id, "reader", 1.0, 1000)
        table["cheap"][t.id] = o(t.id, "cheap", 1.0 if fruit else 0.0, 100)
    pol = bw.fit_knn(table, train, weights=ln.Weights(tokens=0.5, seconds=0.0), k=5, shrink=1.0)
    assert pol.pick(mk(99, "banana cherry fruit", "exam")) == "cheap"
    assert pol.pick(mk(98, "piston motor oil", "exam")) == "reader"
    assert pol.to_json()["kind"] == "knn"


# ------------------------------------------------------------------ the whole evaluation, on fake saved work

def _fake_saved(tmp_path, n_sets=3, per_task=2, odds=True):
    """Saved-style steps for 3 sets made by a live sweep with the odds fake."""
    tasks_all, key_all, steps_all = [], {}, {}
    roster = None
    for k in range(n_sets):
        tasks, key = dev(seed=40 + k, per_task=per_task)
        box, _ = _odds_box(tmp_path, tasks, key, logprobs=8 if odds else 0, name=f"s{k}.jsonl")
        steps = asyncio.run(sweep_batch(box, tasks))
        roster = box.roster
        for t in tasks:
            tid = f"set{k + 1}:{t.id}"
            tasks_all.append(Task(tid, t.passage, t.stem, t.options, t.kind))
            key_all[tid] = key[t.id]
            steps_all[tid] = steps[t.id]
    return tasks_all, key_all, steps_all, roster


def test_evaluate_features_scores_each_switch_alone_and_mixed(tmp_path):
    tasks, key, steps, roster = _fake_saved(tmp_path)
    train = [t.id for t in tasks if not t.id.startswith("set3:")]
    test = [t.id for t in tasks if t.id.startswith("set3:")]
    rep = bw.evaluate_features(tasks, key, steps, roster, train, test, reader=CFG.reader,
                               specs={s.tag: s for s in CFG.ants}, k=3)
    assert rep["gate_data"] and rep["available"] == list(bw.FEATURES)
    labels = [r["config"] for r in rep["rows"]]
    assert labels[0] == "harness (today)" and len(labels) == 16
    assert "harness +gate" in labels and "harness +stop +gate +router +cache" in labels
    today = rep["rows"][0]
    # the fresh count never exceeds the full count
    for r in rep["rows"]:
        assert r["test_fresh"]["tokens"] <= r["test"]["tokens"] + 1e-9
    assert rep["best_mix"] in labels and rep["best_mix"] != "harness (today)"
    lines = bw.table_lines(rep, "T")
    assert len(lines) == 3 + 16 + 1 and "best mix" in lines[-1]
    pub = bw.public(rep)
    json.dumps(pub)
    assert all("choice" not in r for r in pub["rows"]) and "_tabs" not in pub
    # replaying the same routes on the same work gives the same score
    same = bw.swap_score(rep, "harness (today)", steps, roster, tasks, key)
    assert same["test"]["accuracy"] == pytest.approx(today["test"]["accuracy"])
    assert same["test"]["tokens"] == pytest.approx(today["test"]["tokens"])


def test_evaluate_features_without_odds_skips_the_gate(tmp_path):
    tasks, key, steps, roster = _fake_saved(tmp_path, odds=False)
    train = [t.id for t in tasks if not t.id.startswith("set3:")]
    test = [t.id for t in tasks if t.id.startswith("set3:")]
    rep = bw.evaluate_features(tasks, key, steps, roster, train, test, features=("stop", "router"), k=3)
    assert not rep["gate_data"] and rep["available"] == ["stop", "router"] and len(rep["rows"]) == 4


def test_tune_fit_matches_the_bucket_learners_tune(tmp_path):
    tasks, key, steps, roster = _fake_saved(tmp_path, odds=False)
    box = ReplayBox(steps, roster)
    from antelligence.harness.recipes import menu
    table = ln.evaluate([r for r in menu(roster) if r.fits(box)], box, tasks, key)
    names, keep = ln.common(table, tasks)
    w1, _ = ln.tune(table, keep, k=3)
    w2, _, pick = bw.tune_fit(bw.fit_bucket, table, keep, names, k=3)
    assert w1.tokens == w2.tokens and pick["rank"] in (0, 1, 2)


def test_overlay_and_configs():
    old = {"a": {"x": Step("x", "A"), "y": Step("y", "B")}}
    new = {"a": {"x": Step("x", "C")}, "b": {"x": Step("x", "D")}}
    got = bw.overlay(old, new)
    assert got["a"]["x"].answer == "C" and got["a"]["y"].answer == "B" and "b" not in got
    assert old["a"]["x"].answer == "A"
    cs = bw.configs(["stop", "gate"])
    assert [c.label for c in cs] == ["harness (today)", "harness +stop", "harness +gate", "harness +stop +gate"]


# ------------------------------------------------------------------ 5. prompt search

def test_valid_and_parse_prompt():
    assert po.valid(po.ORIGINAL) and all(po.valid(p) for p in po.SEEDS)
    assert not po.valid(po.Prompt("s", 'Reply as JSON: {"answer": "C"} one letter json'))     # names a letter
    assert not po.valid(po.Prompt("s", "answer in one word"))
    got = po.parse_prompt('<think>hm</think> sure: {"system": "Be brief.", "format": "Reply as JSON: {\\"answer\\": '
                          '\\"<one letter>\\"}"}', "r")
    assert got is not None and got.system == "Be brief." and '"answer"' in got.fmt
    assert po.parse_prompt("no json here", "r") is None


def test_frontier_keeps_only_unbeaten_prompts():
    mk = lambda a, t: po.Score(po.Prompt(f"{a}{t}", "f"), a, t, 1, 1, [])  # noqa: E731
    front = po.frontier([mk(0.6, 100), mk(0.7, 200), mk(0.5, 150), mk(0.7, 250)])
    assert [(s.accuracy, s.tokens) for s in front] == [(0.6, 100), (0.7, 200)]


def test_custom_ants_swaps_only_the_fixed_text():
    item = sr.Item("q", "exam", "", "Which?", ("(A) a", "(B) b"))
    spec = CFG.ants[0]
    msgs = bw.custom_ants("Be brief.", 'Reply as JSON: {"answer": "<one letter>"}')(item, "alone", 1, spec)
    base = sr.ant_messages(item, "alone", 1, spec)
    assert msgs[0]["content"] == "Be brief." and ft._format("exam") not in msgs[1]["content"]
    assert msgs[1]["content"].split("Reply as JSON")[0] == base[1]["content"].split("Reply as JSON")[0]


def test_prompt_search_runs_with_fakes_and_reuses_the_original(tmp_path):
    tasks, key = dev(per_task=2)
    tasks = tasks[:8]
    box, prov = _odds_box(tmp_path, tasks, key, name="po.jsonl")
    runner = box.runner
    make = lambda msgs: LiveBox(runner, CFG.reader, CFG.ants, parser=CFG.parser, code=False, logprobs=8,  # noqa
                                ant_messages=msgs)
    ants = [0, 2]
    got = asyncio.run(po.search(make, runner, CFG.reader, tasks, key, ants, rounds=2, per_round=2))
    assert got["original"]["n"] == len(tasks) * len(ants)
    assert len(got["tried"]) >= 4 and got["pick"]["accuracy"] >= got["original"]["accuracy"] - got["tolerance"]
    assert any(t["origin"].startswith("reflection") for t in got["tried"])
    before = prov.calls
    again = asyncio.run(po.score(make, po.ORIGINAL, tasks, key, ants))     # all cached now
    assert prov.calls == before and again.n == len(tasks) * len(ants)


# ------------------------------------------------------------------ the script

def test_features_script_dry_run_end_to_end(tmp_path):
    s = _features_script()
    out = tmp_path / "out"
    assert s.main(["mac", "--dry-run", "--out", str(out), "--po-train", "8", "--po-rounds", "1"]) == 0
    text = (out / "RESULT.txt").read_text()
    for part in ("FREE (saved night answers", "PROBE: mlx-lm fake", "answer odds readable: yes", "CONFSWEEP:",
                 "how sure vs how often right", "GATE (saved answers", "harness +gate", "PROMPT: tried",
                 "hidden final test set was not used"):
        assert part in text, part
    assert "failed:" not in text
    assert (out / "free" / "features.json").exists() and (out / "gate" / "features.json").exists()
    assert (out / "confsweep" / "steps.json").exists() and (out / "promptopt" / "search.json").exists()
    # a second run reuses every call: nothing new is asked
    assert s.main(["mac", "--dry-run", "--out", str(out), "--stages", "confsweep", "--po-train", "8"]) == 0
    assert "CONFSWEEP: 0 new ant calls" in (out / "RESULT.txt").read_text()


def test_features_script_remaps_ants_to_the_saved_column_order(tmp_path):
    s = _features_script()
    live = (AntInfo(0, "a", "f", 0, 1.0), AntInfo(1, "a", "f", 1, 1.0), AntInfo(2, "b", "f", 0, 2.0))
    night = (AntInfo(0, "b", "f", 0, 2.0), AntInfo(1, "a", "f", 1, 1.0), AntInfo(2, "a", "f", 0, 1.0))
    got = s.remap({"q": {"ant0:alone": Step("ant0:alone", "A"), "ant2:alone": Step("ant2:alone", "C"),
                         "reader": Step("reader", "B")}}, live, night, "set2")
    assert set(got) == {"set2:q"}
    assert got["set2:q"]["ant2:alone"].answer == "A" and got["set2:q"]["ant2:alone"].tool == "ant2:alone"
    assert got["set2:q"]["ant0:alone"].answer == "C" and got["set2:q"]["reader"].answer == "B"


def test_features_script_needs_a_server_for_live_stages(tmp_path):
    s = _features_script()
    with pytest.raises(SystemExit):
        s.main(["mac", "--out", str(tmp_path / "o"), "--night", str(tmp_path)])
    with pytest.raises(SystemExit):        # no saved sets there
        s.main(["mac", "--out", str(tmp_path / "o"), "--night", str(tmp_path), "--stages", "free"])


def test_odds_runner_keeps_the_extras_across_a_resume_without_touching_the_frozen_cache_rows(tmp_path):
    tasks, key = dev()
    box, prov = _odds_box(tmp_path, tasks, key, name="r.jsonl")
    tool = ant_tool(0, "alone")
    first = asyncio.run(Ctx(box, tasks[0]).call(tool))
    assert first.sure is not None and "cached_in" in first.payload
    rows = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert all("logprobs" not in r and "cached_tokens" not in r for r in rows)     # the cache rows stay as they were
    assert OddsRunner.extras_of(tmp_path / "r.jsonl").exists()
    calls = prov.calls
    again, _ = _odds_box(tmp_path, tasks, key, name="r.jsonl")                  # a resumed run
    second = asyncio.run(Ctx(again, tasks[0]).call(tool))
    assert again.runner.provider.calls == 0 and prov.calls == calls
    assert second.sure == first.sure and second.payload == first.payload
    # a run seeded with that cache (the prompt search reusing the odds sweep) gets them too
    seeded = OddsRunner(prov, tmp_path / "s.jsonl", check_names=False, seed_paths=[tmp_path / "r.jsonl"])
    b = LiveBox(seeded, CFG.reader, CFG.ants, parser=CFG.parser, code=False, logprobs=8)
    assert asyncio.run(Ctx(b, tasks[0]).call(tool)).sure == first.sure and prov.calls == calls


def test_the_frozen_shared_read_harness_is_untouched():
    import hashlib
    pre = json.loads((REPO / "docs/research/shared-read-20261030/prereg.json").read_text())
    got = hashlib.sha256((REPO / "antelligence/experiments/shared_read.py").read_bytes()).hexdigest()
    assert pre["files"]["harness"] == got
