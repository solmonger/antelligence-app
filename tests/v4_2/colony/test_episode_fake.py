"""run_colony_episode on a fake world: visibility, answer turns, quiet rule, replay, removal replay, barrier."""

import pytest

from antelligence.kernel.board import Board
from antelligence.kernel.colony.episode import (QUEEN_BOARD_KEYS, QUEEN_INPUT_KEYS, ColonyBarrierError, ColonyConfig,
                                                ColonyError, ColonyReplayError, run_colony_episode)
from antelligence.kernel.signal import SignalDraft
from tests.v4_1.vectors import generate_vectors as gv
from tests.v4_2.colony.fake_world import ARMS, BASE_KEYS, FakeWorld, Scripted

LOOKS = {(0, 0, 0): 0.9, (10, 0, 0): 0.7, (20, 0, 0): 0.1}


def mark(kind, pos):
    return SignalDraft(kind=kind, pos=tuple(float(c) for c in pos), ttl=1)


def talker(view):
    """Ant 0 marks and asks at tick 0 and 10; ant 1 acts on any message it is shown; ant 2 does nothing."""
    o, i = view.observation, int(view.agent_id.split("-")[1])
    if i == 0 and view.tick in (0, 10):
        return ({"release": 0}, [mark("tumour_here", (5, 0, 0))],
                {"question": "is_tumour", "pos_vox": [0, 0, 0], "note": "please check"} if view.tick == 0 else None)
    if i == 1 and o.get("messages_in"):
        return ({"release": 1}, [], None)
    return ({"release": 0}, [], None)


def run(arm="S-talk", policy=talker, world=None, **kw):
    seen = []
    w = world or FakeWorld(looks=LOOKS)
    res = run_colony_episode(w, lambda a: Scripted(policy, seen), arm=ARMS[arm], base_keys=BASE_KEYS, **kw)
    return w, res, seen


def test_marks_and_messages_visible_from_next_tick_only():
    w, res, seen = run("S-talk")
    by = {(a, t): o for a, t, o in seen}
    assert "marks_near" in by[("squad-1", 0)] and by[("squad-1", 0)]["marks_near"] == []     # same tick: invisible
    near = by[("squad-1", 5)]["marks_near"]
    assert [(r["kind"], r["authors"]) for r in near if r["kind"] == "tumour_here"] == [("tumour_here", [0])]
    assert by[("squad-2", 5)]["marks_near"] == []                                              # 15 mm away: out of read radius
    assert by[("squad-1", 0)]["messages_in"] == []
    [m] = by[("squad-1", 5)]["messages_in"]                # sent at tick 0, delivered at 1, shown at the next decision turn
    assert m["note"] == '"please check"' and m["type"] == "ask"
    assert by[("squad-1", 10)]["messages_in"] == []        # shown once


def test_answer_turn_runs_on_a_non_decision_tick_and_never_applies():
    w, res, _ = run("S-talk")
    assert all(t % 5 == 0 for _, t in w.applied)           # world.apply was never called off a decision tick
    answers = [e for e in res.colony["action_log"] if e["kind"] == "answer"]
    assert [(e["tick"], e["ant"]) for e in answers] == [(1, 1)]
    msgs = res.colony["messages"]
    assert [m["type"] for m in msgs] == ["ask", "answer"] and msgs[1]["reply_to"] == 1
    assert msgs[1]["to_ant"] == 0 and msgs[1]["tick"] == 1 and msgs[0]["sender_reading"] == pytest.approx(0.9)
    c = res.colony["counters"]
    assert (c["asks_sent"], c["asks_answered"], c["asks_lapsed"], c["asks_refused"]) == (1, 1, 0, 0)


def test_recipient_is_the_nearest_ant_when_nothing_else_applies():
    w, res, _ = run("S-talk")
    assert res.colony["messages"][0]["to_ant"] == 1       # ant 1 is 10 mm away, ant 2 is 20


def test_ask_without_a_reading_to_quote_is_refused_and_counted():
    def mute(view):
        return ({"release": 0}, [], {"question": "is_tumour", "pos_vox": [0, 0, 0]}) if view.agent_id == "squad-0" and view.tick == 0 else ({"release": 0}, [], None)

    w = FakeWorld(looks={})
    _, res, _ = run("S-talk", mute, world=w)
    assert res.colony["counters"]["asks_refused"] == 1 and res.colony["messages"] == []


def test_non_talk_arms_ignore_asks_and_show_no_messages():
    for arm in ("S-solo", "S-marks"):
        w, res, seen = run(arm)
        assert res.colony["messages"] == []
        assert all("messages_in" not in o for _, _, o in seen)


def test_solo_deposits_marks_but_cannot_read_them():
    w, res, seen = run("S-solo")
    assert res.colony["counters"]["marks"].get("tumour_here") == 2
    assert all("marks_near" not in o and "region_status" not in o for _, _, o in seen)


def test_automatic_dosed_mark_on_release_and_judgement_mark_radius():
    def go(view):
        if view.agent_id == "squad-0" and view.tick == 0:
            return ({"release": 1}, [mark("need_more", (3, 0, 0)), mark("need_more", (30, 0, 0)), mark("dosed", (0, 0, 0))], None)
        return ({"release": 0}, [], None)

    _, res, _ = run("S-marks", go)
    kinds = [(r["mark_kind"], r["amount_milli"]) for r in res.colony["marks"]]
    assert ("need_more", 1000) in kinds and ("dosed", 100) in kinds           # 1 of the squad's 10-unit share
    assert res.colony["counters"]["marks_refused"] == 2                        # too far; and `dosed` is not a judgement kind


def test_sensors_are_read_on_every_decision_tick_even_when_the_turn_is_skipped():
    def idle(view):
        return ({"release": 0}, [], None)

    cfg = ColonyConfig(quiet_rule=True, heartbeat_rounds=3)
    w, res, seen = run("S-marks", idle, config=cfg)
    rounds = 40 // 5
    for ant in range(3):
        looks_and_doses = [e for e in res.colony["reading_logs"][str(ant)] if e["kind"] == "dose"]
        assert len(looks_and_doses) == rounds                                   # one reading per decision tick
    sk = res.colony["turns_skipped"]
    turns = {a: sum(1 for t in seen if t[0] == f"squad-{a}") for a in range(3)}
    for a in range(3):
        assert sk[f"ant-{a:02d}"] + turns[a] == rounds
        assert sk[f"ant-{a:02d}"] > 0 and turns[a] >= rounds // 3              # heartbeat keeps a turn every 3rd round
    skipped = [e for e in res.colony["action_log"] if e["kind"] == "skipped"]
    assert len(skipped) == sum(sk.values())
    assert not any(t % 5 for _, t in w.applied)


def test_quiet_rule_off_gives_every_ant_every_turn():
    _, res, seen = run("S-marks")
    assert sum(res.colony["turns_skipped"].values()) == 0 and len(seen) == 3 * 8


def test_quiet_rule_wakes_on_a_new_mark_and_a_message():
    def idle_but_ant0(view):
        if view.agent_id == "squad-0" and view.tick == 0:
            return ({"release": 0}, [mark("tumour_here", (5, 0, 0))], {"question": "is_tumour", "pos_vox": [0, 0, 0]})
        return ({"release": 0}, [], None)

    cfg = ColonyConfig(quiet_rule=True, heartbeat_rounds=100)
    _, res, _ = run("S-talk", idle_but_ant0, config=cfg)
    turns = [(e["tick"], e["ant"], e.get("params", {}).get("wake")) for e in res.colony["action_log"] if e["kind"] == "decision"]
    assert (5, 1, ["mark", "message"]) in turns        # ant 1: ant 0's mark is 5 mm away, and ant 0 asked it
    assert not [t for t in turns if t[1] == 2 and t[0] > 0]  # ant 2 hears nothing: it stays quiet (no heartbeat within 100 rounds)


# --- replay, removal replay, board, barrier ---------------------------------------------------------------------------

def chatty(view):
    """Everyone marks where they stand once; ants 0 and 2 ask; ant 1 releases when it has mail (a message effect)."""
    o, i, t = view.observation, int(view.agent_id.split("-")[1]), view.tick
    emit = [mark("tumour_here", (i * 10, 0, 0))] if t == 5 else []
    ask = None
    if i == 0 and t == 0:
        ask = {"question": "is_tumour", "pos_vox": [0, 0, 0], "note": "n"}
    if i == 2 and t == 10:
        ask = {"question": "is_tumour", "pos_vox": [20, 0, 0]}
    params = {"release": 1 if (i == 1 and o.get("messages_in")) else 0}
    return (params, emit, ask)


def test_replay_reproduces_the_bundle_and_metrics_exactly():
    _, res, _ = run("S-talk", chatty)
    _, again, _ = run("S-talk", chatty)
    assert again.colony == res.colony                       # deterministic
    w2 = FakeWorld(looks=LOOKS)
    rep = run_colony_episode(w2, None, arm=ARMS["S-talk"], base_keys=BASE_KEYS, replay=res.colony)
    assert rep.colony == res.colony and rep.metrics == res.metrics
    # a bundle that went through JSON is the same bundle
    import json
    rep2 = run_colony_episode(FakeWorld(looks=LOOKS), None, arm=ARMS["S-talk"], replay=json.loads(json.dumps(res.colony)))
    assert rep2.colony == res.colony


def test_replay_rejects_a_different_arm_and_needs_a_policy_otherwise():
    _, res, _ = run("S-talk", chatty)
    with pytest.raises(ColonyReplayError):
        run_colony_episode(FakeWorld(looks=LOOKS), None, arm=ARMS["S-marks"], replay=res.colony)
    with pytest.raises(ColonyError):
        run_colony_episode(FakeWorld(looks=LOOKS), None, arm=ARMS["S-talk"])
    with pytest.raises(ColonyError):
        run_colony_episode(FakeWorld(looks=LOOKS), lambda a: None, arm=ARMS["S-talk"], drop={"message": 1})


def test_removal_replay_of_a_message_reports_the_receivers_next_action():
    _, res, _ = run("S-talk", chatty)
    msg = res.colony["messages"][0]
    assert msg["type"] == "ask" and msg["to_ant"] == 1
    seen = []
    eff = run_colony_episode(FakeWorld(looks=LOOKS), lambda a: Scripted(chatty, seen), arm=ARMS["S-talk"],
                             base_keys=BASE_KEYS, replay=res.colony, drop={"message": 1})
    e = eff.effect
    assert e["applied"] and e["receiver"] == 1 and e["changed"] is True
    assert e["baseline_next"]["params"]["release"] == 1 and e["replay_next"]["params"]["release"] == 0
    assert e["first_divergence"] == {"tick": 1, "ant": 1}                 # the answer turn that no longer happens
    assert eff.metrics != res.metrics                         # ant 1 did not release
    # the dropped ask was never answered, but it was sent, numbered and counted
    assert [m["msg_id"] for m in eff.colony["messages"]][:1] == [1]
    assert eff.colony["counters"]["asks_answered"] < res.colony["counters"]["asks_answered"]


def test_removal_replay_of_an_unrelated_message_changes_nothing():
    _, res, _ = run("S-talk", chatty)
    ids = [m["msg_id"] for m in res.colony["messages"] if m["type"] == "answer"]
    assert ids
    seen = []
    eff = run_colony_episode(FakeWorld(looks=LOOKS), lambda a: Scripted(chatty, seen), arm=ARMS["S-talk"], replay=res.colony,
                             drop={"message": ids[0]})
    assert eff.effect["applied"] and eff.effect["changed"] is False       # an answer nobody acts on


def test_removal_replay_of_a_mark():
    _, res, _ = run("S-marks", chatty)
    i = next(k for k, r in enumerate(res.colony["marks"]) if r["mark_kind"] == "tumour_here")
    eff = run_colony_episode(FakeWorld(looks=LOOKS), lambda a: Scripted(chatty, []), arm=ARMS["S-marks"], replay=res.colony,
                             drop={"mark": i})
    assert eff.effect["applied"] and len(eff.colony["marks"]) == len(res.colony["marks"]) - 1
    assert eff.effect["receiver"] is None and eff.effect["changed"] is False
    with pytest.raises(ColonyError):
        run_colony_episode(FakeWorld(looks=LOOKS), lambda a: Scripted(chatty, []), arm=ARMS["S-marks"], replay=res.colony,
                           drop={"mark": 0, "message": 1})


def test_board_receives_mark_message_status_and_need_entries():
    board = gv.build_board("A")
    n0 = len(board)

    def queen(tick, inputs):
        return {"zone_need": {1: "high"}, "colony_note": "go", "need_rows": [
            {"region_id": 1, "need": "high", "need_milli": 900, "delivered_milli": 10}]}

    def asker(view):
        if view.agent_id == "squad-0" and view.tick == 0:
            return ({"release": 0}, [], {"question": "is_tumour", "pos_vox": [10, 0, 0]})
        return ({"release": 0}, [], None)

    class Dosed(FakeWorld):
        def read_sensors(self, i, tick):
            return super().read_sensors(i, tick)[:-1] + [{"kind": "dose", "value": 1.5}]

    w = Dosed(looks=LOOKS)
    w.pos = {0: (10, 0, 0), 1: (11, 0, 0), 2: (40, 0, 0)}                  # ants 0 and 1 stand in zone 1
    res = run_colony_episode(w, lambda a: Scripted(asker, []), arm=ARMS["S-talk"], base_keys=BASE_KEYS, board=board,
                             queen=queen)
    subs = [e.payload["subtype"] for e in board.entries()[n0:]]
    assert {"mark", "message", "status", "need"} <= set(subs)
    status = [e for e in board.entries() if e.payload.get("subtype") == "status"]
    assert len(status) == 1 and status[0].payload["status"] == "done" and status[0].author["agent"] == "harness"
    assert res.colony["counters"]["quorum_events"] == 1 and res.colony["region_status"] == {"1": "done"}
    assert res.colony["need_per_zone"][0] == {"tick": 0, "region_id": 1, "need": "high", "need_milli": 900,
                                              "delivered_milli": 10}
    marks_on_board = [e for e in board.entries() if e.payload.get("subtype") == "mark"]
    assert len(marks_on_board) == len(res.colony["marks"])
    w2 = Dosed(looks=LOOKS)
    w2.pos = dict(w.pos)
    w2.pos.update({0: (10, 0, 0), 1: (11, 0, 0), 2: (40, 0, 0)})
    bare = run_colony_episode(w2, lambda a: Scripted(asker, []), arm=ARMS["S-talk"], queen=queen)
    assert bare.colony == res.colony                               # a board changes nothing about the run


# --- the information barrier (goal 2.8) ----------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list(ARMS))
def test_barrier_every_observation_key_and_queen_input_per_arm(arm):
    spec = ARMS[arm]
    seen_q = []

    def queen(tick, inputs):
        seen_q.append((tick, dict(inputs)))
        return {"zone_need": {1: "high" if tick == 0 else "low"}}

    w, res, seen = run(arm, chatty, queen=queen, world=FakeWorld(looks=LOOKS, ticks=70))
    assert seen
    for ant, tick, obs in seen:
        assert set(obs) == set(BASE_KEYS) | set(spec.allowed_keys), (arm, ant, tick)
    if not spec.colony:
        assert not seen_q and res.colony["marks"] == [] and res.colony["messages"] == []
        return
    for tick, inputs in seen_q:
        want = set(QUEEN_INPUT_KEYS) | (set(QUEEN_BOARD_KEYS) if spec.reads_marks else set())
        assert set(inputs) == want, (arm, tick)
    if spec.reads_marks:
        assert [t for t, _ in seen_q] == [0, 30, 60]
        assert {m["mark_kind"] for _, i in seen_q for m in i["marks"]} <= {"dosed", "dose_ok", "tumour_here", "not_tumour"}
    else:
        assert [t for t, _ in seen_q] == [0]                                  # S-solo: tick-0 need for the whole run
        zn = {tuple(sorted(o["zone_need"].items())) for _, _, o in seen}
        assert zn == {((1, "high"),)}


def test_barrier_solo_gets_no_marks_messages_status_or_shared_flags():
    _, res, seen = run("S-solo", chatty)
    forbidden = {"marks_near", "messages_in", "region_status", "cards_public"}
    assert all(not (forbidden & set(o)) for _, _, o in seen)
    # and the marks it wrote are still countable (goal 6.1: every arm)
    assert res.colony["counters"]["marks"]["tumour_here"] == 3


def test_barrier_raises_when_an_observation_leaks_a_key():
    class Leaky(FakeWorld):
        def observe(self, agent_id, tick):
            o = super().observe(agent_id, tick)
            from antelligence.kernel.types import Observation
            return Observation(data={**o.data, "marks_near": []})

    with pytest.raises(ColonyBarrierError):
        run("S-solo", chatty, world=Leaky(looks=LOOKS))


def test_queen_never_sees_the_world_only_regions_and_board_entries():
    got = []
    run("S-talk", chatty, queen=lambda t, i: got.append(i), world=FakeWorld(looks=LOOKS))
    assert got and all(set(i) == {"tick", "regions", "marks", "quorum_events"} for i in got)


def test_s_arm_is_a_plain_episode():
    _, res, seen = run("S", chatty)
    assert all(set(o) == set(BASE_KEYS) for _, _, o in seen)
    assert res.colony["marks"] == [] and res.colony["reading_logs"] == {"0": [], "1": [], "2": []}


def test_kernel_colony_never_imports_a_world():
    import pathlib
    import re
    root = pathlib.Path(__file__).parents[3] / "antelligence" / "kernel" / "colony"
    for f in root.glob("*.py"):
        assert not re.search(r"^\s*(from|import)\s+antelligence\.worlds", f.read_text(), re.M), f.name
