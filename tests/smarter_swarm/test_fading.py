"""Tests for antelligence.kernel.fading (fixed and smart fading, no checker)."""

from __future__ import annotations

import math

import pytest

from antelligence.kernel.fading import FadeRule, FadingMemory


def test_fixed_mark_halves_every_half_life():
    mem = FadingMemory(FadeRule.fixed(10))
    m = mem.deposit("a", "ant-1", 1, tick=0, amount=8)
    assert m.strength(0) == 8
    assert math.isclose(m.strength(10), 4)
    assert math.isclose(m.strength(30), 1)


def test_marks_are_visible_only_from_the_next_tick():
    mem = FadingMemory(FadeRule.fixed(10))
    mem.deposit("a", "ant-1", 1, tick=5)
    assert mem.at("a", 5) == []
    assert [m.author for m in mem.at("a", 6)] == ["ant-1"]


def test_same_author_same_place_replaces():
    mem = FadingMemory(FadeRule.fixed(10))
    mem.deposit("a", "ant-1", 1, tick=0)
    mem.deposit("a", "ant-1", -1, tick=3)
    assert len(mem) == 1
    assert mem.get("a", "ant-1").value == -1
    mem.deposit("b", "ant-1", 1, tick=3)
    assert len(mem) == 2


def test_tally_sums_strength_per_value():
    mem = FadingMemory(FadeRule.fixed(10))
    mem.deposit("a", "x", 1, tick=0, amount=2)
    mem.deposit("a", "y", 1, tick=0, amount=1)
    mem.deposit("a", "z", -1, tick=0, amount=1)
    t = mem.tally("a", 1)
    assert math.isclose(t[1], 3 * 0.5 ** 0.1) and math.isclose(t[-1], 0.5 ** 0.1)


def test_fixed_rule_counts_feedback_but_never_moves_the_half_life():
    mem = FadingMemory(FadeRule.fixed(10))
    m = mem.deposit("a", "x", 1, tick=0)
    for t in range(1, 6):
        mem.feedback(m, agrees=False, tick=t)
    assert m.half_life == 10 and m.contradicts == 5 and mem.feedback_events == 5


def test_smart_rule_fades_contradicted_marks_faster_and_keeps_confirmed_ones():
    mem = FadingMemory(FadeRule.smart_rule(10))
    bad = mem.deposit("a", "bad", -1, tick=0, amount=8)
    good = mem.deposit("b", "good", 1, tick=0, amount=8)
    for t in range(1, 4):
        mem.feedback(bad, agrees=False, tick=t)
        mem.feedback(good, agrees=True, tick=t)
    assert bad.half_life < 10 < good.half_life
    assert bad.strength(20) < FadingMemory(FadeRule.fixed(10)).deposit("c", "x", -1, 0, amount=8).strength(20)
    assert good.strength(20) > 8 * 0.5 ** 2


def test_half_life_change_never_rewrites_strength_already_lost():
    mem = FadingMemory(FadeRule.smart_rule(10))
    m = mem.deposit("a", "x", 1, tick=0, amount=8)
    before = m.strength(10)
    mem.feedback(m, agrees=False, tick=10)
    assert math.isclose(m.strength(10), before)
    assert math.isclose(m.strength(15), before * 0.5 ** (5 / 5))


def test_half_life_is_clamped():
    rule = FadeRule.smart_rule(10)
    mem = FadingMemory(rule)
    m = mem.deposit("a", "x", 1, tick=0)
    for t in range(1, 40):
        mem.feedback(m, agrees=False, tick=t)
    assert m.half_life == rule.min_half_life
    for t in range(40, 120):
        mem.feedback(m, agrees=True, tick=t)
    assert m.half_life == rule.max_half_life


def test_prune_drops_marks_below_the_floor():
    mem = FadingMemory(FadeRule.fixed(1), floor=0.01)
    mem.deposit("a", "x", 1, tick=0)
    assert mem.prune(3) == 0
    assert mem.prune(10) == 1 and len(mem) == 0


@pytest.mark.parametrize("kw", [dict(half_life=0), dict(contradict=1.5), dict(confirm=0.5),
                                dict(half_life=10, min_half_life=20)])
def test_bad_rules_are_refused(kw):
    with pytest.raises(ValueError):
        FadeRule(**kw)


def test_bad_deposits_are_refused():
    mem = FadingMemory()
    with pytest.raises(ValueError):
        mem.deposit("a", "x", 1, tick=0, amount=0)
    with pytest.raises(ValueError):
        mem.deposit("a", "x", 1, tick=0, half_life=0)
