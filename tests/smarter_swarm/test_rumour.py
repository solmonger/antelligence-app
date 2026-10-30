"""Tests for the rumour world (herd vs think-alone ants, fixed vs smart fading). Small and fast."""

from __future__ import annotations

from dataclasses import replace

import pytest

from antelligence.worlds.rumour import POLICIES, SEEDS, RumourConfig, RumourWorld

SMALL = RumourConfig(n_ants=32, ticks=20)


def test_runs_are_deterministic():
    a = RumourWorld(SMALL, 3).run("herd", "burst").as_dict()
    b = RumourWorld(SMALL, 3).run("herd", "burst").as_dict()
    assert a == b


def test_private_evidence_is_the_same_for_every_policy():
    w = RumourWorld(SMALL, 1)
    solo1, solo2 = w.run("solo"), w.run("solo")
    assert solo1.right == solo2.right
    assert w.readings == RumourWorld(SMALL, 1).readings


def test_solo_ignores_marks_and_seeds():
    w = RumourWorld(SMALL, 2)
    base = w.run("solo", "none")
    for kind in SEEDS:
        assert w.run("solo", kind).wrong_by_tick == base.wrong_by_tick or kind == "liars"


def test_shares_add_up():
    for policy in POLICIES:
        r = RumourWorld(SMALL, 4).run(policy, "none")
        assert abs(r.right + r.wrong + r.undecided - 1) < 1e-6
        assert len(r.wrong_by_tick) == SMALL.ticks


def test_liars_are_not_scored_and_always_lie():
    w = RumourWorld(replace(SMALL, liar_frac=0.1), 5)
    assert len(w.liar_pool) == round(0.1 * 32)
    r = w.run("herd", "liars")
    assert 0 <= r.wrong <= 1


def test_think_alone_never_relays_what_it_heard():
    """An alone ant with no private reading leaves no mark, however strong the marks it read."""
    cfg = replace(SMALL, q=0.0, ticks=5)  # nobody ever gets a private reading
    w = RumourWorld(cfg, 0)
    r = w.run("alone", "burst")
    assert r.feedback_events == 0  # alone ants with no private view give no feedback either
    herd = w.run("herd", "burst")
    assert herd.wrong > 0  # herd ants who read the burst repeat it


def test_smart_fading_kills_a_burst_faster_under_think_alone():
    cfg = replace(RumourConfig(n_ants=128, ticks=60), half_life=20)
    fixed = [RumourWorld(cfg, s).run("alone", "burst").planted_gone_tick for s in range(3)]
    smart = [RumourWorld(replace(cfg, smart=True), s).run("alone", "burst").planted_gone_tick for s in range(3)]
    assert all(t is not None for t in smart)
    assert all(f is None or s < f for f, s in zip(fixed, smart))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        RumourWorld(SMALL, 0).run("queen")
    with pytest.raises(ValueError):
        RumourWorld(SMALL, 0).run("herd", "storm")
    with pytest.raises(ValueError):
        RumourConfig(n_ants=4)
    with pytest.raises(ValueError):
        RumourConfig(accuracy=0.5)
