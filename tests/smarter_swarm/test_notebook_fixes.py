"""Tests for the two notebook fixes in the rumour world: one voice per source, and seen-twice consolidation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from antelligence.worlds.rumour import RumourConfig, RumourWorld

SMALL = RumourConfig(n_ants=48, ticks=30)
SEEDS = (0, 1, 3)


def _print(r) -> str:
    return hashlib.sha256(json.dumps([r.right, r.wrong, r.wrong_by_tick, r.mean_half_life, r.feedback_events]).encode()).hexdigest()[:12]


def _run(cfg, seed, policy, kind):
    world = RumourWorld(cfg, seed)
    result = world.run(policy, kind)
    return world, result


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("policy", ["alone", "graded"])
@pytest.mark.parametrize("kind", ["none", "burst", "liars"])
@pytest.mark.parametrize("notebook", [False, True])
def test_options_off_are_bit_identical(seed, policy, kind, notebook):
    base = replace(SMALL, notebook=notebook)
    explicit = replace(base, notebook_one_voice=False, notebook_consolidate=0.0)
    assert _print(RumourWorld(base, seed).run(policy, kind)) == _print(RumourWorld(explicit, seed).run(policy, kind))


@pytest.mark.parametrize("seed", SEEDS)
def test_one_voice_clips_every_notebook_entry(seed):
    cfg = replace(SMALL, notebook=True, notebook_one_voice=True, q=0.5)
    for policy, cap in (("graded", cfg.grade_cap), ("alone", 1.0)):
        world, _ = _run(cfg, seed, policy, "burst")
        count = 0
        for book in world.notebooks:
            for _, strength, _ in book.values():
                assert strength <= cap + 1e-12
                count += 1
        assert count > 0


def test_one_voice_clips_the_planted_author():
    cfg = replace(SMALL, notebook=True, notebook_one_voice=True)
    assert cfg.planted_amount == 8.0
    for policy, cap in (("graded", cfg.grade_cap), ("alone", 1.0)):
        world, _ = _run(cfg, 1, policy, "burst")
        entries = [book["planted"] for book in world.notebooks if "planted" in book]
        assert entries
        assert all(s <= cap + 1e-12 for _, s, _ in entries)
        # Without the option the planted mark is read well above the cap.
        free, _ = _run(replace(cfg, notebook_one_voice=False), 1, policy, "burst")
        assert max(book["planted"][1] for book in free.notebooks if "planted" in book) > cap


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ["none", "liars"])
def test_one_voice_is_a_no_op_without_a_burst_under_graded(seed, kind):
    cfg = replace(SMALL, notebook=True)
    on = replace(cfg, notebook_one_voice=True)
    assert _print(RumourWorld(cfg, seed).run("graded", kind)) == _print(RumourWorld(on, seed).run("graded", kind))


def test_one_voice_reduces_burst_damage():
    seeds = range(6)
    off_mean = on_mean = 0.0
    for seed in seeds:
        off = RumourWorld(replace(SMALL, q=0.5, notebook=True), seed).run("graded", "burst")
        on = RumourWorld(replace(SMALL, q=0.5, notebook=True, notebook_one_voice=True), seed).run("graded", "burst")
        off_mean += sum(off.wrong_by_tick) / len(off.wrong_by_tick) / len(seeds)
        on_mean += sum(on.wrong_by_tick) / len(on.wrong_by_tick) / len(seeds)
    assert on_mean <= off_mean


def test_consolidate_planted_author_is_never_seen_twice():
    cfg = replace(SMALL, notebook=True, notebook_consolidate=3.0)
    met = 0
    for seed in SEEDS:
        world, _ = _run(cfg, seed, "graded", "burst")
        for book in world.writings:
            if "planted" in book:
                met += 1
                first, twice = book["planted"]
                assert first == 0 and twice is False
    assert met > 0


@pytest.mark.parametrize("policy", ["alone", "graded"])
def test_consolidate_marks_honest_authors_seen_twice(policy):
    cfg = replace(SMALL, notebook=True, notebook_consolidate=3.0)
    world, _ = _run(cfg, 1, policy, "none")
    twice = [k for book in world.writings for k, (_, t) in book.items() if t]
    assert twice
    # A notebook entry is still a 3-tuple.
    assert all(len(v) == 3 for book in world.notebooks for v in book.values())


def test_consolidate_marks_liars_seen_twice_when_met_repeatedly():
    cfg = replace(SMALL, notebook=True, notebook_consolidate=3.0)
    world, _ = _run(cfg, 1, "graded", "liars")
    liars = {str(a) for a in world.liar_pool}
    assert liars
    assert any(t for book in world.writings for k, (_, t) in book.items() if k in liars)


def test_consolidate_negative_raises():
    with pytest.raises(ValueError):
        RumourConfig(notebook_consolidate=-1.0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("policy", ["alone", "graded"])
@pytest.mark.parametrize("kind", ["none", "burst", "liars"])
def test_consolidate_with_notebook_off_has_no_effect(seed, policy, kind):
    off = replace(SMALL, notebook=False)
    with_opt = replace(off, notebook_consolidate=3.0, notebook_one_voice=True)
    assert _print(RumourWorld(off, seed).run(policy, kind)) == _print(RumourWorld(with_opt, seed).run(policy, kind))


@pytest.mark.parametrize("kind", ["none", "burst", "liars"])
def test_solo_ignores_both_options(kind):
    for seed in (0, 1):
        base = RumourWorld(SMALL, seed).run("solo", kind)
        for cfg in (replace(SMALL, notebook=True, notebook_one_voice=True, notebook_consolidate=3.0),
                    replace(SMALL, notebook=False, notebook_one_voice=True, notebook_consolidate=3.0)):
            assert _print(RumourWorld(cfg, seed).run("solo", kind)) == _print(base)
