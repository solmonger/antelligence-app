"""Tests for the field notebook option in the rumour world and its free check. Small and fast."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from antelligence.worlds.rumour import RumourConfig, RumourWorld

REPO = Path(__file__).resolve().parents[2]
SMALL = RumourConfig(n_ants=48, ticks=30)

# Fingerprints (right, wrong, wrong_by_tick, mean half-life, feedback events) taken from the code as it was before the
# notebook option existed, at SMALL, so notebook=False is bit-identical to it.
BEFORE = {
    ("alone", "none", 0): "7694021e6d9f",
    ("alone", "none", 3): "f4dd88ccb050",
    ("alone", "burst", 0): "55231075b54c",
    ("alone", "burst", 3): "2a5a8ff87937",
    ("alone", "liars", 0): "ae91630f8146",
    ("alone", "liars", 3): "2a3a605e28a3",
    ("graded", "none", 0): "01dc688c020a",
    ("graded", "none", 3): "014d661afffd",
    ("graded", "burst", 0): "33185ac9ddf8",
    ("graded", "burst", 3): "b38b24c078ac",
    ("graded", "liars", 0): "515688ce0ab0",
    ("graded", "liars", 3): "6ae853d521bc",
    ("herd", "none", 0): "5174d71491da",
    ("herd", "none", 3): "cdd9d7b5bb63",
    ("herd", "burst", 0): "8c0a3c25ee7c",
    ("herd", "burst", 3): "aa3bf12906b7",
    ("herd", "liars", 0): "f86a649ecb12",
    ("herd", "liars", 3): "9a9a89209424",
    ("solo", "none", 0): "286cc769db3f",
    ("solo", "none", 3): "10a58c649e32",
    ("solo", "burst", 0): "d21090d3d570",
    ("solo", "burst", 3): "b64bae972d6a",
    ("solo", "liars", 0): "a8efdf556ec0",
    ("solo", "liars", 3): "f9729ba1f99f",
}


def _print(r) -> str:
    return hashlib.sha256(json.dumps([r.right, r.wrong, r.wrong_by_tick, r.mean_half_life, r.feedback_events]).encode()).hexdigest()[:12]


@pytest.mark.parametrize("key", sorted(BEFORE))
def test_notebook_off_is_bit_identical_to_the_old_code(key):
    policy, kind, seed = key
    assert _print(RumourWorld(SMALL, seed).run(policy, kind)) == BEFORE[key]
    off = replace(SMALL, notebook=False, notebook_half_life=7.0)  # the half-life means nothing while it is off
    assert _print(RumourWorld(off, seed).run(policy, kind)) == BEFORE[key]


def test_solo_ignores_the_notebook():
    for seed in (0, 1):
        assert _print(RumourWorld(replace(SMALL, notebook=True), seed).run("solo", "burst")) == \
            _print(RumourWorld(SMALL, seed).run("solo", "burst"))


@pytest.mark.parametrize("policy", ["alone", "graded", "herd"])
def test_notebook_counts_each_author_once(policy):
    cfg = replace(SMALL, notebook=True, q=0.5)
    world = RumourWorld(cfg, 2)
    world.run(policy, "burst")
    seen = 0
    for a, book in enumerate(world.notebooks):
        assert str(a) not in book  # an ant never reads its own marks
        assert all(isinstance(k, str) for k in book)  # one entry per author: a dict key, so no author twice
        for value, strength, tick in book.values():
            assert value in (-1, 1) and strength > 0 and 1 <= tick <= cfg.ticks
        seen += len(book)
    assert seen > 0
    # Some ant has met more authors than it could meet at one site over its walk, i.e. the notebook spans sites.
    assert max(len(b) for b in world.notebooks) > 3


def test_a_liar_met_at_many_sites_counts_once():
    """One liar, strong marks, a long walk: the liar's notebook entry is a single (latest) reading per ant."""
    cfg = replace(SMALL, notebook=True, liar_frac=1 / 48, grade_cap=3.0, move_p=1.0, ticks=40)
    world = RumourWorld(cfg, 5)
    world.run("graded", "liars")
    liar = str(world.liar_pool[0])
    holders = [b for b in world.notebooks if liar in b]
    assert holders
    for book in holders:
        assert sum(1 for k in book if k == liar) == 1
        assert book[liar][0] == -world.truth


def test_planted_marks_share_one_author_so_count_once():
    """With no private evidence nobody speaks but the planted marks, which all carry one author name: one entry each."""
    cfg = replace(SMALL, notebook=True, q=0.0, move_p=1.0, burst_frac=0.3)
    world = RumourWorld(cfg, 3)
    world.run("alone", "burst")
    books = [b for b in world.notebooks if b]
    assert books
    for book in books:
        assert list(book) == ["planted"] and book["planted"][0] == -world.truth


def test_notebook_changes_the_answers_when_on():
    cfg = replace(SMALL, q=0.3, social_weight=1.0)
    off = RumourWorld(cfg, 0).run("alone", "none")
    on = RumourWorld(replace(cfg, notebook=True), 0).run("alone", "none")
    assert off.wrong_by_tick != on.wrong_by_tick


def test_what_an_ant_says_does_not_depend_on_the_notebook(monkeypatch):
    from antelligence.kernel import fading

    def marks(cfg):
        seen = []
        real = fading.FadingMemory.deposit

        def spy(self, place, author, value, tick, amount=1.0, *args, **kwargs):
            seen.append((place, author, value, tick, amount))
            return real(self, place, author, value, tick, amount, *args, **kwargs)

        monkeypatch.setattr(fading.FadingMemory, "deposit", spy)
        RumourWorld(cfg, 0).run("alone", "none")
        monkeypatch.undo()
        return seen

    cfg = replace(SMALL, q=0.4)
    assert marks(cfg) == marks(replace(cfg, notebook=True))  # same marks left: sign of private evidence only


def test_negative_notebook_half_life_is_refused():
    with pytest.raises(ValueError):
        RumourConfig(notebook_half_life=-1.0)


def test_zero_notebook_half_life_means_use_half_life():
    a = RumourWorld(replace(SMALL, notebook=True, notebook_half_life=0.0, half_life=10.0), 1).run("alone", "none")
    b = RumourWorld(replace(SMALL, notebook=True, notebook_half_life=10.0, half_life=10.0), 1).run("alone", "none")
    assert a.wrong_by_tick == b.wrong_by_tick


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_check_script_runs_small(tmp_path):
    mod = _load("notebook_check")
    out = tmp_path / "nb.json"
    rc = mod.main(["--tune-n", "32", "--tune-seeds", "0-1", "--test", "32:300-301", "--ticks", "20",
                   "--workers", "1", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert set(data["summary"]["picks"]) == {"alone", "graded", "alone+nb", "graded+nb"}


def test_burst_probe_gives_each_planted_copy_its_own_author(tmp_path):
    probe = _load("notebook_burst_probe")
    cfg = replace(SMALL, notebook=True, q=0.0, move_p=1.0, burst_frac=0.3)
    one = RumourWorld(cfg, 3)
    one.run("alone", "burst")
    from antelligence.kernel import fading
    real = fading.FadingMemory.deposit
    try:
        probe._many_authors()
        many = RumourWorld(cfg, 3)
        many.run("alone", "burst")
    finally:
        fading.FadingMemory.deposit = real
    assert max(len(b) for b in one.notebooks) == 1  # one shared author: a whole burst counts once
    assert max(len(b) for b in many.notebooks) > 1  # one author per site: each copy met counts
    assert all(k.startswith("planted@") for b in many.notebooks for k in b)
