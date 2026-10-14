"""Marks (goal 6.1 / 6.10): decay, distinct authors, same-author refresh, strong / disputed, next-tick visibility."""

import math

import pytest

from antelligence.kernel.board import validate_payload
from antelligence.kernel.colony.marks import KIND_CODES, MarkError, MarkField, MarkParams

P = (10, 10, 10)


def field(voxel_mm=1.0):
    return MarkField("S-marks", MarkParams(voxel_mm=voxel_mm))


def test_decay_halves_at_the_half_life():
    f = field()
    f.deposit(0, "dosed", P, 0, amount=1.0)
    assert f.aggregate("dosed", P, 30).strength == pytest.approx(0.5)
    assert f.aggregate("dosed", P, 60).strength == pytest.approx(0.25)
    f.deposit(1, "need_more", P, 0)
    assert f.aggregate("need_more", P, 15).strength == pytest.approx(0.5)      # half-life 15
    f.deposit(2, "not_tumour", P, 0)
    assert f.aggregate("not_tumour", P, 80).strength == pytest.approx(0.5)     # half-life 80


def test_next_tick_visibility():
    f = field()
    f.deposit(0, "tumour_here", P, 5)
    assert f.aggregate("tumour_here", P, 5).distinct_authors == 0
    assert f.marks_near(P, 5) == []
    assert f.aggregate("tumour_here", P, 6).distinct_authors == 1
    assert len(f.marks_near(P, 6)) == 1


def test_distinct_author_aggregation_and_strong():
    f = field()
    f.deposit(0, "tumour_here", P, 0)
    one = f.aggregate("tumour_here", P, 1)
    assert one.distinct_authors == 1 and not one.strong       # one author is never strong
    f.deposit(1, "tumour_here", (11, 10, 10), 0)               # within 4 mm of P
    two = f.aggregate("tumour_here", P, 1)
    assert two.distinct_authors == 2 and two.authors == (0, 1)
    assert two.strength == pytest.approx(2 * 0.5 ** (1 / 40)) and two.strong


def test_strong_needs_strength_as_well_as_authors():
    f = field()
    f.deposit(0, "need_more", P, 0)
    f.deposit(1, "need_more", P, 0)
    assert f.aggregate("need_more", P, 1).strong
    old = f.aggregate("need_more", P, 16)                      # 2 * 0.5^(16/15) = 0.96 < 1
    assert old.distinct_authors == 2 and old.strength < 1.0 and not old.strong


def test_same_author_refresh_adds_nothing():
    f = field()
    f.deposit(0, "dosed", P, 0)
    f.deposit(0, "dosed", P, 10)
    f.deposit(0, "dosed", (12, 10, 10), 20)
    s = f.aggregate("dosed", P, 21)
    assert s.distinct_authors == 1
    assert s.strength == pytest.approx(0.5 ** (1 / 30))        # only the latest deposit, aged from tick 20
    assert not s.strong


def test_amount_scales_and_is_quantised():
    f = field()
    rec = f.deposit(0, "dosed", P, 0, amount=0.12349)
    assert rec["amount_milli"] == 123
    assert f.aggregate("dosed", P, 1).strength == pytest.approx(0.123 * 0.5 ** (1 / 30))


def test_disputed():
    f = field()
    f.deposit(0, "tumour_here", P, 0)
    assert not f.disputed_at(P, 1)
    f.deposit(1, "not_tumour", (12, 10, 10), 0)
    assert f.disputed_at(P, 1) and f.disputed_at((12, 10, 10), 1)
    assert not f.disputed_at((30, 10, 10), 1)
    row = [r for r in f.marks_near(P, 1) if r["kind"] == "tumour_here"][0]
    assert row["disputed"] is True and row["authors"] == [0]
    assert f.disputed_spots(P, 6.0, 1) == [(10, 10, 10), (12, 10, 10)]


def test_radii_are_millimetres_not_voxels():
    f = field(voxel_mm=2.0)
    f.deposit(0, "dosed", P, 0)
    f.deposit(1, "dosed", (12, 10, 10), 0)                     # 2 voxels = 4 mm: inside the confirm radius
    f.deposit(2, "dosed", (13, 10, 10), 0)                     # 3 voxels = 6 mm: outside confirm, inside read
    assert f.aggregate("dosed", P, 1).authors == (0, 1)
    assert {r["pos_vox"][0] for r in f.marks_near(P, 1)} == {10, 12, 13}
    assert all(r["pos_vox"][0] != 14 for r in f.marks_near(P, 1))


def test_explicit_max_results_does_not_truncate_at_64():
    f = field()
    for ant in range(100):
        f.deposit(ant, "dosed", P, 0)
    assert f.aggregate("dosed", P, 1).distinct_authors == 100


def test_ttl_expiry():
    f = field()
    f.deposit(0, "need_more", P, 0)                            # ttl = ceil(8 * 15) = 120
    assert f.aggregate("need_more", P, 120).distinct_authors == 1
    f.expire(121)
    assert f.aggregate("need_more", P, 121).distinct_authors == 0


def test_records_are_board_payloads_plus_tick():
    f = field()
    rec = f.deposit(3, "slow_uptake", (1, 2, 3), 7, amount=1.0, reading_ppm=250000)
    assert rec == {"subtype": "mark", "mark_kind": "slow_uptake", "ant": 3, "pos_vox": [1, 2, 3],
                   "amount_milli": 1000, "reading_ppm": 250000, "tick": 7}
    validate_payload("evidence", {k: v for k, v in rec.items() if k != "tick"})
    assert KIND_CODES["slow_uptake"] == 3
    assert f.deposit(3, "slow_uptake", (1, 2, 3), 7, amount=1.0, reading_ppm=250000) is None   # exact duplicate
    assert len(f.records) == 1


def test_bad_inputs():
    f = field()
    with pytest.raises(MarkError):
        f.deposit(0, "nonsense", P, 0)
    with pytest.raises(MarkError):
        f.deposit(0, "dosed", P, 0, amount=math.nan)
    with pytest.raises(MarkError):
        MarkParams(half_life={"dosed": 1})
