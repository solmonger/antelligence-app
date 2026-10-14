import pytest

from antelligence.kernel.colony.cards import Card
from antelligence.kernel.colony.notebook import Notebook, NotebookError
from antelligence.kernel.colony.thresholds import Thresholds


def test_notebook_cap_and_text_limit():
    n = Notebook("ant-01")
    for _ in range(7):
        n.add("x" * 200)
    assert n.active_chars() == 1400
    with pytest.raises(NotebookError):
        n.add("y" * 101)
    n.add("y" * 100)
    with pytest.raises(NotebookError):
        n.add("z" * 201)
    with pytest.raises(NotebookError):
        n.patch("L001", "w" * 201)
    n.remove("L002")
    n.patch("L001", "short")
    assert n.active_chars() <= 1500


def test_archive_never_deletes_and_curate_pins():
    n = Notebook("a")
    a, b = n.add("one"), n.add("two")
    n.pin(b)
    assert n.curate(10) == [a]
    assert len(n.lessons) == 2 and n.snapshot() == ["two"]
    n.remove(b)
    assert len(n.lessons) == 2 and n.snapshot() == []


def test_freeze_refuses_all_writes():
    n = Notebook("a")
    lid = n.add("t")
    n.freeze()
    for call in (lambda: n.add("u"), lambda: n.patch(lid, "u"), lambda: n.remove(lid),
                 lambda: n.curate(99), lambda: n.mark_used(lid), lambda: n.pin(lid)):
        with pytest.raises(NotebookError):
            call()
    assert n.snapshot() == ["t"]


def test_round_trip(tmp_path):
    n = Notebook("ant-03")
    ids = [n.add("lesson %d" % i, ["r1"]) for i in range(3)]
    with pytest.raises(NotebookError):
        n.add("q" * 1501)
    n.remove(ids[0])
    n.freeze()
    p = n.save(tmp_path / "nb.json")
    m = Notebook.load(p)
    assert m.to_dict() == n.to_dict() and m.frozen
    with pytest.raises(NotebookError):
        m.add("x")
    ex = m.export_skills()
    assert len(ex) == 2 and all(len(e["name"]) <= 64 and len(e["description"]) <= 1024 for e in ex)


def test_threshold_bounds_and_updates():
    t = Thresholds()
    assert t.theta == {"dose": .5, "explore": .5, "verify": .5}
    for _ in range(10):
        t.update("dose", True)
    assert t.theta["dose"] == pytest.approx(0.1)
    for _ in range(30):
        t.update("explore", False)
    assert t.theta["explore"] == 1.0
    t.reset()
    assert t.theta["dose"] == 0.5


def test_threshold_ties_and_winner():
    t = Thresholds()
    assert t.choose({"dose": .5, "explore": .5, "verify": .5}) == "dose"
    assert t.choose({"dose": 0, "explore": .5, "verify": .5}) == "verify"
    assert t.choose({}) == "dose"
    assert t.choose({"explore": 1.0}) == "explore"
    assert t.probabilities({"dose": .5})["dose"] == pytest.approx(0.5)


def test_card_bands_and_persistence(tmp_path):
    c = Card("ant-03", "S-talk")
    for _ in range(4):
        c.add_claim()
    assert c.public()["band"] == "new"
    c.add_claim()
    for _ in range(4):
        c.confirm()
    c.contradict()
    assert c.band() == "usually right"
    c.contradict()
    assert c.band() == "mixed"
    c.end_run(correct=3, checked=4)
    c.save(tmp_path)
    d = Card.load(tmp_path, "ant-03")
    assert d.to_dict() == c.to_dict() and (tmp_path / "cards" / "ant-03.json").exists()
    d.begin_run()
    assert d.in_run["claims"] == 0 and d.cross_run["claims"] == 5 and d.cross_run["runs"] == 1
    assert set(d.public()) == {"ant_id", "policy", "model", "in_run", "band"}
