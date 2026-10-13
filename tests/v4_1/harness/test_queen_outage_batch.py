"""Queen provider outages through the harness, the batch driver and the cells (mistake-2039): fake providers, fake time."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

from antelligence.experiments import v41_cells
from antelligence.experiments.v41_harness import EpisodeNotStarted
from antelligence.providers.base import ProviderUnavailable
from antelligence.providers.outage import EXIT_QUEEN_UNAVAILABLE, OutagePolicy, QueenUnavailable
from tests.v4_1.harness.conftest import CASE_DEV, build_rig, queen_provider, queen_reply

ROOT = Path(__file__).resolve().parents[3]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, s):
        self.now += s
        await asyncio.sleep(0)          # a real yield, so concurrent episodes interleave

    def policy(self, max_wait=600):
        return OutagePolicy(max_wait_s=max_wait, sleep=self.sleep, clock=self)


def queue_down(clock, until=100):
    return queen_provider(down_until(clock, until))


def down_until(clock, until):
    def respond(req):
        if clock.now < until:
            raise ProviderUnavailable("HTTP 401: invalid, blocked or out of funds", status=401)
        return queen_reply(req)
    return respond


FIELDS = ("queen_fallbacks_provider", "queen_fallbacks_invalid_reply", "queen_fallbacks_cap")


def test_clean_run_carries_zero_fallbacks_and_a_valid_cell(tmp_path):
    res = asyncio.run(build_rig(tmp_path).harness.run("SW-B", CASE_DEV, 1))
    assert all(res.cell[f] == 0 for f in FIELDS) and v41_cells.validate_cell(res.cell) == []
    assert res.run_log["queen_fallbacks"] == {"provider": 0, "invalid_reply": 0, "cap": 0}
    assert res.cell["queen_calls"] > 0 and res.run_log["queen_outage_retries"] == 0


def test_a_short_outage_is_waited_out_with_no_fallback(tmp_path):
    clock = Clock()
    rig = build_rig(tmp_path, queen=queue_down(clock), queen_outage=clock.policy())
    res = asyncio.run(rig.harness.run("SW-B", CASE_DEV, 1))
    assert all(res.cell[f] == 0 for f in FIELDS) and v41_cells.validate_cell(res.cell) == []
    assert res.run_log["queen_outage_retries"] == 3 and res.cell["queen_calls"] > 0
    assert not [e for e in res.board.entries() if e.type == "alarm" and "queen fallback" in e.payload["note"]]


def test_invalid_replies_are_counted_by_cause_in_the_cell(tmp_path):
    rig = build_rig(tmp_path, queen=queen_provider(lambda req: "not json"))
    res = asyncio.run(rig.harness.run("SW-B", CASE_DEV, 1))
    assert res.cell["queen_fallbacks_invalid_reply"] == res.cell["queen_calls"] > 0
    assert res.cell["queen_fallbacks_provider"] == 0 and res.cell["queen_fallbacks_cap"] == 0
    assert v41_cells.validate_cell(res.cell) == []


def test_a_long_outage_errors_the_run_and_never_falls_back(tmp_path):
    clock = Clock()
    rig = build_rig(tmp_path, queen=queen_provider(down_until(clock, 10**9)), queen_outage=clock.policy(600))
    try:
        asyncio.run(rig.harness.run("SW-B", CASE_DEV, 1))
        raise AssertionError("expected QueenUnavailable")
    except QueenUnavailable as exc:
        assert "unavailable" in str(exc)


def test_run_many_stops_starting_episodes_after_the_first_queen_unavailable(tmp_path):
    clock = Clock()
    rig = build_rig(tmp_path, queen=queen_provider(down_until(clock, 10**9)), queen_outage=clock.policy(60))
    reqs = [("SW-B", CASE_DEV, s) for s in (1, 2, 3)] + [("S", CASE_DEV, 1)]
    out = asyncio.run(rig.harness.run_many(reqs, concurrency=1))
    assert isinstance(out[0], QueenUnavailable) and all(isinstance(r, EpisodeNotStarted) for r in out[1:])


def test_in_flight_episodes_finish_or_error_on_their_own(tmp_path):
    clock = Clock()
    rig = build_rig(tmp_path, queen=queen_provider(down_until(clock, 10**9)), queen_outage=clock.policy(60))
    out = asyncio.run(rig.harness.run_many([("SW-B", CASE_DEV, 1), ("SW-B", CASE_DEV, 2), ("S", CASE_DEV, 1)], concurrency=3))
    assert [type(r) for r in out[:2]] == [QueenUnavailable, QueenUnavailable] and not isinstance(out[2], BaseException)


def test_driver_exits_75_writes_no_cell_for_unstarted_runs_and_prints_totals(tmp_path, capsys):
    run = script("v41_run")
    clock = Clock()
    rig = build_rig(tmp_path / "r", queen=queen_provider(down_until(clock, 10**9)), queen_outage=clock.policy(60))
    cells = tmp_path / "cells.jsonl"
    code = run.main(["--config", "unused.json", "--out-dir", str(tmp_path / "o"), "--cells", str(cells), "--concurrency", "1",
                     "--run", f"SW-B:{CASE_DEV}:1", "--run", f"SW-B:{CASE_DEV}:2", "--run", f"SW-B:{CASE_DEV}:3"],
                    harness=rig.harness)
    err = capsys.readouterr().err
    rows = [json.loads(l) for l in cells.read_text().splitlines()]
    assert code == EXIT_QUEEN_UNAVAILABLE == 75
    assert len(rows) == 1 and "QueenUnavailable" in rows[0]["error"]            # the unstarted two have no cell
    assert err.count("NOT STARTED") == 2 and "rerun later" in err
    assert v41_cells.validate_cells(rows) == []


def test_driver_summary_line_and_batch_totals_show_fallbacks(tmp_path, capsys):
    run = script("v41_run")
    rig = build_rig(tmp_path / "r", queen=queen_provider(lambda req: "garbage"))
    code = run.main(["--config", "unused.json", "--out-dir", str(tmp_path / "o"), "--run", f"SW-B:{CASE_DEV}:1"], harness=rig.harness)
    out = capsys.readouterr().out
    assert code == 0 and "fallbacks provider 0 invalid_reply" in out and "batch: 1 ok, 0 failed, 0 not started" in out
    assert "invalid_reply 0" not in out.split("batch:")[1]                          # the totals are non-zero


def test_pre_existing_bundles_without_the_new_fields_still_replay(tmp_path):
    rig = build_rig(tmp_path)
    res = asyncio.run(rig.harness.run("SW-B", CASE_DEV, 1))
    bundle = json.loads(json.dumps(res.bundle))
    for f in FIELDS:
        bundle["extra"]["cell"].pop(f)
    # replay() compares METRIC_FIELDS to the recorded cell: absent new fields must not count as a mismatch
    from antelligence.experiments.v41_harness import METRIC_FIELDS
    assert set(FIELDS) <= set(METRIC_FIELDS)
    recorded = bundle["extra"]["cell"]
    again = res.cell
    mismatched = [f for f in METRIC_FIELDS if recorded.get(f) != again.get(f) and not (f in v41_cells.QUEEN_FALLBACK_FIELDS and f not in recorded)]
    assert mismatched == []
