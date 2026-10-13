"""Episode harness (goal 4.7): determinism, replay, gate order, board-off, Byzantine rule, calibration mode,
log fields, anchor wiring, concurrency. Fake backends, mock anchors, no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from antelligence.anchoring.chain_modes import RoundResult
from antelligence.experiments import v41_cells
from antelligence.experiments.v41_gate import GateError
from antelligence.experiments.v41_harness import (ARM_SPECS, DECISION_LOG_FIELDS, METRIC_FIELDS, RUN_LOG_FIELDS, D1, Q4,
                                                  Q9, HarnessError, KeyVault, Worker, error_cell)
from antelligence.kernel.board import ANCHOR_EVERY, Board, anchor_sizes
from antelligence.kernel.decision import Answer, DecisionError
from antelligence.provenance.bundle import verify_bundle_integrity
from antelligence.providers.decision_backends import CachedDecision, FakeBackend, hashed_answers
from antelligence.providers.wrappers import Cached
from tests.v4_1.harness.conftest import (CASE_DEV, CASE_EVAL, allow_all, build_rig, fake_workers, make_loader,
                                         queen_provider, small_cfg)


def run(rig, arm="SW-B", case=CASE_DEV, seed=1, **kw):
    return asyncio.run(rig.harness.run(arm, case, seed, **kw))


def stable(cell):
    return {k: cell[k] for k in METRIC_FIELDS}


# --- 1 ----------------------------------------------------------------------------------------------------------

def test_episode_deterministic_with_fake_backends(tmp_path):
    vault = KeyVault()
    a = run(build_rig(tmp_path / "a", vault=vault))
    b = run(build_rig(tmp_path / "b", vault=vault))
    assert stable(a.cell) == stable(b.cell) and a.cell["run_id"] == b.cell["run_id"]
    assert a.trace_hash == b.trace_hash
    assert a.board.root() == b.board.root() and a.board.size() == b.board.size() > 50
    assert [{k: r[k] for k in ("squad", "round", "target", "backend", "cited_ids")} for r in a.decision_log] == \
           [{k: r[k] for k in ("squad", "round", "target", "backend", "cited_ids")} for r in b.decision_log]
    c = run(build_rig(tmp_path / "c", vault=vault), seed=2)
    assert c.trace_hash != a.trace_hash                       # the seed sets the entry sites
    assert v41_cells.validate_cell(a.cell) == []


@pytest.mark.parametrize("arm", ["S", "S-prime"])
def test_scripted_arms_have_no_board_and_valid_cells(tmp_path, arm):
    rig = build_rig(tmp_path)
    res = run(rig, arm)
    assert res.board is None and res.cell["board_size"] == 0 and v41_cells.validate_cell(res.cell) == []
    assert rig.calls == []                                    # no model is ever called
    assert run(build_rig(tmp_path / "again"), arm).cell["coverage"] == res.cell["coverage"]


# --- 2 ----------------------------------------------------------------------------------------------------------

def test_replay_reproduces_metrics(tmp_path):
    cache = tmp_path / "cache"
    vault = KeyVault()

    def cached_workers(offline: bool) -> dict:
        live = fake_workers()
        out = {}
        for name, w in live.items():
            inner = w.backend
            if offline:
                inner = FakeBackend(lambda r: (_ for _ in ()).throw(AssertionError("backend called during replay")),
                                    model=w.model)
            out[name] = Worker(CachedDecision(inner, cache / name, offline=offline), w.model)
        return out

    live = build_rig(tmp_path / "live", vault=vault, workers=cached_workers(False),
                     queen=Cached(queen_provider(), cache / "queen.jsonl"))
    res = run(live)
    assert verify_bundle_integrity(res.bundle) and res.bundle["bundle_hash"]
    offline = build_rig(tmp_path / "replay", vault=vault, workers=cached_workers(True),
                        queen=Cached(queen_provider(lambda r: (_ for _ in ()).throw(AssertionError("queen called")))
                                     , cache / "queen.jsonl", offline=True))
    verdict = asyncio.run(offline.harness.replay(res.bundle))
    assert verdict == {"replay_ok": True, "mismatched_fields": [], "trace_hash": res.trace_hash}
    assert res.bundle["extra"]["anchor_sizes"] == anchor_sizes(res.cell["board_size"])
    # a doctored metric is caught
    forged = json.loads(json.dumps(res.bundle))
    forged["extra"]["cell"]["coverage"] = 0.99
    assert asyncio.run(offline.harness.replay(forged))["mismatched_fields"] == ["coverage"]


# --- 3 ----------------------------------------------------------------------------------------------------------

def test_gate_called_before_episode(tmp_path):
    def deny(case):
        raise GateError("no prereg")
    rig = build_rig(tmp_path, gate=deny)
    with pytest.raises(GateError):
        run(rig, "SW-B", CASE_EVAL)
    assert rig.gate_calls == [CASE_EVAL] and rig.loads == [] and rig.calls == []
    ok = build_rig(tmp_path / "ok")
    res = run(ok, "SW-B", CASE_EVAL)
    assert ok.gate_calls == [CASE_EVAL] and res.cell["split"] == "eval"
    # every arm goes through the gate, scripted ones included
    for arm in ("S", "S-prime"):
        r = build_rig(tmp_path / arm, gate=deny)
        with pytest.raises(GateError):
            run(r, arm)
        assert r.loads == []


def test_real_gate_refuses_eval_cases_without_prereg(tmp_path):
    from antelligence.experiments.v41_harness import GateConfig, make_gate
    root = Path(__file__).resolve().parents[3]
    split = json.loads((root / "config/v4_1/split.json").read_text())
    gate = make_gate(GateConfig(root, tmp_path / "PREREG.md", tmp_path / "receipt.json", tmp_path / "G1.md"))
    assert gate(split["dev_a"][0]) == "dev"
    with pytest.raises(GateError):
        gate(split["eval"][0])
    with pytest.raises(GateError):
        gate(split["reserve"][0])


# --- 4 ----------------------------------------------------------------------------------------------------------

def test_board_off_arm_has_no_shared_store(tmp_path):
    rig = build_rig(tmp_path)
    res = run(rig, "SW-noBoard")
    squad_authored = [e for e in res.board.entries() if e.author["agent"].startswith("squad-")]
    assert squad_authored == [] and not [e for e in res.board.entries() if e.type in ("evidence", "decision")]
    assert res.run_log["shared_store"] is False and res.cell["board_size"] == 0 and res.cell["anchors"] == 0
    assert res.cell["citations"] == 0 and res.cell["unread_frac"] is None and res.cell["brief_age_median"] is None
    assert all(r["brief_ids"] == [] and r["cited_ids"] == [] for r in res.decision_log)
    # what the squads read: no other squad's evidence or decisions, no ids to cite
    states = [r.state for w in rig.harness.workers.values() for r in w.backend.requests if "look" not in r.questions]
    assert states and all("looked:" not in s and "chose" not in s for s in states)
    assert all(r.brief_ids == () for w in rig.harness.workers.values() for r in w.backend.requests)
    on_rig = build_rig(tmp_path / "on")
    on = run(on_rig, "SW-B")
    assert any(e.type == "evidence" for e in on.board.entries())
    assert any("looked:" in r.state for w in on_rig.harness.workers.values() for r in w.backend.requests)
    # the queen of the board-off arm saw no squad evidence either
    assert '"evidence":[]' in rig.harness.queen_provider.requests[-1].messages[1]["content"]


# --- 5 ----------------------------------------------------------------------------------------------------------

def liar_evidence(res, crop_p):
    return [e for e in res.board.by_type("evidence") if e.payload["look_p_ppm"] >= 900_000]


def test_byzantine_liars_excluded_below_03(tmp_path):
    rig = build_rig(tmp_path, d1_look=0.05)
    res = run(rig, "SW-Byz-local")
    rep = res.reputation
    liars = sorted({e.author["agent"] for e in liar_evidence(res, None)})
    assert len(liars) == 2
    crop = make_loader()(CASE_DEV)[0]
    for e in liar_evidence(res, None):                         # the lie, exactly as the rule defines it
        v = tuple(e.payload["pos_vox"])
        assert e.payload["p_tumour_ppm"] >= 900_000 and crop.p_tumour[v] < 0.1 and e.sig
    assert all(rep.reputation(a) < 300_000 and rep.excluded(a) for a in liars)
    honest = [f"squad-{i}" for i in range(8) if f"squad-{i}" not in liars]
    assert all(rep.reputation(a) >= 300_000 for a in honest)
    liar_ids = {e.id for e in res.board.entries() if e.author["agent"] in liars}
    for a in liars:                                            # once excluded, none of its entries is in any brief
        t_ex = next(r["tick"] for r in rep.reports if r["author"] == a and r["after_ppm"] < 300_000)
        later = [d for d in res.board.by_type("decision") if d.tick >= t_ex]
        assert later and not any(set(d.payload["brief_ids"]) & {e.id for e in res.board.entries() if e.author["agent"] == a}
                                 for d in later)
        early = [d for d in res.board.by_type("decision") if d.tick < t_ex]
        assert any(set(d.payload["brief_ids"]) & liar_ids for d in early)   # they were read before the exclusion
    # signatures valid: the whole board verifies from its export
    path = tmp_path / "b.jsonl"
    res.board.export_jsonl(path)
    assert Board.load_jsonl(path, verify=True).size() == res.board.size()
    # the same liars on a run without Byzantine squads do not exist
    assert not liar_evidence(run(build_rig(tmp_path / "plain"), "SW-B"), None)


class FakeRepClient:
    def __init__(self):
        self.calls = []

    def enqueue_report(self, run_id, author, claim_id, contradicted):
        self.calls.append((run_id, author, claim_id, contradicted))
        return len(self.calls)


def test_byzantine_chain_arm_reports_every_verdict_to_the_registry(tmp_path):
    client = FakeRepClient()
    vault = KeyVault()
    chain = run(build_rig(tmp_path / "c", vault=vault, reputation_client=client), "SW-Byz-chain")
    assert len(client.calls) == len(chain.reputation.reports) > 0
    assert {c[0] for c in client.calls} == {chain.cell["run_id"]}
    assert all(len(c[1]) == 64 and len(c[2]) == 64 for c in client.calls)       # sha256(pubkey), evidence id
    assert len({(c[1], c[2]) for c in client.calls}) == len(client.calls)       # idempotent per (author, claim)
    local = run(build_rig(tmp_path / "l", vault=vault), "SW-Byz-local")        # same liars, same rule
    assert chain.reputation.score == local.reputation.score
    assert sorted(chain.reputation.excluded_authors()) == sorted(local.reputation.excluded_authors())
    with pytest.raises(HarnessError):
        run(build_rig(tmp_path / "x"), "SW-Byz-chain")


# --- 6 ----------------------------------------------------------------------------------------------------------

def test_calibration_mode_counterfactual_logged_not_used(tmp_path):
    def alt(request):                                           # d1 and Qwen-4B give very different answers
        out = hashed_answers(request)
        return {q: (Answer("noul", 1.0 - a.p_yes, {"yes": 1.0 - a.p_yes, "no": a.p_yes}) if a.type == "noul" else a)
                for q, a in out.items()}
    vault = KeyVault()
    one = run(build_rig(tmp_path / "1", vault=vault), "SW-B", calibration=True)
    two = run(build_rig(tmp_path / "2", vault=vault, d1_answers=alt, q4_answers=alt), "SW-B", calibration=True)
    assert one.trace_hash == two.trace_hash and one.board.root() == two.board.root()
    assert stable(one.cell) == stable(two.cell)
    for r in one.decision_log:
        assert r["backend"] == Q9 and r["router_pick"] == Q9 and r["deferred"] == 0
        cf = r["counterfactual"]
        assert set(cf) == {D1, Q4} and cf[D1]["probs"]["target"] and cf[Q4]["probs"]["amount"]
        assert r["sprime"] and set(r["sprime"]) == {"target", "release", "amount"}
    assert [r["counterfactual"][D1]["release"] for r in one.decision_log] != \
           [r["counterfactual"][D1]["release"] for r in two.decision_log]      # logged, and they differ ...
    assert "calibration" in one.bundle["spec"] and one.bundle["spec"]["calibration"] is True
    assert one.board.header["arm"] == "SW-B:calibration" and one.cell["defer_rate"] == 0.0
    # nothing the counterfactual models said reaches the board: every decision entry is the 9B's
    assert {e.payload["backend"] for e in one.board.by_type("decision")} == {Q9}
    # calibration is for dev cases and the two calibration arms only
    with pytest.raises(HarnessError):
        run(build_rig(tmp_path / "e"), "SW-B", CASE_EVAL, calibration=True)
    with pytest.raises(HarnessError):
        run(build_rig(tmp_path / "f"), "FW", calibration=True)
    base = run(build_rig(tmp_path / "3"), "SW-B", calibration=True)
    assert base.board.header["harness_cfg_sha256"] == run(build_rig(tmp_path / "4"), "SW-staticQueen").board.header["harness_cfg_sha256"]


# --- 7 ----------------------------------------------------------------------------------------------------------

def test_decision_log_fields_complete(tmp_path):
    res = run(build_rig(tmp_path, d1_look=0.4))
    assert len(res.decision_log) == 8 * small_cfg().rounds
    for rec in res.decision_log:
        assert set(DECISION_LOG_FIELDS) <= set(rec)
        assert rec["backend"] and rec["model"] and rec["revision"] and rec["hardware"]
        for k in ("server_ms", "network_ms", "queue_ms", "tokens_in", "tokens_out", "cost_usd"):
            assert isinstance(rec[k], (int, float)) and rec[k] >= 0
        assert rec["router_pick"] in (D1, Q4, Q9) and rec["deferred"] in (0, 1) and isinstance(rec["cited_ids"], list)
        assert 6 <= len(rec["brief_ids"]) <= 12 or len(rec["brief_ids"]) == len(set(rec["brief_ids"]))
        assert rec["calls"] and all("model" in c and "server_ms" in c for c in rec["calls"])
    assert set(RUN_LOG_FIELDS) <= set(res.run_log)
    assert res.run_log["decisions"] == len(res.decision_log) and res.run_log["board_size"] == res.board.size()
    assert v41_cells.validate_cell(res.cell) == []
    # files on disk: board export (round-trips), anchors, decision log, run log, bundle
    d = res.run_dir
    rid = res.cell["run_id"]
    assert Board.load_jsonl(d / f"{rid}.jsonl", verify=True).root() == res.board.root()
    assert len((d / "decisions.jsonl").read_text().splitlines()) == len(res.decision_log)
    assert json.loads((d / f"{rid}.anchors.json").read_text())
    assert set(json.loads((d / "run_log.json").read_text())) >= set(RUN_LOG_FIELDS)
    decisions = res.board.by_type("decision")
    assert len(decisions) == len(res.decision_log) and all(e.payload["brief_ids"] for e in decisions)
    assert res.bundle["trust"]["trust_tier"] == "local_replay" and res.bundle["extra"]["cell"] == res.cell


# --- 8 ----------------------------------------------------------------------------------------------------------

def test_anchor_points_wired(tmp_path):
    rig = build_rig(tmp_path)
    res = run(rig)
    size = res.board.size()
    assert size > 3 * ANCHOR_EVERY
    sizes = [a["size"] for a in res.anchors]
    assert sizes == anchor_sizes(size) == [s for s in range(ANCHOR_EVERY, size, ANCHOR_EVERY)] + [size]
    assert res.cell["anchors"] == len(sizes) and res.bundle["extra"]["anchor_sizes"] == sizes
    rid = bytes.fromhex(res.cell["run_id"])
    for rec in res.anchors:                                    # every system got every anchor
        assert rec["root"] == res.board.root(rec["size"]).hex() and rec["digest"] == res.board.anchor_digest(rec["size"]).hex()
        assert rec["ots_path"] and rec["rekor_uuid"]
        assert rig.lab.registry.root_at[(rid, rec["size"])].hex() == rec["root"]
    assert res.run_log["anchor_latency_s"]["OTS"] and res.run_log["errors"] == []
    # FW is anchored too; the other arms are not
    assert run(build_rig(tmp_path / "fw"), "FW").cell["anchors"] > 3
    for arm in ("SW-noBoard", "SW-noQueen", "SW-staticQueen"):
        assert run(build_rig(tmp_path / arm), arm).cell["anchors"] == 0
    # an anchored arm refuses to run unwired
    with pytest.raises(HarnessError, match="anchor"):
        run(_unwired(tmp_path / "bare"))


def _unwired(tmp_path):
    rig = build_rig(tmp_path)
    rig.harness.anchorer_factory, rig.harness.allow_unanchored = None, False
    return rig


# --- 9 ----------------------------------------------------------------------------------------------------------

def test_concurrent_episodes_isolated(tmp_path):
    vault = KeyVault()
    jobs = [("SW-B", CASE_DEV, 1), ("SW-noBoard", CASE_DEV, 1), ("SW-B", CASE_DEV, 2), ("FW", CASE_DEV, 1),
            ("S", CASE_DEV, 1), ("SW-noQueen", CASE_DEV, 3)]
    seq_rig = build_rig(tmp_path / "seq", vault=vault)
    seq = [run(seq_rig, *j) for j in jobs]
    con_rig = build_rig(tmp_path / "con", vault=vault, workers=fake_workers(latency_s=0.001))
    con = asyncio.run(con_rig.harness.run_many(jobs, concurrency=6))
    assert not [r for r in con if isinstance(r, BaseException)]
    for s, c in zip(seq, con):
        assert stable(s.cell) == stable(c.cell) and s.trace_hash == c.trace_hash
        if s.board is not None:
            assert s.board.root() == c.board.root()
    assert len({c.cell["run_id"] for c in con if c.board is not None}) == len([j for j in jobs if j[0] != "S"])
    boards = [c.board for c in con if c.board is not None]
    assert all(len({e.run_id for e in b.entries()}) == 1 for b in boards)           # no entry crossed into another run
    # a failing episode does not take the others down
    bad = build_rig(tmp_path / "bad", gate=lambda c: allow_all(c) if c == CASE_DEV else (_ for _ in ()).throw(GateError("x")))
    out = asyncio.run(bad.harness.run_many([("SW-B", CASE_DEV, 1), ("SW-B", CASE_EVAL, 1)]))
    assert out[0].cell["arm"] == "SW-B" and isinstance(out[1], GateError)


# --- more arms ------------------------------------------------------------------------------------------------

def test_every_arm_runs_and_cells_validate(tmp_path):
    class FakePost:
        def __init__(self):
            self.n = 0

        def post_round(self, run_id, entries):
            self.n += len(entries)
            return RoundResult(visible=set(range(self.n)), invisible=set(), timed_out=False,
                               latency_ms={e.seq: 40 + e.seq % 7 for e in entries})
    client = FakeRepClient()
    cells = []
    for arm in ARM_SPECS:
        rig = build_rig(tmp_path / arm, post_mode_factory=FakePost, reputation_client=client)
        res = run(rig, arm)
        assert v41_cells.validate_cell(res.cell) == [], arm
        cells.append(res.cell)
    assert v41_cells.validate_cells(cells) == []
    by = {c["arm"]: c for c in cells}
    assert by["SW-D"]["visibility_latency_ms"] is not None and by["SW-B"]["visibility_latency_ms"] is None
    assert by["SW-noQueen"]["queen_calls"] == 0 and by["SW-B"]["queen_calls"] >= 1
    assert by["FW"]["router_top_share"] is None and by["SW-B"]["router_top_share"] is not None
    assert by["S"]["tokens_in"] == 0


def test_mode_d_workers_see_only_included_entries(tmp_path):
    class Lagging:
        """Includes everything but the newest 3 entries of each batch, and times out once."""
        def __init__(self):
            self.visible, self.rounds = set(), 0

        def post_round(self, run_id, entries):
            self.rounds += 1
            self.visible |= {e.seq for e in entries[:-3]}
            return RoundResult(set(self.visible), {e.seq for e in entries[-3:]}, self.rounds == 2, {})
    res = run(build_rig(tmp_path, post_mode_factory=Lagging), "SW-D")
    assert res.run_log["d_timeouts"] >= 1 and any(e.payload["kind"] == "timeout" for e in res.board.by_type("alarm"))
    first_seq_of = {e.id: e.seq for e in res.board.entries()}
    # no decision read an entry that the chain had not yet included when its round started
    assert all(first_seq_of[i] < d.seq for d in res.board.by_type("decision") for i in d.payload["brief_ids"])


def test_missing_flair_variant_and_unsupported_loader(tmp_path):
    rig = build_rig(tmp_path)
    run(rig, "SW-B-missingFLAIR")
    assert rig.loads == [(CASE_DEV, "missing_flair")]
    rig2 = build_rig(tmp_path / "2")
    rig2.harness.load_case = lambda case_id: make_loader()(case_id)      # an adapter without a variant argument
    with pytest.raises(HarnessError, match="missing_flair"):
        run(rig2, "SW-B-missingFLAIR")


def test_fw_arm_uses_only_fw_for_decisions_and_d1_for_looks(tmp_path):
    rig = build_rig(tmp_path)
    res = run(rig, "FW")
    assert {r["backend"] for r in res.decision_log} == {"fw"} and {r["router_pick"] for r in res.decision_log} == {"fw"}
    assert set(rig.calls) == {"d1", "fw"}


def test_missing_workers_or_router_fail_before_loading(tmp_path):
    rig = build_rig(tmp_path)
    rig.harness.router = None
    with pytest.raises(HarnessError, match="router"):
        run(rig)
    assert rig.loads == []


def test_router_and_deferral_drive_the_final_backend(tmp_path):
    res = run(build_rig(tmp_path))
    picks = {r["router_pick"] for r in res.decision_log}
    assert picks <= {D1, Q4, Q9}
    for r in res.decision_log:
        if r["router_pick"] == D1:
            assert r["backend"] == (Q9 if r["deferred"] else D1)
        elif r["router_pick"] == Q4:
            assert r["backend"] in (Q4, Q9) and r["deferred"] == 0
    assert res.cell["defer_rate"] == sum(r["deferred"] for r in res.decision_log) / len(res.decision_log)


def test_key_vault_persists_under_a_ledger_dir(tmp_path):
    v1 = KeyVault(tmp_path / "ledger")
    k1, a1 = v1.get("run-x")
    k2, a2 = KeyVault(tmp_path / "ledger").get("run-x")
    assert k1.pubkeys == k2.pubkeys and a1.fingerprint == a2.fingerprint
    with pytest.raises(Exception):
        KeyVault(tmp_path / "not-a-vault-dir").get("run-y")


def test_error_cell_is_a_valid_excluded_cell():
    cell = error_cell("SW-B", CASE_DEV, 1, "dev", DecisionError("identity mismatch"))
    assert v41_cells.validate_cell(cell) == [] and "identity mismatch" in cell["error"]


# --- review fixes -----------------------------------------------------------------------------------------------

def test_router_model_set_is_checked_per_arm(tmp_path):
    from antelligence.experiments.v41_harness import BONSAI, ROUTER_MODELS_BY_ARM
    from antelligence.experiments.v41_router import BONSAI_ROUTER_MODELS, ROUTER_MODELS
    assert BONSAI_ROUTER_MODELS == (D1, Q4, BONSAI) and ROUTER_MODELS_BY_ARM["SW-Bonsai"] == BONSAI_ROUTER_MODELS
    assert all(ROUTER_MODELS_BY_ARM[a] == ROUTER_MODELS for a in ("SW-B", "SW-noBoard", "SW-D", "SW-Byz-local", "SW-B-missingFLAIR"))
    rig = build_rig(tmp_path)
    res = run(rig, "SW-Bonsai")
    assert {r["router_pick"] for r in res.decision_log} <= set(BONSAI_ROUTER_MODELS)
    assert Q9 not in {r["backend"] for r in res.decision_log}
    wrong = build_rig(tmp_path / "w")
    wrong.harness.router = wrong.harness.bonsai_router          # a Bonsai router on an SW-B run
    with pytest.raises(HarnessError, match="routes"):
        run(wrong, "SW-B")
    assert wrong.loads == []


def test_anchored_arm_set():
    from antelligence.experiments.v41_harness import ANCHORED_ARMS
    assert set(ANCHORED_ARMS) == {"SW-B", "FW", "SW-D", "SW-Byz-local", "SW-Byz-chain"}


def test_mode_d_queen_and_citations_see_only_visible_entries(tmp_path):
    class Hold:
        """Nothing after the first batch is ever included."""
        def __init__(self):
            self.n = 0

        def post_round(self, run_id, entries):
            self.n += 1
            vis = {e.seq for e in entries} if self.n == 1 else set()
            self.seen = getattr(self, "seen", set()) | vis
            return RoundResult(set(self.seen), set(), False, {})
    rig = build_rig(tmp_path, post_mode_factory=Hold)
    res = run(rig, "SW-D")
    visible = {e.seq for e in res.board.entries()[: res.board.size()] if e.type == "knowledge"}
    later = rig.harness.queen_provider.requests
    assert later and all('"evidence":[]' in r.messages[1]["content"] for r in later)   # she never saw squad evidence
    assert res.cell["unread_frac"] is not None
    pos = {e.id: e.seq for e in res.board.entries()}
    first_batch = max(pos[i] for d in res.board.by_type("decision") for i in d.payload["brief_ids"])
    assert first_batch < 60 and visible
