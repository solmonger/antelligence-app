"""The outside verifier: public inputs only, and it catches what each outside log can catch."""

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from antelligence.anchoring.tamper import (
    N_ENTRIES,
    Lab,
    _anchor_all,
    _plan,
    _export,
    expected_detects,
    rebuild,
    run_matrix,
    run_scenario,
    table_mismatches,
)
from web3 import Web3

from antelligence.anchoring.pins import TrustPins
from antelligence.anchoring.verify import OutsideVerifier
from antelligence.kernel.canonical import jcs_bytes
from antelligence.kernel.board import anchor_sizes

ALL = {"OTS", "R", "C"}


def verifier(h, conditions=ALL, directory=None, **kw):
    lab = h["lab"]
    return OutsideVerifier(
        directory or h["dir"], h["plan"], chain=lab.reader(), rekor=lab.rekor, ots_dir=directory or h["dir"],
        header_provider=lab.header_provider(), conditions=conditions, pins=lab.pins(), prereg=lab.prereg_path(),
        study_pem=lab.study_key.pem, **kw,
    )


def test_verifier_accepts_honest_run(honest):
    h = honest
    report = verifier(h).verify()
    assert report["ok"] and report["verdict"] == "verified" and report["plan_authenticated"], json.dumps(report, indent=1)
    run = report["runs"][h["board"].run_id]
    assert run["entries"] == N_ENTRIES and run["unanchored_tail"] == 0
    assert set(run["checks"]) == {"header", "integrity", "signatures", "chain", "rekor", "ots", "anchor_times"}
    assert run["checks"]["rekor"]["rekor_entries"] == len(anchor_sizes(N_ENTRIES))
    assert run["anchors_verified"] == {"chain": 5, "rekor": 4, "ots": 5}
    assert run["checks"]["anchor_times"]["untrusted_time_systems"] == []
    assert "OTS" in run["checks"]["anchor_times"]["systems"] and "R" in run["checks"]["anchor_times"]["systems"]
    m = report["metrics"]
    assert m["network_calls"] > 0 and m["bytes"] > 0 and m["wall_s"] >= 0


def test_verifier_flags_unplanned_and_missing_runs(honest, tmp_path):
    h = honest
    plan = {"study": "v4.1", "runs": [{**h["plan"]["runs"][0], "run_id": "ab" * 32}]}
    report = OutsideVerifier(h["dir"], plan).verify()
    assert not report["ok"]
    assert any("not in the anchored run plan" in p for p in report["problems"])
    assert any("has no export" in p for p in report["problems"])


def test_verifier_detects_sig_tamper(honest):
    h = honest
    path = h["dir"] / f"{h['board'].run_id}.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[10])
    row["sig"] = ("0" if row["sig"][0] != "0" else "1") + row["sig"][1:]
    lines[10] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    report = verifier(h, conditions=set()).verify()          # condition B alone
    run = report["runs"][h["board"].run_id]
    assert not report["ok"] and report["verdict"] == "failed" and not run["checks"]["signatures"]["ok"]
    assert "entry 9: bad or missing signature" in run["checks"]["signatures"]["problems"]
    assert run["checks"]["integrity"]["ok"]                  # the chain itself is still intact


def test_verifier_detects_payload_edit_without_resigning(honest):
    h = honest
    path = h["dir"] / f"{h['board'].run_id}.jsonl"
    text = path.read_text()
    assert '"look_p_ppm"' in text
    path.write_text(re.sub(r'"look_p_ppm":(\d+)', lambda m: f'"look_p_ppm":{int(m.group(1)) + 1}', text, count=1))
    run = verifier(h, conditions=set()).verify()["runs"][h["board"].run_id]
    assert not run["checks"]["integrity"]["ok"]


def test_verifier_detects_root_mismatch_vs_chain(honest, tmp_path):
    """The operator rewrites an entry, re-signs and rebuilds the chain: B passes, the registry's roots do not."""
    h = honest
    forged = rebuild(h["board"], h["keys"], rewrite_at=N_ENTRIES - 3)
    bad = tmp_path / "forged"
    _export(forged, h["akey"], bad)
    assert forged.run_id == h["board"].run_id
    b_only = verifier(h, conditions=set(), directory=bad).verify()
    # valid chain, valid signatures: B cannot tell, so the best it may say is "unverified"
    assert b_only["runs"][forged.run_id]["ok"] and b_only["verdict"] == "unverified" and not b_only["ok"]
    on_chain = verifier(h, conditions={"C"}, directory=bad).verify()
    chk = on_chain["runs"][forged.run_id]["checks"]["chain"]
    assert on_chain["verdict"] == "failed" and any(f"size {N_ENTRIES}: root on chain differs" in p for p in chk["problems"])
    assert on_chain["metrics"]["network_calls"] > 0


def test_verifier_detects_truncation_against_chain_and_rekor(honest, tmp_path):
    h = honest
    short = rebuild(h["board"], h["keys"], keep=N_ENTRIES - 10)
    bad = tmp_path / "short"
    _export(short, h["akey"], bad)
    c = verifier(h, conditions={"C"}, directory=bad).verify()["runs"][short.run_id]["checks"]["chain"]
    assert not c["ok"] and any("truncated" in p for p in c["problems"])
    r = verifier(h, conditions={"R"}, directory=bad).verify()["runs"][short.run_id]["checks"]["rekor"]
    assert not r["ok"] and r["unmatched_entries"] >= 1


def test_verifier_reports_unanchored_tail_without_failing(tmp_path):
    from antelligence.anchoring import mocks

    lab = Lab()
    board, _, akey = mocks.synthetic_run(120, seed=4)
    out = tmp_path / "snap"
    _export(board, akey, out)
    _anchor_all(lab, board, akey, out, [50, 100])
    plan = {"study": "v4.1", "runs": [{"run_id": board.run_id, "pubkeys": board.header["pubkeys"], "anchor_pubkey": akey.fingerprint}]}
    study = lab.single_shot(tmp_path / "s")
    study.anchor("prereg", hashlib.sha256(lab.prereg_doc()).digest())
    study.anchor("runplan", hashlib.sha256(jcs_bytes(plan)).digest())
    lab.confirm_ots(out)
    report = OutsideVerifier(out, plan, chain=lab.reader(), rekor=lab.rekor, ots_dir=out, header_provider=lab.header_provider(),
                             conditions=ALL, pins=lab.pins(), prereg=lab.prereg_path(), study_pem=lab.study_key.pem).verify()
    assert report["ok"] and report["runs"][board.run_id]["unanchored_tail"] == 20


def test_verifier_wrong_deployer_and_wrong_plan_key(honest):
    h = honest
    pins = h["lab"].pins()
    v = verifier(h, conditions={"C"})
    v.pins = TrustPins(pins.registry, "0x" + "11" * 20, pins.code_hash, pins.study_key_sha256)
    chk = v.verify()["runs"][h["board"].run_id]["checks"]["chain"]
    assert not chk["ok"] and "not the pinned deployer" in chk["problems"][0]
    plan = json.loads(json.dumps(h["plan"]))
    plan["runs"][0]["anchor_pubkey"] = "00" * 32
    hdr = OutsideVerifier(h["dir"], plan).verify()["runs"][h["board"].run_id]["checks"]["header"]
    assert not hdr["ok"]


def test_claims_must_postdate_their_anchor(honest):
    h = honest
    rid = h["board"].run_id
    earliest = min(h["lab"].reader().anchor_time(bytes.fromhex(rid), s) for s in anchor_sizes(N_ENTRIES))
    late = [{"id": "ok", "run_id": rid, "covers_size": 50, "made_at": earliest + 10_000}]
    early = [{"id": "backdated", "run_id": rid, "covers_size": 50, "made_at": earliest - 10_000}]
    assert verifier(h, conditions={"C"}, claims=late).verify()["ok"]      # chain times are block timestamps
    bad = verifier(h, conditions={"C"}, claims=early).verify()
    assert bad["verdict"] == "failed" and "backdated" in bad["runs"][rid]["checks"]["anchor_times"]["problems"][0]


def test_verify_script_has_no_private_key_surface():
    """The CLI and the verifier modules never import signing code or accept key material."""
    forbidden = re.compile(r"private|PRIVATE_KEY|\.env|Account|sign_id|sign_digest|LocalSigner|get_private_key|seed", re.I)
    root = Path(__file__).resolve().parents[3]
    for rel in ("scripts/v41_verify_board.py", "antelligence/anchoring/verify.py", "antelligence/anchoring/readers.py"):
        src = (root / rel).read_text()
        code = "\n".join(
            n for n in (ast.unparse(x) for x in ast.walk(ast.parse(src)) if isinstance(x, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Name)))
        )
        assert not forbidden.search(code), rel
        assert "--key" not in src and "--private" not in src


def test_b_only_is_unverified_never_ok(honest):
    report = verifier(honest, conditions=set()).verify()
    assert report["verdict"] == "unverified" and not report["ok"]
    assert any("condition B alone" in u for u in report["unverified"])


def test_empty_plan_fails(honest):
    for plan in ({"study": "v4.1", "runs": []}, {"study": "v4.1"}):
        report = OutsideVerifier(honest["dir"], plan, conditions={"B"}).verify()
        assert report["verdict"] == "failed" and not report["ok"]
        assert any("lists no runs" in p for p in report["problems"])
    empty = honest["tmp"] / "empty"
    empty.mkdir()
    assert not OutsideVerifier(empty, honest["plan"], conditions={"B"}).verify()["ok"]


def test_condition_c_requires_pins(honest):
    lab = honest["lab"]
    with pytest.raises(ValueError, match="pins"):
        OutsideVerifier(honest["dir"], honest["plan"], chain=lab.reader(), conditions={"C"})
    good = lab.pins()
    for field, value, needle in (("registry", "0x" + "ee" * 20, "pinned registry address"),
                                 ("code_hash", "0x" + "00" * 32, "pinned code hash")):
        pins = TrustPins(**{**good.__dict__, field: Web3.to_checksum_address(value) if field == "registry" else value})
        wrong = OutsideVerifier(honest["dir"], honest["plan"], chain=lab.reader(), conditions={"C"}, pins=pins).verify()
        assert wrong["verdict"] == "failed", field
        assert any(needle in p for p in wrong["runs"][honest["board"].run_id]["checks"]["chain"]["problems"]), field


def test_pending_ots_makes_the_run_unverified(tmp_path):
    from antelligence.anchoring import mocks

    lab = Lab()
    board, _, akey = mocks.synthetic_run(60, seed=5)
    out = tmp_path / "x"
    _export(board, akey, out)
    _anchor_all(lab, board, akey, out, [50, 60])              # stamped, never confirmed or upgraded
    plan = _plan(board, akey)
    v = lambda **kw: OutsideVerifier(out, plan, ots_dir=out, conditions={"OTS"}, **kw).verify()   # noqa: E731
    pending = v(header_provider=lab.header_provider())
    assert pending["verdict"] == "unverified" and pending["runs"][board.run_id]["checks"]["ots"]["unverified"]
    no_headers = v()                                          # OTS requested, no header source
    assert no_headers["verdict"] == "unverified" and no_headers["runs"][board.run_id]["checks"]["ots"]["unverified"]
    lab.confirm_ots(out)
    confirmed_no_headers = v()                                # upgraded proofs, still no header source
    assert confirmed_no_headers["verdict"] == "unverified"
    assert "block header source" in confirmed_no_headers["runs"][board.run_id]["checks"]["ots"]["error"]
    assert v(header_provider=lab.header_provider())["runs"][board.run_id]["checks"]["ots"]["ok"]


def test_zero_evidence_fails_each_outside_condition(tmp_path):
    from antelligence.anchoring import mocks

    lab = Lab()
    board, _, akey = mocks.synthetic_run(30, seed=6)         # nothing anchored anywhere
    out = tmp_path / "x"
    _export(board, akey, out)
    plan = _plan(board, akey)
    for cond in ("C", "R", "OTS"):
        report = _only(lab, out, plan, cond)
        assert report["verdict"] == "failed", cond
    # n < 50: the run-end anchor is required, and then it counts
    _anchor_all(lab, board, akey, out, [30])
    lab.confirm_ots(out)
    for cond in ("C", "R", "OTS"):
        run = _only(lab, out, plan, cond)["runs"][board.run_id]
        assert run["status"] == "pass" and run["anchors_verified"][{"C": "chain", "R": "rekor", "OTS": "ots"}[cond]] == 1, cond


def _only(lab, out, plan, cond):
    return OutsideVerifier(
        out, plan, chain=lab.reader() if cond == "C" else None, rekor=lab.rekor if cond == "R" else None, ots_dir=out,
        header_provider=lab.header_provider(), conditions={cond}, pins=lab.pins(),
    ).verify()


def test_rekor_missing_proof_fails_the_run(honest):
    h = honest
    h["lab"].mock_rekor.omit_proofs = True
    report = verifier(h, conditions={"R"}).verify()
    chk = report["runs"][h["board"].run_id]["checks"]["rekor"]
    assert report["verdict"] == "failed" and any("no inclusion proof" in p for p in chk["problems"])


def test_rekor_key_mismatch_fails(honest):
    from antelligence.anchoring.rekor import AnchorKey

    h = honest
    (h["dir"] / f"{h['board'].run_id}.anchor_pub.pem").write_bytes(AnchorKey().pem)    # a different key than the plan's
    chk = verifier(h, conditions={"R"}).verify()["runs"][h["board"].run_id]["checks"]["rekor"]
    assert not chk["ok"] and "does not match" in chk["problems"][0]


def test_rekor_unpinned_log_key_is_unverified_not_ok(honest):
    h = honest
    h["lab"].rekor.pinned_log_key_sha256 = "00" * 32
    h["lab"].rekor._log_key = None
    report = verifier(h, conditions={"R"}).verify()
    chk = report["runs"][h["board"].run_id]["checks"]["rekor"]
    assert report["verdict"] == "unverified" and chk["unverified"] and "pinned" in chk["error"]


def test_unauthenticated_plan_caps_verdict_at_unverified(honest):
    h = honest
    r_only = OutsideVerifier(h["dir"], h["plan"], rekor=h["lab"].rekor, conditions={"R"}).verify()    # no study key
    assert r_only["runs"][h["board"].run_id]["status"] == "pass" and not r_only["plan_authenticated"]
    assert r_only["verdict"] == "unverified"
    altered = {**h["plan"], "study": "v4.1-altered"}
    chain = verifier(h, conditions={"C"})
    chain.plan = altered
    res = chain.verify()
    assert res["verdict"] == "failed" and not res["plan_authenticated"]
    assert any("run-plan digest on chain" in p for p in res["problems"])


def test_network_error_is_unverified_with_the_error_string(honest):
    h = honest
    v = verifier(h, conditions={"C"})
    def boom(*a):
        raise ConnectionError("rpc down")
    v.chain.root_at = boom
    report = v.verify()
    chk = report["runs"][h["board"].run_id]["checks"]["chain"]
    assert report["verdict"] == "unverified" and chk["unverified"] and "rpc down" in chk["error"]
    h["lab"].mock_rekor.get_entry = boom
    r = verifier(h, conditions={"R"}).verify()
    assert r["verdict"] == "unverified" and "rpc down" in r["runs"][h["board"].run_id]["checks"]["rekor"]["error"]


@pytest.mark.parametrize("attack,position,k,tail", [
    ("T1", "early", 0, False), ("T2", "truncate_10", 10, False), ("T3", "middle", 0, False), ("T4", "-", 0, False),
    ("T1", "tail", 0, True), ("T2", "truncate_1", 1, True),
])
def test_single_scenarios_match_expected_table(attack, position, k, tail):
    label = (f"tail_truncate_{k}" if attack == "T2" else "tail") if tail else position
    for o in run_scenario(attack, position, seed=11, k=k, tail=tail):
        assert o.detected == expected_detects(attack, label, o.condition), (attack, label, o.condition)


def test_tamper_matrix_expected_table():
    """B detects none; OTS only T4; R and C detect T1-T4 in the anchored prefix; the unanchored tail nothing."""
    matrix = run_matrix(2)
    assert table_mismatches(matrix) == []
    by = {(r["attack"], r["position"], r["condition"]): r for r in matrix["rows"]}
    assert {r["condition"] for r in matrix["rows"]} == {"B", "OTS", "R", "C"}
    # honest control: no false positives anywhere
    assert all(r["rate"] == 0 for r in matrix["rows"] if r["attack"] == "H")
    for pos in ("early", "middle", "before_last_anchor"):
        for att in ("T1", "T3"):
            assert by[(att, pos, "B")]["rate"] == 0 and by[(att, pos, "OTS")]["rate"] == 0
            assert by[(att, pos, "R")]["rate"] == 1 and by[(att, pos, "C")]["rate"] == 1
    for k in (1, 10, 60):
        assert by[("T2", f"truncate_{k}", "R")]["rate"] == 1 and by[("T2", f"truncate_{k}", "OTS")]["rate"] == 0
    assert [by[("T4", "-", c)]["rate"] for c in ("B", "OTS", "R", "C")] == [0, 1, 1, 1]
    assert all(by[("T1", "tail", c)]["rate"] == 0 for c in ("B", "OTS", "R", "C"))
    # write time vs audit time: the chain refuses a second root at write time, Rekor is only caught at audit
    assert by[("T1", "early", "C")]["when"]["write"] == 2 and by[("T1", "early", "R")]["when"]["audit"] == 2
    row = by[("T1", "early", "R")]
    assert row["wilson"]["low"] > 0 and row["mean_network_calls"] > 0 and row["mean_bytes"] > 0
    assert by[("T1", "tail", "R")]["unanchored_tail_entries"] == [20]
