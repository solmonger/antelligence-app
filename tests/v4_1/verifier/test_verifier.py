"""The outside verifier: public inputs only, and it catches what each outside log can catch."""

import ast
import json
import re
from pathlib import Path

import pytest

from antelligence.anchoring.tamper import (
    N_ENTRIES,
    Lab,
    _anchor_all,
    _export,
    expected_detects,
    rebuild,
    run_matrix,
    run_scenario,
    table_mismatches,
)
from antelligence.anchoring.verify import OutsideVerifier
from antelligence.kernel.board import anchor_sizes

ALL = {"OTS", "R", "C"}


def verifier(h, conditions=ALL, directory=None, **kw):
    lab = h["lab"]
    return OutsideVerifier(
        directory or h["dir"], h["plan"], chain=lab.reader(), rekor=lab.rekor, ots_dir=directory or h["dir"],
        header_provider=lab.header_provider(), conditions=conditions, pinned_deployer=lab.deployer, **kw,
    )


def test_verifier_accepts_honest_run(honest):
    h = honest
    h["lab"].ots_transport.mine_block()               # Bitcoin confirms the calendars' commitments
    client = h["lab"].ots_client()
    for path in sorted(h["dir"].glob("*.ots")):
        assert client.upgrade(path).complete          # "upgrade the .ots proofs at the end"
    report = verifier(h).verify()
    assert report["ok"], json.dumps(report, indent=1)
    run = report["runs"][h["board"].run_id]
    assert run["entries"] == N_ENTRIES and run["unanchored_tail"] == 0
    assert set(run["checks"]) == {"header", "integrity", "signatures", "chain", "rekor", "ots", "anchor_times"}
    assert run["checks"]["rekor"]["rekor_entries"] == len(anchor_sizes(N_ENTRIES))
    assert run["checks"]["ots"]["bitcoin"] == len(anchor_sizes(N_ENTRIES)) and run["checks"]["ots"]["pending"] == 0
    assert "OTS" in run["checks"]["anchor_times"]["systems"]
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
    assert not report["ok"] and not run["checks"]["signatures"]["ok"]
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
    assert b_only["ok"]                                       # valid chain, valid signatures: B cannot tell
    on_chain = verifier(h, conditions={"C"}, directory=bad).verify()
    chk = on_chain["runs"][forged.run_id]["checks"]["chain"]
    assert not on_chain["ok"] and any(f"size {N_ENTRIES}: root on chain differs" in p for p in chk["problems"])
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
    report = OutsideVerifier(out, plan, chain=lab.reader(), rekor=lab.rekor, ots_dir=out, conditions=ALL).verify()
    assert report["ok"] and report["runs"][board.run_id]["unanchored_tail"] == 20


def test_verifier_wrong_deployer_and_wrong_plan_key(honest):
    h = honest
    v = verifier(h, conditions={"C"})
    v.pinned_deployer = "0x" + "11" * 20
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
    assert verifier(h, conditions={"C"}, claims=late).verify()["ok"]
    bad = verifier(h, conditions={"C"}, claims=early).verify()
    assert not bad["ok"] and "backdated" in bad["runs"][rid]["checks"]["anchor_times"]["problems"][0]


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
