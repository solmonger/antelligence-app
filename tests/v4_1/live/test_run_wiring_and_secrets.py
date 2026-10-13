"""scripts/v41_run.py live wiring (stubbed builders) and the guarantee that no code path prints key material."""
import importlib.util
import json
import logging
from pathlib import Path

import pytest

from antelligence.anchoring import keys
from antelligence.anchoring.live import build_live_anchoring
from antelligence.experiments import v41_prod
from antelligence.experiments.v41_harness import BONSAI, D1, FW, Q4, Q9
from tests.v4_1.live.test_budget_and_backends import FAKE_NOUS, FAKE_POD_SECRET, pod_state
from tests.v4_1.live.test_live_anchorer import drive

ROOT = Path(__file__).resolve().parents[3]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_backends_and_anchors_are_wired_from_flags(tmp_path, monkeypatch):
    run = script("v41_run")
    seen = {}

    class StubLive:
        anchorer_factory = staticmethod(lambda b, k, d: None)
        post_mode_factory = staticmethod(lambda: "post")
        reputation_client = "client"

    monkeypatch.setattr(run.live, "build_live_anchoring", lambda **kw: seen.setdefault("anchoring", kw) and StubLive)
    monkeypatch.setattr(run.v41_prod, "build_live_workers", lambda **kw: seen.setdefault("workers", kw) and {})
    monkeypatch.setattr(run.v41_prod, "build_live_queen", lambda **kw: seen.setdefault("queen", kw) and "queen")
    monkeypatch.setattr(run.v41_live, "resolve_loader", lambda spec: (lambda c: None))
    args = run.parser().parse_args(["--backends", "live", "--anchors", "live", "--out-dir", str(tmp_path / "o"),
                                    "--pod-state", "PS", "--bonsai-url", "http://127.0.0.1:9999",
                                    "--run", "SW-B:BRATS_001:1", "--run", "SW-Bonsai:BRATS_001:1", "--run", "S:BRATS_001:1"])
    args.ledger_dir = str(tmp_path / "ledger")
    h = run.build_harness(args, lambda k: None)
    assert seen["workers"]["needed"] == {D1, Q4, Q9, BONSAI}          # S needs nothing; FW is not requested
    assert seen["workers"]["pod_state"] == "PS" and seen["workers"]["bonsai_url"] == "http://127.0.0.1:9999"
    assert seen["queen"] and h.queen_provider == "queen"
    assert h.post_mode_factory is StubLive.post_mode_factory and h.reputation_client == "client"
    assert h.anchorer_factory is StubLive.anchorer_factory


def test_fw_arm_asks_for_the_nous_worker_and_d1_for_local_look(tmp_path, monkeypatch):
    run = script("v41_run")
    seen = {}
    monkeypatch.setattr(run.v41_prod, "build_live_workers", lambda **kw: seen.setdefault("workers", kw) and {})
    monkeypatch.setattr(run.v41_prod, "build_live_queen", lambda **kw: "queen")
    monkeypatch.setattr(run.v41_live, "resolve_loader", lambda spec: (lambda c: None))
    args = run.parser().parse_args(["--backends", "live", "--allow-unanchored", "--out-dir", str(tmp_path), "--run", "FW:BRATS_001:1"])
    run.build_harness(args, lambda k: None)
    assert seen["workers"]["needed"] == {FW, D1}  # FW decisions on Nous; local look is d1 in every swarm arm (goal 4.5)


def test_config_backends_still_need_config(tmp_path):
    run = script("v41_run")
    with pytest.raises(SystemExit, match="--config"):
        run.main(["--out-dir", str(tmp_path), "--run", "S:BRATS_001:1"])


def test_live_anchors_default_to_the_ledger_dir_for_run_keys():
    run = script("v41_run")
    assert run.parser().get_default("anchors") == "none" and "live" in run.parser().parse_args(
        ["--out-dir", "x", "--config", "c", "--run", "S:c:1", "--anchors", "live"]).anchors


# --- no key material on stdout, stderr, logs, errors, queue or ledgers --------------------------------------------------


def test_no_code_path_prints_key_material(live_client, mocks_lab, deployment, tmp_path, capsys, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    signer_hex = deployment["key"].removeprefix("0x").lower()
    secrets_ = [signer_hex, FAKE_NOUS, FAKE_POD_SECRET]
    texts = []

    # 1. a full live anchoring run, with repr/str of everything that holds the signer
    live = build_live_anchoring(client=live_client, ots=mocks_lab.ots_client(), rekor=mocks_lab.rekor, start_pumper=False,
                                archive_path=tmp_path / "anchors.json", drain_timeout_s=60)
    board, anchorer, akey = drive(live, tmp_path, 60)
    texts += [repr(live_client), str(vars(live_client)), repr(live_client.signer), str(vars(live_client.signer)),
              repr(live), repr(anchorer.errors), repr(akey)]

    # 2. error paths: wrong deployer, bad pod state, bad auth file, budget refusal
    live_client.expected_deployer = "0x" + "de" * 20
    with pytest.raises(Exception) as ei:
        live_client.preflight()
    texts.append(str(ei.value))
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"providers": {"nous": {"agent_key": FAKE_NOUS}}}))
    texts.append(repr(v41_prod.read_pod_endpoint(pod_state(tmp_path))))
    for bad in (lambda: v41_prod.read_pod_endpoint(tmp_path / "nope.json"), lambda: v41_prod.read_nous_key(tmp_path / "nope.json")):
        with pytest.raises(v41_prod.ProdConfigError) as ei:
            bad()
        texts.append(str(ei.value))

    # 3. the CLIs
    from scripts import v41_keys

    ledger = tmp_path / "ledger"
    for cmd in (["init-study"], ["fingerprint"], ["audit"], ["delete-run-keys"], ["delete-run-keys", "--confirm"]):
        v41_keys.main(cmd, ledger_dir=ledger)

    # 4. a ledger row for an error that happens to carry the key in its text is scrubbed
    from antelligence.providers.ledger import LedgerWriter

    w = LedgerWriter(tmp_path / "u.jsonl", redact=[FAKE_NOUS])
    w.write(role="worker", backend="nous_fw", model="m", served_model=None, request_hash="h", error=f"401 for key {FAKE_NOUS}")
    texts.append((tmp_path / "u.jsonl").read_text())

    captured = capsys.readouterr()
    texts += [captured.out, captured.err, caplog.text]
    blob = "\n".join(texts)
    for s in secrets_:
        assert s not in blob, "secret appeared in output"
    # nothing written to disk contains the chain signer key or the Nous / pod secrets (queue, archive, ledgers, outboxes)
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.name not in ("auth.json", "v4.1-pod-state.json"):      # the sources themselves
            data = path.read_bytes()
            for s in secrets_:
                assert s.encode() not in data, f"{s[:6]}... found in {path.name}"
    # per-run private files exist only under ledger/ ... and the run keys never reach output
    assert not list(tmp_path.rglob("*.keys.json")) or all("ledger" in p.parts for p in tmp_path.rglob("*.keys.json"))
