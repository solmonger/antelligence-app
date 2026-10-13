"""S2d reviewer findings: raw HTML in the preregistration, duplicate-key deployments JSON, final-anchor requirement."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from antelligence.anchoring import mocks
from antelligence.anchoring.pins import PinError, deployments_at_commit, parse_strict_json, study_key_in
from antelligence.anchoring.verify import OutsideVerifier
from antelligence.anchoring.tamper import Lab, _anchor_all, _export
from tests.v4_1.verifier.test_verifier import ALL, verifier

REPO = Path(__file__).resolve().parents[3]
KEY = "ab" * 32
LINE = f"Study key fingerprint (sha256 of SPKI DER): `{KEY}`"


def test_plain_prereg_passes_and_real_template_has_no_html():
    assert study_key_in(f"# Prereg\n\nSome p < 0.05 text and `<div>` in code.\n\n{LINE}\n") == KEY
    template = (REPO / "docs/v4.1/PREREGISTRATION.template.md").read_text()
    from antelligence.anchoring.pins import reject_raw_html
    reject_raw_html(template)       # must not raise


@pytest.mark.parametrize("forgery", [
    f"<div hidden>\n{LINE}\n</div>\n\n{LINE}\n",
    f"<details>\n<summary>x</summary>\n{LINE}\n</details>\n{LINE}\n",
    f"{LINE}\n\n<!-- hidden -->\n",
    f"{LINE}\n\n`\n\n<div hidden>x</div>\n\n`\n",       # an unterminated span cannot shield later HTML
    f"{LINE}\n\n`a` <span style=display:none>x</span> `b`\n",
])
def test_raw_html_forgeries_rejected(forgery):
    with pytest.raises(PinError, match="raw HTML"):
        study_key_in(forgery)


def test_prereg_script_refuses_html(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v41_prereg", REPO / "scripts/v41_prereg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = {}
    for n in ("world", "harness", "conformal", "split", "router", "revs"):
        p = tmp_path / n
        p.write_text('{"models": [{"name": "m", "revision": "r1"}]}' if n == "revs" else n)
        files[n] = p
    kw = dict(code_sha="a" * 40, world_yaml=files["world"], harness_yaml=files["harness"], conformal_json=files["conformal"],
              split_json=files["split"], router_model_file=files["router"], model_revisions_json=files["revs"],
              contracts={"AnchorRegistry": {"address": "0x1", "code_hash": "0x2"}}, study_key_sha256=KEY)
    assert study_key_in(mod.render_prereg(**kw)) == KEY            # real template renders and passes
    bad = tmp_path / "bad.md"
    bad.write_text((REPO / "docs/v4.1/PREREGISTRATION.template.md").read_text() + "\n<div hidden>x</div>\n")
    with pytest.raises(mod.PreregError, match="raw HTML"):
        mod.render_prereg(**kw, template=bad)


def _git_repo(tmp_path, content: str) -> tuple:
    repo = tmp_path / "repo"
    (repo / "blockchain/deployments").mkdir(parents=True)
    (repo / "blockchain/deployments/net.json").write_text(content)
    env = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), *env, "commit", "-qm", "d"], check=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return repo, sha


def test_duplicate_v4_1_key_rejected(tmp_path):
    repo, sha = _git_repo(tmp_path, '{"v4_1": {"complete": true}, "v4_1": {"complete": false}}')
    with pytest.raises(PinError, match="duplicate key 'v4_1'"):
        deployments_at_commit(repo, sha, "blockchain/deployments/net.json")
    ok_repo, ok_sha = _git_repo(tmp_path / "ok", '{"v4_1": {"complete": true}}')
    assert deployments_at_commit(ok_repo, ok_sha, "blockchain/deployments/net.json") == {"v4_1": {"complete": True}}


@pytest.mark.parametrize("bad", ['{"v4_1": NaN}', '{"v4_1": Infinity}', '{"v4_1": -Infinity}'])
def test_nan_and_infinity_rejected(bad):
    with pytest.raises(ValueError):
        parse_strict_json(bad)


def _tail_setup(tmp_path, n=120):
    lab = Lab()
    board, _, akey = mocks.synthetic_run(n, seed=4)
    out = tmp_path / "snap"
    _export(board, akey, out)
    _anchor_all(lab, board, akey, out, [50, 100])
    plan = {"study": "v4.1", "runs": [{"run_id": board.run_id, "pubkeys": board.header["pubkeys"], "anchor_pubkey": akey.fingerprint}]}
    study = lab.single_shot(tmp_path / "s")
    from antelligence.kernel.canonical import jcs_bytes
    study.anchor("prereg", hashlib.sha256(lab.prereg_doc()).digest())
    study.anchor("runplan", hashlib.sha256(jcs_bytes(plan)).digest())
    lab.confirm_ots(out)
    return lab, board, out, plan


def _verify(lab, out, plan, **kw):
    return OutsideVerifier(out, plan, chain=lab.reader(), rekor=lab.rekor, ots_dir=out, header_provider=lab.header_provider(),
                           conditions=ALL, pins=lab.pins(), prereg=lab.prereg_path(), study_pem=lab.study_key.pem, **kw).verify()


def test_unanchored_tail_surfaced_and_lenient_by_default(tmp_path):
    lab, board, out, plan = _tail_setup(tmp_path)
    report = _verify(lab, out, plan)
    assert report["ok"] and report["unanchored_tail"] == 20
    assert report["unanchored_tail_by_run"] == {board.run_id: 20} and report["require_final_anchor"] is False


def test_require_final_anchor_fails_nonzero_tail(tmp_path):
    lab, board, out, plan = _tail_setup(tmp_path)
    report = _verify(lab, out, plan, require_final_anchor=True)
    run = report["runs"][board.run_id]
    assert not report["ok"] and report["verdict"] == "failed" and report["unanchored_tail"] == 20
    assert run["status"] == "fail" and "not finally anchored" in run["checks"]["final_anchor"]["problems"][0]


def test_require_final_anchor_passes_fully_anchored_run(honest):
    report = verifier(honest, require_final_anchor=True).verify()
    assert report["ok"] and report["unanchored_tail"] == 0
