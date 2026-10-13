import json

import pytest

from antelligence.anchoring import mocks
import hashlib

from antelligence.anchoring.tamper import N_ENTRIES, Lab, _anchor_all, _export, _plan
from antelligence.kernel.canonical import jcs_bytes
from antelligence.kernel.board import anchor_sizes


@pytest.fixture
def honest(tmp_path):
    """An honest signed run anchored to OTS, Rekor and the (mock) registry, plus its public export."""
    lab = Lab()
    board, keys, akey = mocks.synthetic_run(N_ENTRIES, seed=3)
    out = tmp_path / "export"
    _export(board, akey, out)
    _anchor_all(lab, board, akey, out, anchor_sizes(N_ENTRIES))
    plan = _plan(board, akey)
    study = lab.single_shot(tmp_path / "study")
    study.anchor("prereg", hashlib.sha256(lab.prereg_doc()).digest())
    study.anchor("runplan", hashlib.sha256(jcs_bytes(plan)).digest())
    lab.confirm_ots(out)                                    # Bitcoin confirms; proofs upgraded
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    return {"lab": lab, "board": board, "keys": keys, "akey": akey, "dir": out, "plan": plan, "tmp": tmp_path}
