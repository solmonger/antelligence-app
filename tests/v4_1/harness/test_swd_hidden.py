"""SW-D regression: an entry the chain never includes must not reach any brief, the queen's input or a valid citation."""

from __future__ import annotations

import asyncio

from antelligence.anchoring.chain_modes import RoundResult
from antelligence.kernel.briefs import check_citations
from tests.v4_1.harness.conftest import CASE_DEV, build_rig


class HidesHalfAndTails:
    """Never includes about half of the evidence (odd seq) and the last two entries of each batch."""

    def __init__(self):
        self.visible, self.hidden_ids = set(), set()

    def post_round(self, run_id, entries):
        entries = list(entries)
        for pos, e in enumerate(entries):
            tail = pos >= len(entries) - 2
            if tail or (e.type == "evidence" and e.seq % 2 == 1):
                self.hidden_ids.add(e.id)
            else:
                self.visible.add(e.seq)
        return RoundResult(set(self.visible), set(), False, {})


def test_swd_hidden_entries_never_reach_briefs_queen_or_citations(tmp_path):
    posts = []

    def factory():
        posts.append(HidesHalfAndTails())
        return posts[-1]

    rig = build_rig(tmp_path, post_mode_factory=factory)
    res = asyncio.run(rig.harness.run("SW-D", CASE_DEV, 1))
    hidden = posts[0].hidden_ids
    assert hidden and any(e.type == "evidence" and e.id in hidden for e in res.board.entries())   # the test bites

    # 1. read receipts: no hidden id in any brief_ids (decisions and routes)
    receipts = [e for e in res.board.entries() if "brief_ids" in e.payload]
    assert any(e.payload["brief_ids"] for e in receipts)
    assert all(not hidden & set(e.payload["brief_ids"]) for e in receipts)

    # 2. the queen's input never mentions a hidden id
    requests = rig.harness.queen_provider.requests
    assert requests
    assert all(not any(h in m["content"] for h in hidden) for r in requests for m in r.messages)

    # 3. no hidden id is a valid citation
    for d in res.board.by_type("decision"):
        valid = check_citations(d.cites, d.payload["brief_ids"])
        assert not hidden & set(valid.valid)
