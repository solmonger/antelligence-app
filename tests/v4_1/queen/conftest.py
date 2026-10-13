import json

import pytest

from antelligence.kernel.board import Board
from antelligence.kernel.policies.v41_queen import SquadState, V41Queen
from antelligence.providers.wrappers import FakeProvider
from tests.v4_1.vectors import generate_vectors as gv

N_SQUADS = 3


def make_regions(n=5):
    return [
        {"region_id": i, "centroid_dmm": [200 * i + 20, 400, 600], "cls": "ET", "p_tumour_ppm": max(1000, 900000 - 2000 * i),
         "volume_mm3": 1000 + 100 * i, "unc_ppm": 20000}
        for i in range(n)
    ]


def make_squads(n=N_SQUADS):
    return [SquadState(squad=s, pos_vox=(s, 2, 3), payload_ppm=1_000_000) for s in range(n)]


def seed_board(board_on=True, n_knowledge=8):
    board = Board(gv.test_header(), mode="A")
    for r in range(n_knowledge):
        board.append("knowledge", "perception", {
            "region_id": r, "centroid_dmm": [r, 2, 3], "cls": "ET", "p_tumour_ppm": 500000 + r, "volume_mm3": 10,
            "unc_ppm": 1000, "source": "perception", "prov": "unet@1"}, 0)
    if board_on:
        for i in range(6):
            board.append("evidence", f"squad-{i % N_SQUADS}", {
                "squad": i % N_SQUADS, "pos_vox": [1, 2, 3], "look_p_ppm": 100 + i, "p_tumour_ppm": 200, "dose_ug": i, "released_ug": 0}, 1 + i)
    return board


def good_reply(**over):
    reply = {"plan": "split the squads", "guesses": ["region 0 is the core"],
             "routes": [{"squad": s, "region_ids": [s, s + 1], "priority": 2} for s in range(N_SQUADS)],
             "candidates": [0, 1, 2, 3, 4]}
    reply.update(over)
    return json.dumps(reply)


def make_queen(responder=None, board_on=True, **kw):
    provider = FakeProvider(responder or (lambda req: good_reply()), model="fake-queen")
    board = seed_board(board_on)
    queen = V41Queen(provider, "fake-queen", board, n_squads=N_SQUADS, board_on=board_on, **kw)
    return queen, provider, board


@pytest.fixture
def queen_kit():
    return make_queen
