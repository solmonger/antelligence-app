"""Deterministic generator for interfaces.json (the S1/S2 cross-language test vectors).

Run:  .venv/bin/python -m tests.v4_1.vectors.generate_vectors
The board test suite fails if the committed file differs from this generator's output.
Ed25519 (RFC 8032) is deterministic, so signatures are reproducible. The seeds below are
TEST-ONLY constants and are not secrets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from antelligence.kernel.board import (
    Board, anchor_digest, anchor_sizes, consistency_path, inclusion_path, merkle_root, pubkey_hex,
    run_id_of,
)
from antelligence.kernel.canonical import jcs_json

OUT = Path(__file__).with_name("interfaces.json")
AUTHORS = ("perception", "queen", "harness", "router") + tuple(f"squad-{i}" for i in range(8))


def test_seed(author: str) -> bytes:
    return hashlib.sha256(f"antelligence-v4.1-test-seed|{author}".encode()).digest()


def test_seeds() -> Dict[str, bytes]:
    return {a: test_seed(a) for a in AUTHORS}


def test_header() -> Dict[str, Any]:
    return {
        "study": "v4.1", "arm": "d1-board-on", "case": "BraTS2021_00000", "seed": 7, "split": "dev",
        "world_cfg_sha256": hashlib.sha256(b"world-cfg").hexdigest(),
        "harness_cfg_sha256": hashlib.sha256(b"harness-cfg").hexdigest(),
        "pubkeys": {a: pubkey_hex(s) for a, s in test_seeds().items()},
        "anchor_pubkey": pubkey_hex(test_seed("anchor")),
        "code_sha": "0123456789abcdef0123456789abcdef01234567",
    }


def build_board(mode: str) -> Board:
    """Eight entries covering all seven types, UTF-8 text, a cite and a negative number."""
    b = Board(test_header(), mode=mode, keys=test_seeds() if mode == "B" else None)
    model = {"agent": "queen", "model": "qwen3-8b", "rev": "r1"}
    task = b.append("task", model, {"objective": "Find and treat the enhancing tumour core", "squads": [0, 1], "until_tick": 120}, 0)
    k1 = b.append("knowledge", "perception", {
        "region_id": 3, "centroid_dmm": [1204, 873, 455], "cls": "ET", "p_tumour_ppm": 873000,
        "volume_mm3": 5120, "unc_ppm": 41000, "source": "perception", "prov": "unet-v2@abc123"}, 1)
    b.append("evidence", "squad-0", {
        "squad": 0, "pos_vox": [60, 61, 62], "look_p_ppm": 910000, "p_tumour_ppm": 873000,
        "dose_ug": 0, "released_ug": 125000}, 5, cites=[k1.id])
    route = b.append("route", model, {
        "squad": 0, "region_ids": [3, 1], "priority": 2, "brief_ids": [task.id, k1.id]}, 6, cites=[k1.id, task.id])
    b.append("decision", {"agent": "squad-0", "model": "qwen3-8b", "rev": "r1"}, {
        "squad": 0, "round": 1, "target": "A", "release": 1, "amount_level": 3, "p_target_ppm": 640000,
        "p_release_ppm": 710000, "p_amount_ppm": 500000, "backend": "vllm:qwen3-8b", "deferred": 0,
        "router_pick": "slm", "brief_ids": [task.id, k1.id, route.id]}, 7, cites=[route.id])
    b.append("edit", "harness", {
        "round": 1, "patch_sha256": hashlib.sha256(b"patch").hexdigest(), "keys": ["boundaries", "näme"],
        "prediction": {"metric": "dose_error_ppm", "delta_ppm": -15000}, "kept": None}, 8)
    b.append("alarm", "router", {"kind": "defer_rate", "value_ppm": 250000, "note": "défér rate high \U0001F9EA"}, 9)
    b.append("knowledge", "perception", {
        "region_id": 4, "centroid_dmm": [-12, 0, 9007199254740991], "cls": "WT", "p_tumour_ppm": 1000000,
        "volume_mm3": 0, "unc_ppm": 0, "source": "local_look", "prov": "look"}, 10)
    return b


JCS_CASES: List[Any] = [
    {}, [], None, True, 0, -1, 9007199254740991, -9007199254740991, "",
    "a\"b\\c/d\n\t\u0001\u007f", "é€\U0001F600",
    {"b": 1, "a": 2, "": 3, "aa": 4, "é": 5, "\U0001F600": 6, "￿": 7},
    {"z": [1, {"y": None, "x": False}], "A": "B"},
]
JCS_REJECT = [
    {"value": "1.5", "why": "float"}, {"value": "1.0", "why": "float with integral value"},
    {"value": "9007199254740992", "why": "integer above 2^53-1"}, {"value": "-9007199254740992", "why": "integer below -(2^53-1)"},
    {"value": "1e3", "why": "exponent form is a float"},
]


def generate() -> Dict[str, Any]:
    header = test_header()
    run_id = run_id_of(header)
    board_b, board_a = build_board("B"), build_board("A")
    leaves = [bytes.fromhex(e.id) for e in board_b.entries()]
    roots = {str(n): merkle_root(leaves[:n]).hex() for n in (0, 1, 2, 3, 4, 5, 6, 7, 8)}
    inclusion = [
        {"index": i, "size": n, "leaf_id": board_b.get(i).id, "path": [p.hex() for p in inclusion_path(i, leaves[:n])],
         "root": roots[str(n)]}
        for n, i in ((1, 0), (2, 1), (3, 2), (7, 4), (7, 6), (8, 0), (8, 5))
    ]
    consistency = [
        {"old_size": m, "new_size": n, "old_root": roots[str(m)], "new_root": roots[str(n)],
         "proof": [p.hex() for p in consistency_path(m, leaves[:n])]}
        for m, n in ((1, 2), (1, 8), (2, 3), (3, 7), (4, 8), (6, 8), (7, 8), (8, 8), (0, 8))
    ]
    digests = [
        {"run_id": run_id, "size": n, "root": roots[str(n)],
         "preimage_hex": (bytes.fromhex(run_id) + n.to_bytes(8, "big") + bytes.fromhex(roots[str(n)])).hex(),
         "digest": anchor_digest(run_id, n, bytes.fromhex(roots[str(n)])).hex()}
        for n in (1, 3, 8)
    ]
    return {
        "_doc": "Test vectors for docs/v4.1/INTERFACES.md. Generated by tests/v4_1/vectors/generate_vectors.py. Seeds are TEST-ONLY.",
        "jcs": {
            "accept": [{"value": v, "canonical": jcs_json(v)} for v in JCS_CASES],
            "reject": JCS_REJECT,
        },
        "keys": {
            "seed_rule": "seed(author) = sha256(utf8('antelligence-v4.1-test-seed|' + author)); author 'anchor' is the anchor key",
            "seeds_hex": {a: test_seed(a).hex() for a in AUTHORS + ("anchor",)},
            "pubkeys_hex": {a: pubkey_hex(test_seed(a)) for a in AUTHORS + ("anchor",)},
        },
        "header": header,
        "run_id": run_id,
        "header_jcs": jcs_json(header),
        "entries": [
            {"body_jcs": jcs_json({k: v for k, v in e.to_dict().items() if k not in ("id", "sig")}),
             "entry": e.to_dict(), "id": e.id, "sig_mode_b": e.sig, "sig_mode_a": None}
            for e in board_b.entries()
        ],
        "mode_a_ids_equal_mode_b_ids": [e.id for e in board_a.entries()] == [e.id for e in board_b.entries()],
        "merkle": {
            "empty_root": merkle_root([]).hex(),
            "leaves_are": "entry ids as 32 raw bytes; leaf = sha256(0x00||id); node = sha256(0x01||l||r)",
            "roots": roots, "inclusion": inclusion, "consistency": consistency,
        },
        "anchor": {
            "digest_rule": "sha256(runId32 || uint64_be(size) || root32)", "digests": digests,
            "anchor_sizes": {str(n): anchor_sizes(n) for n in (0, 1, 8, 50, 51, 100, 120)},
        },
        "payload_rejects": [
            {"type": "task", "payload": {"objective": "x", "squads": [0], "until_tick": 1.5}, "why": "float"},
            {"type": "task", "payload": {"objective": "x", "squads": [0], "until_tick": 1, "extra": 1}, "why": "extra key"},
            {"type": "alarm", "payload": {"kind": "defer_rate", "value_ppm": 1}, "why": "missing key"},
            {"type": "decision", "payload": "not an object", "why": "payload not an object"},
        ],
    }


def main() -> None:
    OUT.write_text(json.dumps(generate(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
