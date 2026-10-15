"""Shared builders for the P1 tests: hand-made observations in the shapes the colony driver produces."""

from __future__ import annotations

from typing import Any, Dict


def cand(label: str, rid: int, centroid=(10.0, 10.0, 10.0), dist_mm: float = 30.0, p: float = 0.9, volume: int = 4000) -> Dict[str, Any]:
    return {"label": label, "region_id": rid, "centroid_vox": list(centroid), "p_tumour": p, "unc": 0.1, "dist_mm": dist_mm,
            "visited": False, "volume_mm3": volume}


def base_fields() -> Dict[str, Any]:
    """A v4.1 observation (no colony key). voxel_mm is 2.0: A is 10 voxels from the ant, B is 20."""
    return {"local_p_tumour": 0.6, "local_unc": 0.2, "local_dose": 0.5, "local_dose_rel": 0.8, "round": 3, "rounds": 24, "tick": 15,
            "payload_fraction": 0.75, "position_vox": [0, 10, 10], "goal_dist_mm": 42.4, "last_release": {"fraction": 0.25, "tick": 10},
            "squad": 2, "candidates": [cand("A", 1, (10, 10, 10), 20.0), cand("B", 2, (20, 10, 10), 40.0), cand("C", 3, (40, 10, 10), 80.0)]}


def mark_row(kind: str, pos, authors, strong=False, disputed=False, strength=1.0) -> Dict[str, Any]:
    return {"kind": kind, "pos_vox": list(pos), "strength": strength, "distinct_authors": len(authors), "strong": strong,
            "disputed": disputed, "authors": list(authors)}


def colony_fields() -> Dict[str, Any]:
    f = base_fields()
    f.update({
        "look": 0.7, "uptake_reading": {"value": 0.4, "word": "slow", "about_vox": [0, 10, 10]},
        "washout_reading": {"value": 0.03, "word": "fast", "about_vox": [0, 10, 10]},
        "reading_log": [{"tick": 5, "pos_vox": [0, 10, 10], "kind": "look", "value": 0.7, "band": 2, "word": None},
                        {"tick": 10, "pos_vox": [1, 10, 10], "kind": "look", "value": 0.5, "band": 2, "word": None},
                        {"tick": 11, "pos_vox": [60, 60, 60], "kind": "look", "value": 0.0, "band": 0, "word": None}],
        "marks_near": [mark_row("tumour_here", (9, 10, 10), [3, 5], strong=True), mark_row("not_tumour", (9, 10, 11), [6]),
                       mark_row("dosed", (9, 10, 10), [1, 3]), mark_row("need_more", (40, 10, 10), [4])],
        "messages_in": [{"msg_id": 7, "type": "ask", "from_ant": 3, "to_ant": 2, "tick": 14, "about": {"region_id": 2, "pos_vox": None},
                         "question": "is_tumour", "sender_reading": 0.85, "reply_to": None, "answer": None, "answerer_reading": None,
                         "note": '"look at the north edge"'}],
        "zone_need": {"1": "high", "2": "low", "3": "none"}, "colony_note": "Colony note, tick 15: 12% delivered.",
        "region_status": {"1": "open", "2": "open", "3": "open"}, "visited_self": {"regions": [], "walked_count": 3},
        "card_self": {"ant_id": "ant-02", "policy": "S-talk", "model": "script", "rev": "v4.2", "thresholds": {}, "in_run": {}, "cross_run": {}},
        "cards_public": [{"ant_id": f"ant-{i:02d}", "policy": "S-talk", "model": "script", "in_run": {}, "band": b}
                         for i, b in ((0, "new"), (1, "new"), (3, "usually right"), (4, "mixed"), (5, "new"), (6, "mixed"), (7, "new"))],
        "notebook_self": ["Check dose before moving on.", "Trust ants with readings in the zone."]})
    return f
