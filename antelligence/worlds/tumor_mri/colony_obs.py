"""Colony observation keys and the per-arm allowed sets (goal 6.6, 2.8; INTERFACES v4.2 section 7).

``OBS_KEYS`` (v4.1) is unchanged. A key an arm may not see is ABSENT from its observation, not empty. The barrier test
asserts ``set(obs) == set(OBS_KEYS) | ALLOWED[arm]`` for every arm, tick and ant.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from antelligence.kernel.colony.episode import COLONY_KEY_NAMES, ArmSpec
from antelligence.worlds.tumor_mri.world import OBS_KEYS

COLONY_OBS_KEYS: Tuple[str, ...] = ("look", "uptake_reading", "washout_reading", "reading_log", "marks_near",
                                    "messages_in", "zone_need", "colony_note", "region_status", "visited_self",
                                    "card_self", "cards_public", "notebook_self")
assert COLONY_OBS_KEYS == COLONY_KEY_NAMES

_SOLO = ("look", "uptake_reading", "washout_reading", "reading_log", "zone_need", "colony_note", "visited_self",
         "card_self", "notebook_self")
_MARKS = _SOLO + ("marks_near", "region_status", "cards_public")
_TALK = _MARKS + ("messages_in",)

ALLOWED: Dict[str, Tuple[str, ...]] = {
    "S": (),
    "S-solo": _SOLO,
    "S-marks": _MARKS,
    "S-talk": _TALK,
    "S-marks-displaced": _MARKS,
    "S-marks-perfect": _MARKS,
    "S-marks-quiet": _MARKS,
    "S-talk-shuffled": _TALK,
    "S'": (),          # the oracle reads the scorer and F1-F3 directly; it is not an ant arm
}

ANT_ARMS: Tuple[str, ...] = tuple(a for a in ALLOWED if a != "S'")


def expected_keys(arm: str) -> FrozenSet[str]:
    """``OBS_KEYS | ALLOWED[arm]``: exactly the keys an observation in ``arm`` must have."""
    return frozenset(OBS_KEYS) | frozenset(ALLOWED[arm])


def arm_spec(arm: str) -> ArmSpec:
    """The kernel's view of an arm. ``S`` is plain v4.1 (colony off); ``S'`` is not an ant arm."""
    if arm not in ANT_ARMS:
        raise ValueError(f"{arm!r} is not an ant arm (one of {ANT_ARMS})")
    allowed = frozenset(ALLOWED[arm])
    return ArmSpec(name=arm, colony=arm != "S", reads_marks="marks_near" in allowed, reads_talk="messages_in" in allowed,
                   shuffled=arm == "S-talk-shuffled", allowed_keys=allowed)
