"""Colony kernel (v4.2, goal sections 6.1-6.3, 6.5, 6.9): marks, messages, quorum, the quiet rule, the episode driver.

Domain-free: nothing here imports a world. A world reaches this package through the small ``ColonyWorld`` protocol in
:mod:`antelligence.kernel.colony.episode`. Contract: ``docs/v4.2/INTERFACES.md`` sections 1-5, 7, 12.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def seed64(*parts: Any) -> int:
    """INTERFACES v4.2 section 1: ``sha256("|".join(str(p) for p in parts))[:8]`` as a big-endian integer."""
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")


def rng(*parts: Any) -> np.random.Generator:
    """INTERFACES v4.2 section 1: ``parts[0]`` is the stream name. Nothing else seeds a colony draw."""
    return np.random.default_rng(seed64(*parts))
