"""Job thresholds (INTERFACES 10, goal 6.8). Domain-free; our own code."""
from __future__ import annotations

JOBS = ("dose", "verify", "explore")  # tie-break order: dose > verify > explore
START, FLOOR, CEIL, STEP_UP, STEP_DOWN = 0.5, 0.1, 1.0, 0.05, 0.1


class Thresholds:
    """Per-ant response thresholds. Reset at the start of every run."""

    def __init__(self) -> None:
        self.theta = {j: START for j in JOBS}

    def reset(self) -> None:
        self.theta = {j: START for j in JOBS}

    def probabilities(self, stimuli: dict[str, float]) -> dict[str, float]:
        out = {}
        for j in JOBS:
            s2, t2 = float(stimuli.get(j, 0.0)) ** 2, self.theta[j] ** 2
            out[j] = s2 / (s2 + t2) if s2 + t2 > 0 else 0.0
        return out

    def choose(self, stimuli: dict[str, float]) -> str:
        p = self.probabilities(stimuli)
        return max(JOBS, key=lambda j: p[j])  # max keeps first of equals

    def update(self, job: str, success: bool) -> None:
        """Call once per turn spent in `job`."""
        t = self.theta[job]
        self.theta[job] = max(FLOOR, t - STEP_DOWN) if success else min(CEIL, t + STEP_UP)
