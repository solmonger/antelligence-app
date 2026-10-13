"""v4.1 budget projection with the goal's cut orders (brief S5).

Arms (cases x seeds): S, S-prime 80x2; SW-B 80x2; SW-noBoard 80x2; FW 40x1; SW-D 12x1; SW-noQueen,
SW-staticQueen, SW-Bonsai 40x1; SW-Byz-local, SW-Byz-chain 20x1; SW-B-missingFLAIR 20x1.
Budgets: Nous $8 (stop $7.50), RunPod $10 (stop $9.50), testnet 0.04 ETH. ``project`` is pure.
Floors: SW-B and SW-noBoard never below 80x2 cases/seeds, FW never below 30 cases.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Tuple

BUDGETS = {"nous": 8.00, "runpod": 10.00}
STOPS = {"nous": 7.50, "runpod": 9.50}
TESTNET_ETH = 0.04
PLAN: Dict[str, Tuple[int, int]] = {
    "S": (80, 2), "S-prime": (80, 2), "SW-B": (80, 2), "SW-noBoard": (80, 2), "FW": (40, 1), "SW-D": (12, 1),
    "SW-noQueen": (40, 1), "SW-staticQueen": (40, 1), "SW-Bonsai": (40, 1), "SW-Byz-local": (20, 1),
    "SW-Byz-chain": (20, 1), "SW-B-missingFLAIR": (20, 1),
}
DEFAULT_SANDBOX_ROUNDS = 5  # assumption: the goal's starting sandbox depth is not in the brief; pass it explicitly
FLOORS = {"SW-B": 80, "SW-noBoard": 80, "FW": 30}
SEED_FLOORS = {"SW-B": 2, "SW-noBoard": 2}

# (step name, target) in the order the goal fixes. "sandbox" is a pseudo-arm: target = rounds.
NOUS_CUT_ORDER: Tuple[Tuple[str, int], ...] = (
    ("sandbox", 3), ("FW", 30), ("SW-B-missingFLAIR", 20), ("SW-Bonsai", 20), ("SW-staticQueen", 20))
RUNPOD_CUT_ORDER: Tuple[Tuple[str, int], ...] = (
    ("SW-B-missingFLAIR", 10), ("SW-staticQueen", 10), ("SW-noQueen", 10), ("SW-Byz-local", 10),
    ("SW-Byz-chain", 10), ("SW-D", 8))


class FloorViolation(ValueError):
    """A cut would take a protected arm below its floor."""


@dataclasses.dataclass(frozen=True)
class Plan:
    cases: Mapping[str, int]
    seeds: Mapping[str, int]
    sandbox_rounds: int = DEFAULT_SANDBOX_ROUNDS

    @staticmethod
    def default(sandbox_rounds: int = DEFAULT_SANDBOX_ROUNDS) -> "Plan":
        return Plan({a: c for a, (c, _) in PLAN.items()}, {a: s for a, (_, s) in PLAN.items()}, sandbox_rounds)

    def runs(self, arm: str) -> int:
        return self.cases[arm] * self.seeds[arm]


def check_floors(plan: Plan) -> None:
    for arm, floor in FLOORS.items():
        if plan.cases.get(arm, 0) < floor:
            raise FloorViolation(f"{arm} cases {plan.cases.get(arm)} below floor {floor}")
    for arm, floor in SEED_FLOORS.items():
        if plan.seeds.get(arm, 0) < floor:
            raise FloorViolation(f"{arm} seeds {plan.seeds.get(arm)} below floor {floor}")


def totals(plan: Plan, per_run: Mapping[str, Mapping[str, float]], sandbox_per_round: Mapping[str, float]) -> Dict[str, float]:
    """Projected spend by provider. ``per_run[arm][provider]`` is the mean cost of one run."""
    out = {p: 0.0 for p in BUDGETS}
    for arm in plan.cases:
        for p in out:
            out[p] += plan.runs(arm) * per_run.get(arm, {}).get(p, 0.0)
    for p in out:
        out[p] += plan.sandbox_rounds * sandbox_per_round.get(p, 0.0)
    return out


def _apply(plan: Plan, name: str, target: int) -> Plan:
    if name == "sandbox":
        return dataclasses.replace(plan, sandbox_rounds=min(plan.sandbox_rounds, target))
    cases = dict(plan.cases)
    cases[name] = min(cases[name], target)
    return dataclasses.replace(plan, cases=cases)


def project(per_run: Mapping[str, Mapping[str, float]], *, sandbox_per_round: Optional[Mapping[str, float]] = None,
            plan: Optional[Plan] = None, stops: Mapping[str, float] = STOPS) -> Dict[str, Any]:
    """Apply cuts in the goal's order, per provider, until the projection is at or under its stop."""
    sandbox_per_round = sandbox_per_round or {}
    current = plan or Plan.default()
    check_floors(current)
    initial = totals(current, per_run, sandbox_per_round)
    decisions: List[Dict[str, Any]] = []
    for provider, order in (("nous", NOUS_CUT_ORDER), ("runpod", RUNPOD_CUT_ORDER)):
        for name, target in order:
            before = totals(current, per_run, sandbox_per_round)[provider]
            if before <= stops[provider]:
                break
            nxt = _apply(current, name, target)
            if nxt == current:
                continue
            check_floors(nxt)
            current = nxt
            decisions.append({"provider": provider, "cut": f"{name} -> {target}", "projected_before": before,
                              "projected_after": totals(current, per_run, sandbox_per_round)[provider],
                              "stop": stops[provider]})
    final = totals(current, per_run, sandbox_per_round)
    return {"plan": current, "decisions": decisions, "projected_initial": initial, "projected_final": final,
            "stops": dict(stops), "over_after_cuts": [p for p in BUDGETS if final[p] > stops[p]],
            "unpriced_arms": sorted(a for a in current.cases if a not in per_run)}
