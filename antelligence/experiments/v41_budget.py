"""v4.1 budget projection with the goal's cut orders (brief S5).

Arms (cases x seeds): S, S-prime 80x2; SW-B 80x2; SW-noBoard 80x2; FW 40x1; SW-D 12x1; SW-noQueen,
SW-staticQueen, SW-Bonsai 40x1; SW-Byz-local, SW-Byz-chain 20x1; SW-B-missingFLAIR 20x1.
Budgets: Nous $8 (stop $7.50), RunPod $10 (stop $9.50), testnet 0.04 ETH. ``project`` is pure.
Floors: SW-B and SW-noBoard never below 80x2 cases/seeds, FW never below 30 cases.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from antelligence.experiments import v4_budget
from antelligence.kernel.decision import DecisionAbort

BUDGETS = {"nous": 8.00, "runpod": 10.00}
STOPS = {"nous": 7.50, "runpod": 9.50}
TESTNET_ETH = 0.04
PLAN: Dict[str, Tuple[int, int]] = {
    "S": (80, 2), "S-prime": (80, 2), "SW-B": (80, 2), "SW-noBoard": (80, 2), "FW": (40, 1), "SW-D": (12, 1),
    "SW-noQueen": (40, 1), "SW-staticQueen": (40, 1), "SW-Bonsai": (40, 1), "SW-Byz-local": (20, 1),
    "SW-Byz-chain": (20, 1), "SW-B-missingFLAIR": (20, 1),
}
DEFAULT_SANDBOX_ROUNDS = 4  # goal 4.6 and 8: 4 rounds x 60 runs; the first Nous cut is 3 rounds
FLOORS = {"SW-B": 80, "SW-noBoard": 80, "FW": 30}
SEED_FLOORS = {"SW-B": 2, "SW-noBoard": 2}

# Goal 8 cut orders, lead's reading: each numbered step is atomic. A target of 0 drops the arm entirely.
# "sandbox" is a pseudo-arm whose target is the number of rounds.
Step = Tuple[Tuple[str, int], ...]
NOUS_CUT_ORDER: Tuple[Step, ...] = (
    (("sandbox", 3),),
    (("FW", 30),),
    (("SW-B-missingFLAIR", 20), ("SW-Bonsai", 20), ("SW-staticQueen", 20)),
)
RUNPOD_CUT_ORDER: Tuple[Step, ...] = (
    (("SW-B-missingFLAIR", 0),),
    (("SW-staticQueen", 0),),
    (("SW-noQueen", 0),),
    (("SW-Byz-local", 10), ("SW-Byz-chain", 10)),
    (("SW-D", 8),),
)


def step_name(step: Step) -> str:
    return "; ".join(f"{n} -> {v}" for n, v in step)


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


def _apply(plan: Plan, step: Step) -> Plan:
    """Apply one whole step; a cut never raises an arm above its current size."""
    rounds, cases = plan.sandbox_rounds, dict(plan.cases)
    for name, target in step:
        if name == "sandbox":
            rounds = min(rounds, target)
        else:
            cases[name] = min(cases[name], target)
    return dataclasses.replace(plan, cases=cases, sandbox_rounds=rounds)


def project(per_run: Mapping[str, Mapping[str, float]], *, sandbox_per_round: Optional[Mapping[str, float]] = None,
            plan: Optional[Plan] = None, stops: Mapping[str, float] = STOPS) -> Dict[str, Any]:
    """Apply cuts in the goal's order, per provider, until the projection is at or under its stop."""
    sandbox_per_round = sandbox_per_round or {}
    current = plan or Plan.default()
    check_floors(current)
    initial = totals(current, per_run, sandbox_per_round)
    decisions: List[Dict[str, Any]] = []
    for provider, order in (("nous", NOUS_CUT_ORDER), ("runpod", RUNPOD_CUT_ORDER)):
        for step in order:
            before = totals(current, per_run, sandbox_per_round)[provider]
            if before <= stops[provider]:
                break
            nxt = _apply(current, step)
            if nxt == current:
                continue
            check_floors(nxt)
            current = nxt
            decisions.append({"provider": provider, "cut": step_name(step), "projected_before": before,
                              "projected_after": totals(current, per_run, sandbox_per_round)[provider],
                              "stop": stops[provider]})
    final = totals(current, per_run, sandbox_per_round)
    return {"plan": current, "decisions": decisions, "projected_initial": initial, "projected_final": final,
            "stops": dict(stops), "over_after_cuts": [p for p in BUDGETS if final[p] > stops[p]],
            "unpriced_arms": sorted(a for a in current.cases if a not in per_run)}


# --- runtime hard stops (S7) -------------------------------------------------------------------------------------

USAGE_LEDGER = Path("ledger/v4.1-usage.jsonl")
POD_COSTS = Path("ledger/v4.1-pod-costs.jsonl")


class BudgetRefused(DecisionAbort):
    """A paid call was refused because spend reached the hard stop (Nous 7.50, RunPod 9.50).

    A :class:`DecisionAbort`: never retried, never turned into an error decision or a queen fallback."""


def usage_spend(rows: Sequence[Mapping[str, Any]]) -> float:
    """Nous dollars from the usage ledger. Rows with ``cost_usd`` count as written; preflight probe rows (provider
    ``nous``, token ``usage``, no cost field) are priced from the table so they are not free."""
    from antelligence.providers.ledger import PRICES, cost_usd

    total = 0.0
    for row in rows:
        if v4_budget.provider_of(row) != "nous" and row.get("provider") != "nous":
            continue
        if row.get("cost_usd"):
            total += float(row["cost_usd"])
        elif isinstance(row.get("usage"), Mapping) and row.get("model") in PRICES:
            u = row["usage"]
            total += cost_usd(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), PRICES[row["model"]])
    return total


def spend_now(usage: Optional[Path] = USAGE_LEDGER, pods: Optional[Path] = POD_COSTS,
              now: Optional[datetime] = None) -> Dict[str, float]:
    return {"nous": usage_spend(v4_budget.read_jsonl(usage)),
            "runpod": v4_budget.pod_spend(v4_budget.read_jsonl(pods), now)}


def check_or_refuse(provider: str, *, usage: Optional[Path] = USAGE_LEDGER, pods: Optional[Path] = POD_COSTS,
                    pending_usd: float = 0.0, now: Optional[datetime] = None) -> float:
    """Raise :class:`BudgetRefused` when spend (plus a pending cost) is at or past the stop; else return the spend."""
    if provider not in STOPS:
        raise KeyError(f"unknown provider {provider!r}")
    spent = spend_now(usage, pods, now)[provider]
    if spent + pending_usd >= STOPS[provider]:
        raise BudgetRefused(f"{provider} hard stop: ${spent:.4f} spent + ${pending_usd:.4f} pending >= ${STOPS[provider]:.2f}")
    return spent
