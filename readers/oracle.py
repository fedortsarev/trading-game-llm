"""Counterfactual regret: how much better (or worse) did a policy do than a
reference policy, on the *same* deal?

    regret = PnL(model in seat s)  -  PnL(reference in seat s)

both played against the same opponent field on the same (rules, seed). This strips
out card luck and measures skill (design §7.1).

Why the counterfactual is clean here: the default opponent/reference policy,
`FairValueBot`, is *non-reactive* — it prices only off its own cards and the deck
mean, ignoring the tape and others' quotes. So the opponents submit identical orders
in both runs; the only thing that differs is seat s's own policy. (Minor documented
caveat: opponents' realized *fills* can still differ via risk-capping when seat s
trades differently — a mechanical second-order effect. Both runs stay fully
deterministic and reproducible.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agents.base import Agent
from agents.bots import FairValueBot
from engine.rules import Rules
from orchestrator.runner import run_game

from . import metrics

AgentFactory = Callable[[int], Agent]


@dataclass
class SeatResult:
    seed: int
    seat: int
    pnl_model: float
    pnl_reference: float
    regret: float
    calibration_mae: float | None


def _default_factory(pid: int) -> Agent:
    return FairValueBot(pid)


def regret_for_seat(
    model_factory: AgentFactory,
    rules: Rules,
    seed: int,
    seat: int,
    *,
    opponent_factory: AgentFactory = _default_factory,
    reference_factory: AgentFactory = _default_factory,
) -> SeatResult:
    def field(seat_factory: AgentFactory) -> dict[int, Agent]:
        return {pid: (seat_factory(pid) if pid == seat else opponent_factory(pid))
                for pid in range(rules.n_players)}

    model_events = run_game(rules, seed, field(model_factory))
    ref_events = run_game(rules, seed, field(reference_factory))

    pnl_model = metrics.final_pnl(model_events)[seat]
    pnl_ref = metrics.final_pnl(ref_events)[seat]
    calib = metrics.calibration(model_events).get(seat)

    return SeatResult(
        seed=seed,
        seat=seat,
        pnl_model=pnl_model,
        pnl_reference=pnl_ref,
        regret=pnl_model - pnl_ref,
        calibration_mae=calib.mae if calib is not None else None,
    )
