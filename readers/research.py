"""Research sandbox reader.

Per design §6, benchmark and sandbox are the *same engine and the same scoring
machinery* — they differ by a single bit: a benchmark freezes and hashes its config
into a `Suite`, a sandbox does not. So researchers may sweep rules/seeds freely and
still get regret + calibration distributions, just without a content-addressed claim.
"""

from __future__ import annotations

from collections.abc import Sequence

from agents.bots import FairValueBot
from engine.rules import Rules

from .benchmark import MatchupResult, Suite, evaluate
from .oracle import AgentFactory, _default_factory


def research_run(
    model_factory: AgentFactory,
    rules: Rules,
    seeds: Sequence[int],
    *,
    model_id: str = "sandbox",
    opponent_factory: AgentFactory = _default_factory,
    reference_factory: AgentFactory = _default_factory,
    rotate: bool = True,
) -> MatchupResult:
    """Same evaluation as the benchmark, but the config is not frozen/hashed.

    The returned `MatchupResult.suite_hash` is still populated (the machinery is
    identical) but carries no claim of being a pinned suite — vary `rules`/`seeds`
    at will.
    """
    suite = Suite(rules=rules, seeds=tuple(seeds), model_id=model_id)
    return evaluate(
        model_factory, suite,
        opponent_factory=opponent_factory,
        reference_factory=reference_factory,
        rotate=rotate,
    )
