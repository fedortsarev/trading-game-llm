"""Benchmark reader: run a model over many seeds (rotating seats), score regret and
calibration, and report a *distribution* — never a point.

A benchmark claim must be reproducible, so a `Suite` content-addresses the full tuple
that determines a score: rules + seeds + model id + prompt version + protocol version.
A leaderboard entry cites that hash, e.g.
  "model X: regret -2.1 (95% CI [-3.4, -0.8]) over 20 seeds x 4 seats on suite #abc123".

Benchmark vs. research sandbox is a one-bit difference: the suite is frozen+hashed
here; `readers/research.py` runs the same machinery without pinning.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from agents.bots import FairValueBot
from engine.rules import Rules
from protocol.protocol import PROTOCOL_VERSION
from store.log import canonical_json

from .oracle import AgentFactory, SeatResult, _default_factory, regret_for_seat


# --------------------------------------------------------------------------
# Suite — the content-addressed (rules, seeds, model, prompt, protocol) tuple
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Suite:
    rules: Rules
    seeds: tuple[int, ...]
    model_id: str
    prompt_version: str = "n/a"  # hash of the prompt template (LLMs); "n/a" for bots
    protocol_version: str = PROTOCOL_VERSION

    @property
    def hash(self) -> str:
        blob = canonical_json({
            "rules": self.rules.model_dump(),
            "seeds": list(self.seeds),
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "protocol_version": self.protocol_version,
        })
        return hashlib.sha256(blob.encode("ascii")).hexdigest()


# --------------------------------------------------------------------------
# Distribution — we always report a spread, not a single number
# --------------------------------------------------------------------------
@dataclass
class Distribution:
    n: int
    mean: float
    std: float
    min: float
    max: float
    p10: float
    p50: float
    p90: float
    ci95: tuple[float, float]


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def distribution(values: list[float]) -> Distribution:
    n = len(values)
    if n == 0:
        raise ValueError("cannot summarize an empty sample")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    half = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    s = sorted(values)
    return Distribution(
        n=n, mean=mean, std=std, min=s[0], max=s[-1],
        p10=_quantile(s, 0.10), p50=_quantile(s, 0.50), p90=_quantile(s, 0.90),
        ci95=(mean - half, mean + half),
    )


# --------------------------------------------------------------------------
# Matchup result + the evaluation harness
# --------------------------------------------------------------------------
@dataclass
class MatchupResult:
    suite_hash: str
    model_id: str
    records: list[SeatResult] = field(default_factory=list)
    regret: Distribution | None = None
    calibration_mae: Distribution | None = None


def evaluate(
    model_factory: AgentFactory,
    suite: Suite,
    *,
    opponent_factory: AgentFactory = _default_factory,
    reference_factory: AgentFactory = _default_factory,
    rotate: bool = True,
) -> MatchupResult:
    n = suite.rules.n_players
    seats = range(n) if rotate else [0]
    records: list[SeatResult] = []
    for seed in suite.seeds:
        for seat in seats:
            records.append(regret_for_seat(
                model_factory, suite.rules, seed, seat,
                opponent_factory=opponent_factory,
                reference_factory=reference_factory,
            ))

    regrets = [r.regret for r in records]
    maes = [r.calibration_mae for r in records if r.calibration_mae is not None]
    return MatchupResult(
        suite_hash=suite.hash,
        model_id=suite.model_id,
        records=records,
        regret=distribution(regrets),
        calibration_mae=distribution(maes) if maes else None,
    )


def leaderboard_entry(suite: Suite, result: MatchupResult) -> str:
    r = result.regret
    n_seeds = len(suite.seeds)
    n_seats = suite.rules.n_players
    cal = (f", calibration MAE {result.calibration_mae.mean:.2f}"
           if result.calibration_mae else "")
    return (
        f"{suite.model_id}: regret {r.mean:+.2f} "
        f"(95% CI [{r.ci95[0]:+.2f}, {r.ci95[1]:+.2f}], n={r.n}){cal} "
        f"over {n_seeds} seeds x {n_seats} seats "
        f"on suite #{suite.hash[:12]}, prompt #{suite.prompt_version}"
    )


def main() -> None:
    """Offline demo: score a biased FairValueBot against an unbiased reference field."""
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)
    suite = Suite(rules=rules, seeds=tuple(range(20)), model_id="FairValueBot(bias=3)")

    result = evaluate(lambda pid: FairValueBot(pid, bias=3.0), suite)
    print(leaderboard_entry(suite, result))
    print(f"regret  min/p50/max = {result.regret.min:+.1f} / "
          f"{result.regret.p50:+.1f} / {result.regret.max:+.1f}")


if __name__ == "__main__":
    main()
