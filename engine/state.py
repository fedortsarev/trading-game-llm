"""Authoritative game state plus dealing, cash/PnL accounting, and settlement.

Accounting is cash-based and exact:
  buy size @ p  -> cash -= size*p, position += size
  sell size @ p -> cash += size*p, position -= size
Marked PnL = cash + position * mark (mark = last clearing price).
Final PnL  = cash + position * settlement.
This avoids weighted-average-entry bookkeeping and is trivially reproducible.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from .rules import Rules


class GameState(BaseModel):
    rules: Rules
    seed: int

    round: int = 0  # 0 before the first step; rounds are 1-indexed once playing
    next_seq: int = 0  # monotonic event counter

    hands: dict[int, list[int]] = Field(default_factory=dict)  # player_id -> cards
    public_cards: list[int] = Field(default_factory=list)

    positions: dict[int, int] = Field(default_factory=dict)
    cash: dict[int, float] = Field(default_factory=dict)

    tape: list[dict] = Field(default_factory=list)  # public Trade dicts
    last_clearing_price: float | None = None

    settled: bool = False

    # ---- derived ----------------------------------------------------------
    @property
    def settlement_value(self) -> int:
        """Ground-truth settlement = sum of every card in play."""
        return sum(sum(cards) for cards in self.hands.values()) + sum(self.public_cards)

    def marked_pnl(self, pid: int) -> float:
        mark = self.last_clearing_price
        pos = self.positions.get(pid, 0)
        if mark is None:
            return self.cash.get(pid, 0.0)
        return self.cash.get(pid, 0.0) + pos * mark

    def round_rng(self, round_no: int) -> random.Random:
        """Deterministic per-round RNG for auction tie-breaks."""
        return random.Random(self.seed * 1_000_003 + round_no)


def deal(rules: Rules, seed: int) -> GameState:
    """Draw all cards deterministically and initialize accounts."""
    rng = random.Random(seed)
    hands: dict[int, list[int]] = {}
    for pid in range(rules.n_players):
        hands[pid] = [
            rng.randint(rules.card_min, rules.card_max) for _ in range(rules.k_private)
        ]
    public = [rng.randint(rules.card_min, rules.card_max) for _ in range(rules.m_public)]
    return GameState(
        rules=rules,
        seed=seed,
        round=1,
        hands=hands,
        public_cards=public,
        positions={pid: 0 for pid in range(rules.n_players)},
        cash={pid: 0.0 for pid in range(rules.n_players)},
    )
