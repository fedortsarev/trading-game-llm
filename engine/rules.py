"""Parameterized, frozen game rules.

Everything that defines a game's shape lives here. A game is fully determined by
`(Rules, seed)` — that tuple is what a benchmark suite content-addresses.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class Rules(BaseModel, frozen=True):
    n_players: int = 4
    k_private: int = 2  # private cards per player
    m_public: int = 0  # shared public cards
    total_rounds: int = 5  # R

    # Deck: each card value is drawn i.i.d. uniformly from [card_min, card_max].
    card_min: int = 1
    card_max: int = 10

    risk_limit: int = 5  # max absolute position any agent may hold

    @model_validator(mode="after")
    def _check(self) -> "Rules":
        if self.n_players < 2:
            raise ValueError("n_players must be >= 2")
        if self.k_private < 1:
            raise ValueError("k_private must be >= 1")
        if self.m_public < 0:
            raise ValueError("m_public must be >= 0")
        if self.total_rounds < 1:
            raise ValueError("total_rounds must be >= 1")
        if self.card_max < self.card_min:
            raise ValueError("card_max must be >= card_min")
        if self.risk_limit < 1:
            raise ValueError("risk_limit must be >= 1")
        return self

    @property
    def total_cards(self) -> int:
        return self.n_players * self.k_private + self.m_public

    @property
    def card_mean(self) -> float:
        return (self.card_min + self.card_max) / 2.0
