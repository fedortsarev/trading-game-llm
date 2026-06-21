"""Versioned protocol schemas shared by every agent (bots and, later, LLMs).

These are the *only* types that cross the engine <-> agent boundary. Bump
PROTOCOL_VERSION on any breaking change; the version travels inside AgentView so a
log records exactly which contract produced it.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "0.1.0"


class Card(BaseModel):
    value: int


class Trade(BaseModel):
    """A public tape entry: volume cleared at a uniform price in a given round."""

    round: int
    price: float
    size: int


class PublicState(BaseModel):
    round: int
    total_rounds: int
    n_players: int
    # Deck parameters are public knowledge — agents need them to price unknowns.
    k_private: int
    card_min: int
    card_max: int
    public_cards: list[Card]
    tape: list[Trade]
    last_clearing_price: float | None = None


class AgentView(BaseModel):
    """The per-agent slice of state. Contains only this agent's private info."""

    protocol_version: str = PROTOCOL_VERSION
    player_id: int
    own_cards: list[Card]
    position: int
    cash: float
    pnl: float  # marked-to-market with the last clearing price
    risk_limit: int
    public: PublicState


class Quote(BaseModel):
    bid: float | None = None
    ask: float | None = None
    bid_size: int = 0
    ask_size: int = 0


class Take(BaseModel):
    side: Literal["buy", "sell"]
    price: float
    size: int


class Action(BaseModel):
    # `cancel` is intentionally absent in Phase 1: it only has meaning in a
    # continuous order book, which is deferred.
    kind: Literal["quote", "take", "pass"]
    quote: Quote | None = None
    take: Take | None = None
    fair_value_estimate: float | None = None  # for later calibration scoring
    rationale: str | None = None  # spectator/researcher only; never echoed to agents


def export_json_schema() -> dict:
    """Action JSON Schema, for handing to tool-calling models in Phase 2."""
    return Action.model_json_schema()


if __name__ == "__main__":
    print(json.dumps(export_json_schema(), indent=2))
