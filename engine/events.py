"""The typed, append-only event stream — the engine's single output.

Every event carries a visibility tier; readers filter by tier, the engine emits
once. `audience` names the specific agent for AGENT-tier events (else None).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Visibility(str, Enum):
    AGENT = "agent"  # visible only to `audience` (that agent's private info)
    SPECTATOR = "spectator"  # safe to show an audience mid-game
    RESEARCHER = "researcher"  # everything, always


class Event(BaseModel):
    seq: int  # monotonic, per-game
    round: int
    type: str
    payload: dict
    visibility: Visibility
    audience: int | None = None  # player id for AGENT-tier events; else None


# --- Event type catalog (Phase 1) -------------------------------------------
GAME_START = "game_start"
DEAL = "deal"
OBSERVATION_SENT = "observation_sent"
ACTION_RECEIVED = "action_received"
ACTION_REJECTED = "action_rejected"
AUCTION_CLEARED = "auction_cleared"
FILL = "fill"
POSITION_UPDATE = "position_update"
ROUND_END = "round_end"
SETTLEMENT = "settlement"
GAME_END = "game_end"
# model_raw_output is deferred to Phase 2 (no LLM yet).
