"""The single agent interface. Rule-based bots and (later) LLM workers both
implement `act` — the engine and orchestrator never know which is which.
"""

from __future__ import annotations

from typing import Protocol

from protocol.protocol import Action, AgentView


class Agent(Protocol):
    player_id: int

    def act(self, view: AgentView) -> Action:
        ...
