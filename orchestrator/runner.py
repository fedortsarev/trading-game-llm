"""Phase-1 orchestrator: a synchronous, bot-only game loop.

Drives start_game -> R x step -> settle, asking each agent for an action via the
single `Agent.act` interface and persisting every emitted event. asyncio,
per-agent deadlines, and timeout->pass arrive with the LLM worker in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

from agents.base import Agent
from agents.bots import FairValueBot
from engine.engine import agent_view, settle, start_game, step
from engine.events import Event
from engine.rules import Rules
from store.log import EventLogWriter, config_hash


def run_game(rules: Rules, seed: int, agents: dict[int, Agent],
             log_path: str | Path | None = None) -> list[Event]:
    """Play a full game. Returns every event; optionally also writes the JSONL log."""
    all_events: list[Event] = []
    writer = EventLogWriter(log_path) if log_path is not None else None

    def sink(events: list[Event]) -> None:
        for e in events:
            all_events.append(e)
            if writer is not None:
                writer.write(e)

    try:
        state, events = start_game(rules, seed)
        sink(events)

        for _ in range(rules.total_rounds):
            actions = {pid: agents[pid].act(agent_view(state, pid))
                       for pid in range(rules.n_players)}
            state, events = step(state, actions)
            sink(events)

        state, events = settle(state)
        sink(events)
    finally:
        if writer is not None:
            writer.close()

    return all_events


def main() -> None:
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)
    seed = 42
    agents: dict[int, Agent] = {pid: FairValueBot(pid) for pid in range(rules.n_players)}

    log_path = Path("logs") / f"game_{config_hash(rules.model_dump(), seed)[:12]}.jsonl"
    events = run_game(rules, seed, agents, log_path)

    final = next(e for e in reversed(events) if e.type == "settlement")
    print(f"Played {rules.total_rounds}-round game, seed={seed}")
    print(f"Settlement value: {final.payload['settlement_value']}")
    print(f"Final PnL: {final.payload['final_pnl']}")
    print(f"Ranking:   {final.payload['ranking']}")
    print(f"Log:       {log_path}  ({len(events)} events)")


if __name__ == "__main__":
    main()
