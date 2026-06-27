"""Synchronous orchestrator: drives the game loop and persists the event stream.

start_game -> R x step -> settle, asking each agent for an action via the single
`Agent.act` interface. For LLM agents, the orchestrator also stamps the model's raw
output into the same append-only log at RESEARCHER tier (the model call happens in
the worker, outside the pure engine). asyncio + per-agent deadlines are a later
follow-up; today each model call blocks with its own timeout and degrades to pass.
"""

from __future__ import annotations

import os
from pathlib import Path

from agents.base import Agent
from agents.bots import FairValueBot
from engine.engine import agent_view, settle, start_game, step
from engine.events import MODEL_RAW_OUTPUT, Event, Visibility
from engine.rules import Rules
from store.log import EventLogWriter, config_hash


def _raw_output_event(state, round_no: int, agent: Agent) -> Event | None:
    """Build a RESEARCHER-tier model_raw_output event from an agent's last trace.

    Returns None for agents (e.g. bots) that expose no trace. Assigns the next
    monotonic seq and bumps the counter so the log stays strictly ordered.
    """
    trace = getattr(agent, "last_trace", None)
    if trace is None:
        return None
    payload = getattr(agent, "trace_payload", lambda: {})()
    e = Event(
        seq=state.next_seq,
        round=round_no,
        type=MODEL_RAW_OUTPUT,
        payload=payload,
        visibility=Visibility.RESEARCHER,
        audience=None,  # researcher-tier; the player_id rides inside the payload
    )
    state.next_seq += 1
    return e


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
            round_no = state.round
            actions = {pid: agents[pid].act(agent_view(state, pid))
                       for pid in range(rules.n_players)}
            state, events = step(state, actions)
            sink(events)
            # After the game events, stamp any model raw output for this round
            # (RESEARCHER tier), keeping the global seq monotonic.
            raw = [e for pid in range(rules.n_players)
                   if (e := _raw_output_event(state, round_no, agents[pid])) is not None]
            sink(raw)

        state, events = settle(state)
        sink(events)
    finally:
        if writer is not None:
            writer.close()

    return all_events


def _maybe_llm_seat0(player_id: int) -> Agent | None:
    """Seat 0 plays via an LLM if a provider key is set, else None (use a bot).

    ANTHROPIC_API_KEY -> Claude (set TG_MODEL to override the model);
    OPENAI_API_KEY    -> OpenAI.
    """
    from agents.llm_worker import LLMAgent

    if os.environ.get("ANTHROPIC_API_KEY"):
        from agents.llm_client import AnthropicClient
        model = os.environ.get("TG_MODEL", "claude-sonnet-4-6")
        return LLMAgent(player_id, AnthropicClient(model), model)
    if os.environ.get("OPENAI_API_KEY"):
        from agents.llm_client import OpenAIClient
        model = os.environ.get("TG_MODEL", "gpt-4.1")
        return LLMAgent(player_id, OpenAIClient(model), model)
    return None


def main() -> None:
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)
    seed = 42
    agents: dict[int, Agent] = {pid: FairValueBot(pid) for pid in range(rules.n_players)}

    llm = _maybe_llm_seat0(0)
    if llm is not None:
        agents[0] = llm
        print(f"Seat 0 = LLM ({llm.model}); seats 1-3 = FairValueBot")
    else:
        print("No API key found; all seats = FairValueBot "
              "(set ANTHROPIC_API_KEY or OPENAI_API_KEY for an LLM seat)")

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
