import json

from agents.bots import FairValueBot
from agents.llm_client import LLMResult, MockLLMClient
from agents.llm_worker import LLMAgent, action_tool_schema
from engine.events import Visibility
from engine.rules import Rules
from orchestrator.runner import run_game
from protocol.protocol import AgentView, Card, PublicState


def make_view(player_id=0, own=(3, 8)):
    return AgentView(
        player_id=player_id,
        own_cards=[Card(value=v) for v in own],
        position=0,
        cash=0.0,
        pnl=0.0,
        risk_limit=5,
        public=PublicState(
            round=1, total_rounds=5, n_players=4, k_private=2,
            card_min=1, card_max=10, public_cards=[Card(value=4)], tape=[],
            last_clearing_price=None,
        ),
    )


VALID_QUOTE = {
    "kind": "quote",
    "quote": {"bid": 40.0, "ask": 42.0, "bid_size": 1, "ask_size": 1},
    "fair_value_estimate": 41.0,
    "rationale": "centered on fair value",
}


def test_valid_tool_input_becomes_action():
    agent = LLMAgent(0, MockLLMClient([LLMResult(tool_input=dict(VALID_QUOTE))]))
    action = agent.act(make_view())
    assert action.kind == "quote"
    assert action.quote.bid == 40.0 and action.quote.ask == 42.0
    assert action.fair_value_estimate == 41.0
    assert agent.last_trace.attempts == 1
    assert agent.last_trace.error is None
    assert agent.last_trace.final_kind == "quote"


def test_malformed_then_retry_then_pass():
    bad = LLMResult(tool_input={"kind": "leap"})  # invalid literal -> ValidationError
    agent = LLMAgent(0, MockLLMClient([bad, bad]))
    action = agent.act(make_view())
    assert action.kind == "pass"
    assert agent.last_trace.attempts == 2
    assert agent.last_trace.error is not None


def test_recovers_on_retry():
    bad = LLMResult(tool_input={"kind": "leap"})
    good = LLMResult(tool_input=dict(VALID_QUOTE))
    agent = LLMAgent(0, MockLLMClient([bad, good]))
    action = agent.act(make_view())
    assert action.kind == "quote"
    assert agent.last_trace.attempts == 2


def test_client_error_degrades_to_pass():
    agent = LLMAgent(0, MockLLMClient([LLMResult(None, error="ConnectionError: boom")]))
    action = agent.act(make_view())
    assert action.kind == "pass"
    assert agent.last_trace.error is not None


def test_tool_schema_is_self_contained():
    schema = action_tool_schema()
    blob = json.dumps(schema)
    assert "$ref" not in blob
    assert "$defs" not in schema
    # The inlined nested Quote/Take fields survived (quote is `Quote | None`,
    # so it lives under an anyOf rather than directly under properties).
    assert "quote" in schema["properties"]
    assert "bid" in blob and "ask" in blob and "fair_value_estimate" in blob


def test_prompt_only_contains_own_information():
    view = make_view(player_id=0, own=(3, 8))
    system, user = LLMAgent(0, MockLLMClient([])).build_prompt(view)
    snap = json.loads(user.split("(JSON):\n", 1)[1].rsplit("\n\n", 1)[0])
    assert snap["your_cards"] == [3, 8]
    assert snap["unknown_cards_count"] == (4 - 1) * 2
    # The only cards in the snapshot are this agent's own + the public cards;
    # no field enumerates other players' hands.
    assert snap["public_cards"] == [4]
    card_keys = {k for k in snap if "card" in k.lower()}
    assert card_keys == {"your_cards", "your_cards_sum", "public_cards",
                         "unknown_cards_count"}
    assert "hands" not in snap and "players" not in snap


def test_integration_llm_seat_with_bots():
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=3)
    client = MockLLMClient(lambda *a: LLMResult(tool_input=dict(VALID_QUOTE)),
                           model="mock-model")
    agents = {0: LLMAgent(0, client, "mock-model")}
    for pid in range(1, 4):
        agents[pid] = FairValueBot(pid)

    events = run_game(rules, seed=7, agents=agents)

    # Game completed.
    assert any(e.type == "settlement" for e in events)
    # The LLM was actually called once per round.
    assert len(client.calls) == rules.total_rounds

    raw = [e for e in events if e.type == "model_raw_output"]
    assert len(raw) == rules.total_rounds
    for e in raw:
        assert e.visibility == Visibility.RESEARCHER
        assert e.payload["player_id"] == 0  # only the LLM seat produces raw output
        assert e.payload["model"] == "mock-model"

    # Global sequence stays strictly monotonic with the raw-output events interleaved.
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def test_raw_output_never_leaks_to_agent_or_spectator_tier():
    rules = Rules(n_players=4, k_private=2, total_rounds=2)
    client = MockLLMClient(lambda *a: LLMResult(tool_input=dict(VALID_QUOTE)),
                           model="mock-model")
    agents = {0: LLMAgent(0, client, "mock-model")}
    for pid in range(1, 4):
        agents[pid] = FairValueBot(pid)
    events = run_game(rules, seed=1, agents=agents)
    for e in events:
        if e.type == "model_raw_output":
            assert e.visibility == Visibility.RESEARCHER
        if e.visibility == Visibility.AGENT:
            assert "raw_text" not in e.payload
