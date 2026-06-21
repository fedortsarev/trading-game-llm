from agents.bots import FairValueBot
from engine.engine import settle, start_game, step
from engine.rules import Rules
from engine.state import deal
from orchestrator.runner import run_game


def test_settlement_value_is_sum_of_all_cards():
    rules = Rules(n_players=3, k_private=2, m_public=2, total_rounds=3)
    state = deal(rules, seed=7)
    expected = sum(sum(h) for h in state.hands.values()) + sum(state.public_cards)
    assert state.settlement_value == expected


def test_game_is_zero_sum():
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)
    agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
    events = run_game(rules, seed=42, agents=agents)
    final = next(e for e in reversed(events) if e.type == "settlement")
    total = sum(final.payload["final_pnl"].values())
    assert abs(total) < 1e-9


def test_positions_net_to_zero_each_round():
    rules = Rules(n_players=4, k_private=2, total_rounds=4)
    agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
    state, _ = start_game(rules, seed=11)
    for _ in range(rules.total_rounds):
        from engine.engine import agent_view
        actions = {pid: agents[pid].act(agent_view(state, pid))
                   for pid in range(rules.n_players)}
        state, _ = step(state, actions)
        assert sum(state.positions.values()) == 0


def test_final_pnl_matches_cash_plus_position_times_settlement():
    rules = Rules(n_players=3, k_private=2, total_rounds=3)
    agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
    state, _ = start_game(rules, seed=3)
    from engine.engine import agent_view
    for _ in range(rules.total_rounds):
        actions = {pid: agents[pid].act(agent_view(state, pid))
                   for pid in range(rules.n_players)}
        state, _ = step(state, actions)
    sv = state.settlement_value
    _, events = settle(state)
    final = next(e for e in events if e.type == "settlement").payload["final_pnl"]
    for pid in range(rules.n_players):
        expected = state.cash[pid] + state.positions[pid] * sv
        assert abs(final[pid] - expected) < 1e-9
