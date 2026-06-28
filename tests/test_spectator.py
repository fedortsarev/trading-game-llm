from pathlib import Path

from agents.bots import FairValueBot
from engine.engine import agent_view, start_game, step
from engine.events import Visibility
from engine.rules import Rules
from orchestrator.runner import run_game
from readers import spectator


def bot_events(seed=42):
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)
    agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
    return rules, run_game(rules, seed, agents)


def test_renders_full_game():
    rules, events = bot_events()
    text = spectator.render(events)
    assert "TRADING GAME" in text
    assert "SETTLEMENT" in text
    assert "ranking:" in text
    for rnd in range(1, rules.total_rounds + 1):
        assert f"Round {rnd}" in text


def test_consumes_spectator_tier_only():
    # Rendering from the full stream must equal rendering from a pre-filtered
    # spectator-only stream — proof the renderer ignores AGENT/RESEARCHER events.
    _, events = bot_events()
    spec_only = [e for e in events if e.visibility == Visibility.SPECTATOR]
    assert spectator.render(events) == spectator.render(spec_only)


def test_no_private_information_leaks():
    _, events = bot_events()
    text = spectator.render(events)
    # Private/researcher-only artifacts must never appear in the spectator VOD.
    assert "own_cards" not in text
    assert "raw_text" not in text
    assert "hands" not in text

    # The full deal (researcher tier) lists every hand; none may surface verbatim.
    hands = next(e.payload["hands"] for e in events
                 if e.type == "deal" and e.visibility == Visibility.RESEARCHER)
    for pid, cards in hands.items():
        assert str(cards) not in text


def test_reconstructs_from_log_file_alone(tmp_path):
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=4)
    agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
    log = tmp_path / "g.jsonl"
    run_game(rules, 7, agents, log)

    text = spectator.render(spectator.load(log))  # from the file, nothing else
    assert "SETTLEMENT" in text
    assert "Round 4" in text


def test_renderer_has_zero_engine_coupling():
    src = Path(spectator.__file__).read_text()
    # The renderer module must not import the engine package at all. (The optional
    # demo-generation path inside main() lazy-imports it, which is fine.)
    assert "\nfrom engine" not in src
    assert "\nimport engine" not in src


def test_positions_tracked_from_fills():
    _, events = bot_events()
    text = spectator.render(events)
    assert "positions:" in text
