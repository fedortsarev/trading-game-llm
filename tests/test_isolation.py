from agents.bots import FairValueBot
from engine.engine import agent_view, start_game, step
from engine.events import Visibility
from engine.rules import Rules
from orchestrator.runner import run_game
from protocol.protocol import Action, Quote, Take
from store.log import canonical_json


def _play(seed=42):
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)
    agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
    return rules, run_game(rules, seed=seed, agents=agents)


def test_agent_events_never_carry_other_hands():
    rules, events = _play()
    # Ground-truth hands from the researcher-only deal event.
    hands = next(e.payload["hands"] for e in events
                 if e.type == "deal" and e.visibility == Visibility.RESEARCHER)
    hands = {int(k): v for k, v in hands.items()}

    for e in events:
        if e.visibility != Visibility.AGENT:
            continue
        assert e.audience is not None
        blob = canonical_json(e.payload)
        # The full deal table must never appear in an agent-tier event.
        assert '"hands"' not in blob
        # An observation reveals only the addressee's own cards.
        if e.type == "observation_sent":
            own = [c["value"] for c in e.payload["own_cards"]]
            assert own == hands[e.audience]


def test_view_slice_contains_only_own_cards():
    rules, events = _play()
    state, _ = start_game(rules, seed=42)
    hands = {pid: state.hands[pid] for pid in range(rules.n_players)}
    for pid in range(rules.n_players):
        view = agent_view(state, pid)
        assert [c.value for c in view.own_cards] == hands[pid]


def test_rationale_never_at_agent_tier():
    _, events = _play()
    for e in events:
        if e.visibility == Visibility.AGENT:
            assert "rationale" not in e.payload


def test_malformed_actions_degrade_to_pass_without_crashing():
    rules = Rules(n_players=3, k_private=2, total_rounds=1)
    state, _ = start_game(rules, seed=5)
    bad = {
        0: Action(kind="quote", quote=Quote(bid=None, ask=None, bid_size=2, ask_size=0)),
        1: Action(kind="take", take=Take(side="buy", price=10.0, size=0)),
        2: Action(kind="pass"),
    }
    state2, events = step(state, bad)  # must not raise
    rejected = [e for e in events if e.type == "action_rejected"]
    assert {e.payload["player_id"] for e in rejected} == {0, 1}
    # Game advanced normally despite the bad actions.
    assert state2.round == 2
