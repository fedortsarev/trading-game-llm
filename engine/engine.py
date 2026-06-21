"""The pure, deterministic engine: (state, actions) -> (state', [Event]).

`start_game` deals and announces; `step` runs one sealed round through the call
auction; `settle` marks against the true sum. The engine never trusts an agent,
slices state per-agent, and degrades malformed actions to a pass.
"""

from __future__ import annotations

import math

from protocol.protocol import (
    PROTOCOL_VERSION,
    Action,
    AgentView,
    Card,
    PublicState,
    Trade,
)

from . import events as ev
from .auction import Order, clear
from .rules import Rules
from .state import GameState, deal


# --------------------------------------------------------------------------
# View slicing — an agent sees ONLY its own private information.
# --------------------------------------------------------------------------
def agent_view(state: GameState, pid: int) -> AgentView:
    r = state.rules
    public = PublicState(
        round=state.round,
        total_rounds=r.total_rounds,
        n_players=r.n_players,
        k_private=r.k_private,
        card_min=r.card_min,
        card_max=r.card_max,
        public_cards=[Card(value=v) for v in state.public_cards],
        tape=[Trade(**t) for t in state.tape],
        last_clearing_price=state.last_clearing_price,
    )
    return AgentView(
        protocol_version=PROTOCOL_VERSION,
        player_id=pid,
        own_cards=[Card(value=v) for v in state.hands[pid]],
        position=state.positions.get(pid, 0),
        cash=state.cash.get(pid, 0.0),
        pnl=state.marked_pnl(pid),
        risk_limit=r.risk_limit,
        public=public,
    )


# --------------------------------------------------------------------------
# Event helper
# --------------------------------------------------------------------------
def _emit(state: GameState, out: list[ev.Event], type_: str, payload: dict,
          visibility: ev.Visibility, audience: int | None = None) -> None:
    out.append(
        ev.Event(
            seq=state.next_seq,
            round=state.round,
            type=type_,
            payload=payload,
            visibility=visibility,
            audience=audience,
        )
    )
    state.next_seq += 1


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------
def start_game(rules: Rules, seed: int) -> tuple[GameState, list[ev.Event]]:
    state = deal(rules, seed)
    out: list[ev.Event] = []

    # round counter is 1 once dealing is done; game_start/deal logged at round 0.
    state.round = 0
    _emit(state, out, ev.GAME_START, {
        "n_players": rules.n_players,
        "k_private": rules.k_private,
        "m_public": rules.m_public,
        "total_rounds": rules.total_rounds,
        "card_min": rules.card_min,
        "card_max": rules.card_max,
        "risk_limit": rules.risk_limit,
    }, ev.Visibility.SPECTATOR)

    # Public cards are visible to everyone.
    _emit(state, out, ev.DEAL, {"public_cards": state.public_cards},
          ev.Visibility.SPECTATOR)
    # Each agent privately learns its own cards.
    for pid in range(rules.n_players):
        _emit(state, out, ev.DEAL, {"player_id": pid, "own_cards": state.hands[pid]},
              ev.Visibility.AGENT, audience=pid)
    # Researchers see the full deal and the pre-reveal true settlement value.
    _emit(state, out, ev.DEAL, {
        "hands": state.hands,
        "public_cards": state.public_cards,
        "settlement_value": state.settlement_value,
    }, ev.Visibility.RESEARCHER)

    state.round = 1
    return state, out


def _validate(action: Action, view: AgentView) -> tuple[list[Order], str | None]:
    """Expand a (validated) action into orders, or return a rejection reason."""
    pid = view.player_id
    if action.kind == "pass":
        return [], None
    if action.kind == "quote":
        q = action.quote
        if q is None:
            return [], "quote action without quote"
        orders: list[Order] = []
        for side, price, size in (("buy", q.bid, q.bid_size), ("sell", q.ask, q.ask_size)):
            if size < 0:
                return [], "negative size"
            if size > 0:
                if price is None or not math.isfinite(price):
                    return [], "missing/invalid price for sized side"
                orders.append(Order(pid, side, float(price), int(size)))
        return orders, None
    if action.kind == "take":
        t = action.take
        if t is None:
            return [], "take action without take"
        if t.size <= 0:
            return [], "take size must be positive"
        if not math.isfinite(t.price):
            return [], "invalid take price"
        return [Order(pid, t.side, float(t.price), int(t.size))], None
    return [], f"unknown action kind {action.kind!r}"


def step(state: GameState, actions: dict[int, Action]) -> tuple[GameState, list[ev.Event]]:
    """Run one sealed round. Pure: returns a NEW state, never mutates the input."""
    state = state.model_copy(deep=True)
    out: list[ev.Event] = []
    rules = state.rules

    if state.settled or state.round > rules.total_rounds:
        return state, out

    all_orders: list[Order] = []
    for pid in range(rules.n_players):
        view = agent_view(state, pid)
        # Log exactly what was sent to this agent (its private view).
        _emit(state, out, ev.OBSERVATION_SENT, view.model_dump(mode="json"),
              ev.Visibility.AGENT, audience=pid)

        action = actions.get(pid) or Action(kind="pass")
        orders, reason = _validate(action, view)
        if reason is not None:
            # Malformed/illegal -> rejected + treated as pass. Never crashes.
            _emit(state, out, ev.ACTION_REJECTED,
                  {"player_id": pid, "reason": reason, "kind": action.kind},
                  ev.Visibility.SPECTATOR)
            continue
        # Quotes/takes are public once revealed; rationale rides along (spectator).
        _emit(state, out, ev.ACTION_RECEIVED, {
            "player_id": pid,
            "kind": action.kind,
            "quote": action.quote.model_dump() if action.quote else None,
            "take": action.take.model_dump() if action.take else None,
            "fair_value_estimate": action.fair_value_estimate,
            "rationale": action.rationale,
        }, ev.Visibility.SPECTATOR)
        all_orders.extend(orders)

    rng = state.round_rng(state.round)
    result = clear(all_orders, state.positions, rules.risk_limit, rng)

    _emit(state, out, ev.AUCTION_CLEARED, {
        "clearing_price": result.clearing_price,
        "matched_volume": result.matched_volume,
        "risk_capped": result.risk_capped,
    }, ev.Visibility.SPECTATOR)

    # Apply fills (cash accounting) and emit public fills.
    if result.clearing_price is not None and result.matched_volume > 0:
        p = result.clearing_price
        for pid, net in sorted(result.net_fills.items()):
            side = "buy" if net > 0 else "sell"
            size = abs(net)
            state.positions[pid] = state.positions.get(pid, 0) + net
            state.cash[pid] = state.cash.get(pid, 0.0) - net * p
            _emit(state, out, ev.FILL,
                  {"player_id": pid, "side": side, "price": p, "size": size},
                  ev.Visibility.SPECTATOR)
        state.last_clearing_price = p
        state.tape.append({"round": state.round, "price": p, "size": result.matched_volume})

    # Position updates: own to each agent; full table to researchers.
    for pid in range(rules.n_players):
        _emit(state, out, ev.POSITION_UPDATE, {
            "player_id": pid,
            "position": state.positions.get(pid, 0),
            "cash": state.cash.get(pid, 0.0),
            "pnl": state.marked_pnl(pid),
        }, ev.Visibility.AGENT, audience=pid)
    _emit(state, out, ev.POSITION_UPDATE, {
        "positions": dict(state.positions),
        "cash": dict(state.cash),
        "pnl": {pid: state.marked_pnl(pid) for pid in range(rules.n_players)},
    }, ev.Visibility.RESEARCHER)

    _emit(state, out, ev.ROUND_END,
          {"round": state.round, "clearing_price": state.last_clearing_price},
          ev.Visibility.SPECTATOR)

    state.round += 1
    return state, out


def settle(state: GameState) -> tuple[GameState, list[ev.Event]]:
    """Mark all positions to the true sum and rank. Settlement value is now public."""
    state = state.model_copy(deep=True)
    out: list[ev.Event] = []
    if state.settled:
        return state, out

    sv = state.settlement_value
    final = {pid: state.cash.get(pid, 0.0) + state.positions.get(pid, 0) * sv
             for pid in range(state.rules.n_players)}
    ranking = sorted(final, key=lambda pid: (-final[pid], pid))

    _emit(state, out, ev.SETTLEMENT, {
        "settlement_value": sv,
        "final_pnl": final,
        "ranking": ranking,
        "positions": dict(state.positions),
    }, ev.Visibility.SPECTATOR)
    _emit(state, out, ev.GAME_END, {"ranking": ranking}, ev.Visibility.SPECTATOR)

    state.settled = True
    return state, out
