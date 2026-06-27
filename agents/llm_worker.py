"""LLM-powered agent.

Turns an AgentView (this agent's private slice — nothing else) into a prompt, calls
a model with the Action schema as a forced tool, hard-validates the result, retries
once on failure, and otherwise passes. A malformed model never crashes the game.

State is fed as a compact structured snapshot each round, not a growing transcript
(design §8): each call is independent and carries the current view + recent tape.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError

from protocol.protocol import Action, AgentView, export_json_schema

from .llm_client import LLMClient


@dataclass
class LLMTrace:
    player_id: int
    model: str
    attempts: int = 0
    raw_text: str = ""
    tool_input: dict | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    final_kind: str = "pass"


def inline_refs(schema: dict) -> dict:
    """Dereference `$defs`/`$ref` into a self-contained schema for provider tools.

    Some tool-calling APIs validate the input schema strictly and dislike
    `$ref`/`$defs`; inline them so the schema stands alone.
    """
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                target = resolve(copy.deepcopy(defs[name]))
                # merge any sibling keys (e.g. description) onto the resolved target
                merged = {**target, **{k: resolve(v) for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve({k: v for k, v in schema.items() if k != "$defs"})


def action_tool_schema() -> dict:
    return inline_refs(export_json_schema())


SYSTEM_PROMPT = """\
You are a trader in a sealed-bid market-making card game.

Rules:
- Each of the {n_players} players holds {k_private} private card(s). There may also be \
public cards everyone can see. Card values are integers drawn uniformly from \
[{card_min}, {card_max}].
- The traded instrument settles at the SUM OF ALL CARDS in play (every player's private \
cards plus any public cards). You only see your own cards, so you must estimate the \
unknown total.
- Each round is sealed and simultaneous: all players submit at once and the auction \
clears every trade at a single price. You cannot see others' quotes before submitting, \
but the public tape of past clearing prices is informative about what others hold.
- Your edge comes from estimating fair value better than the table and inferring \
others' cards from how prices move. Quote tighter when confident, wider when unsure.
- Position is capped at +/-{risk_limit}. Buying when long / selling when short reduces \
risk.

Each round, call the `submit_action` tool exactly once:
- kind="quote": provide bid/ask prices and sizes (a two-sided market).
- kind="take": cross the market to buy or sell at a stated price/size.
- kind="pass": do nothing this round.
Always include `fair_value_estimate` (your best estimate of the settlement value) and a \
brief `rationale`. The rationale is for analysis only and is never shown to other \
players, so be honest about your reasoning."""


# Pins the prompt template that produced a score — folded into the benchmark Suite
# hash so a leaderboard claim is reproducible against the exact wording used.
PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


class LLMAgent:
    def __init__(self, player_id: int, client: LLMClient, model: str = "", *,
                 timeout: float = 30.0):
        self.player_id = player_id
        self.client = client
        self.model = model or getattr(client, "model", "")
        self.timeout = timeout
        self.last_trace: LLMTrace | None = None

    def build_prompt(self, view: AgentView) -> tuple[str, str]:
        pub = view.public
        system = SYSTEM_PROMPT.format(
            n_players=pub.n_players, k_private=pub.k_private,
            card_min=pub.card_min, card_max=pub.card_max, risk_limit=view.risk_limit,
        )
        unknown_cards = (pub.n_players - 1) * pub.k_private
        snapshot = {
            "round": pub.round,
            "total_rounds": pub.total_rounds,
            "your_player_id": view.player_id,
            "your_cards": [c.value for c in view.own_cards],
            "your_cards_sum": sum(c.value for c in view.own_cards),
            "public_cards": [c.value for c in pub.public_cards],
            "unknown_cards_count": unknown_cards,
            "deck": {"min": pub.card_min, "max": pub.card_max,
                     "mean_per_card": (pub.card_min + pub.card_max) / 2.0},
            "your_position": view.position,
            "your_cash": view.cash,
            "your_pnl_marked": view.pnl,
            "risk_limit": view.risk_limit,
            "last_clearing_price": pub.last_clearing_price,
            "recent_tape": [t.model_dump() for t in pub.tape[-10:]],
        }
        user = (
            "Current state (JSON):\n"
            + json.dumps(snapshot, indent=2)
            + "\n\nDecide your action and call submit_action."
        )
        return system, user

    def act(self, view: AgentView) -> Action:
        system, user = self.build_prompt(view)
        schema = action_tool_schema()
        trace = LLMTrace(player_id=self.player_id, model=self.model)

        nudge = ""
        for attempt in range(2):  # initial try + one retry
            trace.attempts = attempt + 1
            res = self.client.submit(system, user + nudge, schema, timeout=self.timeout)
            trace.raw_text = res.raw_text
            trace.tool_input = res.tool_input
            trace.usage = res.usage
            if res.model:
                trace.model = res.model

            if res.error is not None or res.tool_input is None:
                trace.error = res.error or "no tool input"
                nudge = (f"\n\nYour previous attempt failed: {trace.error}. "
                         "Call submit_action with valid arguments.")
                continue
            try:
                action = Action.model_validate(res.tool_input)
            except ValidationError as e:
                trace.error = f"schema validation failed: {e.error_count()} error(s)"
                nudge = (f"\n\nYour previous tool input was invalid: {e}. "
                         "Fix it and call submit_action again.")
                continue

            trace.error = None
            trace.final_kind = action.kind
            self.last_trace = trace
            return action

        # Exhausted retries -> safe default.
        trace.final_kind = "pass"
        self.last_trace = trace
        return Action(kind="pass", rationale="(auto-pass after model failure)")

    def trace_payload(self) -> dict:
        return asdict(self.last_trace) if self.last_trace is not None else {}
