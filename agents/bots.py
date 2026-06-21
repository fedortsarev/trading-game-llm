"""Reference policies. Deterministic — pure functions of the AgentView.

FairValueBot: estimates the instrument's fair value as
    sum(own cards) + sum(public cards) + E[sum of unknown cards]
where unknown cards are the other players' hands and E uses the known deck mean.
It quotes a symmetric spread around that estimate.
"""

from __future__ import annotations

from protocol.protocol import Action, AgentView, Quote


class FairValueBot:
    def __init__(self, player_id: int, spread: float = 2.0, size: int = 1):
        self.player_id = player_id
        self.spread = spread
        self.size = size

    def fair_value(self, view: AgentView) -> float:
        pub = view.public
        card_mean = (pub.card_min + pub.card_max) / 2.0
        unknown_cards = (pub.n_players - 1) * pub.k_private
        own = sum(c.value for c in view.own_cards)
        public = sum(c.value for c in pub.public_cards)
        return own + public + unknown_cards * card_mean

    def act(self, view: AgentView) -> Action:
        fv = self.fair_value(view)
        half = self.spread / 2.0
        return Action(
            kind="quote",
            quote=Quote(
                bid=fv - half,
                ask=fv + half,
                bid_size=self.size,
                ask_size=self.size,
            ),
            fair_value_estimate=fv,
            rationale=f"FV={fv:.2f} (own+public+{(view.public.n_players-1)*view.public.k_private} unknown @ mean)",
        )
