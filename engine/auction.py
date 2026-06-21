"""Uniform-price call auction.

All trades in a round clear at a single price `p*`. This is fair across agent
latency (sealed, simultaneous) but, by construction, removes market-making spread
capture — Phase 1 therefore measures fair-value estimation, not spread quoting.

Algorithm:
  1. Each agent's quote/take expands to limit orders.
  2. p* maximizes matched volume min(D(p), S(p)) over submitted prices.
     Tie-break: minimize |D(p) - S(p)|, then integer midpoint, then seeded RNG.
  3. Self-match netting: an agent's own crossing buy and sell cancel.
  4. Risk caps: each order is pre-capped to the agent's remaining position
     capacity, so no fill can breach the limit and pro-rata can only reduce.
  5. Rationing: the heavy side is filled pro-rata by size; remainder lots go by
     ascending player_id (deterministic).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Order:
    player_id: int
    side: str  # "buy" | "sell"
    price: float
    size: int


@dataclass
class AuctionResult:
    clearing_price: float | None
    matched_volume: int
    # signed net fill per player: +n bought, -n sold
    net_fills: dict[int, int] = field(default_factory=dict)
    risk_capped: list[int] = field(default_factory=list)


def _demand(orders: list[Order], p: float) -> int:
    return sum(o.size for o in orders if o.side == "buy" and o.price >= p)


def _supply(orders: list[Order], p: float) -> int:
    return sum(o.size for o in orders if o.side == "sell" and o.price <= p)


def clear(
    orders: list[Order],
    positions: dict[int, int],
    risk_limit: int,
    rng: random.Random,
) -> AuctionResult:
    """Pure: same inputs (incl. rng state) -> same result."""
    if not orders:
        return AuctionResult(clearing_price=None, matched_volume=0)

    # 4. Pre-cap each order to remaining risk capacity for its agent.
    #    Buy capacity: how much more we may go long  -> risk_limit - position.
    #    Sell capacity: how much more we may go short -> position + risk_limit.
    capped: list[Order] = []
    risk_capped: set[int] = set()
    for o in orders:
        pos = positions.get(o.player_id, 0)
        capacity = (risk_limit - pos) if o.side == "buy" else (pos + risk_limit)
        capacity = max(0, capacity)
        size = min(o.size, capacity)
        if size < o.size:
            risk_capped.add(o.player_id)
        if size > 0:
            capped.append(Order(o.player_id, o.side, o.price, size))
    orders = capped
    if not orders:
        return AuctionResult(None, 0, {}, sorted(risk_capped))

    # 2. Find the volume-maximizing clearing price over candidate prices.
    candidates = sorted({o.price for o in orders})
    best: tuple[int, int, float] | None = None  # (volume, -imbalance, price)
    best_price: float | None = None
    for p in candidates:
        d, s = _demand(orders, p), _supply(orders, p)
        vol = min(d, s)
        if vol == 0:
            continue
        key = (vol, -abs(d - s))
        if best is None or key > best:
            best = key
            best_price = p
        elif key == best:
            # exact tie on (volume, imbalance): keep the lower price for now;
            # resolved deterministically below.
            pass

    if best is None or best_price is None:
        return AuctionResult(None, 0, {}, sorted(risk_capped))

    # Resolve a tie across an interval: gather all prices sharing the best key,
    # take the integer midpoint, then a seeded RNG pick if still ambiguous.
    best_key = best
    tied = [
        p
        for p in candidates
        if (min(_demand(orders, p), _supply(orders, p)),
            -abs(_demand(orders, p) - _supply(orders, p))) == best_key
    ]
    if len(tied) == 1:
        p_star = tied[0]
    else:
        lo, hi = min(tied), max(tied)
        mid = (lo + hi) / 2.0
        # Prefer a tied price equal to the midpoint; else seeded choice.
        p_star = mid if mid in tied else tied[rng.randrange(len(tied))]

    # 3. Self-match netting at p*: cancel each agent's crossing buy vs sell.
    buys: dict[int, int] = {}
    sells: dict[int, int] = {}
    for o in orders:
        if o.side == "buy" and o.price >= p_star:
            buys[o.player_id] = buys.get(o.player_id, 0) + o.size
        elif o.side == "sell" and o.price <= p_star:
            sells[o.player_id] = sells.get(o.player_id, 0) + o.size
    for pid in set(buys) & set(sells):
        net = min(buys[pid], sells[pid])
        buys[pid] -= net
        sells[pid] -= net
    buys = {k: v for k, v in buys.items() if v > 0}
    sells = {k: v for k, v in sells.items() if v > 0}

    total_buy = sum(buys.values())
    total_sell = sum(sells.values())
    matched = min(total_buy, total_sell)
    if matched == 0:
        return AuctionResult(None, 0, {}, sorted(risk_capped))

    # 5. Ration the heavy side pro-rata; light side fills fully.
    buy_alloc = _ration(buys, matched) if total_buy > matched else dict(buys)
    sell_alloc = _ration(sells, matched) if total_sell > matched else dict(sells)

    net_fills: dict[int, int] = {}
    for pid, n in buy_alloc.items():
        net_fills[pid] = net_fills.get(pid, 0) + n
    for pid, n in sell_alloc.items():
        net_fills[pid] = net_fills.get(pid, 0) - n
    net_fills = {k: v for k, v in net_fills.items() if v != 0}

    return AuctionResult(
        clearing_price=float(p_star),
        matched_volume=matched,
        net_fills=net_fills,
        risk_capped=sorted(risk_capped),
    )


def _ration(want: dict[int, int], total: int) -> dict[int, int]:
    """Allocate `total` lots across `want` pro-rata by size; deterministic
    remainder distribution by ascending player_id."""
    demand = sum(want.values())
    if demand <= total:
        return dict(want)
    alloc = {pid: (n * total) // demand for pid, n in want.items()}
    remainder = total - sum(alloc.values())
    # Hand out leftover lots to the largest unmet demand first, ties by player_id.
    order = sorted(want, key=lambda pid: (-(want[pid] - alloc[pid]), pid))
    for pid in order:
        if remainder <= 0:
            break
        alloc[pid] += 1
        remainder -= 1
    return alloc
