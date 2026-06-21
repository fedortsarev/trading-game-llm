import random

from engine.auction import Order, clear


def rng():
    return random.Random(0)


def test_simple_cross_matches_full_size():
    orders = [Order(0, "buy", 10, 2), Order(1, "sell", 8, 2)]
    res = clear(orders, {0: 0, 1: 0}, risk_limit=5, rng=rng())
    assert res.matched_volume == 2
    assert res.net_fills == {0: 2, 1: -2}
    assert res.clearing_price in {8.0, 10.0}


def test_no_cross_no_trade():
    orders = [Order(0, "buy", 8, 2), Order(1, "sell", 10, 2)]
    res = clear(orders, {0: 0, 1: 0}, risk_limit=5, rng=rng())
    assert res.matched_volume == 0
    assert res.clearing_price is None
    assert res.net_fills == {}


def test_self_match_prevented():
    # pid0 quotes both sides; its own buy and sell must not trade with each other.
    orders = [Order(0, "buy", 10, 3), Order(0, "sell", 8, 2), Order(1, "sell", 9, 1)]
    res = clear(orders, {0: 0, 1: 0}, risk_limit=5, rng=rng())
    assert res.matched_volume == 1
    assert res.net_fills == {0: 1, 1: -1}


def test_rationing_pro_rata():
    orders = [Order(0, "buy", 10, 2), Order(1, "buy", 10, 1), Order(2, "sell", 10, 2)]
    res = clear(orders, {0: 0, 1: 0, 2: 0}, risk_limit=5, rng=rng())
    assert res.matched_volume == 2
    assert res.net_fills[2] == -2
    # Buyers share 2 lots; total bought equals matched volume.
    assert sum(v for v in res.net_fills.values() if v > 0) == 2


def test_risk_limit_caps_fill():
    # pid0 already at +3 with a limit of 5 -> can buy at most 2 more.
    orders = [Order(0, "buy", 10, 5), Order(1, "sell", 10, 5)]
    res = clear(orders, {0: 3, 1: 0}, risk_limit=5, rng=rng())
    assert res.net_fills == {0: 2, 1: -2}
    assert 0 in res.risk_capped


def test_fills_never_breach_limit():
    orders = [Order(0, "buy", 10, 99), Order(1, "sell", 10, 99)]
    res = clear(orders, {0: 0, 1: 0}, risk_limit=5, rng=rng())
    assert abs(res.net_fills.get(0, 0)) <= 5
    assert abs(res.net_fills.get(1, 0)) <= 5
