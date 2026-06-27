from agents.bots import FairValueBot
from engine.rules import Rules
from orchestrator.runner import run_game
from readers import metrics
from readers.benchmark import Suite, distribution, evaluate, leaderboard_entry
from readers.research import research_run


def bot_game(seed=42, **rule_kw):
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5, **rule_kw)
    agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
    return rules, run_game(rules, seed, agents)


# --- metrics ---------------------------------------------------------------
def test_calibration_final_error_matches_hand_computed():
    _, events = bot_game()
    truth = metrics.settlement_value(events)

    # Independently gather each player's last stated estimate from the stream.
    last_est: dict[int, float] = {}
    for e in events:
        if e.type == "action_received" and e.payload["fair_value_estimate"] is not None:
            last_est[e.payload["player_id"]] = e.payload["fair_value_estimate"]

    calib = metrics.calibration(events)
    for pid, est in last_est.items():
        assert calib[pid].final_error == abs(est - truth)
        assert calib[pid].coverage > 0
        assert calib[pid].mae is not None


def test_price_discovery_shape():
    rules, events = bot_game()
    pd = metrics.price_discovery(events)
    assert len(pd.series) == rules.total_rounds
    assert pd.terminal_error is None or pd.terminal_error >= 0


# --- regret ----------------------------------------------------------------
def _suite(seeds=range(5), **rule_kw):
    rules = Rules(n_players=4, k_private=2, total_rounds=3, **rule_kw)
    return Suite(rules=rules, seeds=tuple(seeds), model_id="test")


def test_regret_zero_when_model_is_reference():
    suite = _suite()
    result = evaluate(lambda pid: FairValueBot(pid), suite)
    assert all(r.regret == 0 for r in result.records)
    assert result.regret.mean == 0


def test_regret_nonzero_and_reproducible_for_biased_model():
    suite = _suite()
    r1 = evaluate(lambda pid: FairValueBot(pid, bias=5.0), suite)
    r2 = evaluate(lambda pid: FairValueBot(pid, bias=5.0), suite)
    assert any(rec.regret != 0 for rec in r1.records)
    # Deterministic engine + deterministic bots -> identical regrets on re-run.
    assert [rec.regret for rec in r1.records] == [rec.regret for rec in r2.records]


def test_distribution_is_over_seeds_times_seats():
    seeds = range(5)
    suite = _suite(seeds=seeds)
    result = evaluate(lambda pid: FairValueBot(pid, bias=2.0), suite)
    assert result.regret.n == len(list(seeds)) * suite.rules.n_players
    assert len(result.records) == result.regret.n
    # Aggregates populated.
    d = result.regret
    assert d.min <= d.p50 <= d.max
    assert d.ci95[0] <= d.mean <= d.ci95[1]


def test_distribution_helper_basic():
    d = distribution([1.0, 2.0, 3.0, 4.0])
    assert d.n == 4
    assert d.mean == 2.5
    assert d.min == 1.0 and d.max == 4.0


# --- suite hashing ---------------------------------------------------------
def test_suite_hash_stable_and_sensitive():
    rules = Rules(n_players=4, k_private=2, total_rounds=3)
    base = Suite(rules=rules, seeds=(1, 2, 3), model_id="m", prompt_version="pv1")
    same = Suite(rules=rules, seeds=(1, 2, 3), model_id="m", prompt_version="pv1")
    assert base.hash == same.hash

    assert base.hash != Suite(rules=rules, seeds=(1, 2, 3),
                              model_id="other", prompt_version="pv1").hash
    assert base.hash != Suite(rules=rules, seeds=(1, 2, 3),
                              model_id="m", prompt_version="pv2").hash
    assert base.hash != Suite(rules=rules, seeds=(1, 2, 4),
                              model_id="m", prompt_version="pv1").hash


def test_leaderboard_entry_mentions_hash_and_ci():
    suite = _suite()
    result = evaluate(lambda pid: FairValueBot(pid, bias=2.0), suite)
    entry = leaderboard_entry(suite, result)
    assert suite.hash[:12] in entry
    assert "95% CI" in entry


# --- research sandbox (one-bit difference) --------------------------------
def test_research_run_returns_distribution():
    rules = Rules(n_players=3, k_private=2, total_rounds=3)
    result = research_run(lambda pid: FairValueBot(pid, bias=1.0), rules, range(4))
    assert result.regret.n == 4 * rules.n_players
