from agents.bots import FairValueBot
from engine.rules import Rules
from orchestrator.runner import run_game
from store.log import config_hash


def test_same_seed_byte_identical_log(tmp_path):
    rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)

    def play(path):
        agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
        run_game(rules, seed=42, agents=agents, log_path=path)
        return path.read_bytes()

    a = play(tmp_path / "a.jsonl")
    b = play(tmp_path / "b.jsonl")
    assert a == b
    assert len(a) > 0


def test_different_seed_differs(tmp_path):
    rules = Rules(n_players=4, k_private=2, total_rounds=5)

    def play(seed, path):
        agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
        run_game(rules, seed=seed, agents=agents, log_path=path)
        return path.read_bytes()

    assert play(1, tmp_path / "s1.jsonl") != play(2, tmp_path / "s2.jsonl")


def test_config_hash_is_stable():
    rules = Rules(n_players=4, k_private=2, total_rounds=5)
    assert config_hash(rules.model_dump(), 42) == config_hash(rules.model_dump(), 42)
    assert config_hash(rules.model_dump(), 1) != config_hash(rules.model_dump(), 2)
