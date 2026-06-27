# Trading Game for LLM Agents

A service that lets LLM-powered agents play the **trading game** — the market-making
card game popularized by Gary Stevenson's *The Trading Game*. One engine serves three
goals from a single event stream: an **LLM benchmark**, a **research sandbox**, and a
**spectator product**.

See [`trading-game-design.md`](trading-game-design.md) for the full design.

## The game

Each player is dealt private cards; the traded instrument settles at the **sum of all
cards in play**. You see only your own cards, so you price the instrument by estimating
the unknown total — and refine that estimate by inferring others' cards from how they
trade. Because there's a ground-truth settlement value, we can measure how well price
discovery converged, not merely who made money.

Rounds are **sealed and simultaneous**: every agent submits quotes at once, a
deterministic **call auction** clears them at a single price, results are revealed, and
the next round begins. This makes the game a test of reasoning, not HTTP latency.

## Status — Phase 1 (engine + protocol + scripted bot, no LLM)

Implemented:

- **Pure, deterministic engine** — `step(state, actions) -> (state', [Event])`, seedable
  and exactly replayable.
- **Uniform-price call auction** — volume-maximizing clear, deterministic tie-break,
  self-match prevention, pro-rata rationing, and risk-limit capping.
- **Append-only typed event stream** with `AGENT` / `SPECTATOR` / `RESEARCHER` visibility
  tiers — the single source of truth that all readers consume.
- **Versioned protocol** (Pydantic, `PROTOCOL_VERSION 0.1.0`) with per-agent view slicing.
- **`FairValueBot`** reference policy and a synchronous bot-only orchestrator.
- **JSONL event log** with canonical-JSON config hashing.

> Note: a single-price call auction removes market-making *spread capture*, so Phase 1
> measures **fair-value estimation**, not spread quoting. A continuous order book (and
> with it, market-making dynamics) is deferred to a later phase.

Not yet built: LLM worker, async orchestration with per-agent timeouts, oracle/regret
and calibration scoring, multi-seed runner, benchmark suite pinning, spectator renderer.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                              # create the venv and install deps

uv run python -m orchestrator.runner # play a 5-round, 4-bot game; writes logs/*.jsonl
uv run pytest                        # run the test suite
uv run python -m protocol.protocol   # print the action JSON Schema (for Phase-2 tools)
```

Change players / cards / rounds / seed by editing `Rules(...)` in
`orchestrator/runner.py`, or call `run_game(rules, seed, agents, log_path)` directly.

Every game is fully deterministic in Phase 1: the same seed reproduces the same result
and a byte-identical event log.

## Layout

```
engine/        rules, state/dealing/settlement, call auction, events, pure step()
protocol/      versioned Pydantic schemas + JSON Schema export
agents/        Agent interface + FairValueBot reference policy
orchestrator/  bot-only game loop + event persistence
store/         append-only JSONL log + canonical config hashing
readers/       (Phase 4) benchmark / research / spectator consumers
tests/         settlement, auction mechanics, determinism, isolation
```

## Design invariants

1. The engine is pure and deterministic given `(rules, config, seed)`.
2. The event log is append-only and is the single source of truth.
3. Every event has a visibility tier; readers filter, the engine emits once.
4. No reader ever writes back to the engine.
5. An agent's view contains only its own private information.
6. A malformed or late action degrades to `pass` — it never crashes the game.
