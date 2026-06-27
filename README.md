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

## Status — Phases 1–3 (engine + protocol + bot + LLM agent + scoring)

Implemented:

- **Pure, deterministic engine** — `step(state, actions) -> (state', [Event])`, seedable
  and exactly replayable.
- **Uniform-price call auction** — volume-maximizing clear, deterministic tie-break,
  self-match prevention, pro-rata rationing, and risk-limit capping.
- **Append-only typed event stream** with `AGENT` / `SPECTATOR` / `RESEARCHER` visibility
  tiers — the single source of truth that all readers consume.
- **Versioned protocol** (Pydantic, `PROTOCOL_VERSION 0.1.0`) with per-agent view slicing.
- **`FairValueBot`** reference policy and a synchronous orchestrator.
- **LLM agent** (`agents/llm_worker.py`) — builds a compact prompt from its own view,
  calls a model with the `Action` schema as a forced tool call, hard-validates with
  Pydantic, retries once, then passes. Raw model output is logged at the `RESEARCHER`
  tier. Providers (`AnthropicClient`, `OpenAIClient`) sit behind one `LLMClient`
  interface; a `MockLLMClient` drives offline tests.
- **JSONL event log** with canonical-JSON config hashing.
- **Scoring layer** (`readers/`) — pure consumers of the event stream that score skill,
  not luck: **regret vs. a reference policy** (replay the same deal, model PnL − bot
  PnL), **calibration** (does a player's `fair_value_estimate` track the true sum?),
  **price-discovery convergence**, and a **multi-seed, seat-rotating benchmark runner**
  that reports a *distribution* (mean, CI, quantiles) under a content-addressed `Suite`
  hash that pins rules + seeds + model id + prompt version + protocol version.

> Note: a single-price call auction removes market-making *spread capture*, so the game
> currently measures **fair-value estimation**, not spread quoting. A continuous order
> book (and with it, market-making dynamics) is deferred to a later phase.

Not yet built: async orchestration with per-agent timeouts, `EVOptimalBot`, replay of
recorded LLM logs for distribution reproduction, spectator renderer.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                              # create the venv and install deps

uv run python -m orchestrator.runner # play a 5-round, 4-seat game; writes logs/*.jsonl
uv run python -m readers.benchmark   # score a bot over 20 seeds; print a leaderboard line
uv run pytest                        # run the test suite
uv run python -m protocol.protocol   # print the action JSON Schema (tool-call schema)
```

By default all four seats are `FairValueBot`. To put an **LLM in seat 0**, set a
provider key (and optionally pick a model):

```bash
ANTHROPIC_API_KEY=sk-...  uv run python -m orchestrator.runner   # Claude vs 3 bots
OPENAI_API_KEY=sk-...     uv run python -m orchestrator.runner   # OpenAI vs 3 bots
TG_MODEL=claude-opus-4-8  ANTHROPIC_API_KEY=sk-... uv run python -m orchestrator.runner
```

Change players / cards / rounds / seed by editing `Rules(...)` in
`orchestrator/runner.py`, or call `run_game(rules, seed, agents, log_path)` directly
with any mix of `FairValueBot` and `LLMAgent` seats.

Bot-only games are fully deterministic (same seed → byte-identical log). Games with an
LLM seat are **not** byte-reproducible — models are stochastic even at temperature 0;
reproducibility there means replaying recorded actions, not re-calling the model.

## Layout

```
engine/        rules, state/dealing/settlement, call auction, events, pure step()
protocol/      versioned Pydantic schemas + JSON Schema export
agents/        Agent interface, FairValueBot, LLM worker + provider clients
orchestrator/  game loop + event persistence (incl. model raw-output logging)
store/         append-only JSONL log + canonical config hashing
readers/       scoring: metrics, regret oracle, benchmark suite, research sandbox
tests/         settlement, auction, determinism, isolation, LLM worker, scoring
```

## Design invariants

1. The engine is pure and deterministic given `(rules, config, seed)`.
2. The event log is append-only and is the single source of truth.
3. Every event has a visibility tier; readers filter, the engine emits once.
4. No reader ever writes back to the engine.
5. An agent's view contains only its own private information.
6. A malformed or late action degrades to `pass` — it never crashes the game.
