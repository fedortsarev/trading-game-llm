# Trading Game for LLM Agents — Design Document

> A service that lets LLM-powered agents play the "trading game" (the market-making card game popularized by Gary Stevenson's *The Trading Game*). One engine serves three goals simultaneously: an **LLM benchmark**, a **research sandbox**, and a **spectator product**.

This document is written to be handed to a coding agent. It specifies the architecture, contracts, data schemas, and a phased build plan with acceptance criteria. Where a value is a design choice rather than a hard requirement, it is marked **(config)**.

---

## 1. Core design decisions (read first — everything follows from these)

1. **Discrete sealed rounds, not a continuous market.** LLM calls take seconds with variable latency. A continuous double auction rewards whoever's HTTP call lands first — meaningless and unfair for models. Each round: agents submit sealed, simultaneous quotes → deterministic call-auction match → reveal → repeat. This makes the game a test of *reasoning*, not network speed.

2. **The engine is event-sourced. Its single output is a typed, append-only event stream.** All three goals are *readers* of that stream:
   - **Benchmark** = aggregations over many event logs across seeds.
   - **Research** = queries + instrumentation over logs, plus config injection.
   - **Spectator** = a renderer subscribing live, or replaying a finished log as VOD.

   The engine does not know which reader consumes it. Do **not** special-case "spectator mode" inside the engine (it corrupts reproducibility), and do **not** bolt logging on afterward (the feed and the eval will disagree about what happened). The log is the source of truth from the first commit.

3. **Every event carries a visibility tier.** This generalizes the per-agent information isolation you already need. Readers filter by tier; the engine emits once.

4. **The spectator feed is strictly a reader with no write path back to the engine.** No live channel an agent can read, no human input reaching a model mid-game. Contamination becomes physically impossible, not merely discouraged.

5. **Engine signature is pure and deterministic:** `(state, actions) -> (state', [Event])`. Seedable RNG for the deal and tie-breaks so any game replays exactly.

---

## 2. The game (parameterized)

`N` players **(config)**, each dealt `k` private cards **(config)**; optional `m` public cards **(config)**. The instrument settles at the **sum of all cards in play**.

- Naive fair value = (sum of your cards) + E[sum of unknown cards].
- Edge comes from inferring others' cards from their quotes.
- Key property: there is a **ground-truth settlement value**, so we can measure how well price discovery converged — not merely who made money.

> Note: this is the generic trading-game structure (as used by Citi/Optiver/SIG), not a reproduction of the exact card split in the book. The architecture does not depend on those specifics — they are config.

---

## 3. Round loop

1. **Observe** — engine sends each agent its private view: own cards, position, PnL, risk limit, plus public state (tape, public cards, round number).
2. **Act (sealed, simultaneous)** — each agent returns actions within a deadline: `quote(bid, ask, sizes)`, `take(side, px, sz)`, `cancel`, `pass`.
3. **Match** — run a **call auction**: find the clearing price, match crossing orders. (Cleaner and fairer than a continuous book. A continuous book can be added later if true spread-capture dynamics are wanted.)
4. **Reveal** — broadcast fills, updated tape, updated positions.
5. After `R` rounds **(config)** → **settle** against the true sum, mark all positions, rank by score (§7).

---

## 4. Architecture

### 4.1 Engine
Authoritative, deterministic, no LLM. Pure function `(state, actions) -> (state', [Event])`.
- Seedable RNG for the deal and tie-breaks; games replay exactly from `(rules, config, seed)`.
- Validates and rejects illegal actions; enforces position/risk limits.
- **Never trusts an agent.** Slices state per-agent so no agent can see another's private view.
- Emits a typed event for everything that happens (§5).

### 4.2 Protocol
Tight, **versioned** JSON schema for observation and action. Use Pydantic → free validation + JSON Schema that can be handed straight to tool-calling. Model-agnostic: rule-based bots and LLMs share one interface (`Agent.act(obs) -> actions`).

### 4.3 Agent worker
Takes an observation, builds the prompt, calls the model with the action schema as a tool / structured output, parses, returns actions. One worker per model/config so Opus vs Sonnet vs GPT vs a scripted bot can play the same game. Provider SDKs sit behind the single `Agent.act` interface.

### 4.4 Orchestrator
Drives the loop. Fans observations out in parallel (`asyncio.gather` + per-agent timeout), collects actions by deadline, hands them to the engine, persists every event. Timeout → auto-`pass`.

### 4.5 Event log
Append-only record of every observation, raw model output, action, and fill. This is the replay, the debugger, and the spectator feed. JSONL to start; a real event store later if needed.

---

## 5. Event stream & visibility tiers

The engine emits `Event` objects. Every event carries:

```
Event:
  seq:        int            # monotonic, per-game
  round:      int
  type:       str            # see catalog below
  payload:    dict
  visibility: Visibility     # who may read this event
  audience:   int | None     # if tier is agent-private, which agent (player id); else None
```

### Visibility tiers
- `AGENT` — visible to a specific agent only (use with `audience`). E.g. that agent's own cards.
- `SPECTATOR` — visible to spectators (and researchers). Safe to show an audience mid-game.
- `RESEARCHER` — visible to researchers only. Everything, always.

Readers subscribe to tiers and filter:

| Datum | agent (self) | spectator | researcher |
|---|---|---|---|
| Own cards | ✓ | ✗ | ✓ |
| Other agents' cards | ✗ | ✗ | ✓ |
| Public tape / fills | ✓ | ✓ | ✓ |
| Model stated rationale | ✗ | ✓ | ✓ |
| Raw model output / tokens | ✗ | ✗ | ✓ |
| True settlement value (pre-reveal) | ✗ | ✗ | ✓ |

**Hard rule:** a model's stated reasoning is `researcher: always, spectator: yes, agent: never`. Rationale is the entertainment for spectators and gold for researchers, but it must never leak between players in a benchmark run.

### Event catalog (initial)
`game_start`, `deal` (agent-private per card; researcher sees all), `observation_sent` (agent-private), `model_raw_output` (researcher), `action_received` (rationale → spectator; raw → researcher), `action_rejected`, `auction_cleared` (spectator), `fill` (spectator), `position_update`, `round_end`, `settlement`, `game_end`.

---

## 6. The three readers

All three are pure consumers of the event stream. None writes back to the engine.

- **Benchmark** — aggregates over many logs across seeds. A leaderboard claim must be reproducible: content-address the `(rules, config, seed)` tuple and hash it. A claim reads "Model X scored Y on `trading-game-suite-v1.2` (`#abc123`)" — verifiable, not a vibe. A benchmark **suite** is a frozen set of these tuples.
- **Research sandbox** — queries + instrumentation over logs; may inject config variants freely. Differs from benchmark *only* in that the config tuple is **not** frozen/hashed. Same engine.
- **Spectator** — renders the `SPECTATOR`-tier stream live (on delay for benchmark integrity) or replays a finished log as VOD. Needs nothing from the engine except the rich event stream already emitted. The sealed-round → simultaneous-reveal cadence is *more* dramatic than a continuous-tape blur.

**Benchmark vs sandbox is a one-bit difference: config frozen+hashed, or not.**

---

## 7. Scoring (serves benchmark + research; free because we have ground truth)

Raw PnL is too noisy and too luck-driven to benchmark on. Because the engine knows the true settlement value and replays deterministically:

1. **Counterfactual / regret vs an oracle.** Replay the *exact same deal* with a reference policy (naive fair-value bot, or an EV-optimal bot). Score = model PnL − reference PnL on identical cards. Strips out card luck; measures skill.
2. **Calibration.** Agents emit `fair_value_estimate` each round; score how its trajectory tracks truth. Often more revealing than PnL.
3. **Multi-seed distributions.** One game is high variance. Always report the distribution over many seeded deals per matchup — never a single number. Variance across seeds is itself a research output.

---

## 8. LLM-specific concerns (these bite in practice)

- **Information isolation.** The worker only ever receives its own agent's view. It is trivial to leak others' cards by sharing one state object naively — the engine must slice per-agent. This is the same mechanism as visibility tiers.
- **Structured output.** Use tool calling / constrained JSON. Validate hard, retry once on parse failure, then `pass`. A malformed quote must never crash the game.
- **State as a structured object, not a transcript.** Do not grow a chat history. Feed a compact state snapshot each round (recent tape + running position). Cost = `R × N × context`, so this is money, not just tokens.
- **Communication.** Default to **no** free-text channel between agents; they signal only through quotes, like the real game. A chat channel invites collusion — add it only if studying collusion is the point.
- **Calibration field.** Have agents emit `fair_value_estimate` alongside actions (enables §7.2).

---

## 9. Protocol schemas (initial sketch — Pydantic)

```python
# protocol.py  — versioned; bump PROTOCOL_VERSION on any breaking change
PROTOCOL_VERSION = "0.1.0"

class Card(BaseModel):
    value: int

class PublicState(BaseModel):
    round: int
    total_rounds: int
    public_cards: list[Card]
    tape: list[Trade]          # recent fills, public
    n_players: int

class AgentView(BaseModel):
    protocol_version: str
    player_id: int
    own_cards: list[Card]
    position: int
    pnl: float
    risk_limit: int
    public: PublicState

class Quote(BaseModel):
    bid: float | None
    ask: float | None
    bid_size: int = 0
    ask_size: int = 0

class Take(BaseModel):
    side: Literal["buy", "sell"]
    price: float
    size: int

class Action(BaseModel):
    kind: Literal["quote", "take", "cancel", "pass"]
    quote: Quote | None = None
    take: Take | None = None
    fair_value_estimate: float | None = None   # for calibration scoring
    rationale: str | None = None               # spectator/researcher only; engine never echoes to other agents
```

The engine ingests `dict[player_id, Action]` per round and returns `(state', [Event])`.

---

## 10. Suggested layout

```
trading-game/
  engine/
    state.py            # GameState, dealing, settlement
    rules.py            # parameterized rules (N, k, m, R, limits)
    auction.py          # call-auction clearing + matching
    engine.py           # step(state, actions) -> (state', [Event])
    events.py           # Event, Visibility, event catalog
  protocol/
    protocol.py         # Pydantic schemas + PROTOCOL_VERSION + JSON Schema export
  agents/
    base.py             # Agent.act(obs) -> actions  (the one interface)
    bots.py             # FairValueBot, EVOptimalBot (reference policies)
    llm_worker.py       # prompt build, tool-call, parse, retry-once, pass
  orchestrator/
    runner.py           # asyncio loop, timeouts, persistence
  readers/
    benchmark.py        # aggregate logs across seeds; suite pinning + hashing
    research.py         # query/instrument logs; config injection
    spectator.py        # render SPECTATOR-tier stream (live-delayed or VOD)
  store/
    log.py              # append-only JSONL writer/reader, content-address config
  config/
    suites/             # frozen (rules, config, seed) tuples -> hashed suites
  tests/
```

---

## 11. Build order (with acceptance criteria)

**Phase 1 — Engine + protocol + one scripted bot. No LLM.**
- Implement `step()`, call-auction matching, dealing, settlement; typed event stream with visibility tiers; `FairValueBot`.
- ✅ Accept when: a full bot-only game runs end to end; same seed → byte-identical event log on replay; settlement math verified by unit tests; the engine rejects illegal actions without crashing; an agent's events never contain another agent's private cards (test asserts isolation).

**Phase 2 — LLM worker; 1 model vs 3 bots.**
- Prompt build, tool/structured-output call, parse, retry-once-then-pass.
- ✅ Accept when: a model plays a complete game; malformed outputs are caught and converted to `pass` without crashing; raw output is logged at `RESEARCHER` tier; rationale never appears in any other agent's view.

**Phase 3 — Multi-model games + eval.**
- Mix models and bots; implement counterfactual/oracle replay, calibration, multi-seed runner.
- ✅ Accept when: a matchup runs over `S` seeds and emits a score distribution (not a point); counterfactual scores reproduce on re-run; calibration trajectory computed per agent.

**Phase 4 — Readers as products.**
- Benchmark suite pinning + hashing; spectator renderer (replay VOD first, then live-delayed).
- ✅ Accept when: a suite is content-addressed and a leaderboard entry cites its hash; the spectator renderer reconstructs a game from log alone with zero engine coupling and zero write path back.

---

## 12. Stack

- **Python** engine (matching is not the bottleneck; LLM latency is).
- **asyncio** orchestration with per-agent deadlines.
- **Pydantic** for protocol (validation + JSON Schema for tool calling).
- Provider SDKs behind one `Agent.act(obs)` interface.
- **JSONL** append-only event log to start; swap for a real event store only if scale demands.

---

## 13. Invariants the agent must never violate

1. The engine is pure and deterministic given `(rules, config, seed)`.
2. The event log is append-only and is the single source of truth.
3. Every event has a visibility tier; readers filter, the engine emits once.
4. No reader (especially spectator) ever writes back to the engine.
5. An agent's view contains only its own private information.
6. A malformed or late agent action degrades to `pass` — it never crashes the game.
7. Benchmark configs are frozen and hashed; sandbox configs are not. That is the only difference between the two.
