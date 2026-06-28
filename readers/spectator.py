"""Spectator renderer — replay a finished game as a watchable text VOD.

This is the purest reader: it reconstructs the whole show from the **log alone**,
consuming only `SPECTATOR`-tier events. It has **zero engine coupling** (it imports
nothing from `engine/` and never rebuilds game state) and **no write path** back to
anything — contamination is structurally impossible.

What a spectator may see (and nothing more): public/deal cards, every quote and take
with its stated rationale, the auction clearing price and volume, fills, and the final
settlement + ranking. Private cards (AGENT tier), raw model output and the pre-reveal
true value (RESEARCHER tier) never reach this renderer.
"""

from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

SPECTATOR = "spectator"

# Event type vocabulary (string literals — deliberately not imported from engine,
# to keep the renderer decoupled from the engine package).
GAME_START = "game_start"
DEAL = "deal"
ACTION_RECEIVED = "action_received"
ACTION_REJECTED = "action_rejected"
AUCTION_CLEARED = "auction_cleared"
FILL = "fill"
ROUND_END = "round_end"
SETTLEMENT = "settlement"
GAME_END = "game_end"


def _view(e) -> tuple[str, int, dict, str]:
    """(type, round, payload, visibility) for an Event object or a loaded dict."""
    if hasattr(e, "type"):
        vis = e.visibility.value if hasattr(e.visibility, "value") else e.visibility
        return e.type, e.round, e.payload, vis
    # A dict with no "visibility" key is an already-filtered spectator event,
    # which makes spectator_only() idempotent (load() filters, render() re-filters).
    return e["type"], e["round"], e["payload"], e.get("visibility", SPECTATOR)


def spectator_only(events) -> list[dict]:
    """Filter any event stream down to the spectator-visible slice (as plain dicts)."""
    out = []
    for e in events:
        t, rnd, payload, vis = _view(e)
        if vis == SPECTATOR:
            out.append({"type": t, "round": rnd, "payload": payload})
    return out


def load(path: str | Path) -> list[dict]:
    """Load a finished JSONL log and keep only spectator-tier events."""
    import json

    with Path(path).open("r", encoding="ascii") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    return spectator_only(raw)


def _fmt_action(p: dict) -> str:
    pid = p["player_id"]
    kind = p["kind"]
    fv = p.get("fair_value_estimate")
    fv_str = f"  (FV est {fv:g})" if fv is not None else ""
    rat = p.get("rationale")
    rat_str = f'  "{rat}"' if rat else ""
    if kind == "quote" and p.get("quote"):
        q = p["quote"]
        bid = f"{q['bid']:g}" if q.get("bid") is not None else "--"
        ask = f"{q['ask']:g}" if q.get("ask") is not None else "--"
        body = f"quote  {bid} / {ask}   x{q.get('bid_size', 0)}/{q.get('ask_size', 0)}"
    elif kind == "take" and p.get("take"):
        t = p["take"]
        body = f"take   {t['side']} {t['size']} @ {t['price']:g}"
    else:
        body = "pass"
    return f"  P{pid}  {body}{fv_str}{rat_str}"


def render(events, *, delay: float = 0.0, file=None) -> str:
    """Render the spectator stream to text. Returns the full transcript; also writes
    to `file` (default: returns only). `delay` pauses between rounds for live pacing."""
    spec = spectator_only(events)
    buf = StringIO()

    def out(line: str = "") -> None:
        buf.write(line + "\n")
        if file is not None:
            print(line, file=file, flush=True)

    positions: dict[int, int] = {}
    n_players = 0

    # --- header (game_start + the public deal) ---
    gs = next((e for e in spec if e["type"] == GAME_START), None)
    if gs:
        p = gs["payload"]
        n_players = p.get("n_players", 0)
        public = next((e["payload"].get("public_cards", []) for e in spec
                       if e["type"] == DEAL and "public_cards" in e["payload"]), [])
        out("=" * 52)
        out("  TRADING GAME — spectator replay")
        out(f"  {p.get('n_players')} players · {p.get('total_rounds')} rounds · "
            f"deck {p.get('card_min')}–{p.get('card_max')} · public cards: {public}")
        out("=" * 52)

    # --- group the body by round ---
    rounds = sorted({e["round"] for e in spec
                     if e["type"] in (ACTION_RECEIVED, AUCTION_CLEARED, FILL, ROUND_END)
                     and e["round"] >= 1})
    for rnd in rounds:
        out(f"\n── Round {rnd} " + "─" * (40 - len(str(rnd))))
        for e in spec:
            if e["round"] != rnd:
                continue
            t, p = e["type"], e["payload"]
            if t == ACTION_RECEIVED:
                out(_fmt_action(p))
            elif t == ACTION_REJECTED:
                out(f"  P{p['player_id']}  rejected ({p['reason']}) → pass")
            elif t == AUCTION_CLEARED:
                price = p["clearing_price"]
                if price is None:
                    out("  ⚖  no trade (no crossing orders)")
                else:
                    capped = p.get("risk_capped") or []
                    cap = f"   [risk-capped: {capped}]" if capped else ""
                    out(f"  ⚖  cleared @ {price:g}   volume {p['matched_volume']}{cap}")
            elif t == FILL:
                pid, side, size = p["player_id"], p["side"], p["size"]
                positions[pid] = positions.get(pid, 0) + (size if side == "buy" else -size)
                out(f"  ✔  P{pid} {side} {size} @ {p['price']:g}")
        if positions:
            pos = "  ".join(f"P{i}{positions.get(i, 0):+d}"
                            for i in range(n_players or (max(positions) + 1)))
            out(f"  positions: {pos}")
        if delay:
            time.sleep(delay)

    # --- settlement ---
    st = next((e for e in spec if e["type"] == SETTLEMENT), None)
    if st:
        p = st["payload"]
        pnl = {int(k): v for k, v in p["final_pnl"].items()}
        ranking = [int(x) for x in p["ranking"]]
        out("\n" + "═" * 52)
        out(f"  SETTLEMENT — true value: {p['settlement_value']}")
        for rank, pid in enumerate(ranking):
            tag = "  ← winner" if rank == 0 else ""
            out(f"    P{pid}   PnL {pnl[pid]:+.1f}{tag}")
        out(f"  ranking: {', '.join('P' + str(x) for x in ranking)}")
        out("═" * 52)

    return buf.getvalue()


def _latest_log() -> Path | None:
    logs = sorted(Path("logs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = _latest_log()
        if path is None:
            # No log yet — generate a quick bot game to watch (lazy import keeps the
            # renderer itself free of any engine/orchestrator dependency).
            from agents.bots import FairValueBot
            from engine.rules import Rules
            from orchestrator.runner import run_game
            from store.log import config_hash

            rules = Rules(n_players=4, k_private=2, m_public=1, total_rounds=5)
            agents = {pid: FairValueBot(pid) for pid in range(rules.n_players)}
            path = Path("logs") / f"game_{config_hash(rules.model_dump(), 42)[:12]}.jsonl"
            run_game(rules, 42, agents, path)

    render(load(path), file=sys.stdout)
    print(f"\n(replayed from {path})")


if __name__ == "__main__":
    main()
