"""Pure single-game readers: turn one game's event stream into skill metrics.

These functions are strict consumers of the log — they never touch the engine.
They accept either in-memory `Event` objects (int pid keys) or dicts loaded from a
JSONL log (str pid keys), coercing defensively so both work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.events import (
    ACTION_RECEIVED,
    AUCTION_CLEARED,
    ROUND_END,
    SETTLEMENT,
)


def _fields(e) -> tuple[str, int, dict]:
    """(type, round, payload) for an Event object or a loaded dict."""
    if hasattr(e, "type"):
        return e.type, e.round, e.payload
    return e["type"], e["round"], e["payload"]


def settlement_value(events) -> int:
    for e in events:
        t, _, p = _fields(e)
        if t == SETTLEMENT:
            return int(p["settlement_value"])
    raise ValueError("no settlement event in stream")


def final_pnl(events) -> dict[int, float]:
    for e in events:
        t, _, p = _fields(e)
        if t == SETTLEMENT:
            return {int(k): float(v) for k, v in p["final_pnl"].items()}
    raise ValueError("no settlement event in stream")


def _n_rounds(events) -> int:
    return sum(1 for e in events if _fields(e)[0] == ROUND_END)


# --------------------------------------------------------------------------
# Calibration — does a player's stated fair_value_estimate track the truth?
# --------------------------------------------------------------------------
@dataclass
class CalibrationStats:
    mae: float | None  # mean absolute error over rounds with an estimate
    final_error: float | None  # |last estimate - truth|
    improvement: float | None  # first_error - final_error (>0 means it converged)
    coverage: float  # fraction of rounds the player gave an estimate
    series: list[tuple[int, float, float]] = field(default_factory=list)  # (round, est, abs_err)


def calibration(events) -> dict[int, CalibrationStats]:
    truth = settlement_value(events)
    n_rounds = _n_rounds(events) or 1
    per_player: dict[int, list[tuple[int, float, float]]] = {}
    for e in events:
        t, rnd, p = _fields(e)
        if t != ACTION_RECEIVED:
            continue
        est = p.get("fair_value_estimate")
        if est is None:
            continue
        pid = int(p["player_id"])
        per_player.setdefault(pid, []).append((rnd, float(est), abs(float(est) - truth)))

    out: dict[int, CalibrationStats] = {}
    for pid, series in per_player.items():
        series.sort()
        errs = [e for _, _, e in series]
        out[pid] = CalibrationStats(
            mae=sum(errs) / len(errs) if errs else None,
            final_error=errs[-1] if errs else None,
            improvement=(errs[0] - errs[-1]) if errs else None,
            coverage=len(series) / n_rounds,
            series=series,
        )
    return out


# --------------------------------------------------------------------------
# Price discovery — how fast do clearing prices approach the true sum?
# --------------------------------------------------------------------------
@dataclass
class PriceDiscovery:
    series: list[tuple[int, float | None, float | None]]  # (round, clearing_price, abs_err)
    terminal_error: float | None
    convergence_slope: float | None  # OLS slope of error vs round; <0 means converging


def price_discovery(events) -> PriceDiscovery:
    truth = settlement_value(events)
    series: list[tuple[int, float | None, float | None]] = []
    for e in events:
        t, rnd, p = _fields(e)
        if t != AUCTION_CLEARED:
            continue
        price = p.get("clearing_price")
        err = abs(float(price) - truth) if price is not None else None
        series.append((rnd, price, err))
    series.sort()

    pts = [(r, err) for r, _, err in series if err is not None]
    terminal = pts[-1][1] if pts else None
    slope = _ols_slope(pts) if len(pts) >= 2 else None
    return PriceDiscovery(series=series, terminal_error=terminal, convergence_slope=slope)


def _ols_slope(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom
