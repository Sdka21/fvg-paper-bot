#!/usr/bin/env python3
"""Paper-only Fair Value Gap detector and spot backtester. No live execution."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import ccxt
import numpy as np
import pandas as pd

TIMEFRAME = "15m"
CANDLE_MS = 900_000
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
TERMINAL_STATUSES = {"filled", "invalidated", "used"}


@dataclass
class Zone:
    symbol: str
    direction: str
    created_at: str
    lower: float
    upper: float
    gap_atr: float
    impulse_ratio: float
    volume_ratio: float
    status: str = "active"
    touched_at: str | None = None
    filled_at: str | None = None
    invalidated_at: str | None = None
    confirmed_at: str | None = None
    signal_used: bool = False


@dataclass
class Trade:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    slippage_cost_estimate: float
    net_pnl: float
    exit_reason: str
    zone_lower: float
    zone_upper: float


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    previous = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - previous).abs(),
            (d["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.rolling(14, min_periods=14).mean()
    d["ema_trend"] = d["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    d["vol_median"] = d["volume"].rolling(20, min_periods=20).median()
    return d


def _fvg_candidates(d: pd.DataFrame, i: int, min_gap_atr: float,
                    min_impulse_ratio: float, min_volume_ratio: float,
                    trend_filter: bool = True, volume_filter: bool = True) -> list[dict]:
    if i < 51:
        return []

    a, b, c = d.iloc[i - 2], d.iloc[i - 1], d.iloc[i]
    if not np.isfinite(c["atr"]) or c["atr"] <= 0 or not np.isfinite(b["vol_median"]):
        return []

    candle_range = float(b["high"] - b["low"])
    impulse = abs(float(b["close"] - b["open"])) / candle_range if candle_range > 0 else 0.0
    vol_ratio = float(b["volume"] / b["vol_median"]) if b["vol_median"] > 0 else 0.0

    candidates = []
    if float(c["low"]) > float(a["high"]) and float(b["close"]) > float(b["open"]):
        candidates.append(("bullish", float(a["high"]), float(c["low"])))
    if float(c["high"]) < float(a["low"]) and float(b["close"]) < float(b["open"]):
        candidates.append(("bearish", float(c["high"]), float(a["low"])))

    found = []
    for direction, lower, upper in candidates:
        gap_atr = (upper - lower) / float(c["atr"])
        if gap_atr < min_gap_atr or impulse < min_impulse_ratio:
            continue
        if volume_filter and vol_ratio < min_volume_ratio:
            continue
        if trend_filter:
            if direction == "bullish" and not (c["close"] > c["ema_trend"]):
                continue
            if direction == "bearish" and not (c["close"] < c["ema_trend"]):
                continue

        found.append(
            {
                "symbol": symbol_placeholder if False else "",
                "direction": direction,
                "created_at": pd.Timestamp(
                    int(c["timestamp"]), unit="ms", tz="UTC"
                ).isoformat(),
                "created_index": i,
                "lower": lower,
                "upper": upper,
                "gap_atr": gap_atr,
                "impulse_ratio": impulse,
                "volume_ratio": vol_ratio,
                "status": "active",
                "touched_at": None,
                "filled_at": None,
                "invalidated_at": None,
                "confirmed_at": None,
                "signal_used": False,
            }
        )
    return found


def detect_fvg(df: pd.DataFrame, symbol: str, min_gap_atr: float = 0.12,
               min_impulse_ratio: float = 0.55, min_volume_ratio: float = 1.0,
               trend_filter: bool = True, volume_filter: bool = True) -> list[dict]:
    """Detect qualifying FVGs from closed A/B/C candles."""
    d = indicators(df)
    zones = []
    for i in range(51, len(d)):
        for z in _fvg_candidates(
            d, i, min_gap_atr, min_impulse_ratio, min_volume_ratio,
            trend_filter=trend_filter, volume_filter=volume_filter
        ):
            z["symbol"] = symbol
            zones.append(z)
    return zones


def detect_one(d: pd.DataFrame, i: int, symbol: str, min_gap_atr: float,
               min_impulse_ratio: float, min_volume_ratio: float) -> list[dict]:
    """Online version: evaluate only the latest completed A/B/C trio.

    Indicators are supplied precomputed, so scanning a long history does not
    repeatedly recompute the entire dataframe.
    """
    found = _fvg_candidates(
        d, i, min_gap_atr, min_impulse_ratio, min_volume_ratio,
        trend_filter=True, volume_filter=True
    )
    for z in found:
        z["symbol"] = symbol
    return found


def zone_lifecycle(z: dict, row, stamp: str) -> tuple[bool, bool]:
    """Update one zone using one later closed candle.

    Returns (touched, confirmed). Confirmation is evaluated BEFORE marking a
    deep touch as filled, so a candle that enters the gap and reclaims it can
    generate a valid signal instead of being discarded.
    """
    if z.get("status") in TERMINAL_STATUSES or z.get("signal_used"):
        return False, False

    if z["direction"] == "bullish":
        if float(row["close"]) < z["lower"]:
            z["status"] = "invalidated"
            z["invalidated_at"] = stamp
            return False, False

        touched = float(row["low"]) <= z["upper"] and float(row["high"]) >= z["lower"]
        confirmed = (
            touched
            and float(row["close"]) > z["upper"]
            and float(row["close"]) > float(row["open"])
        )

        if confirmed:
            z["status"] = "confirmed"
            z["confirmed_at"] = stamp
            z["touched_at"] = z["touched_at"] or stamp
            return True, True
        if float(row["low"]) <= z["lower"]:
            z["status"] = "filled"
            z["filled_at"] = stamp
            z["touched_at"] = z["touched_at"] or stamp
            return True, False
        if float(row["low"]) <= z["upper"]:
            z["status"] = "mitigated"
            z["touched_at"] = z["touched_at"] or stamp
            return True, False
        return False, False

    if float(row["close"]) > z["upper"]:
        z["status"] = "invalidated"
        z["invalidated_at"] = stamp
        return False, False

    touched = float(row["high"]) >= z["lower"] and float(row["low"]) <= z["upper"]
    confirmed = (
        touched
        and float(row["close"]) < z["lower"]
        and float(row["close"]) < float(row["open"])
    )

    if confirmed:
        z["status"] = "confirmed"
        z["confirmed_at"] = stamp
        z["touched_at"] = z["touched_at"] or stamp
        return True, True
    if float(row["high"]) >= z["upper"]:
        z["status"] = "filled"
        z["filled_at"] = stamp
        z["touched_at"] = z["touched_at"] or stamp
        return True, False
    if float(row["high"]) >= z["lower"]:
        z["status"] = "mitigated"
        z["touched_at"] = z["touched_at"] or stamp
        return True, False
    return False, False


def update_zones(zones: list[dict], row, stamp: str) -> list[dict]:
    for z in zones:
        zone_lifecycle(z, row, stamp)
    return zones


def latest_valid_fvg(zones: list[dict], direction: str = "bullish") -> dict | None:
    """Return the newest zone still eligible for a signal.

    Active and mitigated zones remain inspectable. Filled, invalidated and
    already-used zones are excluded.
    """
    for z in reversed(zones):
        if z["direction"] != direction:
            continue
        if z.get("status") in TERMINAL_STATUSES or z.get("signal_used"):
            continue
        return z
    return None


def fetch_ohlcv(exchange, symbol: str, days: int) -> pd.DataFrame:
    if symbol not in exchange.markets:
        raise ValueError(f"{symbol} is not listed on {exchange.id}")
    since = exchange.milliseconds() - days * 86_400_000
    cursor, rows = since, []
    while cursor < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(
            symbol, timeframe=TIMEFRAME, since=cursor, limit=1000
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 2:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        raise RuntimeError(f"No OHLCV returned for {symbol}")

    d = (
        pd.DataFrame(rows, columns=COLUMNS)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    now = exchange.milliseconds()
    # Never use the currently-forming 15m candle.
    return d[d["timestamp"] + CANDLE_MS <= now].reset_index(drop=True)


def summarize(trades: list[Trade], initial: float, final: float, max_dd: float) -> dict:
    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    return {
        "trades": len(trades),
        "initial_equity": round(initial, 4),
        "final_equity": round(final, 4),
        "net_pnl": round(final - initial, 4),
        "return_pct": round((final / initial - 1) * 100, 4) if initial else 0.0,
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 4) if len(pnl) else 0.0,
        "profit_factor": round(gp / gl, 4) if gl else (None if not gp else "infinite"),
        "gross_profit": round(gp, 4),
        "gross_loss": round(gl, 4),
        "fees": round(sum(t.fees for t in trades), 4),
        "slippage_cost_estimate": round(sum(t.slippage_cost_estimate for t in trades), 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
    }


def backtest_symbol(df: pd.DataFrame, symbol: str, fee_bps: float, slippage_bps: float,
                    initial_equity: float, risk_fraction: float, reward_risk: float,
                    min_gap_atr: float, min_impulse_ratio: float, min_volume_ratio: float):
    d = indicators(df)
    zones: list[dict] = []
    trades: list[Trade] = []
    equity, peak, max_dd = initial_equity, initial_equity, 0.0
    position = None
    fee, slip = fee_bps / 10_000, slippage_bps / 10_000

    for i in range(51, len(d)):
        row = d.iloc[i]
        stamp = pd.Timestamp(int(row["timestamp"]), unit="ms", tz="UTC").isoformat()

        # Manage positions before evaluating a new entry. Entries are always
        # placed at the next candle's open, never inside the signal candle.
        if position is not None:
            stop_hit = float(row["low"]) <= position["stop"]
            target_hit = float(row["high"]) >= position["target"]
            if stop_hit or target_hit:
                reason = "stop_loss" if stop_hit else "take_profit"
                raw_exit = position["stop"] if stop_hit else position["target"]
                entry_fill = position["entry_ref"] * (1 + slip)
                exit_fill = raw_exit * (1 - slip)
                qty = position["qty"]
                gross = (exit_fill - entry_fill) * qty
                fees = (entry_fill * qty + exit_fill * qty) * fee
                slip_cost = (position["entry_ref"] + raw_exit) * qty * slip
                net = gross - fees
                equity += net
                trades.append(
                    Trade(
                        symbol, position["entry_time"], stamp, entry_fill, exit_fill,
                        qty, gross, fees, slip_cost, net, reason,
                        position["zone"]["lower"], position["zone"]["upper"],
                    )
                )
                position = None
                peak = max(peak, equity)
                max_dd = max(
                    max_dd, (peak - equity) / peak if peak > 0 else 0
                )

        # Detect the current completed A/B/C formation, then lifecycle all
        # older zones using the current candle.
        zones.extend(
            detect_one(
                d, i, symbol, min_gap_atr, min_impulse_ratio, min_volume_ratio
            )
        )
        confirmed_zone_ids = set()
        for z in zones:
            if z["created_index"] < i:
                _, confirmed = zone_lifecycle(z, row, stamp)
                if confirmed:
                    confirmed_zone_ids.add(id(z))

        # Long-only spot. Search newest-to-oldest so the latest valid bullish
        # FVG has priority. A confirmed zone is consumed once it creates an entry.
        if position is None and i + 1 < len(d) and np.isfinite(row["atr"]) and row["atr"] > 0:
            for z in reversed(zones):
                if z["direction"] != "bullish":
                    continue
                if z.get("signal_used") or z.get("status") in TERMINAL_STATUSES:
                    continue
                if id(z) not in confirmed_zone_ids:
                    continue

                entry_ref = float(d.iloc[i + 1]["open"])
                stop = z["lower"] - 0.15 * float(row["atr"])
                if entry_ref <= stop:
                    continue

                risk_unit = entry_ref - stop
                qty = min(
                    max(equity, 0) * risk_fraction / risk_unit,
                    max(equity, 0) * 0.95 / entry_ref,
                )
                if qty <= 0:
                    continue

                position = {
                    "entry_ref": entry_ref,
                    "entry_time": pd.Timestamp(
                        int(d.iloc[i + 1]["timestamp"]), unit="ms", tz="UTC"
                    ).isoformat(),
                    "stop": stop,
                    "target": entry_ref + reward_risk * risk_unit,
                    "qty": qty,
                    "zone": z,
                }
                z["signal_used"] = True
                z["status"] = "used"
                break

    if position is not None:
        row = d.iloc[-1]
        raw_exit = float(row["close"])
        entry_fill = position["entry_ref"] * (1 + slip)
        exit_fill = raw_exit * (1 - slip)
        qty = position["qty"]
        gross = (exit_fill - entry_fill) * qty
        fees = (entry_fill * qty + exit_fill * qty) * fee
        slip_cost = (position["entry_ref"] + raw_exit) * qty * slip
        net = gross - fees
        equity += net
        trades.append(
            Trade(
                symbol,
                position["entry_time"],
                pd.Timestamp(int(row["timestamp"]), unit="ms", tz="UTC").isoformat(),
                entry_fill, exit_fill, qty, gross, fees, slip_cost, net,
                "end_of_sample", position["zone"]["lower"], position["zone"]["upper"],
            )
        )

    return trades, summarize(trades, initial_equity, equity, max_dd), zones


def create_exchange():
    exchange = ccxt.binance(
        {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    )
    exchange.load_markets()
    return exchange


def get_all_spot_symbols(exchange, quote="USDT"):
    """Return every active Binance spot symbol quoted in the requested currency."""
    symbols = []
    for market in exchange.markets.values():
        if (
            market.get("spot")
            and market.get("active", True)
            and market.get("quote") == quote
            and "/" in market["symbol"]
        ):
            symbols.append(market["symbol"])
    return sorted(set(symbols))


def scan_zones(df: pd.DataFrame, symbol: str, min_gap_atr: float,
               min_impulse_ratio: float, min_volume_ratio: float) -> list[dict]:
    """Detect zones and apply lifecycle in chronological order."""
    d = indicators(df)
    zones = []
    for i in range(51, len(d)):
        zones.extend(
            detect_one(
                d, i, symbol, min_gap_atr, min_impulse_ratio, min_volume_ratio
            )
        )
        row = d.iloc[i]
        stamp = pd.Timestamp(int(row["timestamp"]), unit="ms", tz="UTC").isoformat()
        for z in zones:
            if z["created_index"] < i:
                zone_lifecycle(z, row, stamp)
    return zones


def run(args):
    exchange = create_exchange()
    os.makedirs("reports", exist_ok=True)
    all_trades, per_symbol = [], []
    symbols = (
        get_all_spot_symbols(exchange, args.quote)
        if args.all_coins
        else args.symbols
    )
    print(f"Selected {len(symbols)} active Binance spot {args.quote} markets.")

    for symbol in symbols:
        print(f"\n[{symbol}] fetching closed 15m candles for {args.days} days...")
        try:
            df = fetch_ohlcv(exchange, symbol, args.days)
            if args.command == "scan":
                zones = scan_zones(
                    df, symbol, args.min_gap_atr,
                    args.min_impulse_ratio, args.min_volume_ratio
                )
                latest = latest_valid_fvg(zones, "bullish")
                active_count = sum(
                    1 for z in zones
                    if z["direction"] == "bullish"
                    and z["status"] not in TERMINAL_STATUSES
                    and not z.get("signal_used")
                )
                print(f"Closed candles: {len(df)} | FVG zones: {len(zones)} | "
                      f"eligible bullish zones: {active_count}")
                if latest:
                    print(
                        f'LATEST BULLISH FVG: {latest["status"]:10} '
                        f'zone={latest["lower"]:.10g}..{latest["upper"]:.10g} '
                        f'gap/ATR={latest["gap_atr"]:.2f} '
                        f'volume-ratio={latest["volume_ratio"]:.2f} '
                        f'created={latest["created_at"]}'
                    )
                else:
                    print("LATEST BULLISH FVG: NONE")
                for z in zones[-10:]:
                    print(
                        f'{z["direction"]:7} {z["status"]:10} '
                        f'zone={z["lower"]:.10g}..{z["upper"]:.10g} '
                        f'created={z["created_at"]}'
                    )
            else:
                trades, summary, zones = backtest_symbol(
                    df, symbol, args.fee_bps, args.slippage_bps,
                    args.initial_equity, args.risk_fraction, args.reward_risk,
                    args.min_gap_atr, args.min_impulse_ratio, args.min_volume_ratio
                )
                all_trades.extend(trades)
                per_symbol.append({symbol: summary})
                print(json.dumps(summary, indent=2))
        except Exception as exc:
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}")
            if args.command == "backtest":
                per_symbol.append({symbol: {"error": f"{type(exc).__name__}: {exc}"}})

    if args.command == "backtest":
        pd.DataFrame([asdict(t) for t in all_trades]).to_csv(
            "reports/trades.csv", index=False
        )
        report = {
            "config": {
                "timeframe": TIMEFRAME,
                "days": args.days,
                "fee_bps_per_side": args.fee_bps,
                "market_scope": (
                    "all active Binance spot markets"
                    if args.all_coins else "selected symbols"
                ),
                "quote": args.quote,
                "slippage_bps_per_side": args.slippage_bps,
                "initial_equity_per_symbol": args.initial_equity,
                "risk_fraction": args.risk_fraction,
                "reward_risk": args.reward_risk,
                "live_trading": False,
                "position_mode": "long-only spot",
            },
            "per_symbol": per_symbol,
            "combined_trade_count": len(all_trades),
            "combined_net_pnl_across_independent_symbol_accounts": round(
                sum(t.net_pnl for t in all_trades), 4
            ),
        }
        with open("reports/summary.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("\nCombined trades:", len(all_trades))
        print(
            "Note: symbols each start with independent equity; combined P&L "
            "is not one portfolio return."
        )
        print("Reports written to reports/trades.csv and reports/summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="Paper-only multi-market FVG detector/backtester"
    )
    subs = parser.add_subparsers(dest="command", required=True)

    for command in ("scan", "backtest"):
        sp = subs.add_parser(command)
        sp.add_argument(
            "--symbols", nargs="+",
            default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BONK/USDT"],
            help="Used only without --all-coins.",
        )
        sp.add_argument(
            "--all-coins", action="store_true",
            help="Use every active Binance spot market quoted in --quote.",
        )
        sp.add_argument(
            "--quote", default="USDT",
            help="Quote asset used by --all-coins (default: USDT).",
        )
        sp.add_argument("--days", type=int, default=180)
        sp.add_argument("--min-gap-atr", type=float, default=0.12)
        sp.add_argument("--min-impulse-ratio", type=float, default=0.55)
        sp.add_argument("--min-volume-ratio", type=float, default=1.0)

        if command == "backtest":
            sp.add_argument("--fee-bps", type=float, default=10.0)
            sp.add_argument("--slippage-bps", type=float, default=5.0)
            sp.add_argument("--initial-equity", type=float, default=1000.0)
            sp.add_argument("--risk-fraction", type=float, default=0.005)
            sp.add_argument("--reward-risk", type=float, default=2.0)

    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be >= 1")
    if args.command == "backtest" and (args.fee_bps < 0 or args.slippage_bps < 0):
        parser.error("fee/slippage must be non-negative")
    if args.command == "backtest" and not (0 < args.risk_fraction <= 0.05):
        parser.error("--risk-fraction must be > 0 and <= 0.05")

    run(args)


if __name__ == "__main__":
    main()
