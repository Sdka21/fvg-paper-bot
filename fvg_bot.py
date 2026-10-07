#!/usr/bin/env python3
"""ICT-inspired, long-only Binance Spot paper bot and backtester.

Paper-only: public OHLCV data, no API keys, no order submission.
Model: higher-timeframe bias -> sell-side liquidity sweep -> bullish
displacement -> bullish FVG -> retracement/confirmation -> long spot.
"""
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
    ce: float
    gap_atr: float
    impulse_ratio: float
    volume_ratio: float
    sweep_level: float | None = None
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
    stop_price: float
    target_price: float
    zone_lower: float
    zone_upper: float
    sweep_level: float | None


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    prev = d["close"].shift(1)
    tr = pd.concat(
        [(d["high"] - d["low"]),
         (d["high"] - prev).abs(),
         (d["low"] - prev).abs()], axis=1
    ).max(axis=1)
    d["atr"] = tr.rolling(14, min_periods=14).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    d["vol_median"] = d["volume"].rolling(20, min_periods=20).median()
    d["swing_low"] = d["low"].shift(1).rolling(10, min_periods=10).min()
    d["swing_high"] = d["high"].shift(1).rolling(10, min_periods=10).max()
    # Previous completed 1h candle information, deliberately lagged.
    h = (
        d.set_index(pd.to_datetime(d["timestamp"], unit="ms", utc=True))
        .resample("1h")
        .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        .dropna()
    )
    h["ema20"] = h["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    h["ema50"] = h["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    h["bias"] = np.where(
        (h["close"] > h["ema20"]) & (h["ema20"] > h["ema50"]), "bullish",
        np.where((h["close"] < h["ema20"]) & (h["ema20"] < h["ema50"]), "bearish", "neutral")
    )
    idx = pd.to_datetime(d["timestamp"], unit="ms", utc=True).dt.floor("1h")
    h["bias_prev"] = h["bias"].shift(1)
    mapped = h["bias_prev"].reindex(idx, method="ffill").to_numpy()
    d["htf_bias"] = mapped
    return d


def _fvg_candidates(
    d: pd.DataFrame, i: int, min_gap_atr: float, min_impulse_ratio: float,
    min_volume_ratio: float, trend_filter: bool = True,
    volume_filter: bool = True, require_sweep: bool = True
) -> list[dict]:
    if i < 55:
        return []
    a, b, c = d.iloc[i-2], d.iloc[i-1], d.iloc[i]
    if not np.isfinite(c["atr"]) or c["atr"] <= 0 or not np.isfinite(b["vol_median"]):
        return []

    rng = float(b["high"] - b["low"])
    impulse = abs(float(b["close"] - b["open"])) / rng if rng > 0 else 0.0
    vol_ratio = float(b["volume"] / b["vol_median"]) if b["vol_median"] > 0 else 0.0

    # Sell-side liquidity = recent external/internal swing low. Sweep means
    # B takes that low but closes back above it. The displacement candle C
    # must then confirm bullish intent.
    lookback = d.iloc[max(0, i-21):i-1]
    sell_side = float(lookback["low"].min()) if len(lookback) else np.nan
    swept = float(b["low"]) < sell_side and float(b["close"]) > sell_side if np.isfinite(sell_side) else False

    found = []
    if float(c["low"]) > float(a["high"]) and float(b["close"]) > float(b["open"]):
        lower, upper = float(a["high"]), float(c["low"])
        gap_atr = (upper - lower) / float(c["atr"])
        if gap_atr >= min_gap_atr and impulse >= min_impulse_ratio:
            if not volume_filter or vol_ratio >= min_volume_ratio:
                if not trend_filter or c["htf_bias"] in ("bullish", "neutral"):
                    # Either B is the sweep candle or a sweep happened in the
                    # recent five candles before the displacement sequence.
                    recent = d.iloc[max(0, i-6):i]
                    recent_sweep = False
                    sweep_level = sell_side
                    if len(recent) >= 2:
                        prior_low = recent["low"].shift(1).rolling(10, min_periods=2).min()
                        recent_sweep = bool(((recent["low"] < prior_low) & (recent["close"] > prior_low)).any())
                    if not require_sweep or swept or recent_sweep:
                        found.append({
                            "symbol": "", "direction": "bullish",
                            "created_at": pd.Timestamp(int(c["timestamp"]), unit="ms", tz="UTC").isoformat(),
                            "created_index": i, "lower": lower, "upper": upper,
                            "ce": (lower + upper) / 2, "gap_atr": gap_atr,
                            "impulse_ratio": impulse, "volume_ratio": vol_ratio,
                            "sweep_level": sweep_level if (swept or recent_sweep) else None,
                            "status": "active", "touched_at": None, "filled_at": None,
                            "invalidated_at": None, "confirmed_at": None, "signal_used": False
                        })
    return found


def detect_fvg(
    df: pd.DataFrame, symbol: str, min_gap_atr: float = 0.12,
    min_impulse_ratio: float = 0.55, min_volume_ratio: float = 1.0,
    trend_filter: bool = True, volume_filter: bool = True,
    require_sweep: bool = True
) -> list[dict]:
    d = indicators(df)
    zones = []
    for i in range(55, len(d)):
        for z in _fvg_candidates(
            d, i, min_gap_atr, min_impulse_ratio, min_volume_ratio,
            trend_filter, volume_filter, require_sweep
        ):
            z["symbol"] = symbol
            zones.append(z)
    return zones


def detect_one(d, i, symbol, min_gap_atr, min_impulse_ratio, min_volume_ratio, require_sweep=True):
    found = _fvg_candidates(
        d, i, min_gap_atr, min_impulse_ratio, min_volume_ratio,
        True, True, require_sweep
    )
    for z in found:
        z["symbol"] = symbol
    return found


def zone_lifecycle(z: dict, row, stamp: str) -> tuple[bool, bool]:
    if z.get("status") in TERMINAL_STATUSES or z.get("signal_used"):
        return False, False

    # Bullish FVG is invalid if price closes through its lower boundary.
    if float(row["close"]) < z["lower"]:
        z["status"] = "invalidated"
        z["invalidated_at"] = stamp
        return False, False

    touched = float(row["low"]) <= z["upper"] and float(row["high"]) >= z["lower"]
    # Confirmation: price trades into the zone and closes back above CE with
    # a bullish body. This prevents treating a random touch as an entry.
    confirmed = (
        touched and float(row["close"]) > z["ce"] and
        float(row["close"]) > float(row["open"])
    )
    if confirmed:
        z["status"] = "confirmed"
        z["confirmed_at"] = stamp
        z["touched_at"] = z["touched_at"] or stamp
        return True, True

    if touched:
        z["touched_at"] = z["touched_at"] or stamp
        z["status"] = "mitigated"
        return True, False
    return False, False


def update_zones(zones, row, stamp):
    for z in zones:
        zone_lifecycle(z, row, stamp)
    return zones


def latest_valid_fvg(zones, direction="bullish"):
    for z in reversed(zones):
        if z["direction"] == direction and z.get("status") not in TERMINAL_STATUSES and not z.get("signal_used"):
            return z
    return None


def fetch_ohlcv(exchange, symbol, days):
    if symbol not in exchange.markets:
        raise ValueError(f"{symbol} is not listed on {exchange.id}")
    since = exchange.milliseconds() - days * 86_400_000
    cursor, rows = since, []
    while cursor < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < 2:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not rows:
        raise RuntimeError(f"No OHLCV returned for {symbol}")
    d = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("timestamp").sort_values("timestamp")
    now = exchange.milliseconds()
    return d[d["timestamp"] + CANDLE_MS <= now].reset_index(drop=True)


def summarize(trades, initial, final, max_dd):
    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = float(wins.sum()) if len(wins) else 0.0, float(-losses.sum()) if len(losses) else 0.0
    return {
        "trades": len(trades), "initial_equity": round(initial, 4),
        "final_equity": round(final, 4), "net_pnl": round(final-initial, 4),
        "return_pct": round((final/initial-1)*100, 4) if initial else 0.0,
        "win_rate_pct": round(float((pnl > 0).mean()*100), 4) if len(pnl) else 0.0,
        "profit_factor": round(gp/gl, 4) if gl else (None if not gp else "infinite"),
        "gross_profit": round(gp, 4), "gross_loss": round(gl, 4),
        "fees": round(sum(t.fees for t in trades), 4),
        "slippage_cost_estimate": round(sum(t.slippage_cost_estimate for t in trades), 4),
        "max_drawdown_pct": round(max_dd*100, 4)
    }


def _close_position(position, row, reason, fee, slip, equity):
    raw_exit = position["stop"] if reason == "stop_loss" else position["target"]
    entry_fill = position["entry_ref"] * (1 + slip)
    exit_fill = raw_exit * (1 - slip)
    qty = position["qty"]
    gross = (exit_fill-entry_fill)*qty
    fees = (entry_fill*qty+exit_fill*qty)*fee
    slip_cost = (position["entry_ref"]+raw_exit)*qty*slip
    net = gross-fees
    stamp = pd.Timestamp(int(row["timestamp"]), unit="ms", tz="UTC").isoformat()
    trade = Trade(
        position["symbol"], position["entry_time"], stamp, entry_fill, exit_fill,
        qty, gross, fees, slip_cost, net, reason, position["stop"],
        position["target"], position["zone"]["lower"], position["zone"]["upper"],
        position["zone"].get("sweep_level")
    )
    return trade, equity+net


def backtest_symbol(
    df, symbol, fee_bps, slippage_bps, initial_equity, risk_fraction,
    reward_risk, min_gap_atr, min_impulse_ratio, min_volume_ratio,
    stop_atr_buffer=0.15, entry_mode="confirmation"
):
    d = indicators(df)
    zones, trades = [], []
    equity, peak, max_dd = initial_equity, initial_equity, 0.0
    position = None
    fee, slip = fee_bps/10_000, slippage_bps/10_000

    for i in range(55, len(d)-1):
        row = d.iloc[i]
        stamp = pd.Timestamp(int(row["timestamp"]), unit="ms", tz="UTC").isoformat()

        if position is not None:
            stop_hit = float(row["low"]) <= position["stop"]
            target_hit = float(row["high"]) >= position["target"]
            if stop_hit or target_hit:
                reason = "stop_loss" if stop_hit else "take_profit"
                trade, equity = _close_position(position, row, reason, fee, slip, equity)
                trades.append(trade)
                position = None
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak-equity)/peak if peak > 0 else 0)

        # Create the new FVG only after its three candles are closed.
        zones.extend(detect_one(
            d, i, symbol, min_gap_atr, min_impulse_ratio, min_volume_ratio, True
        ))

        confirmed = []
        for z in zones:
            if z["created_index"] < i:
                _, ok = zone_lifecycle(z, row, stamp)
                if ok:
                    confirmed.append(z)

        if position is None and np.isfinite(row["atr"]) and row["atr"] > 0:
            candidates = confirmed if entry_mode == "confirmation" else [
                z for z in reversed(zones)
                if z["direction"] == "bullish"
                and z.get("status") in ("active", "mitigated")
                and not z.get("signal_used")
                and z["lower"] <= float(row["close"]) <= z["upper"]
            ]
            if entry_mode == "confirmation":
                candidates = list(reversed(candidates))

            for z in candidates:
                if z.get("signal_used") or z.get("status") in TERMINAL_STATUSES:
                    continue
                # Spot long only: HTF bearish is rejected.
                if row["htf_bias"] == "bearish":
                    continue
                entry_ref = float(d.iloc[i+1]["open"]) if entry_mode == "confirmation" else float(row["close"])
                sweep = z.get("sweep_level")
                stop_base = sweep if sweep is not None else z["lower"]
                stop = min(float(stop_base), z["lower"]) - stop_atr_buffer*float(row["atr"])
                if entry_ref <= stop:
                    continue
                risk_unit = entry_ref-stop
                qty = min(
                    max(equity,0)*risk_fraction/risk_unit,
                    max(equity,0)*0.95/entry_ref
                )
                if qty <= 0:
                    continue
                position = {
                    "symbol": symbol, "entry_ref": entry_ref,
                    "entry_time": pd.Timestamp(int(d.iloc[i+1]["timestamp"]), unit="ms", tz="UTC").isoformat(),
                    "stop": stop, "target": entry_ref+reward_risk*risk_unit,
                    "qty": qty, "zone": z
                }
                z["signal_used"] = True
                z["status"] = "used"
                break

    if position is not None:
        row = d.iloc[-1]
        trade, equity = _close_position(position, row, "end_of_sample", fee, slip, equity)
        # For end-of-sample the helper uses target; correct it with market close.
        raw_exit = float(row["close"])
        entry_fill = position["entry_ref"]*(1+slip)
        exit_fill = raw_exit*(1-slip)
        qty = position["qty"]
        gross = (exit_fill-entry_fill)*qty
        fees = (entry_fill*qty+exit_fill*qty)*fee
        net = gross-fees
        trade = Trade(
            symbol, position["entry_time"],
            pd.Timestamp(int(row["timestamp"]), unit="ms", tz="UTC").isoformat(),
            entry_fill, exit_fill, qty, gross, fees,
            (position["entry_ref"]+raw_exit)*qty*slip, net, "end_of_sample",
            position["stop"], position["target"], position["zone"]["lower"],
            position["zone"]["upper"], position["zone"].get("sweep_level")
        )
        equity = initial_equity + sum(t.net_pnl for t in trades) + net
        trades.append(trade)

    return trades, summarize(trades, initial_equity, equity, max_dd), zones


def create_exchange():
    public_base = os.getenv("BINANCE_PUBLIC_API_BASE", "https://data-api.binance.vision/api/v3").rstrip("/")
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot", "fetchMarkets": {"types": ["spot"]}}
    })
    exchange.urls["api"]["public"] = public_base
    exchange.load_markets()
    return exchange


def get_all_spot_symbols(exchange, quote="USDT"):
    """Return active crypto spot symbols, excluding tokenised-equity style bases."""
    excluded_suffixes = tuple(
        s.strip().upper()
        for s in os.getenv("EXCLUDED_BASE_SUFFIXES", "B").split(",")
        if s.strip()
    )
    excluded_exact = {
        "AAPL", "AMZN", "COIN", "GOOG", "GOOGL", "META", "MSFT",
        "MSTR", "NFLX", "NVDA", "ORCL", "TSLA",
    }
    symbols = set()
    for m in exchange.markets.values():
        if not m.get("spot") or not m.get("active", True) or m.get("quote") != quote:
            continue
        symbol = m.get("symbol", "")
        base = str(m.get("base") or "").upper()
        if "/" not in symbol or not base:
            continue
        if base in excluded_exact or any(base.endswith(s) for s in excluded_suffixes):
            continue
        symbols.add(symbol)
    return sorted(symbols)


def scan_zones(df, symbol, args):
    d = indicators(df)
    zones = []
    for i in range(55, len(d)):
        zones.extend(detect_one(
            d, i, symbol, args.min_gap_atr, args.min_impulse_ratio,
            args.min_volume_ratio, args.require_sweep
        ))
        row, stamp = d.iloc[i], pd.Timestamp(int(d.iloc[i]["timestamp"]), unit="ms", tz="UTC").isoformat()
        for z in zones:
            if z["created_index"] < i:
                zone_lifecycle(z, row, stamp)
    return zones


def run(args):
    exchange = create_exchange()
    os.makedirs("reports", exist_ok=True)
    symbols = get_all_spot_symbols(exchange, args.quote) if args.all_coins else args.symbols
    print(f"Selected {len(symbols)} active Binance Spot {args.quote} markets.")

    if args.command == "paper":
        # One-shot paper cycle. Scheduled CI can invoke this every 15 minutes.
        args.days = max(args.days, 3)

    all_trades, per_symbol, signals = [], [], []
    for symbol in symbols:
        try:
            df = fetch_ohlcv(exchange, symbol, args.days)
            if len(df) < 80:
                continue
            if args.command in ("scan", "paper"):
                zones = scan_zones(df, symbol, args)
                latest = latest_valid_fvg(zones, "bullish")
                if latest:
                    signal = {
                        "symbol": symbol, "type": "LONG_CANDIDATE",
                        "status": latest["status"], "created_at": latest["created_at"],
                        "zone": [latest["lower"], latest["upper"]],
                        "ce": latest["ce"], "sweep_level": latest.get("sweep_level"),
                        "gap_atr": latest["gap_atr"], "volume_ratio": latest["volume_ratio"]
                    }
                    signals.append(signal)
                    print(json.dumps(signal))
            else:
                trades, summary, _ = backtest_symbol(
                    df, symbol, args.fee_bps, args.slippage_bps,
                    args.initial_equity, args.risk_fraction, args.reward_risk,
                    args.min_gap_atr, args.min_impulse_ratio, args.min_volume_ratio,
                    args.stop_atr_buffer, args.entry_mode
                )
                all_trades.extend(trades)
                per_symbol.append({symbol: summary})
                print(symbol, json.dumps(summary))
        except Exception as exc:
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}")
            if args.command == "backtest":
                per_symbol.append({symbol: {"error": f"{type(exc).__name__}: {exc}"}})

    if args.command == "backtest":
        pd.DataFrame([asdict(t) for t in all_trades]).to_csv("reports/trades.csv", index=False)
        report = {
            "strategy": "ICT-inspired long-only spot",
            "config": vars(args),
            "per_symbol": per_symbol,
            "combined_trade_count": len(all_trades),
            "combined_net_pnl_across_independent_symbol_accounts": round(sum(t.net_pnl for t in all_trades), 4)
        }
        with open("reports/summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    else:
        with open("reports/latest_signals.json", "w", encoding="utf-8") as f:
            json.dump({"strategy":"ICT-inspired long-only spot","signals":signals}, f, indent=2, default=str)
        print(f"LONG candidates: {len(signals)}")


def build_parser():
    p = argparse.ArgumentParser(description="ICT-inspired long-only Binance Spot paper bot")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("scan", "paper", "backtest"):
        s = sub.add_parser(command)
        s.add_argument("--symbols", nargs="+", default=["BTC/USDT","ETH/USDT","SOL/USDT","BONK/USDT"])
        s.add_argument("--all-coins", action="store_true")
        s.add_argument("--quote", default="USDT")
        s.add_argument("--days", type=int, default=180)
        s.add_argument("--min-gap-atr", type=float, default=0.12)
        s.add_argument("--min-impulse-ratio", type=float, default=0.55)
        s.add_argument("--min-volume-ratio", type=float, default=1.0)
        s.add_argument("--require-sweep", action=argparse.BooleanOptionalAction, default=True)
        s.add_argument("--stop-atr-buffer", type=float, default=0.15)
        if command == "backtest":
            s.add_argument("--fee-bps", type=float, default=10.0)
            s.add_argument("--slippage-bps", type=float, default=5.0)
            s.add_argument("--initial-equity", type=float, default=1000.0)
            s.add_argument("--risk-fraction", type=float, default=0.005)
            s.add_argument("--reward-risk", type=float, default=2.0)
            s.add_argument("--entry-mode", choices=["confirmation","touch"], default="confirmation")
    return p


def main():
    args = build_parser().parse_args()
    if args.days < 1:
        raise SystemExit("--days must be >= 1")
    if args.command == "backtest" and not (0 < args.risk_fraction <= 0.05):
        raise SystemExit("--risk-fraction must be > 0 and <= 0.05")
    run(args)


if __name__ == "__main__":
    main()
