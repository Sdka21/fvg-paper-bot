#!/usr/bin/env python3
"""Quick ICT scalp paper trader: 5m execution with 15m bias.

Paper-only, long-only Binance Spot. Market data is read from Binance's
public market-data endpoint; no API key, account access, or live order
endpoint is used.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

STATE_PATH = "reports/scalp_state.json"
TRADES_PATH = "reports/scalp_trades.csv"
EQUITY_PATH = "reports/scalp_equity.csv"
CANDLE_MS = 300_000
DEFAULT_PUBLIC_API = "https://data-api.binance.vision/api/v3"
EXCLUDED_BASE_SUFFIXES = tuple(s.strip().upper() for s in os.getenv("EXCLUDED_BASE_SUFFIXES", "B").split(",") if s.strip())

def is_crypto_spot_market(market, quote):
    """Eligible crypto spot market; excludes tokenised-equity style bases."""
    if not market.get("spot") or not market.get("active", True):
        return False
    if market.get("quote") != quote:
        return False
    base = str(market.get("base") or "").upper()
    if not base or any(base.endswith(suffix) for suffix in EXCLUDED_BASE_SUFFIXES):
        return False
    if base in {"AAPL","AMZN","COIN","GOOGL","GOOG","META","MSFT","MSTR","NFLX","NVDA","ORCL","TSLA"}:
        return False
    return True



def now_iso():
    return datetime.now(timezone.utc).isoformat()


def create_exchange():
    """Create a public-data-only Binance Spot client.

    GitHub-hosted runners can receive HTTP 451 from api.binance.com because of
    regional eligibility rules. Binance documents data-api.binance.vision as a
    no-auth market-data endpoint and it supports exchangeInfo and klines.
    """
    public_base = os.getenv("BINANCE_PUBLIC_API_BASE", DEFAULT_PUBLIC_API).rstrip("/")
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "fetchMarkets": {"types": ["spot"]},
        },
    })
    exchange.urls["api"]["public"] = public_base
    return exchange


def fetch_ohlcv(exchange, symbol, timeframe, days):
    if symbol not in exchange.markets:
        raise ValueError(f"{symbol} is not listed on {exchange.id}")
    since = exchange.milliseconds() - days * 86_400_000
    rows, cursor = [], since
    while cursor < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=cursor, limit=1000
        )
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
    d = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    now = exchange.milliseconds()
    return d[d.timestamp + CANDLE_MS <= now].reset_index(drop=True)


def indicators_5m(d):
    x = d.copy()
    prev = x.close.shift(1)
    tr = pd.concat(
        [(x.high - x.low), (x.high - prev).abs(), (x.low - prev).abs()],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(14, min_periods=14).mean()
    x["vol_median"] = x.volume.rolling(20, min_periods=20).median()

    # Completed 15m candles, lagged by one 15m bar to avoid look-ahead.
    h = (
        x.set_index(pd.to_datetime(x.timestamp, unit="ms", utc=True))
        .resample("15min")
        .agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        })
        .dropna()
    )
    h["ema20"] = h.close.ewm(span=20, adjust=False, min_periods=20).mean()
    h["ema50"] = h.close.ewm(span=50, adjust=False, min_periods=50).mean()
    h["bias"] = np.where(
        (h.close > h.ema20) & (h.ema20 > h.ema50),
        "bullish",
        np.where(
            (h.close < h.ema20) & (h.ema20 < h.ema50),
            "bearish",
            "neutral",
        ),
    )
    h["bias_prev"] = h.bias.shift(1)
    idx = pd.to_datetime(x.timestamp, unit="ms", utc=True).dt.floor("15min")
    x["htf_bias"] = h.bias_prev.reindex(idx, method="ffill").to_numpy()
    return x


def detect_scalp_fvg(d, i, symbol, min_gap_atr, min_impulse, min_volume):
    """Detect the precise three-candle bullish FVG ending at candle i."""
    if i < 55:
        return None
    a, b, c = d.iloc[i - 2], d.iloc[i - 1], d.iloc[i]
    if (
        not np.isfinite(c.atr)
        or c.atr <= 0
        or not np.isfinite(b.vol_median)
        or b.vol_median <= 0
    ):
        return None

    rng = float(b.high - b.low)
    impulse = abs(float(b.close - b.open)) / rng if rng > 0 else 0.0
    volume_ratio = float(b.volume / b.vol_median)

    # Bullish three-candle FVG: candle C low is above candle A high.
    if float(c.low) <= float(a.high) or float(b.close) <= float(b.open):
        return None

    lower, upper = float(a.high), float(c.low)
    gap_atr = (upper - lower) / float(c.atr)
    if (
        gap_atr < min_gap_atr
        or impulse < min_impulse
        or volume_ratio < min_volume
        or c.htf_bias == "bearish"
    ):
        return None

    # Sell-side liquidity sweep before/around displacement.
    look = d.iloc[max(0, i - 21):i - 1]
    sell_side = float(look.low.min()) if len(look) else np.nan
    swept = (
        np.isfinite(sell_side)
        and float(b.low) < sell_side
        and float(b.close) > sell_side
    )

    recent = d.iloc[max(0, i - 6):i]
    recent_sweep = False
    if len(recent) >= 3:
        prior = recent.low.shift(1).rolling(10, min_periods=2).min()
        recent_sweep = bool(
            ((recent.low < prior) & (recent.close > prior)).any()
        )

    if not (swept or recent_sweep):
        return None

    return {
        "symbol": symbol,
        "created_index": i,
        "created_at": pd.Timestamp(
            int(c.timestamp), unit="ms", tz="UTC"
        ).isoformat(),
        "lower": lower,
        "upper": upper,
        "ce": (lower + upper) / 2,
        "sweep_level": sell_side,
        "gap_atr": gap_atr,
        "impulse_ratio": impulse,
        "volume_ratio": volume_ratio,
    }


def latest_entry_fvg(
    d, symbol, min_gap_atr, min_impulse, min_volume, latest_index
):
    """Return the newest FVG whose first valid confirmation is this candle.

    This fixes the old behaviour that only inspected the newest three candles.
    Older, still-unmitigated FVGs are considered newest-to-oldest and can be
    used when the latest completed 5m candle retraces into them.
    """
    for i in range(latest_index, 54, -1):
        z = detect_scalp_fvg(
            d, i, symbol, min_gap_atr, min_impulse, min_volume
        )
        if not z:
            continue

        # The zone must survive all candles between creation and the current
        # candle. A close through the lower edge invalidates it.
        invalidated = False
        already_confirmed = False
        for j in range(i + 1, latest_index + 1):
            row = d.iloc[j]
            if float(row.close) < z["lower"]:
                invalidated = True
                break
            touched = (
                float(row.low) <= z["upper"]
                and float(row.high) >= z["lower"]
            )
            confirmed = (
                touched
                and float(row.close) > z["ce"]
                and float(row.close) > float(row.open)
            )
            if confirmed:
                if j != latest_index:
                    already_confirmed = True
                break

        if invalidated or already_confirmed:
            continue

        row = d.iloc[latest_index]
        touched = float(row.low) <= z["upper"] and float(row.high) >= z["lower"]
        confirmed = (
            touched
            and float(row.close) > z["ce"]
            and float(row.close) > float(row.open)
        )
        if confirmed:
            z["confirmed_at"] = pd.Timestamp(
                int(row.timestamp), unit="ms", tz="UTC"
            ).isoformat()
            return z
    return None


def load_state(initial, max_open):
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        s.setdefault("equity", initial)
        s.setdefault("cash", initial)
        s.setdefault("open_positions", [])
        s.setdefault("trades", [])
        s.setdefault("processed", {})
        s.setdefault("equity_curve", [])
        s.setdefault("max_open_positions", max_open)
        s.setdefault("capital_deployment", 0.90)
        s.setdefault("last_signals", [])
        s.setdefault("data_status", "unknown")
        s.setdefault("last_error", None)
        s.setdefault("last_prices", {})
        s.setdefault("scan_errors", {})
        s.setdefault("symbols_scanned", 0)
        s.setdefault("symbols_with_data", 0)
        return s

    return {
        "strategy": "ICT quick scalp 5m execution / 15m bias",
        "started_at": now_iso(),
        "last_run_at": None,
        "equity": initial,
        "cash": initial,
        "open_positions": [],
        "trades": [],
        "processed": {},
        "equity_curve": [],
        "max_open_positions": max_open,
        "capital_deployment": 0.90,
        "last_signals": [],
        "data_status": "new",
        "last_error": None,
        "last_prices": {},
        "scan_errors": {},
        "symbols_scanned": 0,
        "symbols_with_data": 0,
    }


def save_state(s):
    os.makedirs("reports", exist_ok=True)
    # Atomic replacement prevents a cancelled runner from leaving a truncated
    # JSON state file that breaks the next scheduled cycle.
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, default=str)
    os.replace(tmp, STATE_PATH)
    pd.DataFrame(s["trades"]).to_csv(TRADES_PATH, index=False)
    pd.DataFrame(s["equity_curve"]).to_csv(EQUITY_PATH, index=False)


def close_position(s, p, exit_price, reason, fee_bps, slip_bps, stamp):
    fee, slip = fee_bps / 10000, slip_bps / 10000
    entry_fill = p["entry_price"] * (1 + slip)
    exit_fill = exit_price * (1 - slip)
    qty = p["qty"]
    gross = (exit_fill - entry_fill) * qty
    fees = (entry_fill * qty + exit_fill * qty) * fee
    net = gross - fees
    s["cash"] += qty * exit_fill - exit_fill * qty * fee
    s["trades"].append({
        "symbol": p["symbol"],
        "entry_time": p["entry_time"],
        "exit_time": stamp,
        "entry_price": entry_fill,
        "exit_price": exit_fill,
        "quantity": qty,
        "gross_pnl": gross,
        "fees": fees,
        "net_pnl": net,
        "exit_reason": reason,
        "stop_price": p["stop_price"],
        "target_price": p["target_price"],
        "zone_lower": p["zone_lower"],
        "zone_upper": p["zone_upper"],
        "sweep_level": p.get("sweep_level"),
        "slippage_cost_estimate": (p["entry_price"] + exit_price) * qty * slip,
    })


def mark_error_and_save(state, stamp, exc):
    state["last_run_at"] = stamp
    state["data_status"] = "error"
    state["last_error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "timestamp": stamp,
    }
    state["last_signals"] = []
    save_state(state)
    print(
        f"SCALP DATA ERROR {type(exc).__name__}: {exc}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-coins", action="store_true")
    ap.add_argument("--quote", default="USDT")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--initial-equity", type=float, default=1000.0)
    ap.add_argument("--risk-fraction", type=float, default=0.005)
    ap.add_argument("--reward-risk", type=float, default=1.5)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--stop-atr-buffer", type=float, default=0.10)
    ap.add_argument("--min-gap-atr", type=float, default=0.10)
    ap.add_argument("--min-impulse-ratio", type=float, default=0.60)
    ap.add_argument("--min-volume-ratio", type=float, default=1.10)
    ap.add_argument("--max-open-positions", type=int, default=9)
    ap.add_argument("--capital-deployment", type=float, default=0.90)
    args = ap.parse_args()

    stamp_now = now_iso()
    state = load_state(args.initial_equity, args.max_open_positions)
    state["max_open_positions"] = args.max_open_positions
    state["capital_deployment"] = args.capital_deployment

    try:
        exchange = create_exchange()
        exchange.load_markets()
    except Exception as exc:
        # Keep the previous portfolio intact and publish a diagnostic state
        # instead of crashing before reports exist.
        mark_error_and_save(state, stamp_now, exc)
        return

    eligible_symbols = (
        [s for s, market in exchange.markets.items() if is_crypto_spot_market(market, args.quote)]
        if args.all_coins
        else ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    open_symbols = {p["symbol"] for p in state["open_positions"]}
    symbols = sorted(set(eligible_symbols) | open_symbols)
    allowed_new_entries = set(eligible_symbols)

    latest = dict(state.get("last_prices", {}))
    signals = []
    successful_symbols = 0
    scan_errors = {}

    for symbol in symbols:
        try:
            raw = fetch_ohlcv(exchange, symbol, "5m", args.days)
            if len(raw) < 100:
                continue

            successful_symbols += 1
            d = indicators_5m(raw)
            row = d.iloc[-1]
            stamp = pd.Timestamp(
                int(row.timestamp), unit="ms", tz="UTC"
            ).isoformat()
            latest[symbol] = float(row.close)
            state["last_prices"][symbol] = float(row.close)

            for p in list(state["open_positions"]):
                if p["symbol"] != symbol:
                    continue
                stop = float(row.low) <= p["stop_price"]
                target = float(row.high) >= p["target_price"]
                if stop or target:
                    reason = "stop_loss" if stop else "take_profit"
                    exit_price = (
                        p["stop_price"] if stop else p["target_price"]
                    )
                    close_position(
                        state, p, exit_price, reason,
                        args.fee_bps, args.slippage_bps, stamp
                    )
                    state["open_positions"].remove(p)
                    print(f"CLOSED {symbol} {reason} exit={exit_price}")

            ts = int(row.timestamp)
            if state["processed"].get(symbol) == ts:
                continue

            z = latest_entry_fvg(
                d, symbol, args.min_gap_atr,
                args.min_impulse_ratio, args.min_volume_ratio, len(d) - 1
            )

            if (
                z
                and symbol in allowed_new_entries
                and len(state["open_positions"]) < args.max_open_positions
                and row.htf_bias != "bearish"
                and row.low <= z["upper"]
                and row.close > z["ce"]
                and row.close > row.open
            ):
                entry = float(row.close)
                stop = (
                    min(float(z["sweep_level"]), z["lower"])
                    - args.stop_atr_buffer * float(row.atr)
                )
                risk = entry - stop
                if risk > 0 and state["cash"] > 0:
                    risk_cash = state["equity"] * args.risk_fraction
                    deployed = sum(
                        p["qty"] * p["entry_price"]
                        for p in state["open_positions"]
                    )
                    room = max(
                        0,
                        state["equity"] * args.capital_deployment - deployed,
                    )
                    slot = (
                        state["equity"] * args.capital_deployment
                        / max(args.max_open_positions, 1)
                    )
                    slip = args.slippage_bps / 10000
                    fee = args.fee_bps / 10000
                    qty = min(
                        risk_cash / risk,
                        min(slot, room)
                        / (entry * (1 + slip) * (1 + fee)),
                        state["cash"]
                        / (entry * (1 + slip) * (1 + fee)),
                    )
                    if qty > 0:
                        target = entry + args.reward_risk * risk
                        entry_cost = (
                            qty * entry * (1 + slip) * (1 + fee)
                        )
                        state["cash"] -= entry_cost
                        p = {
                            "symbol": symbol,
                            "entry_time": stamp,
                            "entry_price": entry,
                            "qty": qty,
                            "stop_price": stop,
                            "target_price": target,
                            "zone_lower": z["lower"],
                            "zone_upper": z["upper"],
                            "sweep_level": z["sweep_level"],
                        }
                        state["open_positions"].append(p)
                        signals.append({
                            "symbol": symbol,
                            "type": "SCALP_LONG_OPENED",
                            "time": stamp,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "quantity": qty,
                            "fvg": [z["lower"], z["upper"]],
                            "created_at": z["created_at"],
                        })
                        print(
                            f"OPENED {symbol} entry={entry} "
                            f"stop={stop} target={target}"
                        )

            state["processed"][symbol] = ts

        except Exception as exc:
            scan_errors[symbol] = f"{type(exc).__name__}: {exc}"
            print(
                f"ERROR {symbol}: {type(exc).__name__}: {exc}",
                flush=True,
            )

    state["scan_errors"] = scan_errors
    state["symbols_scanned"] = len(symbols)
    state["symbols_with_data"] = successful_symbols
    state["eligible_new_entry_symbols"] = len(eligible_symbols)
    state["excluded_market_count"] = max(0, len(exchange.markets) - len(eligible_symbols))

    value = state["cash"] + sum(
        p["qty"] * latest.get(p["symbol"], p["entry_price"])
        for p in state["open_positions"]
    )
    state["equity"] = value
    state["last_run_at"] = stamp_now
    state["last_signals"] = signals
    state["data_status"] = (
        "ok" if successful_symbols and not scan_errors else
        "partial" if successful_symbols else "no_data"
    )
    state["last_error"] = None if successful_symbols else {
        "type": "NoMarketData",
        "message": "No symbol returned usable OHLCV data.",
        "timestamp": stamp_now,
    }
    state["equity_curve"].append({
        "timestamp": stamp_now,
        "equity": state["equity"],
        "cash": state["cash"],
        "open_positions": len(state["open_positions"]),
        "symbols_scanned": len(symbols),
        "symbols_with_data": successful_symbols,
        "symbols_with_errors": len(scan_errors),
        "eligible_new_entry_symbols": len(eligible_symbols),
    })
    save_state(state)

    print(
        f"SCALP RUN {stamp_now} symbols={len(symbols)} "
        f"symbols_with_data={successful_symbols} "
        f"open_positions={len(state['open_positions'])} "
        f"trades={len(state['trades'])} "
        f"equity={state['equity']:.4f} cash={state['cash']:.4f}"
    )


if __name__ == "__main__":
    main()
