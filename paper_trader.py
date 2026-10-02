#!/usr/bin/env python3
"""Persistent ICT-inspired long-only spot paper trader.

No live orders. State is stored in reports/paper_state.json so scheduled
GitHub Actions runs continue the same simulated portfolio.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fvg_bot import (
    CANDLE_MS, create_exchange, fetch_ohlcv, get_all_spot_symbols,
    indicators, detect_one, zone_lifecycle
)

STATE_PATH = "reports/paper_state.json"
TRADES_PATH = "reports/paper_trades.csv"
EQUITY_PATH = "reports/paper_equity.csv"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_state(initial_equity: float, max_open_positions: int):
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("equity", initial_equity)
        state.setdefault("cash", initial_equity)
        state.setdefault("open_positions", [])
        state.setdefault("trades", [])
        state.setdefault("processed", {})
        state.setdefault("zones", {})
        state.setdefault("equity_curve", [])
        state.setdefault("started_at", now_iso())
        state.setdefault("max_open_positions", max_open_positions)
        return state
    return {
        "strategy": "ICT-inspired long-only spot paper trader",
        "started_at": now_iso(),
        "last_run_at": None,
        "equity": initial_equity,
        "cash": initial_equity,
        "open_positions": [],
        "trades": [],
        "processed": {},
        "zones": {},
        "equity_curve": [],
        "max_open_positions": max_open_positions,
    }


def save_state(state):
    os.makedirs("reports", exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    pd.DataFrame(state["trades"]).to_csv(TRADES_PATH, index=False)
    pd.DataFrame(state["equity_curve"]).to_csv(EQUITY_PATH, index=False)


def mark_to_market(state, latest_prices):
    value = state["cash"]
    for p in state["open_positions"]:
        price = latest_prices.get(p["symbol"], p["entry_price"])
        value += p["qty"] * price
    state["equity"] = value


def close_position(state, p, exit_price, reason, fee_bps, slippage_bps, stamp):
    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0
    entry_fill = p["entry_price"] * (1 + slip)
    exit_fill = exit_price * (1 - slip)
    qty = p["qty"]
    gross = (exit_fill - entry_fill) * qty
    fees = (entry_fill * qty + exit_fill * qty) * fee
    net = gross - fees
    state["cash"] += qty * exit_fill - exit_fill * qty * fee
    state["trades"].append({
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
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all-coins", action="store_true")
    p.add_argument("--quote", default="USDT")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--initial-equity", type=float, default=1000.0)
    p.add_argument("--risk-fraction", type=float, default=0.005)
    p.add_argument("--reward-risk", type=float, default=2.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--stop-atr-buffer", type=float, default=0.15)
    p.add_argument("--min-gap-atr", type=float, default=0.12)
    p.add_argument("--min-impulse-ratio", type=float, default=0.55)
    p.add_argument("--min-volume-ratio", type=float, default=1.0)
    p.add_argument("--max-open-positions", type=int, default=9)
    p.add_argument("--capital-deployment", type=float, default=0.90,
                   help="Maximum fraction of current equity deployed across open positions.")
    args = p.parse_args()

    exchange = create_exchange()
    symbols = get_all_spot_symbols(exchange, args.quote) if args.all_coins else [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BONK/USDT"
    ]
    state = load_state(args.initial_equity, args.max_open_positions)
    state["max_open_positions"] = args.max_open_positions
    state["capital_deployment"] = args.capital_deployment
    signals = []
    latest_prices = {}
    stamp_now = now_iso()

    for symbol in symbols:
        try:
            df = fetch_ohlcv(exchange, symbol, max(args.days, 3))
            if len(df) < 80:
                continue
            d = indicators(df)
            row = d.iloc[-1]
            stamp = pd.Timestamp(int(row["timestamp"]), unit="ms", tz="UTC").isoformat()
            latest_prices[symbol] = float(row["close"])

            # Manage any existing position first, using the newly closed candle.
            for pos in list(state["open_positions"]):
                if pos["symbol"] != symbol:
                    continue
                stop_hit = float(row["low"]) <= pos["stop_price"]
                target_hit = float(row["high"]) >= pos["target_price"]
                if stop_hit or target_hit:
                    reason = "stop_loss" if stop_hit else "take_profit"
                    exit_price = pos["stop_price"] if stop_hit else pos["target_price"]
                    close_position(state, pos, exit_price, reason, args.fee_bps, args.slippage_bps, stamp)
                    state["open_positions"].remove(pos)
                    print(f"CLOSED {symbol} {reason} exit={exit_price}")

            # Do not re-process the same closed candle.
            if state["processed"].get(symbol) == int(row["timestamp"]):
                continue

            zones = []
            for i in range(55, len(d)):
                zones.extend(detect_one(
                    d, i, symbol, args.min_gap_atr,
                    args.min_impulse_ratio, args.min_volume_ratio, True
                ))
            # Rebuild lifecycle using the complete recent window, but only
            # permit a new entry on the latest closed candle.
            for z in zones:
                for i in range(z["created_index"] + 1, len(d)):
                    lifecycle_row = d.iloc[i]
                    lifecycle_stamp = pd.Timestamp(
                        int(lifecycle_row["timestamp"]), unit="ms", tz="UTC"
                    ).isoformat()
                    zone_lifecycle(z, lifecycle_row, lifecycle_stamp)

            candidates = []
            for z in reversed(zones):
                if z["direction"] != "bullish" or z.get("signal_used"):
                    continue
                if z.get("status") == "confirmed" and z.get("confirmed_at") == stamp:
                    candidates.append(z)

            if candidates and len(state["open_positions"]) < state["max_open_positions"]:
                z = candidates[0]
                if row["htf_bias"] != "bearish" and np.isfinite(row["atr"]) and row["atr"] > 0:
                    entry_price = float(row["close"])
                    sweep = z.get("sweep_level")
                    stop_base = sweep if sweep is not None else z["lower"]
                    stop = min(float(stop_base), z["lower"]) - args.stop_atr_buffer * float(row["atr"])
                    risk_unit = entry_price - stop
                    if risk_unit > 0 and state["cash"] > 0:
                        # Compound from current marked equity. Each slot targets ~10% of
                        # current equity, with the portfolio capped at 90% deployed.
                        risk_cash = state["equity"] * args.risk_fraction
                        deployed = sum(p["qty"] * p["entry_price"] for p in state["open_positions"])
                        deployment_room = max(0.0, state["equity"] * args.capital_deployment - deployed)
                        slot_allocation = state["equity"] * args.capital_deployment / max(state["max_open_positions"], 1)
                        allocation = min(slot_allocation, deployment_room)
                        qty = min(risk_cash / risk_unit, allocation / (entry_price * (1 + args.slippage_bps / 10000.0)),
                                  state["cash"] / (entry_price * (1 + args.slippage_bps / 10000.0)))
                        if qty > 0:
                            target = entry_price + args.reward_risk * risk_unit
                            cost = qty * entry_price * (1 + args.slippage_bps / 10000.0)
                            state["cash"] -= cost
                            state["open_positions"].append({
                                "symbol": symbol,
                                "entry_time": stamp,
                                "entry_price": entry_price,
                                "qty": qty,
                                "stop_price": stop,
                                "target_price": target,
                                "zone_lower": z["lower"],
                                "zone_upper": z["upper"],
                                "sweep_level": sweep,
                            })
                            z["signal_used"] = True
                            z["status"] = "used"
                            signal = {
                                "symbol": symbol,
                                "type": "LONG_OPENED",
                                "time": stamp,
                                "entry": entry_price,
                                "stop": stop,
                                "target": target,
                                "quantity": qty,
                                "risk_cash": risk_cash,
                                "fvg": [z["lower"], z["upper"]],
                                "sweep_level": sweep,
                            }
                            signals.append(signal)
                            print(json.dumps(signal))

            state["processed"][symbol] = int(row["timestamp"])

        except Exception as exc:
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}")

    mark_to_market(state, latest_prices)
    state["last_run_at"] = stamp_now
    state["equity_curve"].append({
        "timestamp": stamp_now,
        "equity": state["equity"],
        "cash": state["cash"],
        "open_positions": len(state["open_positions"]),
    })
    state["last_signals"] = signals
    save_state(state)

    print(f"PAPER RUN {stamp_now}")
    print(f"symbols={len(symbols)} open_positions={len(state['open_positions'])} "
          f"trades={len(state['trades'])} equity={state['equity']:.4f} cash={state['cash']:.4f}")
    for pos in state["open_positions"]:
        print(json.dumps({"OPEN_POSITION": pos}))
    if not signals:
        print("No new paper entry this cycle.")


if __name__ == "__main__":
    main()
