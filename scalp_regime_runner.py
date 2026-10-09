#!/usr/bin/env python3
"""Run the ICT scalp paper trader with a BTC market-regime guard.

The trader remains paper-only and long-only. Open positions are still managed
when BTC is bearish/neutral, but new long entries are disabled until BTC's
completed 15m regime is bullish.
"""
from __future__ import annotations

import os

import scalp_paper_trader as trader


def btc_regime() -> tuple[str, str]:
    exchange = trader.create_exchange()
    exchange.load_markets()
    raw = trader.fetch_ohlcv(exchange, "BTC/USDT", "5m", 2)
    if len(raw) < 100:
        raise RuntimeError("BTC/USDT did not return enough completed 5m candles")
    data = trader.indicators_5m(raw)
    row = data.iloc[-1]
    regime = str(row.get("htf_bias"))
    stamp = str(row["timestamp"])
    return regime, stamp


def main() -> None:
    try:
        regime, stamp = btc_regime()
    except Exception as exc:
        # Fail closed for new longs when the market-regime reference cannot be
        # established. The underlying trader still manages existing positions.
        print(f"BTC REGIME UNKNOWN; NEW LONGS DISABLED: {type(exc).__name__}: {exc}")
        trader.latest_entry_fvg = lambda *args, **kwargs: None
        trader.main()
        return

    print(f"BTC MARKET REGIME: {regime} at completed candle {stamp}")
    if regime != "bullish":
        print("BTC regime is not bullish; NEW LONGS DISABLED for this cycle.")
        trader.latest_entry_fvg = lambda *args, **kwargs: None
    else:
        print("BTC regime is bullish; qualified long setups are allowed.")

    trader.main()


if __name__ == "__main__":
    main()
