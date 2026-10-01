# FVG Paper Bot

A standalone, paper-only Fair Value Gap detector and 15-minute crypto spot backtester.

## Safety
- No live orders, private API keys, wallet signing, or trading credentials are used.
- Uses public OHLCV candles from Binance Spot through CCXT.
- Backtest results are historical measurements, not a promise or forecast of profitability.

## FVG definition
For three closed candles A, B, C: bullish FVG if `low[C] > high[A]`; bearish FVG if `high[C] < low[A]`. The middle candle must meet impulse-body, volume, and trend filters. Zones are tracked as active, partially mitigated, filled, or invalidated.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Backtest
```bash
python fvg_bot.py backtest --symbols BTC/USDT ETH/USDT SOL/USDT --days 180
```
Defaults: 15m candles, 10 bps taker fee per side, 5 bps slippage per side, $1,000 starting equity per symbol, 0.5% risk per trade, 2R target. Reports are written to `reports/`. Each symbol is an independent test account; do not interpret summed P&L as one portfolio return.

## Scan
```bash
python fvg_bot.py scan --symbols BTC/USDT ETH/USDT SOL/USDT BONK/USDT
```

## Filters
- 1h EMA trend filter
- Middle candle volume versus rolling median
- Middle candle body/range impulse threshold
- Minimum gap size relative to ATR
- Retest/close confirmation before simulated entry

## Limitations
Candle data cannot reveal exact intrabar sequencing. The backtest assumes stop-first if stop and target are touched in the same candle, includes fees and adverse slippage in fills, and does not model spread, market impact, taxes, or correlated portfolio risk. Validate on out-of-sample data before relying on results.
