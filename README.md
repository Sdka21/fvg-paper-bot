# ICT Long-Only Spot Paper Bot

A standalone, paper-only ICT-inspired trading system for Binance Spot.

It does not submit live orders, use private keys, or require exchange credentials.

## Strategy

1. Higher-timeframe bias — 1h bullish/neutral structure using EMA20/EMA50.
2. Sell-side liquidity — recent swing-low liquidity is identified.
3. Liquidity sweep — price trades below that liquidity and closes back above it.
4. Bullish displacement — an impulsive bullish candle is required.
5. Bullish FVG — exact three-candle imbalance: low[C] > high[A].
6. FVG quality — minimum gap/ATR, displacement-body and volume filters.
7. FVG lifecycle — active -> mitigated -> confirmed, or invalidated.
8. Newest valid FVG priority — searches newest-to-oldest and does not reuse a consumed zone.
9. Entry — default confirmation: price retraces into the FVG and closes back above its CE with a bullish body. Touch mode is available for research.
10. Risk — stop below swept liquidity/FVG plus an ATR buffer; position size uses a fixed equity risk fraction.
11. Target — configurable R multiple.
12. Spot only — no shorts, leverage, futures or margin.

This is an ICT-inspired systematic model, not a claim that ICT concepts are profitable.

## Commands

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scan all Binance USDT spot markets:
```bash
python fvg_bot.py scan --all-coins --quote USDT --days 7
```

One-shot paper signal cycle:
```bash
python fvg_bot.py paper --all-coins --quote USDT --days 7
```

Backtest all Binance USDT spot markets:
```bash
python fvg_bot.py backtest --all-coins --quote USDT --days 180
```

Research touch entries instead of confirmation:
```bash
python fvg_bot.py backtest --all-coins --entry-mode touch
```

## Default risk model

- 15-minute execution timeframe
- 1-hour directional filter
- 0.5% equity risk per trade
- Maximum position notional: 95% of available equity
- 2R default target
- 10 bps fee per side
- 5 bps slippage per side
- 0.15 ATR stop buffer
- Minimum FVG size: 0.12 ATR
- Minimum displacement body/range: 0.55
- Minimum volume ratio: 1.0
- Sell-side sweep required by default

These are research defaults and should be tested rather than treated as optimized parameters.

## Reports

Backtests write reports/trades.csv and reports/summary.json.
Scan/paper runs write reports/latest_signals.json.
The paper scanner intentionally produces signals only. It does not place exchange orders.

## Important backtest assumptions

OHLCV candles cannot reveal exact intrabar order. When a candle touches both stop and target, the backtest treats the stop as occurring first. Fees and adverse slippage are included. Spread, market impact, taxes and portfolio correlation are not fully modeled.

With --all-coins, each symbol is currently evaluated as an independent account. Summing those independent P&Ls is not a portfolio return.

Validate the strategy out-of-sample and with paper trading before considering any real-money use.