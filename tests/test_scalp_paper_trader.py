import json

import numpy as np
import pandas as pd

import scalp_paper_trader as scalp


def synthetic_5m():
    rows = []
    start = pd.Timestamp("2026-10-07T12:00:00Z")
    for i in range(60):
        rows.append([
            int((start + pd.Timedelta(minutes=5 * i)).timestamp() * 1000),
            100.0, 101.0, 99.0, 100.0, 100.0,
        ])

    # A/B/C bullish FVG with a sell-side sweep on B.
    rows[57] = [rows[57][0], 100.0, 100.0, 99.5, 99.8, 100.0]
    rows[58] = [rows[58][0], 99.5, 101.0, 95.0, 100.5, 100.0]
    rows[59] = [rows[59][0], 101.5, 102.0, 101.5, 101.8, 100.0]
    return pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )


def test_binance_public_market_data_endpoint(monkeypatch):
    monkeypatch.delenv("BINANCE_PUBLIC_API_BASE", raising=False)
    exchange = scalp.create_exchange()
    assert exchange.urls["api"]["public"] == scalp.DEFAULT_PUBLIC_API


def test_three_candle_fvg_and_latest_confirmation():
    d = scalp.indicators_5m(synthetic_5m())
    z = scalp.detect_scalp_fvg(
        d, len(d) - 1, "TEST/USDT",
        min_gap_atr=0.1, min_impulse=0.1, min_volume=0.5,
    )
    assert z is not None
    assert z["lower"] == 100.0
    assert z["upper"] == 101.5

    latest = scalp.latest_entry_fvg(
        d, "TEST/USDT", 0.1, 0.1, 0.5, len(d) - 1
    )
    assert latest is not None
    assert latest["confirmed_at"]


def test_state_error_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(scalp, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(scalp, "TRADES_PATH", str(tmp_path / "trades.csv"))
    monkeypatch.setattr(scalp, "EQUITY_PATH", str(tmp_path / "equity.csv"))

    state = scalp.load_state(1000.0, 9)
    scalp.mark_error_and_save(
        state, "2026-10-07T12:00:00+00:00",
        RuntimeError("market data unavailable"),
    )

    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["data_status"] == "error"
    assert saved["last_error"]["message"] == "market data unavailable"
