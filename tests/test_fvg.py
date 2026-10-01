import unittest
import pandas as pd

from fvg_bot import detect_fvg, latest_valid_fvg, summarize, zone_lifecycle


def candles(rows):
    start = 1_700_000_000_000
    return pd.DataFrame(
        [[start + i * 900_000, *row] for i, row in enumerate(rows)],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


class ICTTests(unittest.TestCase):
    def test_detects_three_candle_bullish_gap_without_sweep_when_disabled(self):
        rows = [[100, 101, 99, 100, 1000] for _ in range(60)]
        rows[-3] = [100, 101, 99, 100, 1000]
        rows[-2] = [100, 104, 100, 104, 3000]
        rows[-1] = [104, 105, 103, 104.5, 1500]
        zones = detect_fvg(
            candles(rows), "TEST/USDT",
            trend_filter=False, volume_filter=False,
            require_sweep=False, min_impulse_ratio=0.4, min_gap_atr=0.01
        )
        self.assertTrue(any(z["direction"] == "bullish" for z in zones))

    def test_flat_data_has_no_gap(self):
        self.assertEqual(
            detect_fvg(candles([[100, 101, 99, 100, 1000] for _ in range(80)]),
                       "TEST/USDT", require_sweep=False), []
        )

    def test_deep_mitigation_reclaim_is_confirmed(self):
        zone = {
            "direction": "bullish", "lower": 101.0, "upper": 103.0, "ce": 102.0,
            "status": "active", "touched_at": None, "filled_at": None,
            "invalidated_at": None, "confirmed_at": None, "signal_used": False
        }
        row = pd.Series({"open": 103.0, "high": 105.0, "low": 100.5, "close": 104.0})
        touched, confirmed = zone_lifecycle(zone, row, "2026-10-01T12:00:00+00:00")
        self.assertTrue(touched)
        self.assertTrue(confirmed)
        self.assertEqual(zone["status"], "confirmed")

    def test_close_below_fvg_invalidates_long_zone(self):
        zone = {
            "direction": "bullish", "lower": 101.0, "upper": 103.0, "ce": 102.0,
            "status": "active", "touched_at": None, "filled_at": None,
            "invalidated_at": None, "confirmed_at": None, "signal_used": False
        }
        row = pd.Series({"open": 101.5, "high": 102, "low": 99, "close": 100})
        touched, confirmed = zone_lifecycle(zone, row, "2026-10-01T12:00:00+00:00")
        self.assertFalse(touched)
        self.assertFalse(confirmed)
        self.assertEqual(zone["status"], "invalidated")

    def test_latest_valid_fvg_prefers_newest_non_terminal_zone(self):
        zones = [
            {"direction":"bullish","status":"filled","signal_used":False},
            {"direction":"bullish","status":"active","signal_used":False},
            {"direction":"bullish","status":"mitigated","signal_used":False},
            {"direction":"bullish","status":"invalidated","signal_used":False},
        ]
        self.assertIs(zones[2], latest_valid_fvg(zones))

    def test_used_zone_is_not_reused(self):
        zones = [
            {"direction":"bullish","status":"used","signal_used":True},
            {"direction":"bullish","status":"active","signal_used":False},
        ]
        self.assertIs(zones[1], latest_valid_fvg(zones))

    def test_empty_summary(self):
        s = summarize([], 1000, 1000, 0)
        self.assertEqual(s["trades"], 0)
        self.assertEqual(s["net_pnl"], 0)
        self.assertEqual(s["win_rate_pct"], 0)


if __name__ == "__main__":
    unittest.main()
