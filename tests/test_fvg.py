import unittest
import pandas as pd

from fvg_bot import detect_fvg, summarize


def candles(rows):
    start = 1_700_000_000_000
    return pd.DataFrame(
        [[start + i * 900_000, *row] for i, row in enumerate(rows)],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


class FVGTests(unittest.TestCase):
    def test_detects_three_candle_bullish_gap(self):
        rows = []
        for _ in range(55):
            rows.append([100, 101, 99, 100, 1000])
        # A: high 101; B: strong bullish impulse; C: low 103 -> gap [101, 103].
        rows[-3] = [100, 101, 99, 100, 1000]
        rows[-2] = [100, 104, 100, 104, 3000]
        rows[-1] = [104, 105, 103, 104.5, 1500]
        df = candles(rows)
        zones = detect_fvg(df, "TEST/USDT", trend_filter=False,
                           volume_filter=False, min_impulse_ratio=0.4,
                           min_gap_atr=0.01)
        self.assertTrue(any(z["direction"] == "bullish" and
                            z["lower"] == 101 and z["upper"] == 103
                            for z in zones))

    def test_flat_data_has_no_gap(self):
        df = candles([[100, 101, 99, 100, 1000] for _ in range(80)])
        self.assertEqual(detect_fvg(df, "TEST/USDT"), [])

    def test_empty_summary(self):
        s = summarize([], 1000, 1000, 0)
        self.assertEqual(s["trades"], 0)
        self.assertEqual(s["net_pnl"], 0)
        self.assertEqual(s["win_rate_pct"], 0)


if __name__ == "__main__":
    unittest.main()
