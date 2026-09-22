"""验证时间窗口、递归预测边界、季节基线及指标。"""
import unittest
import numpy as np
import pandas as pd
from src.data_gen import generate_traffic, steps_per_day
from src.features import build_supervised, FEATURE_COLS
from src.models import recursive_gbr_forecast, seasonal_forecast
from src.evaluate import regression_metrics


class Recorder:
    def __init__(self):
        self.rows = []
    def predict(self, X):
        self.rows.append(X.iloc[0].to_numpy())
        return np.array([10000.0 + len(self.rows)])


class ForecastTests(unittest.TestCase):
    def test_seed_reproducible(self):
        pd.testing.assert_frame_equal(generate_traffic(days=2), generate_traffic(days=2))

    def test_irregular_or_duplicate_time_rejected(self):
        df = generate_traffic(days=2)
        self.assertEqual(steps_per_day(df), 96)
        with self.assertRaises(ValueError):
            steps_per_day(df.drop(index=3))
        with self.assertRaises(ValueError):
            steps_per_day(pd.concat([df.iloc[:1], df]))

    def test_invalid_generator_settings(self):
        for kwargs in ({'interval_min': 0}, {'interval_min': 7}, {'days': 0}):
            with self.assertRaises(ValueError):
                generate_traffic(**kwargs)

    def test_feature_window_matches_training_formula(self):
        for spd in (1, 4, 24):
            n = spd * 10
            df = pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=n+1,
                              freq=pd.Timedelta(days=1)/spd),
                              'traffic': np.arange(n+1, dtype=float)})
            expected = build_supervised(df, spd).iloc[-1][FEATURE_COLS].to_numpy(dtype=float)
            recorder = Recorder()
            recursive_gbr_forecast(recorder, df.traffic.iloc[:-1], df.timestamp.iloc[-1:], spd)
            np.testing.assert_allclose(recorder.rows[0], expected)

    def test_multiday_forecast_uses_predictions_for_unknown_lags(self):
        spd = 4
        recorder = Recorder()
        future = pd.date_range('2025-02-01', periods=10, freq='6h')
        history = np.arange(40, dtype=float)
        predictions = recursive_gbr_forecast(recorder, history, future, spd)
        self.assertEqual(recorder.rows[4][3], predictions[0])
        self.assertEqual(recorder.rows[5][3], predictions[1])
        np.testing.assert_array_equal(history, np.arange(40, dtype=float))

    def test_seasonal_forecast_repeats_only_training_cycle(self):
        np.testing.assert_array_equal(seasonal_forecast([1,2,3,4,5], 7, 3),
                                      [3,4,5,3,4,5,3])
        with self.assertRaises(ValueError):
            seasonal_forecast([1], 3, 7)

    def test_metrics_and_zero_mape(self):
        m = regression_metrics([0, 2, 4], [1, 3, 4])
        self.assertAlmostEqual(m['MAE'], 2/3)
        self.assertAlmostEqual(m['RMSE'], np.sqrt(2/3))
        self.assertEqual(m['MAPE'], 25.0)
        self.assertTrue(np.isnan(regression_metrics([0], [1])['MAPE']))


if __name__ == '__main__':
    unittest.main()
