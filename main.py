"""合成交通序列：固定起点预测最后五天，不使用测试期流量构造特征。"""
from pathlib import Path
import argparse
import pandas as pd
from src.data_gen import generate_traffic, steps_per_day
from src.features import build_supervised, FEATURE_COLS
from src import models
from src.evaluate import regression_metrics
from src.visualize import plot_forecast, plot_metrics

HERE = Path(__file__).resolve().parent


def main(test_days: int = 5):
    if not 1 <= test_days <= 30:
        raise ValueError("test_days must be between 1 and 30.")
    data_dir, out_dir = HERE / "data", HERE / "outputs"
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    df = generate_traffic()
    spd = steps_per_day(df)
    n_test = test_days * spd
    history, test = df.iloc[:-n_test], df.iloc[-n_test:]
    # 先切分，再仅用训练数据构造监督特征。
    train = build_supervised(history, spd)
    gbr = models.fit_gbr(train[FEATURE_COLS], train["traffic"].values)
    arima = models.fit_arima(history["traffic"])
    forecasts = {
        "Naive(week)": models.seasonal_forecast(history["traffic"], n_test, 7 * spd),
        "GradientBoosting": models.recursive_gbr_forecast(
            gbr, history["traffic"], test["timestamp"], spd),
        "ARIMA(1,1,1)": arima.forecast(steps=n_test),
    }
    results = {name: regression_metrics(test["traffic"], prediction)
               for name, prediction in forecasts.items()}
    df.to_csv(data_dir / "traffic_sample.csv", index=False)
    table = pd.DataFrame(results).T
    table.index.name = "model"
    table.to_csv(out_dir / "metrics.csv")
    predictions = test.rename(columns={"traffic": "synthetic_actual"}).copy()
    for name, prediction in forecasts.items():
        predictions[name] = prediction
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    plot_forecast(test["timestamp"], test["traffic"], forecasts, out_dir / "forecast.png")
    plot_metrics(results, out_dir / "model_compare.png")
    print("Synthetic data | fixed-origin forecast | no observed test values used")
    print(table.round(3).to_string())
    print(f"Outputs: {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, default=5)
    main(parser.parse_args().test_days)
