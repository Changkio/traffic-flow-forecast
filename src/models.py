"""固定预测起点的多步预测：各方法只能使用起点之前的流量观测。"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from .features import FEATURE_COLS


def fit_gbr(X, y):
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, random_state=42,
    )
    model.fit(X, y)
    return model


def fit_arima(series, order=(1, 1, 1)):
    return ARIMA(np.asarray(series, dtype=float), order=order).fit()


def seasonal_forecast(train_series, horizon, period):
    """重复训练区间末尾一个周期，不读取测试区间真实值。"""
    values = np.asarray(train_series, dtype=float)
    if period < 1 or len(values) < period or horizon < 1:
        raise ValueError("Need a positive horizon and a full seasonal history.")
    return np.resize(values[-period:], horizon)


def recursive_gbr_forecast(model, train_series, future_timestamps, spd):
    """已知未来日历时间；未知流量滞后项由先前预测递归补齐。"""
    history = list(np.asarray(train_series, dtype=float))
    if spd < 1 or len(history) < 7 * spd:
        raise ValueError("At least seven days of training history are required.")
    predictions = []
    for timestamp in pd.to_datetime(future_timestamps):
        # 与 features.py 的 shift(spd).rolling(spd) 保持相同时间窗口。
        rolling_window = history[len(history)-2*spd+1:len(history)-spd+1]
        row = [timestamp.hour + timestamp.minute / 60.0, timestamp.weekday(),
               int(timestamp.weekday() >= 5), history[-spd], history[-7*spd],
               np.mean(rolling_window)]
        prediction = float(model.predict(pd.DataFrame([row], columns=FEATURE_COLS))[0])
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions)
