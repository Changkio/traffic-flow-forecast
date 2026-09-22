"""生成带日/周周期的模拟交通流量数据。

仅用于演示，不代表任何实际道路或观测设备。
"""
import numpy as np
import pandas as pd


def generate_traffic(interval_min: int = 15, days: int = 90, seed: int = 42) -> pd.DataFrame:
    if not isinstance(interval_min, int) or interval_min <= 0 or 1440 % interval_min:
        raise ValueError("interval_min must be a positive integer divisor of 1440.")
    if not isinstance(days, int) or days <= 0:
        raise ValueError("days must be a positive integer.")
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * 60 / interval_min)
    t = pd.date_range("2025-01-01", periods=steps, freq=f"{interval_min}min")
    hour = (t.hour + t.minute / 60.0).to_numpy()

    # 日周期：早高峰 ~8 点、晚高峰 ~18 点
    morning = 30.0 * np.exp(-((hour - 8.0) / 1.8) ** 2)
    evening = 35.0 * np.exp(-((hour - 18.0) / 2.2) ** 2)
    level = 20.0 + morning + evening

    # 周周期：周末流量明显下降
    weekend = t.weekday.to_numpy() >= 5
    level[weekend] *= 0.55

    # 缓慢上升趋势 + 随机噪声
    trend = np.linspace(0.0, 6.0, steps)
    noise = rng.normal(0.0, 2.5, steps)
    traffic = np.clip(level + trend + noise, 0.0, None)

    return pd.DataFrame({"timestamp": t, "traffic": traffic})


def steps_per_day(df: pd.DataFrame) -> int:
    """要求时间戳严格递增、等间隔，且采样间隔整除一天。"""
    timestamps = pd.to_datetime(df["timestamp"], errors="raise")
    deltas = timestamps.diff().iloc[1:]
    if len(deltas) == 0 or timestamps.isna().any():
        raise ValueError("At least two valid timestamps are required.")
    delta = deltas.iloc[0]
    if delta <= pd.Timedelta(0) or not (deltas == delta).all():
        raise ValueError("Timestamps must be sorted, unique and regularly sampled.")
    day = pd.Timedelta(days=1)
    if day % delta != pd.Timedelta(0):
        raise ValueError("Sampling interval must divide one day exactly.")
    return int(day / delta)
