"""构造监督学习特征：时间属性 + 滞后特征 + 滚动统计。"""
import pandas as pd


def build_supervised(df: pd.DataFrame, spd: int) -> pd.DataFrame:
    d = df.copy()
    ts = d["timestamp"]
    d["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    d["dow"] = ts.dt.weekday
    d["is_weekend"] = d["dow"].isin([5, 6]).astype(int)

    # 滞后：1 天前、7 天前同一时刻
    d["lag_1d"] = d["traffic"].shift(spd)
    d["lag_7d"] = d["traffic"].shift(spd * 7)
    # 一天滞后后的日窗口：t-2*spd+1 ... t-spd（训练与递归预测定义相同）
    d["roll_mean"] = d["traffic"].shift(spd).rolling(spd).mean()

    return d.dropna().reset_index(drop=True)


FEATURE_COLS = ["hour", "dow", "is_weekend", "lag_1d", "lag_7d", "roll_mean"]
