"""预测误差指标。"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE 在真实值为零时无定义；这些点仅从 MAPE 中排除。
    valid = np.abs(y_true) > 1e-8
    mape = (np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0
            if valid.any() else float("nan"))
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
