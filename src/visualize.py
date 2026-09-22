"""结果可视化（图表标签使用英文，避免无中文字体时乱码）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_forecast(timestamps, actual, forecasts: dict, path: str):
    plt.figure(figsize=(11, 4.8))
    plt.plot(timestamps, actual, color="black", lw=2, label="Actual")
    styles = ["--", ":", "-."]
    for i, (name, pred) in enumerate(forecasts.items()):
        plt.plot(timestamps, pred, styles[i % len(styles)], lw=1.6, label=name)
    plt.title("Synthetic Traffic: Fixed-origin Multi-step Forecast")
    plt.xlabel("Time")
    plt.ylabel("Synthetic traffic (arbitrary units)")
    plt.legend(ncol=4, fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metrics(results: dict, path: str):
    names = list(results.keys())
    mae = [results[n]["MAE"] for n in names]
    rmse = [results[n]["RMSE"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    plt.figure(figsize=(8, 4.2))
    plt.bar(x - w / 2, mae, w, label="MAE", color="#4C78A8")
    plt.bar(x + w / 2, rmse, w, label="RMSE", color="#F58518")
    plt.xticks(x, names, rotation=10)
    plt.ylabel("Error (synthetic units)")
    plt.title("Model Error Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
