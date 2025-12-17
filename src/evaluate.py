"""Baseline models and evaluation metrics."""

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Allow running from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import load_consumption_data, preprocess_household_hourly  # noqa: E402
from src.utils import temporal_split  # noqa: E402

TARGET_COL = "load"

__all__ = [
    "mae",
    "rmse",
    "mape",
    "evaluate_forecasts",
    "sarima_rolling_forecast",
    "baseline_persistence",
    "baseline_previous_day",
    "baseline_moving_average",
    "baseline_sarima",
    "run_all_baselines",
]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Metrics computed on the original (inverse-transformed) scale
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def _prepare_series(
    data_path: Path,
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = load_consumption_data(str(data_path))
    hourly = preprocess_household_hourly(df)
    train_df, val_df, test_df = temporal_split(hourly, train_end=train_end, val_end=val_end)
    return train_df[TARGET_COL], val_df[TARGET_COL], test_df[TARGET_COL]


def _build_samples(series: pd.Series, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create aligned (y_true, last_value) arrays for naive baselines."""
    values = series.values
    n = len(values)
    samples = n - horizon
    if samples <= 0:
        raise ValueError("Series too short for requested horizon.")
    y_true = np.stack([values[i + 1 : i + 1 + horizon] for i in range(samples)], axis=0)
    last_values = values[:samples]
    return y_true, last_values


def baseline_persistence(series: pd.Series, horizon: int = 24) -> Dict[str, float]:
    y_true, last_values = _build_samples(series, horizon)
    y_pred = np.repeat(last_values[:, None], horizon, axis=1)
    return evaluate_forecasts(y_true, y_pred)


def baseline_previous_day(series: pd.Series, horizon: int = 24) -> Dict[str, float]:
    values = series.values
    n = len(values)
    start = 24  # need data from t-24
    samples = n - start - horizon
    if samples <= 0:
        raise ValueError("Series too short for previous-day baseline.")
    y_true = np.stack([values[start + i + 1 : start + i + 1 + horizon] for i in range(samples)], axis=0)
    ref = np.stack([values[start + i + 1 - 24 : start + i + 1 - 24 + horizon] for i in range(samples)], axis=0)
    return evaluate_forecasts(y_true, ref)


def baseline_moving_average(series: pd.Series, horizon: int = 24, window: int = 24) -> Dict[str, float]:
    values = series.values
    n = len(values)
    start = window
    samples = n - start - horizon
    if samples <= 0:
        raise ValueError("Series too short for moving average baseline.")
    y_true = np.stack([values[start + i + 1 : start + i + 1 + horizon] for i in range(samples)], axis=0)
    means = []
    for i in range(samples):
        window_vals = values[start + i - window + 1 : start + i + 1]
        means.append(np.mean(window_vals))
    means = np.asarray(means)
    y_pred = np.repeat(means[:, None], horizon, axis=1)
    return evaluate_forecasts(y_true, y_pred)


def baseline_sarima(
    train_series: pd.Series,
    test_series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24),
    horizon: int = 24,
    max_points: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Fit SARIMA on train (optionally truncated) and forecast rolling on test horizon-by-horizon.
    Warning: can be slow; use max_points to limit training sample.
    """
    y_true, y_pred = sarima_rolling_forecast(
        train_series=train_series,
        test_series=test_series,
        order=order,
        seasonal_order=seasonal_order,
        horizon=horizon,
        warm_start=0,
        max_points=max_points,
        max_steps=max_steps,
    )
    return evaluate_forecasts(y_true, y_pred)


def sarima_rolling_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    *,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24),
    horizon: int = 24,
    warm_start: int = 0,
    stride: int = 1,
    max_points: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling multi-step SARIMA forecast.

    Fit once on train, optionally append `warm_start` first test observations (to align with windowed models),
    then iterate through the test set and forecast `horizon` steps ahead at each origin.

    Notes:
      - This returns the aligned arrays (y_true, y_pred) with shape (n_samples, horizon).
      - `max_points` and `max_steps` are optional runtime controls; set to None for full evaluation.
    """
    train_vals = train_series.values
    if max_points is not None:
        train_vals = train_vals[-max_points:]

    model = SARIMAX(
        train_vals,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    if stride <= 0:
        raise ValueError("stride must be >= 1")

    test_vals = test_series.values
    if warm_start < 0:
        raise ValueError("warm_start must be >= 0")
    if warm_start > 0:
        if len(test_vals) < warm_start:
            raise ValueError("Test series too short for warm_start.")
        # Update the filtered state with the first warm_start observations (no refit)
        res = res.append(test_vals[:warm_start], refit=False)

    samples = len(test_vals) - warm_start - horizon + 1
    if samples <= 0:
        raise ValueError("Test series too short for requested horizon/warm_start.")

    # When stride > 1, we evaluate only one forecast every `stride` points (faster),
    # but still forecast `horizon` steps ahead.
    steps = (samples + stride - 1) // stride
    if max_steps is not None:
        steps = min(steps, max_steps)
    y_true = np.empty((steps, horizon), dtype=float)
    y_pred = np.empty((steps, horizon), dtype=float)

    for step_idx in range(steps):
        i = step_idx * stride
        fc = res.forecast(steps=horizon)
        y_pred[step_idx, :] = np.asarray(fc)
        start = warm_start + i
        y_true[step_idx, :] = test_vals[start : start + horizon]
        # Reveal next `stride` observations and advance state (except after last iteration)
        if step_idx < steps - 1:
            chunk = test_vals[start : start + stride]
            res = res.append(chunk, refit=False)

    return y_true, y_pred


def run_all_baselines(
    data_path: Path,
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
    horizon: int = 24,
    moving_window: int = 24,
    sarima: bool = False,
) -> Dict[str, Dict[str, float]]:
    train_series, val_series, test_series = _prepare_series(data_path, train_end=train_end, val_end=val_end)

    results = {
        "persistence": baseline_persistence(test_series, horizon=horizon),
        "previous_day": baseline_previous_day(test_series, horizon=horizon),
        "moving_average": baseline_moving_average(test_series, horizon=horizon, window=moving_window),
    }
    if sarima:
        results["sarima"] = baseline_sarima(train_series, test_series, horizon=horizon, max_points=5000)
    return results


def main() -> None:
    data_path = REPO_ROOT / "data" / "consumption_data.csv"
    results = run_all_baselines(
        data_path=data_path,
        train_end=None,
        val_end=None,
        horizon=24,
        moving_window=24,
        sarima=False,  # set True if you want SARIMA (can be slow)
    )
    for name, metrics in results.items():
        print(f"== {name} ==")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")
        print()


if __name__ == "__main__":
    main()
