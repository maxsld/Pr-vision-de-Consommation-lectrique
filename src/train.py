"""Training entrypoint: scaled LSTM forecasting with inverse-transformed metrics and baseline comparison."""

from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Allow running this script from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import (  # noqa: E402
    add_lag_features,
    add_time_features,
    load_consumption_data,
    load_opsd_weather,
    merge_weather_features,
    preprocess_household_hourly,
)
from src.evaluate import (
    baseline_moving_average,
    baseline_persistence,
    baseline_previous_day,
    evaluate_forecasts,
)  # noqa: E402
from src.models import LSTMForecaster  # noqa: E402
from src.utils import temporal_split  # noqa: E402

TARGET_COL = "load"


def make_sequences(
    df,
    target_col: str,
    input_width: int,
    forecast_horizon: int = 24,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sliding windows from a time-indexed DataFrame.

    X shape: [samples, input_width, n_features]
    y shape: [samples, forecast_horizon]
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    feature_values = df[feature_cols].values
    target_values = df[target_col].values

    X_list, y_list = [], []
    total_len = len(df)
    max_start = total_len - input_width - forecast_horizon + 1
    if max_start <= 0:
        raise ValueError("Not enough data to create sequences with the requested window sizes.")

    for start in range(max_start):
        end_input = start + input_width
        end_target = end_input + forecast_horizon
        X_list.append(feature_values[start:end_input])
        y_list.append(target_values[end_input:end_target])

    return np.asarray(X_list), np.asarray(y_list)


def process_data(
    data_path: Path,
    weather_path: Optional[Path] = None,
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
):
    """Load, preprocess, add features, split. Returns DataFrames and feature list (no scaling yet)."""
    raw = load_consumption_data(str(data_path))
    hourly = preprocess_household_hourly(raw)

    # Feature engineering: calendar/cyclic + lags + rolling means
    hourly = add_time_features(hourly)
    hourly = add_lag_features(
        hourly,
        target_col=TARGET_COL,
        lags=(1, 24),
        rolling_windows=(3, 24),
        dropna=True,
    )

    # Optional weather merge
    if weather_path and weather_path.exists():
        temp = load_opsd_weather(str(weather_path))
        hourly = merge_weather_features(hourly, temp)
        # Drop weather column if no overlap to avoid empty DataFrame
        weather_col = temp.name or "temperature"
        if weather_col in hourly.columns and hourly[weather_col].notna().sum() == 0:
            hourly = hourly.drop(columns=[weather_col])

    # Drop rows with missing target/lag-derived features but keep rows even if weather is NaN
    hourly = hourly.dropna(subset=[TARGET_COL])

    train_df, val_df, test_df = temporal_split(hourly, train_end=train_end, val_end=val_end)
    feature_cols = [c for c in hourly.columns if c != TARGET_COL]
    return train_df, val_df, test_df, feature_cols


def main() -> None:
    data_path = REPO_ROOT / "data" / "consumption_data.csv"
    weather_candidates = [
        REPO_ROOT / "data" / "weather.csv",
        REPO_ROOT / "data" / "weather_opsd.csv",
        REPO_ROOT / "data" / "weather_data.csv",
    ]
    weather_path = next((p for p in weather_candidates if p.exists()), None)

    train_df, val_df, test_df, feature_cols = process_data(
        data_path=data_path,
        weather_path=weather_path,
        train_end=None,
        val_end=None,
    )

    # Choose a window size for LSTM training (reduced to 24 for faster iteration)
    window = 24
    forecast_horizon = 24

    # Fit separate scalers on train only (features vs target)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Flatten time dimension to fit feature scaler
    train_feat = train_df[feature_cols].values
    scaler_x.fit(train_feat)

    # Fit target scaler on train target only
    scaler_y.fit(train_df[[TARGET_COL]])

    # Transform splits
    def _apply_scalers(df):
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler_x.transform(df[feature_cols])
        df_scaled[TARGET_COL] = scaler_y.transform(df[[TARGET_COL]])
        return df_scaled

    train_scaled = _apply_scalers(train_df)
    val_scaled = _apply_scalers(val_df)
    test_scaled = _apply_scalers(test_df)

    # Build sequences on scaled data
    X_train, y_train = make_sequences(
        train_scaled,
        TARGET_COL,
        input_width=window,
        forecast_horizon=forecast_horizon,
        feature_cols=feature_cols,
    )
    X_val, y_val = make_sequences(
        val_scaled,
        TARGET_COL,
        input_width=window,
        forecast_horizon=forecast_horizon,
        feature_cols=feature_cols,
    )
    X_test, y_test = make_sequences(
        test_scaled,
        TARGET_COL,
        input_width=window,
        forecast_horizon=forecast_horizon,
        feature_cols=feature_cols,
    )

    print(f"Scaled X_train_seq.shape: {X_train.shape} | Scaled y_train_seq.shape: {y_train.shape}")
    print(f"Scaled X_val_seq.shape:   {X_val.shape} | Scaled y_val_seq.shape:   {y_val.shape}")
    print(f"Scaled X_test_seq.shape:  {X_test.shape} | Scaled y_test_seq.shape:  {y_test.shape}")

    # Build DataLoaders
    batch_size = 128
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_dim=len(feature_cols), hidden_size=64, num_layers=1, horizon=forecast_horizon, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_epochs = 50
    patience = 5
    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        # Early stopping
        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation on val/test with best model
    def _predict(loader):
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        return np.vstack(trues), np.vstack(preds)

    y_val_true_scaled, y_val_pred_scaled = _predict(val_loader)
    y_test_true_scaled, y_test_pred_scaled = _predict(test_loader)

    # Inverse transform predictions/targets to original scale for metric computation
    def _invert(y_scaled):
        flat = y_scaled.reshape(-1, 1)
        inv = scaler_y.inverse_transform(flat).reshape(y_scaled.shape)
        return inv

    y_val_true = _invert(y_val_true_scaled)
    y_val_pred = _invert(y_val_pred_scaled)
    y_test_true = _invert(y_test_true_scaled)
    y_test_pred = _invert(y_test_pred_scaled)

    print(f"Inverse-transformed y_val shape: {y_val_true.shape}, y_pred shape: {y_val_pred.shape}")
    print(f"Inverse-transformed y_test shape: {y_test_true.shape}, y_pred shape: {y_test_pred.shape}")

    # Global metrics for LSTM
    lstm_val_metrics = evaluate_forecasts(y_val_true, y_val_pred)
    lstm_test_metrics = evaluate_forecasts(y_test_true, y_test_pred)
    print("Validation metrics (original scale):", lstm_val_metrics)
    print("Test metrics (original scale):", lstm_test_metrics)

    # Baselines on original test series (persistence, previous day, moving average)
    test_series = test_df[TARGET_COL]
    baseline_results = {
        "persistence": baseline_persistence(test_series, horizon=forecast_horizon),
        "previous_day": baseline_previous_day(test_series, horizon=forecast_horizon),
        "moving_average": baseline_moving_average(test_series, horizon=forecast_horizon, window=24),
        "lstm": lstm_test_metrics,
    }

    # Save comparison table
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    table = pd.DataFrame(baseline_results).T
    table.to_csv(results_dir / "metrics_global.csv")
    print("\nGlobal metrics (test):")
    print(table)

    # Per-horizon MAE for LSTM (h=1..24)
    horizons = np.arange(1, forecast_horizon + 1)
    mae_per_h = [np.mean(np.abs(y_test_true[:, h - 1] - y_test_pred[:, h - 1])) for h in horizons]
    pd.DataFrame({"horizon": horizons, "MAE": mae_per_h}).to_csv(results_dir / "metrics_per_horizon_lstm.csv", index=False)

    # Error analysis by hour-of-day and day-of-week using target timestamps
    idx = test_df.index
    n_samples = y_test_true.shape[0]
    target_times = []
    for s in range(n_samples):
        start = window + s
        horizon_times = idx[start : start + forecast_horizon]
        if len(horizon_times) < forecast_horizon:
            break
        target_times.append(horizon_times)
    target_times = target_times[: y_test_true.shape[0]]
    target_times = np.stack(target_times, axis=0)

    abs_err = np.abs(y_test_true - y_test_pred)
    flat_times = target_times.reshape(-1)
    flat_err = abs_err.reshape(-1)
    df_err = pd.DataFrame({"ts": flat_times, "abs_err": flat_err})
    df_err["hour"] = pd.to_datetime(df_err["ts"]).dt.hour
    df_err["dow"] = pd.to_datetime(df_err["ts"]).dt.dayofweek

    mae_hour = df_err.groupby("hour")["abs_err"].mean()
    mae_dow = df_err.groupby("dow")["abs_err"].mean()
    mae_hour.to_csv(results_dir / "mae_by_hour.csv")
    mae_dow.to_csv(results_dir / "mae_by_dow.csv")

    # Visualizations
    def _plot_24h(sample_idx: int, title: str, outfile: Path):
        true = y_test_true[sample_idx]
        pred = y_test_pred[sample_idx]
        hrs = np.arange(1, forecast_horizon + 1)
        plt.figure(figsize=(10, 4))
        plt.plot(hrs, true, label="True", marker="o")
        plt.plot(hrs, pred, label="Pred", marker="x")
        plt.title(title)
        plt.xlabel("Horizon (hour)")
        plt.ylabel("Load")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()

    _plot_24h(0, "24h forecast sample", results_dir / "plot_24h_sample.png")
    _plot_24h(len(y_test_true) // 2, "24h forecast sample (mid test)", results_dir / "plot_24h_sample_mid.png")

    # For a week: stitch consecutive 24h forecasts spaced by 24 samples
    def _plot_week(start_idx: int, title: str, outfile: Path):
        days = 7
        preds = []
        trues = []
        for d in range(days):
            idx_forecast = start_idx + d * 24
            if idx_forecast >= len(y_test_true):
                break
            pred = y_test_pred[idx_forecast]
            true = y_test_true[idx_forecast]
            horizon_times = target_times[idx_forecast]
            preds.append(pd.Series(pred, index=horizon_times))
            trues.append(pd.Series(true, index=horizon_times))
        if not preds:
            return
        pred_series = pd.concat(preds).sort_index()
        true_series = pd.concat(trues).sort_index()
        plt.figure(figsize=(12, 4))
        plt.plot(true_series.index, true_series.values, label="True")
        plt.plot(pred_series.index, pred_series.values, label="Pred", alpha=0.8)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Load")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()

    _plot_week(0, "Week forecast (start of test)", results_dir / "plot_week_start.png")
    _plot_week(len(y_test_true) // 2, "Week forecast (mid test)", results_dir / "plot_week_mid.png")

    mae_hour.plot(kind="bar", figsize=(10, 4), title="MAE by hour of day")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(results_dir / "mae_by_hour.png")
    plt.close()

    mae_dow.plot(kind="bar", figsize=(8, 4), title="MAE by day of week")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(results_dir / "mae_by_dow.png")
    plt.close()

    # Persist the target scaler for later inference
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_y, results_dir / "scaler_y.pkl")


if __name__ == "__main__":
    main()
