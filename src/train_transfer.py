"""Train on one country and evaluate forecasts on a distant country."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
from src.evaluate import evaluate_forecasts  # noqa: E402
from src.models import LSTMForecaster  # noqa: E402
from src.train import make_sequences  # noqa: E402
from src.utils import temporal_split  # noqa: E402

TARGET_COL = "load"


def build_feature_frame(
    country: str,
    data_path: Path,
    weather_path: Optional[Path],
    lags: Sequence[int] = (1, 24),
    rolling_windows: Sequence[int] = (3, 24),
    use_weather: bool = True,
    holidays_country: Optional[str] = None,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """Load, clean, enrich features for a given country (no split yet)."""
    raw = load_consumption_data(str(data_path), country=country)
    hourly = preprocess_household_hourly(raw)
    hourly = add_time_features(hourly, holidays_country=holidays_country or country)
    hourly = add_lag_features(hourly, target_col=TARGET_COL, lags=lags, rolling_windows=rolling_windows, dropna=True)

    if use_weather and weather_path and weather_path.exists():
        try:
            temp = load_opsd_weather(
                str(weather_path),
                datetime_col="utc_timestamp",
                temperature_col=f"{country}_temperature",
            )
            hourly = merge_weather_features(hourly, temp)
            if "temperature" in hourly.columns and hourly["temperature"].notna().sum() == 0:
                hourly = hourly.drop(columns=["temperature"])
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Weather not merged for {country}: {exc}")

    hourly = hourly.dropna(subset=[TARGET_COL])
    feature_cols = [c for c in hourly.columns if c != TARGET_COL]
    return hourly, feature_cols


def align_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_features: Iterable[str],
    test_features: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[str]]:
    """Keep only the common feature space between train and transfer countries."""
    common = sorted(set(train_features).intersection(set(test_features)))
    if not common:
        raise ValueError("No common feature columns between the two countries.")
    train_df = train_df[common + [TARGET_COL]]
    test_df = test_df[common + [TARGET_COL]]
    return train_df, test_df, common


def fit_scalers(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[StandardScaler, StandardScaler]:
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(train_df[feature_cols])
    scaler_y.fit(train_df[[TARGET_COL]])
    return scaler_x, scaler_y


def apply_scalers(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
) -> pd.DataFrame:
    scaled = df.copy()
    scaled[feature_cols] = scaler_x.transform(df[feature_cols])
    scaled[TARGET_COL] = scaler_y.transform(df[[TARGET_COL]])
    return scaled


def fit_lstm(
    feature_dim: int,
    forecast_horizon: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-3,
) -> LSTMForecaster:
    """Train LSTM with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(
        input_dim=feature_dim,
        hidden_size=64,
        num_layers=1,
        horizon=forecast_horizon,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=128,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=128,
        shuffle=False,
    )

    for epoch in range(1, max_epochs + 1):
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
        print(f"[lstm] Epoch {epoch:02d} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("[lstm] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_scaled(model: LSTMForecaster, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def evaluate_split(
    model: LSTMForecaster,
    X: np.ndarray,
    y: np.ndarray,
    scaler_y: StandardScaler,
    forecast_horizon: int,
    target_times: Optional[np.ndarray] = None,
) -> Tuple[dict, pd.DataFrame, Optional[pd.DataFrame]]:
    """Run inference, invert scaling, compute metrics and optional error breakdown."""
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=256,
        shuffle=False,
    )
    y_true_scaled, y_pred_scaled = _predict_scaled(model, loader)

    def _invert(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, 1)
        inv = scaler_y.inverse_transform(flat).reshape(arr.shape)
        return inv

    y_true = _invert(y_true_scaled)
    y_pred = _invert(y_pred_scaled)

    metrics = evaluate_forecasts(y_true, y_pred)
    horizons = np.arange(1, forecast_horizon + 1)
    per_horizon = pd.DataFrame({"horizon": horizons, "MAE": [np.mean(np.abs(y_true[:, h - 1] - y_pred[:, h - 1])) for h in horizons]})

    error_by_time = None
    if target_times is not None:
        abs_err = np.abs(y_true - y_pred)
        flat_times = target_times.reshape(-1)
        flat_err = abs_err.reshape(-1)
        df_err = pd.DataFrame({"ts": flat_times, "abs_err": flat_err})
        df_err["hour"] = pd.to_datetime(df_err["ts"]).dt.hour
        df_err["dow"] = pd.to_datetime(df_err["ts"]).dt.dayofweek
        error_by_time = pd.DataFrame(
            {
                "mae_by_hour": df_err.groupby("hour")["abs_err"].mean(),
                "mae_by_dow": df_err.groupby("dow")["abs_err"].mean(),
            }
        )
    return metrics, per_horizon, error_by_time


def build_target_times(index: pd.DatetimeIndex, window: int, horizon: int) -> np.ndarray:
    """Build target timestamps aligned with the sliding windows."""
    target_times = []
    for s in range(len(index)):
        start_idx = window + s
        horizon_times = index[start_idx : start_idx + horizon]
        if len(horizon_times) < horizon:
            break
        target_times.append(horizon_times)
    return np.stack(target_times, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on one country and evaluate on another.")
    parser.add_argument("--train-country", type=str, required=True, help="Country code used for training.")
    parser.add_argument("--test-country", type=str, required=True, help="Country code used for transfer evaluation.")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "consumption_data.csv")
    parser.add_argument("--weather-path", type=Path, default=REPO_ROOT / "data" / "weather_data.csv")
    parser.add_argument("--train-end", type=str, default=None, help="Optional date boundary for training set (inclusive).")
    parser.add_argument("--val-end", type=str, default=None, help="Optional date boundary for validation set (inclusive).")
    parser.add_argument("--input-width", type=int, default=24, help="History window size.")
    parser.add_argument("--forecast-horizon", type=int, default=24, help="Forecast horizon (hours).")
    parser.add_argument("--results-dir", type=Path, default=None, help="Where to write metrics (defaults to results/transfer_*).")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-weather", action="store_true", help="Disable weather merge.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or (REPO_ROOT / "results" / f"transfer_{args.train_country}_to_{args.test_country}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing data for train country={args.train_country} and test country={args.test_country}")
    train_df_full, train_features = build_feature_frame(
        country=args.train_country,
        data_path=args.data_path,
        weather_path=None if args.no_weather else args.weather_path,
    )
    test_df_full, test_features = build_feature_frame(
        country=args.test_country,
        data_path=args.data_path,
        weather_path=None if args.no_weather else args.weather_path,
    )

    train_df_full, test_df_full, feature_cols = align_features(train_df_full, test_df_full, train_features, test_features)
    print(f"Using {len(feature_cols)} common features: {feature_cols}")

    train_df, val_df, test_df = temporal_split(train_df_full, train_end=args.train_end, val_end=args.val_end)

    scaler_x, scaler_y = fit_scalers(train_df, feature_cols)
    train_scaled = apply_scalers(train_df, feature_cols, scaler_x, scaler_y)
    val_scaled = apply_scalers(val_df, feature_cols, scaler_x, scaler_y)
    test_scaled = apply_scalers(test_df, feature_cols, scaler_x, scaler_y)
    transfer_scaled = apply_scalers(test_df_full, feature_cols, scaler_x, scaler_y)

    window = args.input_width
    horizon = args.forecast_horizon

    X_train, y_train = make_sequences(train_scaled, TARGET_COL, input_width=window, forecast_horizon=horizon, feature_cols=feature_cols)
    X_val, y_val = make_sequences(val_scaled, TARGET_COL, input_width=window, forecast_horizon=horizon, feature_cols=feature_cols)
    X_test, y_test = make_sequences(test_scaled, TARGET_COL, input_width=window, forecast_horizon=horizon, feature_cols=feature_cols)
    X_transfer, y_transfer = make_sequences(transfer_scaled, TARGET_COL, input_width=window, forecast_horizon=horizon, feature_cols=feature_cols)

    print(
        f"Shapes | train {X_train.shape} | val {X_val.shape} | in-country test {X_test.shape} | transfer test {X_transfer.shape}"
    )

    model = fit_lstm(
        feature_dim=len(feature_cols),
        forecast_horizon=horizon,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
    )

    target_times_in_country = build_target_times(test_df.index, window=window, horizon=horizon)
    in_country_metrics, in_country_per_h, in_country_error = evaluate_split(
        model,
        X_test,
        y_test,
        scaler_y=scaler_y,
        forecast_horizon=horizon,
        target_times=target_times_in_country,
    )

    target_times_transfer = build_target_times(test_df_full.index, window=window, horizon=horizon)
    transfer_metrics, transfer_per_h, transfer_error = evaluate_split(
        model,
        X_transfer,
        y_transfer,
        scaler_y=scaler_y,
        forecast_horizon=horizon,
        target_times=target_times_transfer,
    )

    pd.DataFrame({"train_country_test": in_country_metrics, "transfer_test": transfer_metrics}).to_csv(
        results_dir / "metrics_transfer.csv"
    )
    in_country_per_h.to_csv(results_dir / "metrics_per_horizon_in_country.csv", index=False)
    transfer_per_h.to_csv(results_dir / "metrics_per_horizon_transfer.csv", index=False)
    if in_country_error is not None:
        in_country_error.to_csv(results_dir / "error_breakdown_in_country.csv")
    if transfer_error is not None:
        transfer_error.to_csv(results_dir / "error_breakdown_transfer.csv")

    print("== In-country test metrics ==")
    for k, v in in_country_metrics.items():
        print(f"{k}: {v:.3f}")

    print("\n== Transfer test metrics ==")
    for k, v in transfer_metrics.items():
        print(f"{k}: {v:.3f}")

    print(f"\nSaved results to {results_dir}")


if __name__ == "__main__":
    main()
