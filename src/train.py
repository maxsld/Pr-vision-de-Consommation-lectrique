"""Training entrypoint placeholder: prepare sequences and show dataset shapes."""

from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

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
from src.utils import scale_splits, temporal_split  # noqa: E402

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
    train_end: str = "2008-12-31",
    val_end: str = "2009-12-31",
):
    """Load, preprocess, add features, split, and scale. Returns scaled splits and feature list."""
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

    hourly = hourly.dropna()

    train_df, val_df, test_df = temporal_split(hourly, train_end=train_end, val_end=val_end)

    feature_cols = [c for c in hourly.columns if c != TARGET_COL]
    train_scaled, val_scaled, test_scaled, _ = scale_splits(
        train_df,
        val_df,
        test_df,
        columns=feature_cols,
        scaler_type="standard",
    )

    return train_scaled, val_scaled, test_scaled, feature_cols


def main() -> None:
    data_path = REPO_ROOT / "data" / "consumption_data.csv"
    weather_candidates = [
        REPO_ROOT / "data" / "weather.csv",
        REPO_ROOT / "data" / "weather_opsd.csv",
        REPO_ROOT / "data" / "weather_data.csv",
    ]
    weather_path = next((p for p in weather_candidates if p.exists()), None)

    train_scaled, val_scaled, test_scaled, feature_cols = process_data(
        data_path=data_path,
        weather_path=weather_path,
        train_end="2008-12-31",
        val_end="2009-12-31",
    )

    window_sizes = [24, 48, 168]  # 1 jour, 2 jours, 1 semaine
    forecast_horizon = 24

    for win in window_sizes:
        X_train, y_train = make_sequences(
            train_scaled,
            TARGET_COL,
            input_width=win,
            forecast_horizon=forecast_horizon,
            feature_cols=feature_cols,
        )
        X_val, y_val = make_sequences(
            val_scaled,
            TARGET_COL,
            input_width=win,
            forecast_horizon=forecast_horizon,
            feature_cols=feature_cols,
        )
        X_test, y_test = make_sequences(
            test_scaled,
            TARGET_COL,
            input_width=win,
            forecast_horizon=forecast_horizon,
            feature_cols=feature_cols,
        )

        print(f"[fenêtre={win}h] X_train_seq.shape: {X_train.shape} | y_train_seq.shape: {y_train.shape}")
        print(f"[fenêtre={win}h] X_val_seq.shape:   {X_val.shape} | y_val_seq.shape:   {y_val.shape}")
        print(f"[fenêtre={win}h] X_test_seq.shape:  {X_test.shape} | y_test_seq.shape:  {y_test.shape}")


if __name__ == "__main__":
    main()
