"""Data loading, preprocessing, and feature engineering utilities."""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

UCI_SUB_METERING_PREFIX = "Sub_metering"

__all__ = [
    "load_consumption_data",
    "load_uci_household",
    "preprocess_household_hourly",
    "add_time_features",
    "add_lag_features",
    "merge_weather_features",
    "split_time_series",
    "load_opsd_weather",
]


def load_consumption_data(
    path: str,
    load_column: Optional[str] = None,
    country: str = "FR",
    timestamp_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load consumption dataset (UCI legacy or OPSD time series) and return a DataFrame indexed by datetime with a single
    column named 'load'.

    - If Date/Time columns are present (UCI format), they are merged.
    - If utc/cet timestamps are present (OPSD), parses them and selects a load column.
    """
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, na_values="?", low_memory=False)

    # Legacy UCI format: Date + Time columns
    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H:%M:%S",
            dayfirst=True,
            errors="coerce",
        )
        df = df.drop(columns=["Date", "Time"])
        df = df.set_index("datetime").sort_index()
        target_col = load_column or "Global_active_power"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in UCI dataset.")
        df = df.rename(columns={target_col: "load"})
        return df

    # OPSD format: look for timestamp column
    if timestamp_column is None:
        ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
        if not ts_candidates:
            raise ValueError("No timestamp column found in dataset.")
        timestamp_column = ts_candidates[0]

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_column])
    df = df.set_index(timestamp_column).sort_index()
    df.index = df.index.tz_convert(None)

    if load_column is None:
        preferred = [
            f"{country}_load_actual_entsoe_transparency",
            f"{country}_load_actual",
            f"{country}_load_forecast_entsoe_transparency",
        ]
        load_column = next((c for c in preferred if c in df.columns), None)
        if load_column is None:
            load_column = next((c for c in df.columns if "load_actual" in c), None)
    if load_column is None or load_column not in df.columns:
        raise ValueError("No suitable load column found in dataset; specify load_column explicitly.")

    out = df[[load_column]].rename(columns={load_column: "load"})
    return out


def load_uci_household(path: str) -> pd.DataFrame:
    """Backwards compatibility wrapper."""
    return load_consumption_data(path)


def preprocess_household_hourly(
    df: pd.DataFrame,
    resample_rule: str = "1h",
    max_missing_ratio: float = 0.5,
    ffill_limit: int = 2,
    bfill_limit: int = 1,
) -> pd.DataFrame:
    """
    Clean and resample the UCI household data to hourly frequency.

    - Merge duplicate timestamps, sort by time.
    - Coerce numeric columns, mark invalid as NaN.
    - Resample: mean for power/voltage/intensity, sum for sub-metering.
    - Fill small gaps (ffill/bfill with limits), drop rows too sparse.
    """
    df = df[~df.index.duplicated(keep="first")].sort_index()
    numeric_cols = df.columns.tolist()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    sum_cols = [c for c in numeric_cols if c.startswith(UCI_SUB_METERING_PREFIX)]
    mean_cols = [c for c in numeric_cols if c not in sum_cols]

    agg_map = {c: "sum" for c in sum_cols}
    agg_map.update({c: "mean" for c in mean_cols})

    hourly = df.resample(resample_rule).agg(agg_map)
    hourly = hourly.dropna(how="all")

    if ffill_limit or bfill_limit:
        hourly = hourly.ffill(limit=ffill_limit).bfill(limit=bfill_limit)

    missing_ratio = hourly.isna().mean(axis=1)
    hourly = hourly[missing_ratio <= max_missing_ratio]
    hourly = hourly.dropna()
    return hourly


def add_time_features(df: pd.DataFrame, holidays_country: Optional[str] = "FR") -> pd.DataFrame:
    """
    Add calendar and cyclic time features.

    - hour (0-23), dayofweek (0-6), month (1-12)
    - is_weekend, is_holiday (if holidays package is available)
    - hour_sin, hour_cos (cyclic encoding)
    - dayofweek_sin, dayofweek_cos (cyclic encoding)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex to add time features.")

    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = out["dayofweek"] >= 5

    if holidays_country:
        try:
            import holidays  # type: ignore

            holiday_calendar = holidays.country_holidays(holidays_country)
            out["is_holiday"] = idx.normalize().isin(holiday_calendar)
        except Exception:
            out["is_holiday"] = False
    else:
        out["is_holiday"] = False

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int] = (1, 24),
    rolling_windows: Sequence[int] = (3, 24),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Add lagged targets and rolling means on the target column.

    - Lags: e.g., y(t-1), y(t-24)
    - Rolling means: e.g., rolling_mean_3h, rolling_mean_24h
    """
    out = df.copy()
    if target_col not in out.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    for lag in lags:
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    for win in rolling_windows:
        out[f"{target_col}_rollmean_{win}h"] = out[target_col].shift(1).rolling(win, min_periods=win).mean()

    if dropna:
        out = out.dropna()
    return out


def merge_weather_features(
    df: pd.DataFrame,
    weather: pd.Series,
    how: str = "left",
    suffix: str = "temp",
) -> pd.DataFrame:
    """
    Merge a weather series (e.g., temperature) onto the main DataFrame by timestamp.

    Args:
        df: main time-indexed DataFrame.
        weather: Series indexed by datetime (e.g., temperature).
        how: join type (default left).
        suffix: suffix/name used for the weather column if not already named.
    """
    if weather.name is None:
        weather = weather.rename(suffix)
    # Align timezone awareness (drop tz info to match consumption which is naive)
    if getattr(weather.index, "tz", None) is not None:
        weather.index = weather.index.tz_convert(None)
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    joined = df.join(weather, how=how)
    return joined


def load_opsd_weather(
    path: str,
    datetime_col: Optional[str] = None,
    temperature_col: Optional[str] = None,
    resample_rule: str = "1h",
) -> pd.Series:
    """
    Load OPSD aggregated weather data and return an hourly temperature series.

    Args:
        path: CSV path.
        datetime_col: name of the datetime column (auto-detects if None).
        temperature_col: name of the temperature column (auto-detects if None, looks for 'temp').
        resample_rule: resampling rule (default 1h).
    """
    df = pd.read_csv(path, low_memory=False)

    if datetime_col is None:
        candidates = [c for c in df.columns if "time" in c.lower()]
        if not candidates:
            datetime_col = df.columns[0]
        else:
            datetime_col = candidates[0]
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found in weather data.")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.set_index(datetime_col).sort_index()

    if temperature_col is None:
        temp_candidates = [c for c in df.columns if "temp" in c.lower()]
        if not temp_candidates:
            raise ValueError("No temperature column found (searched for substring 'temp').")
        temperature_col = temp_candidates[0]
    if temperature_col not in df.columns:
        raise ValueError(f"Temperature column '{temperature_col}' not found in weather data.")

    temp = pd.to_numeric(df[temperature_col], errors="coerce")
    # Drop timezone if present
    if getattr(temp.index, "tz", None) is not None:
        temp.index = temp.index.tz_convert(None)
    temp = temp.resample(resample_rule).mean()
    return temp.rename("temperature")


def split_time_series(
    df: pd.DataFrame,
    train_end: str,
    val_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split into train/val/test by date boundaries.

    Args:
        df: time-indexed DataFrame.
        train_end: last timestamp for the training set (inclusive).
        val_end: last timestamp for the validation set (inclusive).
                 If None, split 70/15/15 by proportion.
    """
    if val_end is None:
        n = len(df)
        train_cut = int(n * 0.7)
        val_cut = int(n * 0.85)
        train = df.iloc[:train_cut]
        val = df.iloc[train_cut:val_cut]
        test = df.iloc[val_cut:]
        return train, val, test

    train_boundary = pd.Timestamp(train_end)
    val_boundary = pd.Timestamp(val_end)

    train = df[df.index <= train_boundary]
    val = df[(df.index > train_boundary) & (df.index <= val_boundary)]
    test = df[df.index > val_boundary]
    return train, val, test
