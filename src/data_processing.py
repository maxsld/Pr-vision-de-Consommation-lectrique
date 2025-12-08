"""Data loading and preprocessing utilities for UCI household power consumption."""

from typing import Optional, Tuple

import pandas as pd


UCI_SUB_METERING_PREFIX = "Sub_metering"

__all__ = [
    "load_uci_household",
    "preprocess_household_hourly",
    "split_time_series",
]


def load_uci_household(path: str) -> pd.DataFrame:
    """
    Load the UCI Household Power Consumption dataset.

    Expects the raw text file with ';' separator and Date/Time columns.
    """
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        low_memory=False,
        parse_dates={"datetime": ["Date", "Time"]},
        infer_datetime_format=True,
    )
    df = df.set_index("datetime").sort_index()
    return df


def preprocess_household_hourly(
    df: pd.DataFrame,
    resample_rule: str = "1H",
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
