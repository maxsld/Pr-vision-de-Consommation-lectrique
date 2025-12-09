"""Utility helpers for splits and scaling."""

from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Allow running from anywhere by adding repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import split_time_series

__all__ = ["temporal_split", "scale_splits", "make_scaler"]


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split train/val/test without shuffle.

    If val_end is provided: train <= train_end, val in (train_end, val_end], test > val_end.
    If val_end is None: fallback to 70/15/15 proportion.
    """
    return split_time_series(df, train_end=train_end, val_end=val_end)


def make_scaler(scaler_type: str = "standard", feature_range: Tuple[float, float] = (0.0, 1.0)):
    """Instantiate a scaler from sklearn."""
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler(feature_range=feature_range)
    raise ValueError(f"Unknown scaler_type: {scaler_type}")


def scale_splits(
    train: pd.DataFrame,
    val: Optional[pd.DataFrame],
    test: Optional[pd.DataFrame],
    columns: Optional[Sequence[str]] = None,
    scaler_type: str = "standard",
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], object]:
    """
    Fit a scaler on train only, apply to val/test, return scaled copies and the scaler.

    Args:
        columns: columns to scale (default: all columns).
        scaler_type: "standard" or "minmax".
    """
    cols = list(columns) if columns is not None else list(train.columns)
    scaler = make_scaler(scaler_type, feature_range)

    train_scaled = train.copy()
    train_scaled[cols] = scaler.fit_transform(train[cols])

    val_scaled = None
    test_scaled = None
    if val is not None:
        val_scaled = val.copy()
        val_scaled[cols] = scaler.transform(val[cols])
    if test is not None:
        test_scaled = test.copy()
        test_scaled[cols] = scaler.transform(test[cols])

    return train_scaled, val_scaled, test_scaled, scaler


