"""Plot load curves for two chosen countries from the OPSD dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# Allow running this script from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing import load_consumption_data, preprocess_household_hourly  # noqa: E402


def load_country_series(path: Path, country: str) -> pd.Series:
    """Load and clean a single country load series."""
    df = load_consumption_data(str(path), country=country)
    hourly = preprocess_household_hourly(df)
    return hourly["load"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot load curves for two countries.")
    parser.add_argument("--country-a", required=True, help="First country code (e.g., FR).")
    parser.add_argument("--country-b", required=True, help="Second country code (e.g., SE).")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "consumption_data.csv")
    parser.add_argument("--start", type=str, default=None, help="Optional start datetime (e.g., 2018-01-01).")
    parser.add_argument("--end", type=str, default=None, help="Optional end datetime (e.g., 2019-01-01).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG file (default: results/plot_<A>_vs_<B>.png).",
    )
    return parser.parse_args()


def maybe_slice(series: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
    if start:
        series = series[series.index >= pd.to_datetime(start)]
    if end:
        series = series[series.index <= pd.to_datetime(end)]
    return series


def main() -> None:
    args = parse_args()
    out_path = args.output or (REPO_ROOT / "results" / f"plot_{args.country_a}_vs_{args.country_b}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s_a = load_country_series(args.data_path, args.country_a)
    s_b = load_country_series(args.data_path, args.country_b)

    s_a = maybe_slice(s_a, args.start, args.end)
    s_b = maybe_slice(s_b, args.start, args.end)

    # Align on common timestamps
    df = pd.concat(
        [s_a.rename(args.country_a), s_b.rename(args.country_b)],
        axis=1,
        join="inner",
    ).dropna()

    if df.empty:
        raise ValueError("No overlapping data between the two series (after slicing).")

    plt.figure(figsize=(12, 4))
    df.plot(ax=plt.gca())
    plt.title(f"Load comparison: {args.country_a} vs {args.country_b}")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
