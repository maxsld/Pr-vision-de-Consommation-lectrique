"""CLI entrypoint to generate exploratory plots."""

from pathlib import Path
import sys

# Allow running the script from any working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eda import run_eda


if __name__ == "__main__":
    data_path = REPO_ROOT / "data" / "household_power_consumption.txt"
    results_dir = REPO_ROOT / "results"
    run_eda(data_path=data_path, results_dir=results_dir)
