"""CLI entrypoint to generate exploratory plots."""

from pathlib import Path
import sys

# Allow running the script from any working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eda import run_eda


if __name__ == "__main__":
    data_path = REPO_ROOT / "data" / "consumption_data.csv"
    results_dir = REPO_ROOT / "results"
    # Try to locate a weather file if present
    weather_candidates = [
        REPO_ROOT / "data" / "weather.csv",
        REPO_ROOT / "data" / "weather_opsd.csv",
        REPO_ROOT / "data" / "weather_data.csv",
    ]
    weather_path = next((p for p in weather_candidates if p.exists()), None)
    if weather_path is None:
        print("Aucun fichier météo détecté dans data/ (weather.csv / weather_opsd.csv / weather_data.csv). EDA météo ignorée.")
    else:
        print(f"EDA météo avec : {weather_path.name}")

    run_eda(
        data_path=data_path,
        results_dir=results_dir,
        show=True,
        weather_path=weather_path,
    )
