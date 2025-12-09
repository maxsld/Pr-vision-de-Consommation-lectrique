"""Download raw datasets into the local data directory."""

from pathlib import Path
import argparse
import sys
import urllib.request

# Allow running the script from any working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

WEATHER_URL = "https://data.open-power-system-data.org/index.php?package=weather_data&version=2020-09-16&action=customDownload&resource=0&filter%5B_contentfilter_utc_timestamp%5D%5Bfrom%5D=2006-12-16&filter%5B_contentfilter_utc_timestamp%5D%5Bto%5D=2010-11-26&filter%5BVariable%5D%5B%5D=radiation_diffuse_horizontal&filter%5BVariable%5D%5B%5D=radiation_direct_horizontal&filter%5BVariable%5D%5B%5D=temperature&filter%5BCountry%5D%5B%5D=AT&filter%5BCountry%5D%5B%5D=HU&filter%5BCountry%5D%5B%5D=IE&filter%5BCountry%5D%5B%5D=IT&filter%5BCountry%5D%5B%5D=LT&filter%5BCountry%5D%5B%5D=LU&filter%5BCountry%5D%5B%5D=LV&filter%5BCountry%5D%5B%5D=NL&filter%5BCountry%5D%5B%5D=PL&filter%5BCountry%5D%5B%5D=PT&filter%5BCountry%5D%5B%5D=RO&filter%5BCountry%5D%5B%5D=SE&filter%5BCountry%5D%5B%5D=SI&filter%5BCountry%5D%5B%5D=0&filter%5BCountry%5D%5B%5D=BE&filter%5BCountry%5D%5B%5D=BG&filter%5BCountry%5D%5B%5D=CH&filter%5BCountry%5D%5B%5D=CZ&filter%5BCountry%5D%5B%5D=DE&filter%5BCountry%5D%5B%5D=DK&filter%5BCountry%5D%5B%5D=EE&filter%5BCountry%5D%5B%5D=ES&filter%5BCountry%5D%5B%5D=FI&filter%5BCountry%5D%5B%5D=FR&filter%5BCountry%5D%5B%5D=GB&filter%5BCountry%5D%5B%5D=GR&filter%5BCountry%5D%5B%5D=HR&filter%5BCountry%5D%5B%5D=SK&filter%5BResolution%5D%5B%5D=Country&downloadCSV=Download+CSV"
WEATHER_DEST = DATA_DIR / "weather_data.csv"

# UCI Individual Household Electric Power Consumption (raw text file)
UCI_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
UCI_DEST = DATA_DIR / "consumption_data.csv"


def _format_size(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:,.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:,.1f}TB"


def stream_download(url: str, dest: Path) -> None:
    """Download with a simple progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as output:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        downloaded = 0
        chunk_size = 1024 * 64

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)

            if total_bytes:
                percent = downloaded / total_bytes
                bar_len = 30
                filled = int(bar_len * percent)
                bar = "#" * filled + "-" * (bar_len - filled)
                sys.stdout.write(
                    f"\r[{bar}] {percent*100:5.1f}% ({_format_size(downloaded)}/{_format_size(total_bytes)})"
                )
            else:
                sys.stdout.write(f"\rDownloaded {_format_size(downloaded)}")
            sys.stdout.flush()

        sys.stdout.write("\n")


def download_weather(force: bool = False) -> None:
    if WEATHER_DEST.exists() and not force:
        print(f"Weather data already exists at {WEATHER_DEST}")
        return
    print(f"Downloading weather data from {WEATHER_URL}")
    stream_download(WEATHER_URL, WEATHER_DEST)
    print(f"Saved weather data to {WEATHER_DEST}")


def download_uci_power_consumption(force: bool = False) -> None:
    if UCI_DEST.exists() and not force:
        print(f"UCI power consumption data already exists at {UCI_DEST}")
        return

    print(f"Downloading UCI dataset from {UCI_URL}")
    stream_download(UCI_URL, UCI_DEST)
    print(f"Saved UCI data to {UCI_DEST}")


def slice_weather_years(min_year: int = 2006, max_year: int = 2010) -> None:
    if not WEATHER_DEST.exists():
        print("Weather file not found; skipping slicing.")
        return

    tmp_path = WEATHER_DEST.with_suffix(".tmp")
    kept = 0
    total = 0

    with WEATHER_DEST.open("r") as src, tmp_path.open("w") as dst:
        header = src.readline()
        if not header:
            print("Weather file is empty; skipping slicing.")
            return
        dst.write(header)

        for line in src:
            total += 1
            year_str = line[:4]
            try:
                year = int(year_str)
            except ValueError:
                continue

            if min_year <= year <= max_year:
                dst.write(line)
                kept += 1

    tmp_path.replace(WEATHER_DEST)
    print(f"Sliced weather data to {min_year}-{max_year}: kept {kept} of {total} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets into data/")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist",
    )
    parser.add_argument(
        "--no-slice-weather",
        action="store_true",
        help="Skip slicing weather data to 2006-2010",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_weather(force=args.force)
    if not args.no_slice_weather:
        slice_weather_years()
    download_uci_power_consumption(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
