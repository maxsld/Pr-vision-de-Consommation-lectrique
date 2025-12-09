#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Télécharge les jeux de données bruts dans data/ (sans options CLI)."""

from __future__ import annotations

import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple
import urllib.request
import urllib.error

import pandas as pd

# ---------------------------------------------------------------------
# Chemins / ressources
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

WEATHER_URL = "https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv"
WEATHER_DEST = DATA_DIR / "weather_data.csv"

UCI_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
UCI_DEST = DATA_DIR / "consumption_data.csv"

RESOURCES = [
    ("Weather", WEATHER_URL, WEATHER_DEST),
    ("UCI", UCI_URL, UCI_DEST),
]

DATE_RANGE: Tuple[datetime, datetime] = (
    datetime(2015, 1, 1),
    datetime(2019, 12, 31, 23, 59, 59),
)

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:.1f}B"

def _format_eta(seconds: float) -> str:
    if math.isinf(seconds) or math.isnan(seconds):
        return "--:--"
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

def _open(url: str, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "dataset-downloader/mini/1.0", **(headers or {})},
    )
    return urllib.request.urlopen(req, timeout=timeout)

# ---------------------------------------------------------------------
# Téléchargement simple mais robuste
# ---------------------------------------------------------------------
def download_with_resume(name: str, url: str, dest: Path, *, retries: int = 3, timeout: int = 30) -> None:
    """Télécharge url→dest avec reprise (.part), progress et validations basiques."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"✓ {name}: déjà présent ({dest})")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0

    attempt = 0
    while True:
        attempt += 1
        try:
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            mode = "ab" if resume_from else "wb"

            with _open(url, headers=headers, timeout=timeout) as resp, tmp.open(mode) as out:
                status = getattr(resp, "status", None)
                # si reprise demandée mais non acceptée → repartir de zéro
                if resume_from and status != 206:
                    tmp.unlink(missing_ok=True)
                    resume_from = 0
                    headers = {}
                    mode = "wb"
                    print(f"↻ {name}: reprise non supportée, redémarrage…")
                    continue  # relance la boucle avec une nouvelle requête

                # Content-Length = taille du *segment* renvoyé
                seg_len = resp.headers.get("Content-Length")
                seg_len = int(seg_len) if seg_len else None

                downloaded = resume_from
                expected_total = None
                # On tente d'inférer la taille totale si seg_len connu
                if seg_len is not None:
                    expected_total = resume_from + seg_len

                start = last = time.time()
                bytes_since = 0
                bar_len = 30
                chunk_size = 1024 * 1024  # 1 MiB

                print(f"↓ {name}: téléchargement vers {dest.name}")
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    bytes_since += len(chunk)

                    now = time.time()
                    if now - last >= 0.1:
                        speed = bytes_since / (now - last + 1e-9)
                        last = now
                        bytes_since = 0

                        if expected_total:
                            pct = min(1.0, downloaded / expected_total)
                            filled = int(bar_len * pct)
                            bar = "#" * filled + "-" * (bar_len - filled)
                            remaining = (expected_total - downloaded) / (speed + 1e-9)
                            sys.stdout.write(
                                f"\r   [{bar}] {pct*100:5.1f}% "
                                f"({_format_size(downloaded)}/{_format_size(expected_total)}) "
                                f"{_format_size(speed)}/s ETA {_format_eta(remaining)}"
                            )
                        else:
                            sys.stdout.write(
                                f"\r   Téléchargé {_format_size(downloaded)} "
                                f"à {_format_size(speed)}/s"
                            )
                        sys.stdout.flush()
                sys.stdout.write("\n")

            # Validation de taille simple si on la connaît
            if expected_total is not None and downloaded != expected_total:
                raise IOError(f"Taille inattendue: {downloaded} vs {expected_total}")

            tmp.replace(dest)  # écriture atomique
            print(f"✅ {name}: terminé → {dest}")
            return

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, IOError) as e:
            if attempt > retries:
                print(f"❌ {name}: échec après {retries} tentatives ({e})")
                # on laisse .part pour permettre une reprise lors d’un prochain run
                raise
            sleep = min(30, 2 ** (attempt - 1))
            print(f"⚠️  {name}: erreur «{e}», nouvelle tentative #{attempt} dans {sleep}s…")
            time.sleep(sleep)

# ---------------------------------------------------------------------
# Découpage temporel
# ---------------------------------------------------------------------
def _detect_sep(path: Path) -> str:
    with path.open("r") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") >= first_line.count(",") else ","


def _detect_datetime_column(path: Path, sep: str) -> str:
    header = pd.read_csv(path, sep=sep, nrows=0)
    for col in header.columns:
        low = col.lower()
        if "time" in low or "timestamp" in low or low == "date":
            return col
    return header.columns[0]


def slice_date_range(path: Path, start: datetime, end: datetime) -> None:
    """
    Découpe un CSV selon une colonne de temps détectée (time/timestamp/date) pour ne garder que [start, end].
    Parcourt en chunks pour limiter la mémoire.
    """
    if not path.exists():
        print(f"⚠️  Fichier introuvable pour découpe : {path}")
        return

    sep = _detect_sep(path)
    date_col = _detect_datetime_column(path, sep)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    wrote_header = False
    total_kept = 0
    for chunk in pd.read_csv(path, sep=sep, parse_dates=[date_col], chunksize=200_000, low_memory=False):
        # Nettoie timezone pour comparer proprement
        if getattr(chunk[date_col].dt, "tz", None) is not None:
            chunk[date_col] = chunk[date_col].dt.tz_localize(None)

        mask = (chunk[date_col] >= start) & (chunk[date_col] <= end)
        filtered = chunk.loc[mask]
        if filtered.empty:
            continue
        filtered.to_csv(tmp_path, mode="a", header=not wrote_header, index=False, sep=sep)
        wrote_header = True
        total_kept += len(filtered)

    if not wrote_header:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Aucune donnée entre {start.date()} et {end.date()} pour {path.name}")

    tmp_path.replace(path)
    print(f"✂️  {path.name}: gardé {total_kept} lignes entre {start.date()} et {end.date()}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    try:
        for name, url, dest in RESOURCES:
            download_with_resume(name, url, dest, retries=3, timeout=30)
        # Découpe sur la plage 2015-2019
        start, end = DATE_RANGE
        for dest in (WEATHER_DEST, UCI_DEST):
            try:
                slice_date_range(dest, start, end)
            except Exception as e:
                print(f"⚠️  Découpe ignorée pour {dest.name}: {e}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrompu par l’utilisateur.")
        return 130
    except Exception as e:
        print(f"Erreur: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
