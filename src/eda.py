"""Exploratory data analysis utilities for household power consumption."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.data_processing import load_uci_household, preprocess_household_hourly

plt.style.use("seaborn-v0_8-darkgrid")

DEFAULT_WINDOWS: Dict[str, Tuple[str, str]] = {
    "semaine_jan_2010": ("2010-01-04", "2010-01-10"),
    "journee_2010-01-05": ("2010-01-05", "2010-01-05 23:59:59"),
    "mois_jan_2010": ("2010-01-01", "2010-01-31"),
    "hiver_2009_2010": ("2009-12-01", "2010-02-28"),
    "ete_2010": ("2010-07-01", "2010-08-31"),
}

__all__ = ["run_eda", "plot_window", "plot_global_daily", "plot_hourly_profiles"]


def _save(fig: plt.Figure, outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_global_daily(cons: pd.Series, outfile: Path | None = None) -> None:
    """Plot daily mean consumption over the full period."""
    global_daily = cons.resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    global_daily.plot(ax=ax, color="tab:blue", linewidth=1)
    ax.set_title("Consommation moyenne quotidienne (kW)")
    ax.set_xlabel("")
    ax.set_ylabel("kW")
    if outfile:
        _save(fig, outfile)
    else:
        plt.show()


def plot_window(cons: pd.Series, start: str, end: str, title: str, outfile: Path | None = None) -> None:
    """Plot consumption on a specific time window."""
    window = cons.loc[start:end]
    fig, ax = plt.subplots(figsize=(12, 4))
    window.plot(ax=ax, color="tab:orange", linewidth=1)
    ax.set_title(f"{title} ({start} -> {end})")
    ax.set_xlabel("")
    ax.set_ylabel("kW")
    if outfile:
        _save(fig, outfile)
    else:
        plt.show()


def plot_hourly_profiles(cons: pd.Series, outfile_hourly: Path | None = None, outfile_weekend: Path | None = None) -> None:
    """Plot hourly mean profile and compare weekday vs weekend."""
    # All days
    hod = cons.groupby(cons.index.hour).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    hod.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Profil moyen par heure")
    ax.set_xlabel("Heure du jour")
    ax.set_ylabel("kW")
    if outfile_hourly:
        _save(fig, outfile_hourly)
    else:
        plt.show()

    # Weekday vs weekend
    tmp = cons.to_frame("load_kw")
    tmp["hour"] = tmp.index.hour
    tmp["is_weekend"] = tmp.index.weekday >= 5
    pivot = tmp.groupby(["is_weekend", "hour"]).mean().unstack(0)["load_kw"]
    pivot.columns = ["Semaine", "Weekend"]

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(ax=ax)
    ax.set_title("Profil horaire moyen : semaine vs weekend")
    ax.set_xlabel("Heure du jour")
    ax.set_ylabel("kW")
    if outfile_weekend:
        _save(fig, outfile_weekend)
    else:
        plt.show()


def plot_monthly(cons: pd.Series, outfile: Path | None = None) -> None:
    """Plot monthly mean consumption to visualize seasonality."""
    monthly = cons.resample("M").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, color="tab:green", linewidth=1)
    ax.set_title("Consommation moyenne mensuelle (kW)")
    ax.set_xlabel("")
    ax.set_ylabel("kW")
    if outfile:
        _save(fig, outfile)
    else:
        plt.show()


def run_eda(
    data_path: Path = Path("data/household_power_consumption.txt"),
    results_dir: Path = Path("results"),
) -> None:
    """
    Run exploratory plots and save them under results/.

    Generates:
    - courbe globale (moyenne quotidienne)
    - fenêtres (semaine, journée, mois, hiver, été)
    - profils horaires + weekend
    - saisonnalité mensuelle
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw = load_uci_household(str(data_path))
    hourly = preprocess_household_hourly(raw)
    cons = hourly["Global_active_power"].rename("load_kw")

    results_dir.mkdir(parents=True, exist_ok=True)

    plot_global_daily(cons, results_dir / "eda_global_daily.png")

    plot_window(cons, *DEFAULT_WINDOWS["semaine_jan_2010"], "Semaine (janvier 2010)", results_dir / "eda_semaine.png")
    plot_window(cons, *DEFAULT_WINDOWS["journee_2010-01-05"], "Journée (5 janvier 2010)", results_dir / "eda_journee.png")
    plot_window(cons, *DEFAULT_WINDOWS["mois_jan_2010"], "Mois (janvier 2010)", results_dir / "eda_mois.png")
    plot_window(cons, *DEFAULT_WINDOWS["hiver_2009_2010"], "Hiver (dec 2009 - fev 2010)", results_dir / "eda_hiver.png")
    plot_window(cons, *DEFAULT_WINDOWS["ete_2010"], "Ete (juil-aout 2010)", results_dir / "eda_ete.png")

    plot_hourly_profiles(
        cons,
        outfile_hourly=results_dir / "eda_profil_horaire.png",
        outfile_weekend=results_dir / "eda_profil_weekend.png",
    )

    plot_monthly(cons, results_dir / "eda_mensuel.png")


if __name__ == "__main__":
    run_eda()
