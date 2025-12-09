"""Exploratory data analysis utilities for household power consumption."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.data_processing import (
    load_opsd_weather,
    load_uci_household,
    preprocess_household_hourly,
)

plt.style.use("seaborn-v0_8-darkgrid")

DEFAULT_WINDOWS: Dict[str, Tuple[str, str]] = {
    "semaine_jan_2010": ("2010-01-04", "2010-01-10"),
    "journee_2010-01-05": ("2010-01-05", "2010-01-05 23:59:59"),
    "mois_jan_2010": ("2010-01-01", "2010-01-31"),
    "hiver_2009_2010": ("2009-12-01", "2010-02-28"),
    "ete_2010": ("2010-07-01", "2010-08-31"),
}

__all__ = ["run_eda", "plot_window", "plot_global_daily", "plot_hourly_profiles"]


def _finalize(fig: plt.Figure, outfile: Path | None, show: bool) -> None:
    fig.tight_layout()
    if outfile:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_global_daily(
    series: pd.Series,
    title: str,
    ylabel: str,
    outfile: Path | None = None,
    show: bool = True,
    resample_rule: str = "D",
    color: str = "tab:blue",
) -> None:
    """Plot daily mean values over the full period."""
    daily = series.resample(resample_rule).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    daily.plot(ax=ax, color=color, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def plot_window(
    series: pd.Series,
    start: str,
    end: str,
    title: str,
    ylabel: str,
    outfile: Path | None = None,
    show: bool = True,
    color: str = "tab:orange",
) -> None:
    """Plot a specific time window."""
    window = series.loc[start:end]
    fig, ax = plt.subplots(figsize=(12, 4))
    window.plot(ax=ax, color=color, linewidth=1)
    ax.set_title(f"{title} ({start} -> {end})")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def plot_hourly_profiles(
    cons: pd.Series,
    outfile_hourly: Path | None = None,
    outfile_weekend: Path | None = None,
    show: bool = True,
) -> None:
    """Plot hourly mean profile and compare weekday vs weekend."""
    # All days
    hod = cons.groupby(cons.index.hour).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    hod.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Profil moyen par heure")
    ax.set_xlabel("Heure du jour")
    ax.set_ylabel("kW")
    _finalize(fig, outfile_hourly, show)

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
    _finalize(fig, outfile_weekend, show)


def plot_monthly(
    series: pd.Series,
    title: str,
    ylabel: str,
    outfile: Path | None = None,
    show: bool = True,
    color: str = "tab:green",
) -> None:
    """Plot monthly mean values to visualize seasonality."""
    monthly = series.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, color=color, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def run_eda(
    data_path: Path = Path("data/household_power_consumption.txt"),
    results_dir: Path = Path("results"),
    show: bool = True,
    weather_path: Path | None = None,
) -> None:
    """
    Run exploratory plots and save them under results/.

    Generates:
    - courbe globale (moyenne quotidienne)
    - fenêtres (semaine, journée, mois, hiver, été)
    - profils horaires + weekend
    - saisonnalité mensuelle
    Optionnel : même plots pour la température si un fichier météo est fourni.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw = load_uci_household(str(data_path))
    hourly = preprocess_household_hourly(raw)
    cons = hourly["Global_active_power"].rename("load_kw")

    results_dir.mkdir(parents=True, exist_ok=True)

    plot_global_daily(
        cons,
        title="Consommation moyenne quotidienne (kW)",
        ylabel="kW",
        outfile=results_dir / "eda_global_daily.png",
        show=show,
    )

    plot_window(
        cons,
        *DEFAULT_WINDOWS["semaine_jan_2010"],
        title="Semaine (janvier 2010)",
        ylabel="kW",
        outfile=results_dir / "eda_semaine.png",
        show=show,
    )
    plot_window(
        cons,
        *DEFAULT_WINDOWS["journee_2010-01-05"],
        title="Journée (5 janvier 2010)",
        ylabel="kW",
        outfile=results_dir / "eda_journee.png",
        show=show,
    )
    plot_window(
        cons,
        *DEFAULT_WINDOWS["mois_jan_2010"],
        title="Mois (janvier 2010)",
        ylabel="kW",
        outfile=results_dir / "eda_mois.png",
        show=show,
    )
    plot_window(
        cons,
        *DEFAULT_WINDOWS["hiver_2009_2010"],
        title="Hiver (dec 2009 - fev 2010)",
        ylabel="kW",
        outfile=results_dir / "eda_hiver.png",
        show=show,
    )
    plot_window(
        cons,
        *DEFAULT_WINDOWS["ete_2010"],
        title="Ete (juil-aout 2010)",
        ylabel="kW",
        outfile=results_dir / "eda_ete.png",
        show=show,
    )

    plot_hourly_profiles(
        cons,
        outfile_hourly=results_dir / "eda_profil_horaire.png",
        outfile_weekend=results_dir / "eda_profil_weekend.png",
        show=show,
    )

    plot_monthly(
        cons,
        title="Consommation moyenne mensuelle (kW)",
        ylabel="kW",
        outfile=results_dir / "eda_mensuel.png",
        show=show,
    )

    # Optional weather EDA
    if weather_path and Path(weather_path).exists():
        temp = load_opsd_weather(str(weather_path))
        temp = temp.rename("temperature_degC")
        plot_global_daily(
            temp,
            title="Température moyenne quotidienne (°C)",
            ylabel="°C",
            outfile=results_dir / "eda_weather_daily.png",
            show=show,
            color="tab:red",
        )
        plot_window(
            temp,
            *DEFAULT_WINDOWS["semaine_jan_2010"],
            title="Température - semaine (janvier 2010)",
            ylabel="°C",
            outfile=results_dir / "eda_weather_semaine.png",
            show=show,
            color="tab:red",
        )
        plot_window(
            temp,
            *DEFAULT_WINDOWS["journee_2010-01-05"],
            title="Température - journée (5 janvier 2010)",
            ylabel="°C",
            outfile=results_dir / "eda_weather_journee.png",
            show=show,
            color="tab:red",
        )
        plot_window(
            temp,
            *DEFAULT_WINDOWS["mois_jan_2010"],
            title="Température - mois (janvier 2010)",
            ylabel="°C",
            outfile=results_dir / "eda_weather_mois.png",
            show=show,
            color="tab:red",
        )
        plot_window(
            temp,
            *DEFAULT_WINDOWS["hiver_2009_2010"],
            title="Température - hiver (dec 2009 - fev 2010)",
            ylabel="°C",
            outfile=results_dir / "eda_weather_hiver.png",
            show=show,
            color="tab:red",
        )
        plot_window(
            temp,
            *DEFAULT_WINDOWS["ete_2010"],
            title="Température - été (juil-aout 2010)",
            ylabel="°C",
            outfile=results_dir / "eda_weather_ete.png",
            show=show,
            color="tab:red",
        )
        plot_monthly(
            temp,
            title="Température moyenne mensuelle (°C)",
            ylabel="°C",
            outfile=results_dir / "eda_weather_mensuel.png",
            show=show,
            color="tab:red",
        )


if __name__ == "__main__":
    run_eda()
