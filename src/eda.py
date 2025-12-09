"""Exploratory data analysis focused on simple consumption time series (monde entier, plage auto)."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Allow running this module directly from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Preprocessing helpers are unused here; kept minimal dependencies.

plt.style.use("seaborn-v0_8-darkgrid")

__all__ = [
    "run_eda",
    "plot_series",
    "plot_window",
    "plot_global_daily",
    "plot_weekday_weekend_hourly",
    "plot_daily_by_weekday",
    "plot_monthly_profile",
    "plot_dayofyear_profile",
    "plot_rolling_mean",
    "plot_week_heatmap",
]


def _finalize(fig: plt.Figure, outfile: Path | None, show: bool) -> None:
    fig.tight_layout()
    if outfile:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_series(
    series: pd.Series,
    title: str,
    ylabel: str,
    outfile: Path | None = None,
    show: bool = True,
    color: str = "tab:blue",
    max_points: int | None = None,
) -> None:
    if max_points is not None and len(series) > max_points:
        series = series.iloc[-max_points:]
    fig, ax = plt.subplots(figsize=(12, 4))
    series.plot(ax=ax, color=color, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def plot_global_daily(
    series: pd.Series,
    title: str,
    ylabel: str,
    outfile: Path | None = None,
    show: bool = True,
    resample_rule: str = "D",
    color: str = "tab:blue",
) -> None:
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
    window = series.loc[start:end]
    fig, ax = plt.subplots(figsize=(12, 4))
    window.plot(ax=ax, color=color, linewidth=1)
    ax.set_title(f"{title} ({start} -> {end})")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def plot_weekday_weekend_hourly(
    series: pd.Series,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    tmp = series.to_frame("load_mw")
    tmp["hour"] = tmp.index.hour
    tmp["is_weekend"] = tmp.index.weekday >= 5
    pivot = tmp.groupby(["is_weekend", "hour"]).mean().unstack(0)["load_mw"]
    pivot = pivot.rename(columns={False: "Semaine", True: "Weekend"})

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(ax=ax)
    ax.set_title("Profil horaire : semaine vs weekend")
    ax.set_xlabel("Heure")
    ax.set_ylabel("MW")
    _finalize(fig, outfile, show)


def plot_daily_by_weekday(
    series: pd.Series,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    daily = series.resample("D").mean()
    pivot = daily.to_frame("load_mw")
    pivot["dayofweek"] = pivot.index.dayofweek
    pivot["is_weekend"] = pivot["dayofweek"] >= 5
    by_dow = pivot.groupby("dayofweek")["load_mw"].mean()
    labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    fig, ax = plt.subplots(figsize=(8, 4))
    by_dow.plot(kind="bar", ax=ax, color=["tab:blue"] * 5 + ["tab:orange", "tab:orange"])
    ax.set_title("Consommation moyenne quotidienne par jour de semaine")
    ax.set_xlabel("Jour")
    ax.set_ylabel("MW")
    ax.set_xticklabels(labels, rotation=0)
    _finalize(fig, outfile, show)


def plot_monthly(
    series: pd.Series,
    title: str,
    ylabel: str,
    outfile: Path | None = None,
    show: bool = True,
    color: str = "tab:green",
) -> None:
    monthly = series.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, color=color, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def plot_monthly_profile(
    series: pd.Series,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    monthly = series.resample("M").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, color="tab:green", linewidth=1)
    ax.set_title("Profil mensuel moyen (charges mensuelles)")
    ax.set_xlabel("")
    ax.set_ylabel("MW")
    _finalize(fig, outfile, show)


def plot_dayofyear_profile(
    series: pd.Series,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    daily = series.resample("D").mean().to_frame("load_mw")
    daily["doy"] = daily.index.dayofyear
    profile = daily.groupby("doy")["load_mw"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    profile.plot(ax=ax, color="tab:purple", linewidth=1)
    ax.set_title("Profil moyen par jour de l'année")
    ax.set_xlabel("Jour de l'année")
    ax.set_ylabel("MW")
    _finalize(fig, outfile, show)


def plot_rolling_mean(
    series: pd.Series,
    window: int,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    roll = series.resample("D").mean().rolling(window, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    roll.plot(ax=ax, color="tab:red", linewidth=1.2)
    ax.set_title(f"Moyenne mobile {window} jours")
    ax.set_xlabel("")
    ax.set_ylabel("MW")
    _finalize(fig, outfile, show)


def plot_week_heatmap(
    series: pd.Series,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    daily = series.resample("D").mean()
    weekly = daily.to_frame("load_mw")
    weekly["year"] = weekly.index.year
    weekly["week"] = weekly.index.isocalendar().week.astype(int)
    pivot = weekly.pivot_table(index="week", columns="year", values="load_mw", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(pivot, aspect="auto", cmap="coolwarm", origin="lower")
    ax.set_title("Chaleur hebdomadaire (moyenne journalière)")
    ax.set_xlabel("Année")
    ax.set_ylabel("Semaine ISO")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="MW")
    _finalize(fig, outfile, show)


def _load_global_series(data_path: Path) -> pd.Series:
    """
    Charge le fichier OPSD et agrège la charge de tous les pays (somme des colonnes *load_actual*).
    """
    # Reprend la logique de parsing de load_consumption_data pour l'horodatage
    df = pd.read_csv(data_path, sep=";", na_values="?", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(data_path, na_values="?", low_memory=False)

    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            format="%d/%m/%Y %H:%M:%S",
            dayfirst=True,
            errors="coerce",
        )
        df = df.drop(columns=["Date", "Time"])
        time_col = "datetime"
    else:
        ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
        if not ts_candidates:
            raise ValueError("Aucune colonne temporelle trouvée pour agréger la consommation globale.")
        time_col = ts_candidates[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col])
        df[time_col] = df[time_col].dt.tz_convert(None)

    df = df.set_index(time_col).sort_index()

    load_cols = [c for c in df.columns if "load_actual" in c.lower()]
    if not load_cols:
        raise ValueError("Aucune colonne 'load_actual' trouvée pour construire la charge globale.")

    load_df = df[load_cols].apply(pd.to_numeric, errors="coerce")
    global_series = load_df.sum(axis=1).resample("1H").mean().dropna()
    global_series = global_series.rename("load_mw")
    if global_series.empty:
        raise ValueError(f"Aucune donnée disponible dans {data_path}")
    return global_series


def run_eda(
    data_path: Path = Path("data/consumption_data.csv"),
    results_dir: Path = Path("results"),
    show: bool = True,
) -> None:
    """Generate simple time series plots for global consumption with automatic date range."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    cons = _load_global_series(data_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_date = cons.index.min().normalize()
    end_date = cons.index.max().normalize()
    # Archive plage temporelle utilisée pour référence rapide
    (results_dir / "eda_time_range.txt").write_text(f"start={start_date.date()}, end={end_date.date()}")

    plot_series(
        cons,
        title=f"Consommation horaire monde ({start_date.date()} -> {end_date.date()})",
        ylabel="MW",
        outfile=results_dir / "eda_series_full.png",
        show=show,
        max_points=20000,
    )
    plot_global_daily(
        cons,
        title="Consommation moyenne quotidienne - Monde",
        ylabel="MW",
        outfile=results_dir / "eda_global_daily.png",
        show=show,
    )
    plot_monthly_profile(
        cons,
        outfile=results_dir / "eda_monthly_profile.png",
        show=show,
    )
    plot_dayofyear_profile(
        cons,
        outfile=results_dir / "eda_dayofyear_profile.png",
        show=show,
    )
    plot_rolling_mean(
        cons,
        window=30,
        outfile=results_dir / "eda_rolling_30d.png",
        show=show,
    )

    # Fenêtres dynamiques sur la plage réelle
    mid_date = start_date + (end_date - start_date) / 2
    first_week = (start_date, start_date + pd.Timedelta(days=6))
    mid_week_start = (mid_date - pd.Timedelta(days=3)).normalize()
    mid_week_end = mid_week_start + pd.Timedelta(days=6)
    last_week = (end_date - pd.Timedelta(days=6), end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    plot_window(
        cons,
        *first_week,
        title="Fenêtre : première semaine disponible",
        ylabel="MW",
        outfile=results_dir / "eda_semaine_debut.png",
        show=show,
    )
    plot_window(
        cons,
        mid_week_start,
        mid_week_end,
        title="Fenêtre : semaine milieu de période",
        ylabel="MW",
        outfile=results_dir / "eda_semaine_milieu.png",
        show=show,
    )
    plot_window(
        cons,
        *last_week,
        title="Fenêtre : dernière semaine disponible",
        ylabel="MW",
        outfile=results_dir / "eda_semaine_fin.png",
        show=show,
    )
    plot_weekday_weekend_hourly(
        cons,
        outfile=results_dir / "eda_weekday_weekend_hourly.png",
        show=show,
    )
    plot_daily_by_weekday(
        cons,
        outfile=results_dir / "eda_daily_by_weekday.png",
        show=show,
    )
    plot_week_heatmap(
        cons,
        outfile=results_dir / "eda_week_heatmap.png",
        show=show,
    )

if __name__ == "__main__":
    run_eda()
