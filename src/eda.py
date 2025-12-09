"""Exploratory data analysis utilities for household power consumption."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data_processing import (
    add_lag_features,
    add_time_features,
    load_opsd_weather,
    load_uci_household,
    merge_weather_features,
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

__all__ = [
    "run_eda",
    "plot_window",
    "plot_global_daily",
    "plot_hourly_profiles",
    "plot_histogram",
    "plot_box_by",
    "compute_basic_stats",
    "plot_decomposition",
    "plot_correlation_matrix",
]


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


def compute_basic_stats(series: pd.Series, outfile: Path | None = None) -> pd.Series:
    """Compute basic statistics (mean, median, min, max, quantiles). Optionally save to CSV."""
    stats = {
        "mean": series.mean(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
    }
    quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    quantiles.index = [f"quantile_{int(q * 100)}" for q in quantiles.index]
    stats_series = pd.concat([pd.Series(stats), quantiles])
    if outfile:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        stats_series.to_csv(outfile, header=["value"])
    return stats_series


def plot_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str = "Fréquence",
    outfile: Path | None = None,
    show: bool = True,
    bins: int = 50,
    color: str = "tab:purple",
) -> None:
    """Plot histogram of the target variable."""
    fig, ax = plt.subplots(figsize=(10, 4))
    series.dropna().plot(kind="hist", bins=bins, ax=ax, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def plot_box_by(
    series: pd.Series,
    groups: List[int],
    labels: List[str] | None,
    title: str,
    ylabel: str,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    """Plot boxplots of the series grouped by a categorical/time bucket."""
    tmp = pd.DataFrame({"value": series.values, "group": groups})
    unique_groups = sorted(pd.unique(tmp["group"]))
    data = [tmp.loc[tmp["group"] == g, "value"].dropna().values for g in unique_groups]

    if labels is None or len(labels) != len(unique_groups):
        labels = [str(g) for g in unique_groups]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _finalize(fig, outfile, show)


def plot_decomposition(
    series: pd.Series,
    period: int,
    title_prefix: str,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    """
    Seasonal decomposition (trend + seasonal + resid).

    Args:
        series: time series (DatetimeIndex).
        period: seasonal period (e.g., 24 for daily seasonality on hourly data).
        title_prefix: base title for the plot.
    """
    decomp = seasonal_decompose(series.dropna(), model="additive", period=period, extrapolate_trend="freq")
    fig = decomp.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"{title_prefix} (période={period})", fontsize=14)
    _finalize(fig, outfile, show)


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
    """Plot a correlation matrix for selected columns."""
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title(title)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    # Annotate values
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
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
    hourly_feat = add_time_features(hourly)
    hourly_feat = add_lag_features(
        hourly_feat,
        target_col="Global_active_power",
        lags=(),
        rolling_windows=(24,),
        dropna=True,
    )
    cons = hourly_feat["Global_active_power"].rename("load_kw")

    results_dir.mkdir(parents=True, exist_ok=True)

    # Basic stats and distributions
    compute_basic_stats(cons, results_dir / "eda_stats_consumption.csv")
    plot_histogram(
        cons,
        title="Distribution de la consommation (kW)",
        xlabel="kW",
        outfile=results_dir / "eda_hist_consumption.png",
        show=show,
    )
    plot_box_by(
        cons,
        groups=cons.index.hour.tolist(),
        labels=[str(h) for h in range(24)],
        title="Boxplot consommation par heure",
        ylabel="kW",
        outfile=results_dir / "eda_box_hour.png",
        show=show,
    )
    plot_box_by(
        cons,
        groups=cons.index.dayofweek.tolist(),
        labels=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
        title="Boxplot consommation par jour de semaine",
        ylabel="kW",
        outfile=results_dir / "eda_box_dow.png",
        show=show,
    )
    plot_box_by(
        cons,
        groups=cons.index.month.tolist(),
        labels=[str(m) for m in range(1, 13)],
        title="Boxplot consommation par mois",
        ylabel="kW",
        outfile=results_dir / "eda_box_month.png",
        show=show,
    )

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
    # Decompositions: daily seasonality (24h) and yearly (24*365 approx)
    plot_decomposition(
        cons,
        period=24,
        title_prefix="Décomposition quotidienne (saisonnalité 24h)",
        outfile=results_dir / "eda_decomp_daily.png",
        show=show,
    )
    plot_decomposition(
        cons.resample("D").mean(),
        period=365,
        title_prefix="Décomposition annuelle (saisonnalité 365j)",
        outfile=results_dir / "eda_decomp_yearly.png",
        show=show,
    )

    # Optional weather EDA and merge
    if weather_path and Path(weather_path).exists():
        temp = load_opsd_weather(str(weather_path))
        temp = temp.rename("temperature_degC")
        hourly_feat = merge_weather_features(hourly_feat, temp)
        compute_basic_stats(temp, results_dir / "eda_stats_weather.csv")
        plot_histogram(
            temp,
            title="Distribution de la température (°C)",
            xlabel="°C",
            outfile=results_dir / "eda_hist_weather.png",
            show=show,
            color="tab:red",
        )
        plot_box_by(
            temp,
            groups=temp.index.hour.tolist(),
            labels=[str(h) for h in range(24)],
            title="Boxplot température par heure",
            ylabel="°C",
            outfile=results_dir / "eda_weather_box_hour.png",
            show=show,
        )
        plot_box_by(
            temp,
            groups=temp.index.dayofweek.tolist(),
            labels=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
            title="Boxplot température par jour de semaine",
            ylabel="°C",
            outfile=results_dir / "eda_weather_box_dow.png",
            show=show,
        )
        plot_box_by(
            temp,
            groups=temp.index.month.tolist(),
            labels=[str(m) for m in range(1, 13)],
            title="Boxplot température par mois",
            ylabel="°C",
            outfile=results_dir / "eda_weather_box_month.png",
            show=show,
        )
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
        plot_decomposition(
            temp,
            period=24,
            title_prefix="Décomposition température (24h)",
            outfile=results_dir / "eda_weather_decomp_daily.png",
            show=show,
        )
    else:
        temp = None

    # Correlation matrix (cons, calendar, rolling, temp if available)
    corr_cols = [
        "Global_active_power",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "Global_active_power_rollmean_24h",
    ]
    if temp is not None and "temperature_degC" in hourly_feat.columns:
        corr_cols.append("temperature_degC")
    corr_cols = [c for c in corr_cols if c in hourly_feat.columns]
    if corr_cols:
        corr_df = hourly_feat[corr_cols].dropna()
        plot_correlation_matrix(
            corr_df,
            corr_cols,
            title="Matrice de corrélation (consommation + calendrier + météo)",
            outfile=results_dir / "eda_corr_matrix.png",
            show=show,
        )
        corr_df.corr().to_csv(results_dir / "eda_corr_matrix.csv")


if __name__ == "__main__":
    run_eda()
