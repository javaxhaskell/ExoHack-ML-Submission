from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loading import SUPPLEMENTARY_COLS, TARGET_COLS, SpectralData
from src.features import regime_feature_table, sorted_spectral_arrays


def _style_axis(ax) -> None:
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_target_distributions(targets: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for idx, col in enumerate(TARGET_COLS):
        ax = axes.flat[idx]
        ax.hist(targets[col], bins=50, color="#1f77b4", alpha=0.85)
        ax.set_title(col)
        _style_axis(ax)
    _save(fig, Path(output_path))


def plot_target_correlation(targets: pd.DataFrame, output_path: str | Path) -> None:
    corr = targets[TARGET_COLS].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, vmin=-0.2, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(TARGET_COLS)), TARGET_COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(TARGET_COLS)), TARGET_COLS)
    for row in range(len(TARGET_COLS)):
        for col in range(len(TARGET_COLS)):
            ax.text(col, row, f"{corr[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    _style_axis(ax)
    _save(fig, Path(output_path))


def plot_metadata_distributions(supplementary: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for idx, col in enumerate(SUPPLEMENTARY_COLS):
        ax = axes.flat[idx]
        values = np.log10(supplementary[col])
        ax.hist(values, bins=50, color="#2ca02c", alpha=0.85)
        ax.set_title(f"log10({col})")
        _style_axis(ax)
    _save(fig, Path(output_path))


def plot_metadata_target_relationships(train_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    relationships = [
        ("planet_orbital_period", "planet_temp"),
        ("planet_distance", "planet_temp"),
        ("star_radius_m", "planet_temp"),
        ("planet_mass_kg", "log_H2O"),
        ("planet_orbital_period", "log_CO2"),
        ("star_radius_m", "log_NH3"),
    ]

    sample = train_df.sample(n=min(len(train_df), 8000), random_state=42).copy()

    for ax, (feature, target) in zip(axes.flat, relationships):
        x = np.log10(sample[feature])
        y = sample[target]
        corr = sample[[feature, target]].corr(method="spearman").iloc[0, 1]
        ax.scatter(x, y, s=3, alpha=0.18, color="#1f77b4")
        ax.set_xlabel(f"log10({feature})")
        ax.set_ylabel(target)
        ax.set_title(f"Spearman={corr:.3f}")
        _style_axis(ax)

    _save(fig, Path(output_path))


def plot_spectrum_noise_summary(spectral_data: SpectralData, output_path: str | Path) -> None:
    wavelength, _, spectrum, noise = sorted_spectral_arrays(spectral_data)
    spec_p10, spec_p50, spec_p90 = np.percentile(spectrum, [10, 50, 90], axis=0)
    noise_p10, noise_p50, noise_p90 = np.percentile(noise, [10, 50, 90], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].fill_between(wavelength, spec_p10, spec_p90, alpha=0.25, color="#1f77b4")
    axes[0].plot(wavelength, spec_p50, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Spectrum")
    axes[0].set_yscale("log")
    axes[0].set_title("Spectrum median and 10-90 percentile band")
    _style_axis(axes[0])

    axes[1].fill_between(wavelength, noise_p10, noise_p90, alpha=0.25, color="#d62728")
    axes[1].plot(wavelength, noise_p50, color="#d62728", linewidth=1.5)
    axes[1].set_ylabel("Noise")
    axes[1].set_xlabel("Wavelength")
    axes[1].set_yscale("log")
    axes[1].set_title("Noise median and 10-90 percentile band")
    _style_axis(axes[1])

    _save(fig, Path(output_path))


def plot_snr_diagnostics(spectral_data: SpectralData, output_path: str | Path) -> None:
    wavelength, _, spectrum, noise = sorted_spectral_arrays(spectral_data)
    snr = spectrum / noise
    snr_p10, snr_p50, snr_p90 = np.percentile(snr, [10, 50, 90], axis=0)
    mean_snr = snr.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].fill_between(wavelength, snr_p10, snr_p90, alpha=0.25, color="#9467bd")
    axes[0].plot(wavelength, snr_p50, color="#9467bd", linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_title("Per-wavelength SNR")
    axes[0].set_xlabel("Wavelength")
    axes[0].set_ylabel("SNR")
    _style_axis(axes[0])

    axes[1].hist(mean_snr, bins=60, color="#ff7f0e", alpha=0.85)
    axes[1].set_xscale("log")
    axes[1].set_title("Mean SNR per planet")
    axes[1].set_xlabel("Mean SNR")
    axes[1].set_ylabel("Count")
    _style_axis(axes[1])

    _save(fig, Path(output_path))


def plot_regime_scatter(
    supplementary: pd.DataFrame,
    spectral_data: SpectralData,
    targets: pd.DataFrame,
    regime_groups: pd.DataFrame,
    output_path: str | Path,
) -> None:
    regime_table = regime_feature_table(supplementary, spectral_data).merge(
        regime_groups[["planet_ID", "regime_group"]],
        on="planet_ID",
        how="left",
        validate="one_to_one",
    )
    regime_table = regime_table.merge(targets[["planet_ID", "planet_temp"]], on="planet_ID", how="left")
    sample = regime_table.sample(n=min(len(regime_table), 12000), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    scatter_a = axes[0].scatter(
        sample["log_spec_mean"],
        sample["spectral_slope"],
        c=sample["planet_temp"],
        s=4,
        alpha=0.35,
        cmap="viridis",
    )
    axes[0].set_xlabel("log_spec_mean")
    axes[0].set_ylabel("spectral_slope")
    axes[0].set_title("Regime plane colored by temperature")
    _style_axis(axes[0])
    fig.colorbar(scatter_a, ax=axes[0], shrink=0.8)

    axes[1].scatter(
        sample["log_spec_mean"],
        sample["spectral_slope"],
        c=sample["regime_group"],
        s=4,
        alpha=0.35,
        cmap="tab20",
    )
    axes[1].set_xlabel("log_spec_mean")
    axes[1].set_ylabel("spectral_slope")
    axes[1].set_title("Regime plane colored by grouped split bins")
    _style_axis(axes[1])

    _save(fig, Path(output_path))


def plot_representative_regime_spectra(
    supplementary: pd.DataFrame,
    spectral_data: SpectralData,
    targets: pd.DataFrame,
    regime_groups: pd.DataFrame,
    output_path: str | Path,
    n_regimes: int = 6,
) -> None:
    wavelength, _, spectrum, _ = sorted_spectral_arrays(spectral_data)
    order = np.argsort(spectral_data.wavelength)

    regime_table = regime_feature_table(supplementary, spectral_data).merge(
        regime_groups[["planet_ID", "regime_group"]],
        on="planet_ID",
        how="left",
        validate="one_to_one",
    )
    regime_table = regime_table.merge(targets[["planet_ID", "planet_temp"]], on="planet_ID", how="left")

    group_counts = regime_table["regime_group"].value_counts().head(n_regimes)
    fig, ax = plt.subplots(figsize=(10, 5))

    for regime_group in group_counts.index:
        mask = regime_table["regime_group"].to_numpy() == regime_group
        representative = np.median(spectral_data.spectrum[mask][:, order], axis=0)
        label = f"group {int(regime_group)} | n={int(mask.sum())} | temp={regime_table.loc[mask, 'planet_temp'].mean():.0f}K"
        ax.plot(wavelength, representative, linewidth=1.2, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Median spectrum")
    ax.set_title("Representative spectra from the largest regime groups")
    ax.legend(fontsize=7, ncol=2)
    _style_axis(ax)
    _save(fig, Path(output_path))
