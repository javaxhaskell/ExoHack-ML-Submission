from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loading import SUPPLEMENTARY_COLS, TARGET_COLS, SpectralData


EPS_SPECTRUM = 1e-8
EPS_NOISE = 1e-12


@dataclass(frozen=True)
class FeatureBundle:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_groups: dict[str, list[str]]
    wavelength: np.ndarray
    width: np.ndarray
    target_cols: list[str]


def spectrum_column_names(prefix: str = "spectrum_bin") -> list[str]:
    return [f"{prefix}_{idx:02d}" for idx in range(52)]


def noise_column_names(prefix: str = "noise_bin") -> list[str]:
    return [f"{prefix}_{idx:02d}" for idx in range(52)]


def flatten_spectral_data(spectral_data: SpectralData) -> pd.DataFrame:
    spectrum_frame = pd.DataFrame(spectral_data.spectrum, columns=spectrum_column_names())
    noise_frame = pd.DataFrame(spectral_data.noise, columns=noise_column_names())
    return pd.concat(
        [pd.DataFrame({"planet_ID": spectral_data.planet_ids}), spectrum_frame, noise_frame],
        axis=1,
    )


def build_flat_training_table(
    supplementary: pd.DataFrame,
    targets: pd.DataFrame,
    spectral_data: SpectralData,
) -> pd.DataFrame:
    flat_spectra = flatten_spectral_data(spectral_data)
    merged = supplementary.merge(flat_spectra, on="planet_ID", how="inner", validate="one_to_one")
    merged = merged.merge(targets, on="planet_ID", how="inner", validate="one_to_one")
    return merged.sort_values("planet_ID").reset_index(drop=True)


def build_flat_test_table(supplementary: pd.DataFrame, spectral_data: SpectralData) -> pd.DataFrame:
    flat_spectra = flatten_spectral_data(spectral_data)
    merged = supplementary.merge(flat_spectra, on="planet_ID", how="inner", validate="one_to_one")
    return merged.sort_values("planet_ID").reset_index(drop=True)


def log10_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    return np.log10(df[SUPPLEMENTARY_COLS].copy())


def log10_spectrum_features(spectral_data: SpectralData) -> np.ndarray:
    return np.log10(spectral_data.spectrum + EPS_SPECTRUM)


def log10_noise_features(spectral_data: SpectralData) -> np.ndarray:
    return np.log10(spectral_data.noise + EPS_NOISE)


def signal_to_noise_matrix(spectral_data: SpectralData) -> np.ndarray:
    return spectral_data.spectrum / spectral_data.noise


def sorted_spectral_arrays(spectral_data: SpectralData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(spectral_data.wavelength)
    wavelength = spectral_data.wavelength[order]
    width = spectral_data.width[order]
    spectrum = spectral_data.spectrum[:, order]
    noise = spectral_data.noise[:, order]
    return wavelength, width, spectrum, noise


def spectral_summary_features(spectral_data: SpectralData) -> pd.DataFrame:
    wavelength, _, spectrum, noise = sorted_spectral_arrays(spectral_data)
    snr = spectrum / noise
    log_spectrum = np.log10(spectrum + EPS_SPECTRUM)
    log_noise = np.log10(noise + EPS_NOISE)
    log_snr = np.log10(snr)

    n_bins = wavelength.shape[0]
    band_edges = [0, n_bins // 3, 2 * n_bins // 3, n_bins]

    short_slice = slice(band_edges[0], band_edges[1])
    mid_slice = slice(band_edges[1], band_edges[2])
    long_slice = slice(band_edges[2], band_edges[3])

    short_mean = log_spectrum[:, short_slice].mean(axis=1)
    mid_mean = log_spectrum[:, mid_slice].mean(axis=1)
    long_mean = log_spectrum[:, long_slice].mean(axis=1)

    summary = pd.DataFrame(
        {
            "planet_ID": spectral_data.planet_ids,
            "log_spec_mean": log_spectrum.mean(axis=1),
            "log_spec_std": log_spectrum.std(axis=1),
            "band_short_mean": short_mean,
            "band_mid_mean": mid_mean,
            "band_long_mean": long_mean,
            "spectral_slope": long_mean - short_mean,
            "band_contrast": mid_mean - 0.5 * (short_mean + long_mean),
            "log_noise_mean": log_noise.mean(axis=1),
            "log_noise_std": log_noise.std(axis=1),
            "noise_slope": log_noise[:, long_slice].mean(axis=1) - log_noise[:, short_slice].mean(axis=1),
            "log_snr_mean": log_snr.mean(axis=1),
            "log_snr_std": log_snr.std(axis=1),
        }
    )
    return summary.sort_values("planet_ID").reset_index(drop=True)


def regime_feature_table(supplementary: pd.DataFrame, spectral_data: SpectralData) -> pd.DataFrame:
    metadata = supplementary[["planet_ID"] + SUPPLEMENTARY_COLS].copy()
    for col in SUPPLEMENTARY_COLS:
        metadata[f"log10_{col}"] = np.log10(metadata[col])
    summary = spectral_summary_features(spectral_data)
    merged = metadata.merge(summary, on="planet_ID", how="inner", validate="one_to_one")
    return merged.sort_values("planet_ID").reset_index(drop=True)


def _phase2_spectral_frame(spectral_data: SpectralData) -> pd.DataFrame:
    wavelength, width, spectrum, noise = sorted_spectral_arrays(spectral_data)
    log_spectrum = np.log10(spectrum + EPS_SPECTRUM)
    log_noise = np.log10(noise + EPS_NOISE)
    log_snr = log_spectrum - log_noise
    dlog_spectrum = np.diff(log_spectrum, axis=1)
    dlog_noise = np.diff(log_noise, axis=1)

    spectral_cols = {
        **{f"logspec_{idx:02d}": log_spectrum[:, idx] for idx in range(log_spectrum.shape[1])},
        **{f"lognoise_{idx:02d}": log_noise[:, idx] for idx in range(log_noise.shape[1])},
        **{f"logsnr_{idx:02d}": log_snr[:, idx] for idx in range(log_snr.shape[1])},
        **{f"dlogspec_{idx:02d}": dlog_spectrum[:, idx] for idx in range(dlog_spectrum.shape[1])},
        **{f"dlognoise_{idx:02d}": dlog_noise[:, idx] for idx in range(dlog_noise.shape[1])},
    }
    frame = pd.concat(
        [pd.DataFrame({"planet_ID": spectral_data.planet_ids}), pd.DataFrame(spectral_cols)],
        axis=1,
    )
    summary = spectral_summary_features(spectral_data)
    frame = frame.merge(summary, on="planet_ID", how="inner", validate="one_to_one")
    return frame.sort_values("planet_ID").reset_index(drop=True)


def _phase2_metadata_frame(supplementary: pd.DataFrame) -> pd.DataFrame:
    frame = supplementary[["planet_ID"]].copy()
    for col in SUPPLEMENTARY_COLS:
        frame[f"meta_log_{col}"] = np.log10(supplementary[col])

    frame["meta_eqtemp_proxy"] = (
        np.log10(supplementary["star_temperature"])
        + 0.5 * np.log10(supplementary["star_radius_m"])
        - 0.5 * np.log10(supplementary["planet_distance"])
    )
    frame["meta_flux_proxy"] = (
        4.0 * np.log10(supplementary["star_temperature"])
        + 2.0 * np.log10(supplementary["star_radius_m"])
        - 2.0 * np.log10(supplementary["planet_distance"])
    )
    frame["meta_density_proxy"] = np.log10(supplementary["planet_mass_kg"]) - 3.0 * np.log10(
        supplementary["planet_radius_m"]
    )
    return frame.sort_values("planet_ID").reset_index(drop=True)


def _phase2_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    meta_cols = [col for col in frame.columns if col.startswith("meta_")]
    summary_cols = [
        "log_spec_mean",
        "log_spec_std",
        "band_short_mean",
        "band_mid_mean",
        "band_long_mean",
        "spectral_slope",
        "band_contrast",
        "log_noise_mean",
        "log_noise_std",
        "noise_slope",
        "log_snr_mean",
        "log_snr_std",
    ]
    spectrum_cols = [col for col in frame.columns if col.startswith("logspec_")]
    noise_cols = [col for col in frame.columns if col.startswith("lognoise_")]
    snr_cols = [col for col in frame.columns if col.startswith("logsnr_")]
    deriv_cols = [col for col in frame.columns if col.startswith("dlogspec_") or col.startswith("dlognoise_")]

    return {
        "metadata": meta_cols,
        "spectral_summary": summary_cols,
        "spectrum": spectrum_cols,
        "noise": noise_cols,
        "snr": snr_cols,
        "derivatives": deriv_cols,
        "spectrum_noise": spectrum_cols + noise_cols,
        "spectrum_noise_metadata": spectrum_cols + noise_cols + meta_cols,
        "spectrum_noise_metadata_derivatives": spectrum_cols + noise_cols + meta_cols + deriv_cols,
        "metadata_summaries": meta_cols + summary_cols,
        "summary_only": summary_cols,
        "spectrum_noise_summaries_metadata": spectrum_cols + noise_cols + summary_cols + meta_cols,
    }


def build_phase2_feature_bundle(
    train_supplementary: pd.DataFrame,
    train_targets: pd.DataFrame,
    train_spectral_data: SpectralData,
    test_supplementary: pd.DataFrame,
    test_spectral_data: SpectralData,
) -> FeatureBundle:
    train_meta = _phase2_metadata_frame(train_supplementary)
    test_meta = _phase2_metadata_frame(test_supplementary)
    train_spec = _phase2_spectral_frame(train_spectral_data)
    test_spec = _phase2_spectral_frame(test_spectral_data)

    train_frame = train_meta.merge(train_spec, on="planet_ID", how="inner", validate="one_to_one")
    train_frame = train_frame.merge(train_targets, on="planet_ID", how="inner", validate="one_to_one")
    train_frame = train_frame.sort_values("planet_ID").reset_index(drop=True)

    test_frame = test_meta.merge(test_spec, on="planet_ID", how="inner", validate="one_to_one")
    test_frame = test_frame.sort_values("planet_ID").reset_index(drop=True)

    feature_groups = _phase2_feature_groups(train_frame)
    wavelength, width, _, _ = sorted_spectral_arrays(train_spectral_data)
    return FeatureBundle(
        train_frame=train_frame,
        test_frame=test_frame,
        feature_groups=feature_groups,
        wavelength=wavelength,
        width=width,
        target_cols=TARGET_COLS.copy(),
    )
