from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


TARGET_COLS = ["planet_temp", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
SUPPLEMENTARY_COLS = [
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "star_temperature",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_radius_m",
    "planet_surface_gravity",
]


@dataclass(frozen=True)
class SpectralData:
    planet_ids: np.ndarray
    spectrum: np.ndarray
    noise: np.ndarray
    wavelength: np.ndarray
    width: np.ndarray


def parse_planet_id(group_name: str) -> int:
    """Extract the integer planet id from an HDF5 group like 'Planet_21988'."""
    prefix, planet_id = group_name.rsplit("_", maxsplit=1)
    if prefix != "Planet":
        raise ValueError(f"Unexpected HDF5 group name: {group_name}")
    return int(planet_id)


def read_csv_table(csv_path: str | Path, sort_by_planet_id: bool = True) -> pd.DataFrame:
    """Read a challenge CSV and drop notebook-export index columns."""
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if sort_by_planet_id and "planet_ID" in df.columns:
        df = df.sort_values("planet_ID").reset_index(drop=True)

    return df


def load_spectral_data(hdf5_path: str | Path, sort_by_planet_id: bool = True) -> SpectralData:
    """
    Load spectra/noise arrays and preserve numeric planet ordering.

    Numeric sorting is mandatory for the public test file because raw HDF5 key
    iteration is lexicographic there: Planet_0, Planet_1, Planet_10, ...
    """
    hdf5_path = Path(hdf5_path)

    with h5py.File(hdf5_path, "r") as handle:
        keys = list(handle.keys())
        if sort_by_planet_id:
            keys = sorted(keys, key=parse_planet_id)

        first_key = keys[0]
        n_planets = len(keys)
        n_bins = handle[first_key]["instrument_spectrum"].shape[0]

        spectrum = np.zeros((n_planets, n_bins), dtype=np.float64)
        noise = np.zeros((n_planets, n_bins), dtype=np.float64)
        planet_ids = np.zeros(n_planets, dtype=np.int64)

        for row_idx, key in enumerate(keys):
            group = handle[key]
            spectrum[row_idx] = group["instrument_spectrum"][:]
            noise[row_idx] = group["instrument_noise"][:]
            planet_ids[row_idx] = parse_planet_id(key)

        wavelength = handle[first_key]["instrument_wlgrid"][:]
        width = handle[first_key]["instrument_width"][:]

    return SpectralData(
        planet_ids=planet_ids,
        spectrum=spectrum,
        noise=noise,
        wavelength=wavelength,
        width=width,
    )


def load_training_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, SpectralData]:
    """Load sorted training supplementary data, targets, and spectra."""
    data_dir = Path(data_dir)
    supplementary = read_csv_table(data_dir / "Training_supplementary_data.csv")
    targets = read_csv_table(data_dir / "Training_targets.csv")
    spectra = load_spectral_data(data_dir / "Training_SpectralData.hdf5")
    return supplementary, targets, spectra


def load_test_data(data_dir: str | Path) -> tuple[pd.DataFrame, SpectralData]:
    """Load sorted test supplementary data and spectra."""
    data_dir = Path(data_dir)
    supplementary = read_csv_table(data_dir / "Test_supplementary_data.csv")
    spectra = load_spectral_data(data_dir / "Test_SpectralData.hdf5")
    return supplementary, spectra


def merge_training_tables(supplementary: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Create a single training frame keyed by planet_ID."""
    merged = supplementary.merge(targets, on="planet_ID", how="inner", validate="one_to_one")
    return merged.sort_values("planet_ID").reset_index(drop=True)
