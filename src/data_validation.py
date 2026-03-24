from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

from src.data_loading import (
    SUPPLEMENTARY_COLS,
    TARGET_COLS,
    load_test_data,
    load_training_data,
    merge_training_tables,
    parse_planet_id,
)


def assert_unique_no_missing(df, name: str) -> None:
    if "planet_ID" not in df.columns:
        raise ValueError(f"{name} is missing planet_ID")
    if not df["planet_ID"].is_unique:
        raise ValueError(f"{name} contains duplicated planet_ID values")
    if int(df.isna().sum().sum()) != 0:
        raise ValueError(f"{name} contains missing values")


def assert_id_alignment(expected_ids: np.ndarray, observed_ids: np.ndarray, name: str) -> None:
    if expected_ids.shape != observed_ids.shape:
        raise ValueError(f"{name} shape mismatch: {expected_ids.shape} vs {observed_ids.shape}")
    if not np.array_equal(expected_ids, observed_ids):
        raise ValueError(f"{name} planet_ID ordering mismatch")


def raw_hdf5_key_order_matches_numeric_sort(hdf5_path: str | Path) -> bool:
    with h5py.File(hdf5_path, "r") as handle:
        raw_keys = list(handle.keys())
    raw_ids = np.array([parse_planet_id(key) for key in raw_keys], dtype=np.int64)
    return bool(np.all(raw_ids[:-1] <= raw_ids[1:]))


def top_abs_spearman_correlations(train_df, top_k: int = 5) -> dict[str, list[dict[str, float | str]]]:
    results: dict[str, list[dict[str, float | str]]] = {}
    for target in TARGET_COLS:
        scored = []
        for feature in SUPPLEMENTARY_COLS:
            corr = float(spearmanr(train_df[feature], train_df[target]).statistic)
            scored.append(
                {
                    "feature": feature,
                    "spearman": corr,
                    "abs_spearman": abs(corr),
                }
            )
        results[target] = sorted(scored, key=lambda item: item["abs_spearman"], reverse=True)[:top_k]
    return results


def run_phase1_audit(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    train_supp, train_targets, train_spectra = load_training_data(data_dir)
    test_supp, test_spectra = load_test_data(data_dir)
    train_df = merge_training_tables(train_supp, train_targets)

    assert_unique_no_missing(train_supp, "training supplementary data")
    assert_unique_no_missing(train_targets, "training targets")
    assert_unique_no_missing(test_supp, "test supplementary data")
    assert_id_alignment(train_targets["planet_ID"].to_numpy(), train_supp["planet_ID"].to_numpy(), "training CSV alignment")
    assert_id_alignment(train_targets["planet_ID"].to_numpy(), train_spectra.planet_ids, "training spectral alignment")
    assert_id_alignment(test_supp["planet_ID"].to_numpy(), test_spectra.planet_ids, "test spectral alignment")

    combined_ids = np.concatenate(
        [train_targets["planet_ID"].to_numpy(dtype=np.int64), test_supp["planet_ID"].to_numpy(dtype=np.int64)]
    )

    target_summary = (
        train_targets[TARGET_COLS]
        .describe()
        .loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        .T
        .round(6)
        .to_dict(orient="index")
    )

    return {
        "train_rows": int(len(train_targets)),
        "test_rows": int(len(test_supp)),
        "n_total_planets": int(len(combined_ids)),
        "global_planet_id_min": int(combined_ids.min()),
        "global_planet_id_max": int(combined_ids.max()),
        "complete_contiguous_id_range": bool(
            np.array_equal(np.sort(combined_ids), np.arange(combined_ids.min(), combined_ids.max() + 1))
        ),
        "train_test_overlap_count": int(
            np.intersect1d(
                train_targets["planet_ID"].to_numpy(dtype=np.int64),
                test_supp["planet_ID"].to_numpy(dtype=np.int64),
            ).size
        ),
        "spectrum_shape_train": tuple(int(v) for v in train_spectra.spectrum.shape),
        "noise_shape_train": tuple(int(v) for v in train_spectra.noise.shape),
        "wavelength_bins": int(train_spectra.wavelength.shape[0]),
        "shared_wavelength_grid_train_test": bool(
            np.allclose(train_spectra.wavelength, test_spectra.wavelength)
            and np.allclose(train_spectra.width, test_spectra.width)
        ),
        "raw_hdf5_iteration_is_numeric_train": raw_hdf5_key_order_matches_numeric_sort(
            data_dir / "Training_SpectralData.hdf5"
        ),
        "raw_hdf5_iteration_is_numeric_test": raw_hdf5_key_order_matches_numeric_sort(
            data_dir / "Test_SpectralData.hdf5"
        ),
        "target_summary": target_summary,
        "top_metadata_correlations": top_abs_spearman_correlations(train_df),
    }
