from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src.features import regime_feature_table


def _qcut_codes(values: pd.Series | np.ndarray, n_bins: int) -> np.ndarray:
    series = pd.Series(values)
    codes = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    return codes.to_numpy(dtype=np.int64)


def make_random_folds(planet_ids: np.ndarray, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    fold_assignment = np.full(len(planet_ids), -1, dtype=np.int64)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy = np.zeros(len(planet_ids), dtype=np.int8)
    for fold, (_, val_idx) in enumerate(splitter.split(dummy)):
        fold_assignment[val_idx] = fold

    return pd.DataFrame({"planet_ID": planet_ids, "fold": fold_assignment}).sort_values("planet_ID").reset_index(drop=True)


def make_regime_group_folds(
    supplementary: pd.DataFrame,
    spectral_data,
    n_splits: int = 5,
    n_bins: int = 4,
) -> pd.DataFrame:
    regime_table = regime_feature_table(supplementary, spectral_data)

    depth_code = _qcut_codes(regime_table["band_mid_mean"], n_bins=n_bins)
    slope_code = _qcut_codes(regime_table["spectral_slope"], n_bins=n_bins)
    orbit_code = _qcut_codes(regime_table["log10_planet_orbital_period"], n_bins=n_bins)

    group_code = depth_code * (n_bins**2) + slope_code * n_bins + orbit_code

    fold_assignment = np.full(len(regime_table), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(regime_table), dtype=np.int8)
    for fold, (_, val_idx) in enumerate(splitter.split(dummy, groups=group_code)):
        fold_assignment[val_idx] = fold

    frame = regime_table[["planet_ID"]].copy()
    frame["regime_group"] = group_code
    frame["fold"] = fold_assignment
    return frame.sort_values("planet_ID").reset_index(drop=True)


def make_edge_regime_holdout(
    supplementary: pd.DataFrame,
    spectral_data,
    holdout_fraction: float = 0.15,
) -> pd.DataFrame:
    regime_table = regime_feature_table(supplementary, spectral_data)
    score_cols = [
        "log10_star_temperature",
        "log10_planet_orbital_period",
        "log10_planet_radius_m",
        "log10_planet_surface_gravity",
        "log_spec_mean",
        "log_spec_std",
        "spectral_slope",
        "band_contrast",
        "log_noise_mean",
        "log_snr_mean",
        "log_snr_std",
    ]

    robust_z = []
    for col in score_cols:
        values = regime_table[col].to_numpy(dtype=np.float64)
        median = np.median(values)
        iqr = np.subtract(*np.percentile(values, [75, 25]))
        scale = iqr if iqr > 0 else values.std()
        robust_z.append(np.abs((values - median) / (scale + 1e-9)))

    edge_score = np.sqrt(np.mean(np.square(np.column_stack(robust_z)), axis=1))
    threshold = float(np.quantile(edge_score, 1.0 - holdout_fraction))
    is_holdout = edge_score >= threshold

    frame = regime_table[["planet_ID"]].copy()
    frame["edge_score"] = edge_score
    frame["is_holdout"] = is_holdout
    return frame.sort_values("planet_ID").reset_index(drop=True)


def summarize_fold_balance(assignments: pd.DataFrame, fold_col: str = "fold") -> pd.DataFrame:
    counts = assignments.groupby(fold_col)["planet_ID"].size().rename("n_planets")
    fractions = (counts / counts.sum()).rename("fraction")
    return pd.concat([counts, fractions], axis=1).reset_index()
