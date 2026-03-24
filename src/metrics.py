from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


TARGET_COLS = ["planet_temp", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
NON_PARAM_COLS = {"public_key", "planet_ID"}
TRAINING_MEAN = np.array(
    [1203.40224666, -5.99997486, -6.50670597, -5.99946094, -4.49307449, -6.49032295],
    dtype=np.float64,
)
TRAINING_STD = np.array(
    [683.34122277, 1.73346792, 1.44476115, 1.74095922, 0.86326402, 1.44037952],
    dtype=np.float64,
)


def _score_split(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray) -> dict[str, float | np.ndarray]:
    z = (y_true - mu) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))

    z_ref = y_true
    crps_ref = z_ref * (2 * norm.cdf(z_ref) - 1) + 2 * norm.pdf(z_ref) - 1.0 / np.sqrt(np.pi)

    crps_per_param = crps.mean(axis=0)
    crps_ref_per_param = crps_ref.mean(axis=0)
    score_per_param = 1.0 - crps_per_param / crps_ref_per_param

    return {
        "score": float(score_per_param.mean()),
        "mean_crps": float(crps.mean()),
        "crps_per_param": crps_per_param,
        "score_per_param": score_per_param,
    }


def _target_index(target_cols: list[str] | None, n_cols: int) -> np.ndarray:
    if target_cols is None:
        if n_cols != len(TARGET_COLS):
            raise ValueError(f"target_cols is required for arrays with {n_cols} columns")
        return np.arange(len(TARGET_COLS), dtype=np.int64)
    return np.array([TARGET_COLS.index(col) for col in target_cols], dtype=np.int64)


def normalize_targets(values: np.ndarray, target_cols: list[str] | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    idx = _target_index(target_cols, values.shape[1])
    return (values - TRAINING_MEAN[idx]) / TRAINING_STD[idx]


def normalize_std(values: np.ndarray, target_cols: list[str] | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    idx = _target_index(target_cols, values.shape[1])
    return values / TRAINING_STD[idx]


def score_numpy(
    y_true: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    target_cols: list[str] | None = None,
) -> dict[str, float | np.ndarray]:
    y_true = normalize_targets(y_true, target_cols=target_cols)
    mu = normalize_targets(mu, target_cols=target_cols)
    std = normalize_std(std, target_cols=target_cols)

    if not (y_true.shape == mu.shape == std.shape):
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, mu={mu.shape}, std={std.shape}")
    if np.any(std <= 0):
        raise ValueError(f"All std values must be strictly positive. Found {int(np.sum(std <= 0))} non-positive entries.")

    return _score_split(y_true, mu, std)


def compute_participant_score(y_true: pd.DataFrame, mu: pd.DataFrame, std: pd.DataFrame) -> dict[str, float | np.ndarray]:
    param_cols = [col for col in y_true.columns if col not in NON_PARAM_COLS]
    return score_numpy(
        y_true[param_cols].to_numpy(dtype=np.float64),
        mu[param_cols].to_numpy(dtype=np.float64),
        std[param_cols].to_numpy(dtype=np.float64),
        target_cols=param_cols,
    )


def array_to_submission(arr: np.ndarray, planet_ids=None) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(TARGET_COLS):
        raise ValueError(f"Expected shape (N, {len(TARGET_COLS)}), got {arr.shape}.")
    if planet_ids is None:
        planet_ids = np.arange(arr.shape[0], dtype=np.int64)

    frame = pd.DataFrame(arr, columns=TARGET_COLS)
    frame.insert(0, "planet_ID", planet_ids)
    return frame
