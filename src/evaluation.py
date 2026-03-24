from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, r2_score

from src.metrics import TARGET_COLS as FULL_TARGET_COLS
from src.metrics import TRAINING_MEAN, TRAINING_STD, score_numpy


@dataclass(frozen=True)
class EvaluationResult:
    aggregate: pd.DataFrame
    per_target: pd.DataFrame
    sigma_by_target: pd.DataFrame


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _gaussian_nll_normalized(y_true_norm: np.ndarray, mu_norm: np.ndarray, sigma_norm: np.ndarray) -> np.ndarray:
    z = (y_true_norm - mu_norm) / sigma_norm
    return 0.5 * np.log(2.0 * np.pi * np.square(sigma_norm)) + 0.5 * np.square(z)


def fit_plugin_constant_sigma(y_true: np.ndarray, mu: np.ndarray, target_cols: list[str]) -> np.ndarray:
    """Fit one positive constant sigma per target by minimizing Gaussian CRPS."""
    sigmas = np.zeros(len(target_cols), dtype=np.float64)
    target_idx = np.array([FULL_TARGET_COLS.index(col) for col in target_cols], dtype=np.int64)
    train_mean = TRAINING_MEAN[target_idx]
    train_std = TRAINING_STD[target_idx]
    y_norm = (y_true - train_mean) / train_std
    mu_norm = (mu - train_mean) / train_std

    for idx, _ in enumerate(target_cols):
        residual_norm = y_norm[:, idx] - mu_norm[:, idx]
        scale_guess = float(np.std(residual_norm))
        lower = max(scale_guess * 0.1, 1e-3)
        upper = max(scale_guess * 3.0, lower * 10.0)

        def objective(sigma_norm: float) -> float:
            sigma = np.full((y_norm.shape[0], 1), sigma_norm, dtype=np.float64)
            return float(
                score_numpy(
                    y_true[:, [idx]],
                    mu[:, [idx]],
                    sigma * train_std[idx],
                    target_cols=[target_cols[idx]],
                )["mean_crps"]
            )

        result = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-4})
        sigmas[idx] = float(result.x) * train_std[idx]

    return sigmas


def summarize_predictions(
    y_true: np.ndarray,
    mu: np.ndarray,
    target_cols: list[str],
    candidate_name: str,
    scheme_name: str,
    sigma_mode: str = "plugin_constant",
) -> EvaluationResult:
    if sigma_mode != "plugin_constant":
        raise ValueError(f"Unsupported sigma_mode: {sigma_mode}")

    sigma_per_target = fit_plugin_constant_sigma(y_true, mu, target_cols)
    sigma_matrix = np.repeat(sigma_per_target.reshape(1, -1), y_true.shape[0], axis=0)

    score = score_numpy(y_true, mu, sigma_matrix, target_cols=target_cols)

    target_idx = np.array([FULL_TARGET_COLS.index(col) for col in target_cols], dtype=np.int64)
    train_mean = TRAINING_MEAN[target_idx]
    train_std = TRAINING_STD[target_idx]
    y_norm = (y_true - train_mean) / train_std
    mu_norm = (mu - train_mean) / train_std
    sigma_norm = sigma_matrix / train_std
    nll = _gaussian_nll_normalized(y_norm, mu_norm, sigma_norm)

    aggregate = pd.DataFrame(
        [
            {
                "candidate": candidate_name,
                "scheme": scheme_name,
                "target_scope": ",".join(target_cols),
                "primary_metric": float(score["score"]),
                "mean_crps": float(score["mean_crps"]),
                "mean_mae": float(np.mean(np.abs(y_true - mu))),
                "mean_rmse": float(np.mean([_rmse(y_true[:, i], mu[:, i]) for i in range(mu.shape[1])])),
                "mean_r2": float(np.mean([r2_score(y_true[:, i], mu[:, i]) for i in range(mu.shape[1])])),
                "mean_nll_norm": float(nll.mean()),
                "sigma_mode": sigma_mode,
            }
        ]
    )

    per_target_rows = []
    for idx, target in enumerate(target_cols):
        per_target_rows.append(
            {
                "candidate": candidate_name,
                "scheme": scheme_name,
                "target": target,
                "primary_metric": float(score["score_per_param"][idx]),
                "crps": float(score["crps_per_param"][idx]),
                "mae": float(mean_absolute_error(y_true[:, idx], mu[:, idx])),
                "rmse": _rmse(y_true[:, idx], mu[:, idx]),
                "r2": float(r2_score(y_true[:, idx], mu[:, idx])),
                "nll_norm": float(nll[:, idx].mean()),
                "sigma_plugin": float(sigma_per_target[idx]),
            }
        )

    sigma_frame = pd.DataFrame(
        {
            "candidate": candidate_name,
            "scheme": scheme_name,
            "target": target_cols,
            "sigma_plugin": sigma_per_target,
        }
    )

    return EvaluationResult(
        aggregate=aggregate,
        per_target=pd.DataFrame(per_target_rows),
        sigma_by_target=sigma_frame,
    )
