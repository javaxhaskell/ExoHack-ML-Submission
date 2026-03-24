from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from collections.abc import Callable


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def evaluate_single_target_cv(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: pd.DataFrame,
    fold_assignment: pd.DataFrame,
    target_cols: list[str],
    model_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    folds = fold_assignment["fold"].to_numpy(dtype=np.int64)
    rows: list[dict[str, float | str]] = []

    for target in target_cols:
        y_values = y[target].to_numpy(dtype=np.float64)
        oof = np.zeros(len(y_values), dtype=np.float64)
        for fold in np.unique(folds):
            val_mask = folds == fold
            train_mask = ~val_mask
            model = model_factory()
            model.fit(X[train_mask], y_values[train_mask])
            oof[val_mask] = model.predict(X[val_mask])

        rows.append(
            {
                "model": model_name,
                "scheme": scheme_name,
                "target": target,
                "r2": float(r2_score(y_values, oof)),
                "rmse": _rmse(y_values, oof),
                "mae": float(mean_absolute_error(y_values, oof)),
            }
        )

    return pd.DataFrame(rows)


def evaluate_multioutput_cv(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: pd.DataFrame,
    fold_assignment: pd.DataFrame,
    target_cols: list[str],
    model_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    folds = fold_assignment["fold"].to_numpy(dtype=np.int64)
    y_values = y[target_cols].to_numpy(dtype=np.float64)
    oof = np.zeros_like(y_values)

    for fold in np.unique(folds):
        val_mask = folds == fold
        train_mask = ~val_mask
        model = model_factory()
        model.fit(X[train_mask], y_values[train_mask])
        oof[val_mask] = model.predict(X[val_mask])

    rows = []
    for target_idx, target in enumerate(target_cols):
        rows.append(
            {
                "model": model_name,
                "scheme": scheme_name,
                "target": target,
                "r2": float(r2_score(y_values[:, target_idx], oof[:, target_idx])),
                "rmse": _rmse(y_values[:, target_idx], oof[:, target_idx]),
                "mae": float(mean_absolute_error(y_values[:, target_idx], oof[:, target_idx])),
            }
        )

    return pd.DataFrame(rows)


def evaluate_single_target_holdout(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    target_cols: list[str],
    model_name: str,
    scheme_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for target in target_cols:
        y_values = y[target].to_numpy(dtype=np.float64)
        model = model_factory()
        model.fit(X[train_mask], y_values[train_mask])
        pred = model.predict(X[val_mask])
        truth = y_values[val_mask]
        rows.append(
            {
                "model": model_name,
                "scheme": scheme_name,
                "target": target,
                "r2": float(r2_score(truth, pred)),
                "rmse": _rmse(truth, pred),
                "mae": float(mean_absolute_error(truth, pred)),
            }
        )

    return pd.DataFrame(rows)
