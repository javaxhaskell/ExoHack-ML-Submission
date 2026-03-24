from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.metrics import TRAINING_STD, TARGET_COLS, score_numpy
from src.models.builders import make_extra_trees, passthrough_columns, svd_with_side_features


EPS_SIGMA = 1e-6
EPS_RESID = 1e-8
ABS_RESID_TO_SIGMA = np.sqrt(np.pi / 2.0)
DEFAULT_COVERAGE_LEVELS = (0.5, 0.68, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class ScaleFloorModel:
    scale: float
    floor: float


@dataclass(frozen=True)
class IsotonicSigmaModel:
    isotonic: IsotonicRegression
    scale: float
    floor: float


@dataclass(frozen=True)
class BlendModel:
    linear: LinearRegression
    scale: float
    floor: float


@dataclass(frozen=True)
class HardnessInflationModel:
    gamma: float
    floor: float
    mean: float
    std: float


def _target_std(target_col: str) -> float:
    return float(TRAINING_STD[TARGET_COLS.index(target_col)])


def _positive(values: np.ndarray, floor: float = EPS_SIGMA) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), floor)


def target_crps(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    target_col: str,
) -> float:
    return float(
        score_numpy(
            y_true.reshape(-1, 1),
            mu.reshape(-1, 1),
            _positive(sigma).reshape(-1, 1),
            target_cols=[target_col],
        )["mean_crps"]
    )


def target_score(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    target_col: str,
) -> float:
    return float(
        score_numpy(
            y_true.reshape(-1, 1),
            mu.reshape(-1, 1),
            _positive(sigma).reshape(-1, 1),
            target_cols=[target_col],
        )["score"]
    )


def fit_constant_sigma(y_true: np.ndarray, mu: np.ndarray, target_col: str) -> float:
    residual = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(mu, dtype=np.float64))
    scale_guess = max(float(np.median(residual) * ABS_RESID_TO_SIGMA), EPS_SIGMA)
    lower = max(scale_guess * 0.1, EPS_SIGMA)
    upper = max(scale_guess * 5.0, lower * 10.0)

    def objective(sigma_value: float) -> float:
        sigma = np.full_like(residual, sigma_value, dtype=np.float64)
        return target_crps(np.asarray(y_true), np.asarray(mu), sigma, target_col)

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-6})
    return float(result.x)


def fit_scale_floor_from_raw_sigma(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma_raw: np.ndarray,
    target_col: str,
) -> ScaleFloorModel:
    sigma_raw = _positive(sigma_raw)
    sigma_guess = fit_constant_sigma(y_true, mu, target_col)
    spread_guess = max(float(np.median(sigma_raw)), EPS_SIGMA)
    init_scale = sigma_guess / spread_guess
    init = np.array([np.log(max(init_scale, EPS_SIGMA)), np.log(max(sigma_guess * 0.05, EPS_SIGMA))], dtype=np.float64)

    def objective(theta: np.ndarray) -> float:
        scale = float(np.exp(theta[0]))
        floor = float(np.exp(theta[1]))
        sigma = np.sqrt(np.square(scale * sigma_raw) + floor * floor)
        return target_crps(np.asarray(y_true), np.asarray(mu), sigma, target_col)

    result = minimize(objective, x0=init, method="L-BFGS-B")
    scale = float(np.exp(result.x[0]))
    floor = float(np.exp(result.x[1]))
    return ScaleFloorModel(scale=scale, floor=floor)


def apply_scale_floor(sigma_raw: np.ndarray, model: ScaleFloorModel) -> np.ndarray:
    sigma_raw = _positive(sigma_raw)
    sigma = np.sqrt(np.square(model.scale * sigma_raw) + model.floor * model.floor)
    return _positive(sigma)


def fit_isotonic_sigma_map(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma_raw: np.ndarray,
    target_col: str,
) -> IsotonicSigmaModel:
    sigma_raw = _positive(sigma_raw)
    sigma_proxy = _positive(np.abs(np.asarray(y_true) - np.asarray(mu)) * ABS_RESID_TO_SIGMA)
    floor = max(EPS_SIGMA, 1e-4 * _target_std(target_col))
    isotonic = IsotonicRegression(y_min=floor, out_of_bounds="clip")
    isotonic.fit(sigma_raw, sigma_proxy)
    sigma_iso = _positive(isotonic.predict(sigma_raw), floor=floor)
    scale_model = fit_scale_floor_from_raw_sigma(y_true, mu, sigma_iso, target_col)
    return IsotonicSigmaModel(
        isotonic=isotonic,
        scale=scale_model.scale,
        floor=scale_model.floor,
    )


def apply_isotonic_sigma_map(sigma_raw: np.ndarray, model: IsotonicSigmaModel) -> np.ndarray:
    sigma_iso = _positive(model.isotonic.predict(_positive(sigma_raw)), floor=model.floor)
    sigma = np.sqrt(np.square(model.scale * sigma_iso) + model.floor * model.floor)
    return _positive(sigma, floor=model.floor)


def build_summary_sigma_pipeline(feature_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("select", passthrough_columns(feature_cols)),
            ("model", make_extra_trees(n_estimators=96, max_features="sqrt", min_samples_leaf=4)),
        ]
    )


def build_svd_sigma_pipeline(raw_cols: list[str], side_cols: list[str], n_components: int = 24) -> Pipeline:
    return Pipeline(
        [
            ("features", svd_with_side_features(raw_cols, passthrough_cols=side_cols, n_components=n_components)),
            ("model", make_extra_trees(n_estimators=96, max_features="sqrt", min_samples_leaf=4)),
        ]
    )


def _split_from_fold_column(df: pd.DataFrame, fold_col: str) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = sorted(df[fold_col].dropna().unique().tolist())
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        val_idx = np.flatnonzero(df[fold_col].to_numpy() == fold)
        train_idx = np.flatnonzero(df[fold_col].to_numpy() != fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def crossfit_residual_sigma(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    residual_feature: str = "abs_residual",
    fold_col: str = "fold",
    builder: str = "summary",
    raw_feature_cols: list[str] | None = None,
    n_components: int = 24,
    use_isotonic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if builder == "summary":
        def pipeline_builder() -> Pipeline:
            return build_summary_sigma_pipeline(feature_cols)
    elif builder == "svd":
        if raw_feature_cols is None:
            raise ValueError("raw_feature_cols is required for builder='svd'")

        def pipeline_builder() -> Pipeline:
            return build_svd_sigma_pipeline(raw_feature_cols, feature_cols, n_components=n_components)
    else:
        raise ValueError(f"Unsupported residual builder: {builder}")

    train_target = np.log(_positive(train_df[residual_feature].to_numpy(dtype=np.float64), floor=EPS_RESID))
    val_sigma_raw = np.zeros(len(val_df), dtype=np.float64)
    train_sigma_raw = np.zeros(len(train_df), dtype=np.float64)

    if fold_col in train_df.columns and train_df[fold_col].nunique() > 1:
        inner_splits = _split_from_fold_column(train_df, fold_col)
    else:
        indices = np.arange(len(train_df), dtype=np.int64)
        thirds = np.array_split(indices, 3)
        inner_splits = []
        for shard in thirds:
            val_idx = shard
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
            inner_splits.append((train_idx, val_idx))

    for inner_train_idx, inner_val_idx in inner_splits:
        model = pipeline_builder()
        model.fit(train_df.iloc[inner_train_idx], train_target[inner_train_idx])
        pred = np.exp(model.predict(train_df.iloc[inner_val_idx])) * ABS_RESID_TO_SIGMA
        train_sigma_raw[inner_val_idx] = _positive(pred)

    train_sigma_raw = _positive(train_sigma_raw)
    if use_isotonic:
        calibrator = fit_isotonic_sigma_map(
            y_true=train_df[target_col].to_numpy(dtype=np.float64),
            mu=train_df["mu_primary"].to_numpy(dtype=np.float64),
            sigma_raw=train_sigma_raw,
            target_col=target_col,
        )
        model = pipeline_builder()
        model.fit(train_df, train_target)
        val_sigma_raw = np.exp(model.predict(val_df)) * ABS_RESID_TO_SIGMA
        return (
            apply_isotonic_sigma_map(train_sigma_raw, calibrator),
            apply_isotonic_sigma_map(val_sigma_raw, calibrator),
        )

    calibrator = fit_scale_floor_from_raw_sigma(
        y_true=train_df[target_col].to_numpy(dtype=np.float64),
        mu=train_df["mu_primary"].to_numpy(dtype=np.float64),
        sigma_raw=train_sigma_raw,
        target_col=target_col,
    )
    model = pipeline_builder()
    model.fit(train_df, train_target)
    val_sigma_raw = np.exp(model.predict(val_df)) * ABS_RESID_TO_SIGMA
    return (
        apply_scale_floor(train_sigma_raw, calibrator),
        apply_scale_floor(val_sigma_raw, calibrator),
    )


def fit_log_linear_blend(
    y_true: np.ndarray,
    mu: np.ndarray,
    component_matrix: np.ndarray,
    target_col: str,
) -> BlendModel:
    linear = fit_raw_log_linear_blend(y_true, mu, component_matrix)
    sigma_raw = predict_raw_log_linear_blend(component_matrix, linear)
    scale_model = fit_scale_floor_from_raw_sigma(y_true, mu, sigma_raw, target_col)
    return BlendModel(linear=linear, scale=scale_model.scale, floor=scale_model.floor)


def apply_log_linear_blend(component_matrix: np.ndarray, model: BlendModel) -> np.ndarray:
    sigma_raw = predict_raw_log_linear_blend(component_matrix, model.linear)
    sigma = np.sqrt(np.square(model.scale * sigma_raw) + model.floor * model.floor)
    return _positive(sigma, floor=model.floor)


def fit_raw_log_linear_blend(y_true: np.ndarray, mu: np.ndarray, component_matrix: np.ndarray) -> LinearRegression:
    component_matrix = _positive(component_matrix)
    linear = LinearRegression(positive=True)
    linear.fit(np.log(component_matrix), np.log(_positive(np.abs(y_true - mu) * ABS_RESID_TO_SIGMA)))
    return linear


def predict_raw_log_linear_blend(component_matrix: np.ndarray, linear: LinearRegression) -> np.ndarray:
    component_matrix = _positive(component_matrix)
    return _positive(np.exp(linear.predict(np.log(component_matrix))))


def fit_hardness_inflation(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma_base: np.ndarray,
    hardness: np.ndarray,
    target_col: str,
) -> HardnessInflationModel:
    sigma_base = _positive(sigma_base)
    hardness = np.asarray(hardness, dtype=np.float64)
    hardness_mean = float(hardness.mean())
    hardness_std = float(hardness.std() + 1e-9)
    hardness_z = np.maximum((hardness - hardness_mean) / hardness_std, 0.0)
    constant_sigma = fit_constant_sigma(y_true, mu, target_col)
    init = np.array([0.0, np.log(max(constant_sigma * 0.05, EPS_SIGMA))], dtype=np.float64)

    def objective(theta: np.ndarray) -> float:
        gamma = float(np.exp(theta[0]))
        floor = float(np.exp(theta[1]))
        sigma = np.sqrt(np.square(sigma_base * (1.0 + gamma * hardness_z)) + floor * floor)
        return target_crps(np.asarray(y_true), np.asarray(mu), sigma, target_col)

    result = minimize(objective, x0=init, method="L-BFGS-B")
    return HardnessInflationModel(
        gamma=float(np.exp(result.x[0])),
        floor=float(np.exp(result.x[1])),
        mean=hardness_mean,
        std=hardness_std,
    )


def apply_hardness_inflation(sigma_base: np.ndarray, hardness: np.ndarray, model: HardnessInflationModel) -> np.ndarray:
    hardness = np.asarray(hardness, dtype=np.float64)
    hardness_z = np.maximum((hardness - model.mean) / model.std, 0.0)
    sigma = np.sqrt(np.square(_positive(sigma_base) * (1.0 + model.gamma * hardness_z)) + model.floor * model.floor)
    return _positive(sigma, floor=model.floor)


def coverage_table(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
) -> pd.DataFrame:
    z = np.abs((np.asarray(y_true) - np.asarray(mu)) / _positive(sigma))
    rows = []
    for level in levels:
        threshold = float(norm.ppf((1.0 + level) / 2.0))
        rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": float(np.mean(z <= threshold)),
            }
        )
    return pd.DataFrame(rows)


def reliability_table(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sigma": _positive(np.asarray(sigma)),
            "abs_residual": np.abs(np.asarray(y_true) - np.asarray(mu)),
            "squared_residual": np.square(np.asarray(y_true) - np.asarray(mu)),
        }
    ).sort_values("sigma")
    df["bin"] = pd.qcut(df["sigma"], q=min(n_bins, df["sigma"].nunique()), labels=False, duplicates="drop")
    summary = (
        df.groupby("bin", as_index=False)
        .agg(
            n=("sigma", "size"),
            sigma_mean=("sigma", "mean"),
            empirical_sigma_abs=("abs_residual", lambda x: float(np.mean(x) * ABS_RESID_TO_SIGMA)),
            empirical_sigma_rmse=("squared_residual", lambda x: float(np.sqrt(np.mean(x)))),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )
    return summary


def standardized_residuals(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (np.asarray(y_true) - np.asarray(mu)) / _positive(sigma)
