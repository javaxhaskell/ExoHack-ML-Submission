#!/usr/bin/env python
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning

from src.calibration import (
    apply_hardness_inflation,
    apply_isotonic_sigma_map,
    apply_log_linear_blend,
    apply_scale_floor,
    coverage_table,
    crossfit_residual_sigma,
    fit_constant_sigma,
    fit_hardness_inflation,
    fit_isotonic_sigma_map,
    fit_log_linear_blend,
    fit_raw_log_linear_blend,
    fit_scale_floor_from_raw_sigma,
    predict_raw_log_linear_blend,
    reliability_table,
    standardized_residuals,
    target_crps,
    target_score,
)
from src.data_loading import load_test_data, load_training_data
from src.features import FeatureBundle, build_phase2_feature_bundle
from src.metrics import TARGET_COLS, score_numpy
from src.models.catalog import phase2_candidates
from src.train import load_phase1_schemes


warnings.filterwarnings("ignore", category=ConvergenceWarning)


DATA_DIR = ROOT / "Hackathon_training"
REPORT_DIR = ROOT / "outputs" / "reports"
CALIB_DIR = ROOT / "outputs" / "calibration"
SIGNAL_DIR = CALIB_DIR / "backbone_signals"
OOF_SYSTEM_DIR = CALIB_DIR / "oof_systems"
DIAG_DIR = CALIB_DIR / "diagnostics"

TEMP_PRIMARY = "temp_rf_meta"
TEMP_BACKUP = "temp_et_meta"
ABUND_PRIMARY = "abun_sep_et_spec_noise_meta_deriv"
ABUND_BACKUPS = ["abun_sep_et_spec_noise_meta", "abun_sep_et_spec_noise"]
TEMP_TARGET = "planet_temp"
ABUNDANCE_TARGETS = ["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
ALL_SCHEMES = ["random_5fold", "regime_group_5fold", "edge_holdout_15pct"]
RETAINED_TEMP_METHODS = {
    "constant",
    "tree_raw",
    "tree_scaled",
    "disagree_raw",
    "disagree_scaled",
    "resid_summary",
    "blend_summary",
    "blend_summary_iso",
}
RETAINED_TEMP_BACKUP_METHODS = RETAINED_TEMP_METHODS.copy()
RETAINED_ABUND_METHODS = {
    "constant",
    "tree_raw",
    "tree_scaled",
    "disagree_raw",
    "disagree_scaled",
    "resid_summary",
    "resid_svd",
    "blend_svd",
    "blend_svd_iso",
    "blend_svd_hardness",
}


@dataclass(frozen=True)
class SystemSpec:
    name: str
    temperature_backbone: str
    temperature_method: str
    abundance_methods: dict[str, str]


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def format_frame(df: pd.DataFrame, decimals: int = 4) -> str:
    display = df.copy()
    numeric_cols = display.select_dtypes(include=["number", "bool"]).columns
    display[numeric_cols] = display[numeric_cols].round(decimals)
    return display.to_string(index=False)


def candidate_map():
    return {candidate.name: candidate for candidate in phase2_candidates()}


def _transform_features(fitted_pipeline, frame: pd.DataFrame) -> np.ndarray:
    transformed = frame
    for _, step in fitted_pipeline.steps[:-1]:
        transformed = step.transform(transformed)
    return np.asarray(transformed, dtype=np.float64)


def _tree_prediction_stats(fitted_pipeline, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _transform_features(fitted_pipeline, frame)
    model = fitted_pipeline.named_steps["model"]
    tree_predictions = np.column_stack([tree.predict(X) for tree in model.estimators_]).astype(np.float64)
    mu = tree_predictions.mean(axis=1)
    tree_std = tree_predictions.std(axis=1, ddof=1)
    tree_iqr = np.subtract(*np.quantile(tree_predictions, [0.75, 0.25], axis=1))
    return mu, tree_std, tree_iqr


def _align_scheme(train_frame: pd.DataFrame, scheme_df: pd.DataFrame) -> pd.DataFrame:
    merged = train_frame[["planet_ID"]].merge(scheme_df, on="planet_ID", how="left", validate="one_to_one")
    if merged.isna().any().any():
        missing = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"Missing scheme alignment for columns: {missing}")
    return merged


def generate_tree_oof_signals(
    candidate,
    feature_bundle: FeatureBundle,
    scheme_name: str,
    scheme_df: pd.DataFrame,
    out_path: Path,
    reuse_existing: bool = True,
) -> pd.DataFrame:
    if reuse_existing and out_path.exists():
        return pd.read_csv(out_path)

    if candidate.training_mode != "independent":
        raise ValueError(f"Phase 3 signal generation expects independent models, got {candidate.training_mode}")

    train_frame = feature_bundle.train_frame.copy()
    aligned_scheme = _align_scheme(train_frame, scheme_df)
    oof = train_frame[["planet_ID"]].copy()
    for target in candidate.target_cols:
        oof[f"mu_{target}"] = np.nan
        oof[f"tree_std_{target}"] = np.nan
        oof[f"tree_iqr_{target}"] = np.nan
    oof["is_valid"] = False

    if "fold" in aligned_scheme.columns:
        for fold in sorted(aligned_scheme["fold"].unique().tolist()):
            val_mask = aligned_scheme["fold"].to_numpy() == fold
            train_mask = ~val_mask
            train_df = train_frame.loc[train_mask].reset_index(drop=True)
            val_df = train_frame.loc[val_mask].reset_index(drop=True)
            for target in candidate.target_cols:
                model = candidate.build_pipeline(feature_bundle.feature_groups)
                model.fit(train_df, train_df[target].to_numpy(dtype=np.float64))
                mu, tree_std, tree_iqr = _tree_prediction_stats(model, val_df)
                oof.loc[val_mask, f"mu_{target}"] = mu
                oof.loc[val_mask, f"tree_std_{target}"] = tree_std
                oof.loc[val_mask, f"tree_iqr_{target}"] = tree_iqr
            oof.loc[val_mask, "is_valid"] = True
    elif "is_holdout" in aligned_scheme.columns:
        val_mask = aligned_scheme["is_holdout"].to_numpy(dtype=bool)
        train_mask = ~val_mask
        train_df = train_frame.loc[train_mask].reset_index(drop=True)
        val_df = train_frame.loc[val_mask].reset_index(drop=True)
        for target in candidate.target_cols:
            model = candidate.build_pipeline(feature_bundle.feature_groups)
            model.fit(train_df, train_df[target].to_numpy(dtype=np.float64))
            mu, tree_std, tree_iqr = _tree_prediction_stats(model, val_df)
            oof.loc[val_mask, f"mu_{target}"] = mu
            oof.loc[val_mask, f"tree_std_{target}"] = tree_std
            oof.loc[val_mask, f"tree_iqr_{target}"] = tree_iqr
        oof.loc[val_mask, "is_valid"] = True
    else:
        raise ValueError(f"Unsupported scheme columns for {scheme_name}: {aligned_scheme.columns.tolist()}")

    export = oof.merge(aligned_scheme, on="planet_ID", how="left", validate="one_to_one")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_path, index=False, compression="gzip")
    return export


def load_prefixed_signals(candidate_name: str, scheme_name: str) -> pd.DataFrame:
    path = SIGNAL_DIR / f"{candidate_name}__{scheme_name}.csv.gz"
    frame = pd.read_csv(path)
    rename = {
        col: f"{candidate_name}__{col}"
        for col in frame.columns
        if col != "planet_ID"
    }
    return frame.rename(columns=rename)


def build_phase3_base_frame(feature_bundle: FeatureBundle, schemes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = feature_bundle.train_frame.copy()
    regime = schemes["regime_group_5fold"][["planet_ID", "regime_group"]].copy()
    edge = schemes["edge_holdout_15pct"][["planet_ID", "edge_score", "is_holdout"]].copy()
    base = base.merge(regime, on="planet_ID", how="left", validate="one_to_one")
    base = base.merge(edge, on="planet_ID", how="left", validate="one_to_one")
    base["is_edge_regime"] = base["is_holdout"].astype(bool)
    edge_mean = float(base["edge_score"].mean())
    edge_std = float(base["edge_score"].std() + 1e-9)
    base["edge_score_z"] = (base["edge_score"] - edge_mean) / edge_std
    return base.sort_values("planet_ID").reset_index(drop=True)


def build_temperature_frame(base_frame: pd.DataFrame, scheme_name: str, primary_name: str) -> pd.DataFrame:
    other_name = TEMP_BACKUP if primary_name == TEMP_PRIMARY else TEMP_PRIMARY
    primary = load_prefixed_signals(primary_name, scheme_name)
    other = load_prefixed_signals(other_name, scheme_name)

    keep_cols = [col for col in other.columns if col.startswith(f"{other_name}__")]
    frame = base_frame.merge(primary, on="planet_ID", how="left", validate="one_to_one")
    frame = frame.merge(other[["planet_ID"] + keep_cols], on="planet_ID", how="left", validate="one_to_one")

    frame["mu_primary"] = frame[f"{primary_name}__mu_{TEMP_TARGET}"]
    frame["tree_std_primary"] = frame[f"{primary_name}__tree_std_{TEMP_TARGET}"]
    frame["tree_iqr_primary"] = frame[f"{primary_name}__tree_iqr_{TEMP_TARGET}"]
    frame["mu_backup_1"] = frame[f"{other_name}__mu_{TEMP_TARGET}"]
    frame["tree_std_backup_1"] = frame[f"{other_name}__tree_std_{TEMP_TARGET}"]
    preds = np.column_stack([frame["mu_primary"].to_numpy(dtype=np.float64), frame["mu_backup_1"].to_numpy(dtype=np.float64)])
    frame["disagree_std"] = np.std(preds, axis=1, ddof=1)
    frame["disagree_range"] = np.abs(frame["mu_primary"] - frame["mu_backup_1"])
    frame["mu_norm"] = (frame["mu_primary"] - frame[TEMP_TARGET].mean()) / (frame[TEMP_TARGET].std() + 1e-9)
    frame["residual"] = frame[TEMP_TARGET] - frame["mu_primary"]
    frame["abs_residual"] = np.abs(frame["residual"])
    frame["squared_residual"] = np.square(frame["residual"])
    frame["track"] = "temperature"
    frame["primary_backbone"] = primary_name
    frame["target"] = TEMP_TARGET
    frame["is_valid_primary"] = frame[f"{primary_name}__is_valid"].astype(bool)
    if f"{primary_name}__fold" in frame.columns:
        frame["fold"] = frame[f"{primary_name}__fold"]
    if f"{primary_name}__is_holdout" in frame.columns:
        frame["is_holdout_eval"] = frame[f"{primary_name}__is_holdout"].astype(bool)
    return frame


def build_abundance_frame(base_frame: pd.DataFrame, scheme_name: str, target_col: str) -> pd.DataFrame:
    signal_frames = [load_prefixed_signals(ABUND_PRIMARY, scheme_name)]
    signal_frames.extend(load_prefixed_signals(name, scheme_name) for name in ABUND_BACKUPS)

    frame = base_frame.copy()
    for signal_frame in signal_frames:
        keep_cols = [col for col in signal_frame.columns if col != "planet_ID"]
        frame = frame.merge(signal_frame[["planet_ID"] + keep_cols], on="planet_ID", how="left", validate="one_to_one")

    frame["mu_primary"] = frame[f"{ABUND_PRIMARY}__mu_{target_col}"]
    frame["tree_std_primary"] = frame[f"{ABUND_PRIMARY}__tree_std_{target_col}"]
    frame["tree_iqr_primary"] = frame[f"{ABUND_PRIMARY}__tree_iqr_{target_col}"]

    backup_pred_cols = []
    backup_std_cols = []
    for idx, name in enumerate(ABUND_BACKUPS, start=1):
        mu_col = f"{name}__mu_{target_col}"
        std_col = f"{name}__tree_std_{target_col}"
        frame[f"mu_backup_{idx}"] = frame[mu_col]
        frame[f"tree_std_backup_{idx}"] = frame[std_col]
        backup_pred_cols.append(f"mu_backup_{idx}")
        backup_std_cols.append(f"tree_std_backup_{idx}")

    pred_cols = ["mu_primary"] + backup_pred_cols
    pred_matrix = frame[pred_cols].to_numpy(dtype=np.float64)
    frame["disagree_std"] = np.std(pred_matrix, axis=1, ddof=1)
    frame["disagree_range"] = pred_matrix.max(axis=1) - pred_matrix.min(axis=1)
    frame["tree_std_backup_mean"] = frame[backup_std_cols].mean(axis=1)
    frame["mu_norm"] = (frame["mu_primary"] - frame[target_col].mean()) / (frame[target_col].std() + 1e-9)
    frame["residual"] = frame[target_col] - frame["mu_primary"]
    frame["abs_residual"] = np.abs(frame["residual"])
    frame["squared_residual"] = np.square(frame["residual"])
    frame["track"] = "abundance"
    frame["primary_backbone"] = ABUND_PRIMARY
    frame["target"] = target_col
    frame["is_valid_primary"] = frame[f"{ABUND_PRIMARY}__is_valid"].astype(bool)
    if f"{ABUND_PRIMARY}__fold" in frame.columns:
        frame["fold"] = frame[f"{ABUND_PRIMARY}__fold"]
    if f"{ABUND_PRIMARY}__is_holdout" in frame.columns:
        frame["is_holdout_eval"] = frame[f"{ABUND_PRIMARY}__is_holdout"].astype(bool)
    return frame


def sigma_side_columns() -> list[str]:
    return [
        "mu_primary",
        "mu_norm",
        "tree_std_primary",
        "tree_iqr_primary",
        "disagree_std",
        "disagree_range",
        "edge_score",
        "edge_score_z",
        "regime_group",
        "is_edge_regime",
    ]


def summary_feature_columns(bundle: FeatureBundle) -> list[str]:
    return bundle.feature_groups["metadata_summaries"] + ["edge_score", "edge_score_z", "regime_group", "is_edge_regime"]


def raw_abundance_feature_columns(bundle: FeatureBundle) -> list[str]:
    return bundle.feature_groups["spectrum_noise_metadata_derivatives"]


def compute_sigma_method_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    track: str,
    target_col: str,
    summary_cols: list[str],
    side_cols: list[str],
    raw_cols: list[str],
) -> dict[str, np.ndarray]:
    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    mu_train = train_df["mu_primary"].to_numpy(dtype=np.float64)

    sigma_map: dict[str, np.ndarray] = {}

    constant_sigma = fit_constant_sigma(y_train, mu_train, target_col)
    sigma_map["constant"] = np.full(len(val_df), constant_sigma, dtype=np.float64)

    tree_train_raw = train_df["tree_std_primary"].to_numpy(dtype=np.float64)
    tree_val_raw = val_df["tree_std_primary"].to_numpy(dtype=np.float64)
    sigma_map["tree_raw"] = np.maximum(tree_val_raw, 1e-6)
    tree_scale_model = fit_scale_floor_from_raw_sigma(y_train, mu_train, tree_train_raw, target_col)
    tree_train_scaled = apply_scale_floor(tree_train_raw, tree_scale_model)
    tree_val_scaled = apply_scale_floor(tree_val_raw, tree_scale_model)
    sigma_map["tree_scaled"] = tree_val_scaled

    disagree_train_raw = train_df["disagree_std"].to_numpy(dtype=np.float64)
    disagree_val_raw = val_df["disagree_std"].to_numpy(dtype=np.float64)
    sigma_map["disagree_raw"] = np.maximum(disagree_val_raw, 1e-6)
    disagree_scale_model = fit_scale_floor_from_raw_sigma(y_train, mu_train, disagree_train_raw, target_col)
    disagree_train_scaled = apply_scale_floor(disagree_train_raw, disagree_scale_model)
    disagree_val_scaled = apply_scale_floor(disagree_val_raw, disagree_scale_model)
    sigma_map["disagree_scaled"] = disagree_val_scaled

    residual_side_cols = list(dict.fromkeys(summary_cols + side_cols))
    resid_summary_train, resid_summary_val = crossfit_residual_sigma(
        train_df=train_df,
        val_df=val_df,
        feature_cols=residual_side_cols,
        target_col=target_col,
        builder="summary",
        use_isotonic=False,
    )
    sigma_map["resid_summary"] = resid_summary_val

    blend_summary_components_train = np.column_stack([tree_train_scaled, disagree_train_scaled, resid_summary_train])
    blend_summary_components_val = np.column_stack([tree_val_scaled, disagree_val_scaled, resid_summary_val])
    blend_summary_model = fit_log_linear_blend(y_train, mu_train, blend_summary_components_train, target_col)
    sigma_map["blend_summary"] = apply_log_linear_blend(blend_summary_components_val, blend_summary_model)

    blend_summary_linear = fit_raw_log_linear_blend(y_train, mu_train, blend_summary_components_train)
    blend_summary_raw_train = predict_raw_log_linear_blend(blend_summary_components_train, blend_summary_linear)
    blend_summary_raw_val = predict_raw_log_linear_blend(blend_summary_components_val, blend_summary_linear)
    blend_summary_iso = fit_isotonic_sigma_map(y_train, mu_train, blend_summary_raw_train, target_col)
    sigma_map["blend_summary_iso"] = apply_isotonic_sigma_map(blend_summary_raw_val, blend_summary_iso)

    if track == "abundance":
        resid_svd_train, resid_svd_val = crossfit_residual_sigma(
            train_df=train_df,
            val_df=val_df,
            feature_cols=residual_side_cols,
            target_col=target_col,
            builder="svd",
            raw_feature_cols=raw_cols,
            n_components=24,
            use_isotonic=False,
        )
        sigma_map["resid_svd"] = resid_svd_val

        blend_svd_components_train = np.column_stack([tree_train_scaled, disagree_train_scaled, resid_svd_train])
        blend_svd_components_val = np.column_stack([tree_val_scaled, disagree_val_scaled, resid_svd_val])
        blend_svd_model = fit_log_linear_blend(y_train, mu_train, blend_svd_components_train, target_col)
        sigma_map["blend_svd"] = apply_log_linear_blend(blend_svd_components_val, blend_svd_model)

        blend_svd_linear = fit_raw_log_linear_blend(y_train, mu_train, blend_svd_components_train)
        blend_svd_raw_train = predict_raw_log_linear_blend(blend_svd_components_train, blend_svd_linear)
        blend_svd_raw_val = predict_raw_log_linear_blend(blend_svd_components_val, blend_svd_linear)
        blend_svd_iso = fit_isotonic_sigma_map(y_train, mu_train, blend_svd_raw_train, target_col)
        blend_svd_train_iso = apply_isotonic_sigma_map(blend_svd_raw_train, blend_svd_iso)
        blend_svd_val_iso = apply_isotonic_sigma_map(blend_svd_raw_val, blend_svd_iso)
        sigma_map["blend_svd_iso"] = blend_svd_val_iso

        hardness_model = fit_hardness_inflation(
            y_true=y_train,
            mu=mu_train,
            sigma_base=blend_svd_train_iso,
            hardness=train_df["edge_score"].to_numpy(dtype=np.float64),
            target_col=target_col,
        )
        sigma_map["blend_svd_hardness"] = apply_hardness_inflation(
            sigma_base=blend_svd_val_iso,
            hardness=val_df["edge_score"].to_numpy(dtype=np.float64),
            model=hardness_model,
        )

    return sigma_map


def evaluate_track_target(
    random_frame: pd.DataFrame,
    eval_frames: dict[str, pd.DataFrame],
    track: str,
    target_col: str,
    summary_cols: list[str],
    side_cols: list[str],
    raw_cols: list[str],
    backbone_name: str,
    retained_methods: set[str],
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows = []
    retained_predictions: dict[tuple[str, str], pd.DataFrame] = {}

    train_edge_source = random_frame.loc[~random_frame["is_edge_regime"] & random_frame["is_valid_primary"]].copy().reset_index(drop=True)

    for scheme_name in ALL_SCHEMES:
        scheme_frame = eval_frames[scheme_name].copy()
        if scheme_name == "edge_holdout_15pct":
            val_frame = scheme_frame.loc[scheme_frame["is_valid_primary"]].copy().reset_index(drop=True)
            sigma_map = compute_sigma_method_predictions(
                train_df=train_edge_source,
                val_df=val_frame,
                track=track,
                target_col=target_col,
                summary_cols=summary_cols,
                side_cols=side_cols,
                raw_cols=raw_cols,
            )
            scheme_predictions = {name: np.full(len(val_frame), values, dtype=np.float64) if np.isscalar(values) else values for name, values in sigma_map.items()}
            scheme_index = val_frame.index.to_numpy()
            eval_index = val_frame.index.to_numpy()
        else:
            working = scheme_frame.loc[scheme_frame["is_valid_primary"]].copy().reset_index(drop=True)
            method_names = None
            scheme_predictions: dict[str, np.ndarray] = {}
            for fold in sorted(working["fold"].unique().tolist()):
                train_df = working.loc[working["fold"] != fold].reset_index(drop=True)
                val_df = working.loc[working["fold"] == fold].reset_index(drop=True)
                sigma_map = compute_sigma_method_predictions(
                    train_df=train_df,
                    val_df=val_df,
                    track=track,
                    target_col=target_col,
                    summary_cols=summary_cols,
                    side_cols=side_cols,
                    raw_cols=raw_cols,
                )
                if method_names is None:
                    method_names = list(sigma_map.keys())
                    scheme_predictions = {name: np.full(len(working), np.nan, dtype=np.float64) for name in method_names}
                val_positions = working.index[working["fold"] == fold].to_numpy()
                for name, sigma_values in sigma_map.items():
                    scheme_predictions[name][val_positions] = sigma_values
            val_frame = working
            eval_index = val_frame.index.to_numpy()

        y_true = val_frame[target_col].to_numpy(dtype=np.float64)
        mu = val_frame["mu_primary"].to_numpy(dtype=np.float64)

        for method_name, sigma in scheme_predictions.items():
            coverage = coverage_table(y_true, mu, sigma)
            z = standardized_residuals(y_true, mu, sigma)
            reliability = reliability_table(y_true, mu, sigma)
            rows.append(
                {
                    "track": track,
                    "primary_backbone": backbone_name,
                    "target": target_col,
                    "scheme": scheme_name,
                    "method": method_name,
                    "primary_metric": target_score(y_true, mu, sigma, target_col),
                    "mean_crps": target_crps(y_true, mu, sigma, target_col),
                    "sigma_mean": float(np.mean(sigma)),
                    "sigma_median": float(np.median(sigma)),
                    "coverage_50": float(coverage.loc[coverage["nominal_coverage"].eq(0.5), "empirical_coverage"].iloc[0]),
                    "coverage_68": float(coverage.loc[coverage["nominal_coverage"].eq(0.68), "empirical_coverage"].iloc[0]),
                    "coverage_90": float(coverage.loc[coverage["nominal_coverage"].eq(0.9), "empirical_coverage"].iloc[0]),
                    "coverage_95": float(coverage.loc[coverage["nominal_coverage"].eq(0.95), "empirical_coverage"].iloc[0]),
                    "z_mean": float(np.mean(z)),
                    "z_std": float(np.std(z)),
                    "z_abs_p95": float(np.quantile(np.abs(z), 0.95)),
                    "sigma_abs_resid_spearman": float(spearmanr(sigma, np.abs(y_true - mu)).correlation),
                    "reliability_gap_abs": float(np.mean(np.abs(reliability["sigma_mean"] - reliability["empirical_sigma_abs"]))),
                }
            )

            if method_name in retained_methods:
                retained_predictions[(scheme_name, method_name)] = val_frame[
                    ["planet_ID", target_col, "mu_primary"]
                ].copy()
                retained_predictions[(scheme_name, method_name)]["sigma"] = sigma

    return pd.DataFrame(rows), retained_predictions


def rank_stability(metric_frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rank_rows = []
    for keys, sub in metric_frame.groupby(group_cols, dropna=False):
        ranked = sub.sort_values("primary_metric", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        rank_rows.append(ranked)
    ranks = pd.concat(rank_rows, ignore_index=True)
    summary = (
        ranks.groupby([col for col in ranks.columns if col not in {"scheme", "primary_metric", "rank"}], as_index=False)
        .agg(avg_rank=("rank", "mean"), std_rank=("rank", "std"), avg_primary=("primary_metric", "mean"))
        .fillna(0.0)
    )
    return summary.sort_values(["avg_rank", "avg_primary"], ascending=[True, False]).reset_index(drop=True)


def write_track_diagnostics(
    predictions: dict[tuple[str, str], pd.DataFrame],
    track: str,
    backbone_name: str,
    target_col: str,
) -> None:
    for (scheme_name, method_name), frame in predictions.items():
        out_dir = DIAG_DIR / track / backbone_name / method_name / scheme_name
        out_dir.mkdir(parents=True, exist_ok=True)
        y_true = frame[target_col].to_numpy(dtype=np.float64)
        mu = frame["mu_primary"].to_numpy(dtype=np.float64)
        sigma = frame["sigma"].to_numpy(dtype=np.float64)

        reliability = reliability_table(y_true, mu, sigma)
        coverage = coverage_table(y_true, mu, sigma)
        z = standardized_residuals(y_true, mu, sigma)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(reliability["sigma_mean"], reliability["empirical_sigma_abs"], marker="o", label="Empirical sigma (|r|)")
        axes[0].plot(reliability["sigma_mean"], reliability["empirical_sigma_rmse"], marker="s", label="Empirical sigma (RMSE)")
        upper = float(max(reliability["sigma_mean"].max(), reliability["empirical_sigma_rmse"].max()))
        axes[0].plot([0, upper], [0, upper], linestyle="--", color="black", linewidth=1)
        axes[0].set_title(f"{target_col}: Reliability")
        axes[0].set_xlabel("Predicted sigma")
        axes[0].set_ylabel("Empirical sigma")
        axes[0].legend()

        axes[1].plot(coverage["nominal_coverage"], coverage["empirical_coverage"], marker="o")
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        axes[1].set_title(f"{target_col}: Coverage")
        axes[1].set_xlabel("Nominal coverage")
        axes[1].set_ylabel("Empirical coverage")
        fig.tight_layout()
        fig.savefig(out_dir / f"{target_col}_reliability.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(z, bins=40, density=True, alpha=0.75, color="#3a6ea5")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{target_col}: Standardized residuals")
        ax.set_xlabel("(y - mu) / sigma")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(out_dir / f"{target_col}_z_hist.png", dpi=160)
        plt.close(fig)


def inventory_signal_rows(frame: pd.DataFrame, target_col: str, backbone_name: str) -> pd.DataFrame:
    signal_cols = [
        "tree_std_primary",
        "tree_iqr_primary",
        "disagree_std",
        "disagree_range",
        "edge_score",
        "edge_score_z",
        "log_spec_mean",
        "log_spec_std",
        "spectral_slope",
        "band_contrast",
        "log_noise_mean",
        "log_noise_std",
        "log_snr_mean",
        "log_snr_std",
        "meta_eqtemp_proxy",
        "meta_flux_proxy",
        "meta_density_proxy",
        "mu_norm",
    ]
    rows = []
    for col in signal_cols:
        if col not in frame.columns:
            continue
        corr = spearmanr(frame[col], frame["abs_residual"], nan_policy="omit").correlation
        rows.append(
            {
                "backbone": backbone_name,
                "target": target_col,
                "signal": col,
                "spearman_abs_residual": float(corr),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_abs_residual", key=lambda s: s.abs(), ascending=False)


def build_system_frame(
    scheme_name: str,
    temp_prediction: pd.DataFrame,
    abundance_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frame = temp_prediction[["planet_ID", TEMP_TARGET, "mu_primary", "sigma"]].copy()
    frame = frame.rename(columns={"mu_primary": f"mu_{TEMP_TARGET}", "sigma": f"sigma_{TEMP_TARGET}"})

    for target_col, pred in abundance_predictions.items():
        add = pred[["planet_ID", target_col, "mu_primary", "sigma"]].copy()
        add = add.rename(columns={"mu_primary": f"mu_{target_col}", "sigma": f"sigma_{target_col}"})
        frame = frame.merge(add, on="planet_ID", how="inner", validate="one_to_one")

    frame = frame.sort_values("planet_ID").reset_index(drop=True)
    return frame


def evaluate_system_frame(frame: pd.DataFrame) -> dict[str, float]:
    y = frame[TARGET_COLS].to_numpy(dtype=np.float64)
    mu = frame[[f"mu_{target}" for target in TARGET_COLS]].to_numpy(dtype=np.float64)
    sigma = frame[[f"sigma_{target}" for target in TARGET_COLS]].to_numpy(dtype=np.float64)
    score = score_numpy(y, mu, sigma, target_cols=TARGET_COLS)
    return {
        "primary_metric": float(score["score"]),
        "mean_crps": float(score["mean_crps"]),
    }


def export_system_oof(system_name: str, scheme_name: str, frame: pd.DataFrame) -> None:
    export = frame[["planet_ID"]].copy()
    for target_col in TARGET_COLS:
        export[f"mu_{target_col}"] = frame[f"mu_{target_col}"]
        export[f"sigma_{target_col}"] = frame[f"sigma_{target_col}"]
    OOF_SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    export.to_csv(OOF_SYSTEM_DIR / f"{system_name}__{scheme_name}.csv.gz", index=False, compression="gzip")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    candidates = candidate_map()
    train_supp, train_targets, train_spectra = load_training_data(DATA_DIR)
    test_supp, test_spectra = load_test_data(DATA_DIR)
    feature_bundle = build_phase2_feature_bundle(train_supp, train_targets, train_spectra, test_supp, test_spectra)
    schemes = load_phase1_schemes(ROOT / "outputs" / "folds")
    base_frame = build_phase3_base_frame(feature_bundle, schemes)
    side_cols = sigma_side_columns()
    summary_cols = summary_feature_columns(feature_bundle)
    abundance_raw_cols = raw_abundance_feature_columns(feature_bundle)

    selected_signals = [TEMP_PRIMARY, TEMP_BACKUP, ABUND_PRIMARY] + ABUND_BACKUPS
    for candidate_name in selected_signals:
        for scheme_name in ALL_SCHEMES:
            generate_tree_oof_signals(
                candidate=candidates[candidate_name],
                feature_bundle=feature_bundle,
                scheme_name=scheme_name,
                scheme_df=schemes[scheme_name],
                out_path=SIGNAL_DIR / f"{candidate_name}__{scheme_name}.csv.gz",
                reuse_existing=True,
            )

    metrics_rows = []
    signal_inventory_rows = []
    retained_prediction_store: dict[tuple[str, str, str, str], pd.DataFrame] = {}

    temp_frames_rf = {scheme: build_temperature_frame(base_frame, scheme, TEMP_PRIMARY) for scheme in ALL_SCHEMES}
    temp_rf_metrics, temp_rf_predictions = evaluate_track_target(
        random_frame=temp_frames_rf["random_5fold"],
        eval_frames=temp_frames_rf,
        track="temperature",
        target_col=TEMP_TARGET,
        summary_cols=summary_cols,
        side_cols=side_cols,
        raw_cols=[],
        backbone_name=TEMP_PRIMARY,
        retained_methods=RETAINED_TEMP_METHODS,
    )
    metrics_rows.append(temp_rf_metrics)
    signal_inventory_rows.append(inventory_signal_rows(temp_frames_rf["random_5fold"], TEMP_TARGET, TEMP_PRIMARY))
    write_track_diagnostics(temp_rf_predictions, "temperature", TEMP_PRIMARY, TEMP_TARGET)
    for key, frame in temp_rf_predictions.items():
        retained_prediction_store[("temperature", TEMP_PRIMARY, TEMP_TARGET, key[1], key[0])] = frame

    temp_frames_et = {scheme: build_temperature_frame(base_frame, scheme, TEMP_BACKUP) for scheme in ALL_SCHEMES}
    temp_et_metrics, temp_et_predictions = evaluate_track_target(
        random_frame=temp_frames_et["random_5fold"],
        eval_frames=temp_frames_et,
        track="temperature",
        target_col=TEMP_TARGET,
        summary_cols=summary_cols,
        side_cols=side_cols,
        raw_cols=[],
        backbone_name=TEMP_BACKUP,
        retained_methods=RETAINED_TEMP_BACKUP_METHODS,
    )
    metrics_rows.append(temp_et_metrics)
    signal_inventory_rows.append(inventory_signal_rows(temp_frames_et["random_5fold"], TEMP_TARGET, TEMP_BACKUP))
    write_track_diagnostics(temp_et_predictions, "temperature", TEMP_BACKUP, TEMP_TARGET)
    for key, frame in temp_et_predictions.items():
        retained_prediction_store[("temperature", TEMP_BACKUP, TEMP_TARGET, key[1], key[0])] = frame

    abundance_prediction_store: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    abundance_metric_frames = []
    for target_col in ABUNDANCE_TARGETS:
        abundance_frames = {scheme: build_abundance_frame(base_frame, scheme, target_col) for scheme in ALL_SCHEMES}
        metric_frame, retained = evaluate_track_target(
            random_frame=abundance_frames["random_5fold"],
            eval_frames=abundance_frames,
            track="abundance",
            target_col=target_col,
            summary_cols=summary_cols,
            side_cols=side_cols + ["tree_std_backup_mean", "mu_backup_1", "mu_backup_2"],
            raw_cols=abundance_raw_cols,
            backbone_name=ABUND_PRIMARY,
            retained_methods=RETAINED_ABUND_METHODS,
        )
        abundance_metric_frames.append(metric_frame)
        signal_inventory_rows.append(inventory_signal_rows(abundance_frames["random_5fold"], target_col, ABUND_PRIMARY))
        write_track_diagnostics(retained, "abundance", ABUND_PRIMARY, target_col)
        for key, frame in retained.items():
            abundance_prediction_store[(ABUND_PRIMARY, target_col, key[1], key[0])] = frame

    metrics_df = pd.concat(metrics_rows + abundance_metric_frames, ignore_index=True)
    metrics_df.to_csv(CALIB_DIR / "phase3_target_sigma_metrics.csv", index=False)

    inventory_df = pd.concat(signal_inventory_rows, ignore_index=True)
    inventory_df.to_csv(CALIB_DIR / "phase3_signal_inventory.csv", index=False)

    track_summary = (
        metrics_df.groupby(["track", "primary_backbone", "target", "method"], as_index=False)
        .agg(
            avg_primary=("primary_metric", "mean"),
            worst_primary=("primary_metric", "min"),
            avg_reliability_gap=("reliability_gap_abs", "mean"),
            avg_z_std=("z_std", "mean"),
        )
        .sort_values(["track", "target", "avg_primary"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    track_summary.to_csv(CALIB_DIR / "phase3_target_sigma_summary.csv", index=False)

    stability_df = rank_stability(
        metrics_df[["track", "primary_backbone", "target", "scheme", "method", "primary_metric"]],
        group_cols=["track", "primary_backbone", "target", "scheme"],
    )
    stability_df.to_csv(CALIB_DIR / "phase3_target_sigma_rank_stability.csv", index=False)

    temp_summary = track_summary[track_summary["track"].eq("temperature")].copy()
    best_temp_row = temp_summary.iloc[0]
    best_temp_backbone = str(best_temp_row["primary_backbone"])
    best_temp_method = str(best_temp_row["method"])

    abundance_summary = track_summary[track_summary["track"].eq("abundance")].copy()
    best_method_by_target = (
        abundance_summary.sort_values(["target", "avg_primary"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()[["target", "method"]]
    )
    best_method_map = dict(zip(best_method_by_target["target"], best_method_by_target["method"]))
    best_method_by_target.to_csv(CALIB_DIR / "phase3_best_sigma_method_per_target.csv", index=False)

    system_specs = [
        SystemSpec(
            name="baseline_constant",
            temperature_backbone=TEMP_PRIMARY,
            temperature_method="constant",
            abundance_methods={target: "constant" for target in ABUNDANCE_TARGETS},
        ),
        SystemSpec(
            name="scaled_spread",
            temperature_backbone=TEMP_PRIMARY,
            temperature_method="tree_scaled",
            abundance_methods={target: "tree_scaled" for target in ABUNDANCE_TARGETS},
        ),
        SystemSpec(
            name="blend_uniform",
            temperature_backbone=TEMP_PRIMARY,
            temperature_method="blend_summary_iso",
            abundance_methods={target: "blend_svd_iso" for target in ABUNDANCE_TARGETS},
        ),
        SystemSpec(
            name="target_specific_best",
            temperature_backbone=best_temp_backbone,
            temperature_method=best_temp_method,
            abundance_methods=best_method_map,
        ),
        SystemSpec(
            name="temp_et_backup",
            temperature_backbone=TEMP_BACKUP,
            temperature_method="blend_summary_iso",
            abundance_methods=best_method_map,
        ),
    ]

    system_rows = []
    for system in system_specs:
        for scheme_name in ALL_SCHEMES:
            temp_frame = retained_prediction_store[
                ("temperature", system.temperature_backbone, TEMP_TARGET, system.temperature_method, scheme_name)
            ]
            abundance_frames = {
                target: abundance_prediction_store[(ABUND_PRIMARY, target, system.abundance_methods[target], scheme_name)]
                for target in ABUNDANCE_TARGETS
            }
            system_frame = build_system_frame(scheme_name, temp_frame, abundance_frames)
            export_system_oof(system.name, scheme_name, system_frame)
            metrics = evaluate_system_frame(system_frame)
            system_rows.append(
                {
                    "system": system.name,
                    "scheme": scheme_name,
                    "temperature_backbone": system.temperature_backbone,
                    "temperature_method": system.temperature_method,
                    "abundance_method_signature": "|".join(f"{target}:{system.abundance_methods[target]}" for target in ABUNDANCE_TARGETS),
                    **metrics,
                }
            )

    system_df = pd.DataFrame(system_rows)
    system_df.to_csv(CALIB_DIR / "phase3_system_metrics.csv", index=False)
    system_stability = rank_stability(system_df[["system", "scheme", "primary_metric"]], group_cols=["scheme"])
    system_stability.to_csv(CALIB_DIR / "phase3_system_rank_stability.csv", index=False)

    calibration_gain_rows = []
    baseline_metrics = system_df[system_df["system"].eq("baseline_constant")][["scheme", "primary_metric"]].rename(
        columns={"primary_metric": "baseline_primary"}
    )
    for system_name in system_df["system"].unique():
        merged = system_df[system_df["system"].eq(system_name)][["scheme", "primary_metric"]].merge(
            baseline_metrics,
            on="scheme",
            how="left",
            validate="one_to_one",
        )
        merged["gain_vs_constant"] = merged["primary_metric"] - merged["baseline_primary"]
        merged.insert(0, "system", system_name)
        calibration_gain_rows.append(merged)
    calibration_gain_df = pd.concat(calibration_gain_rows, ignore_index=True)
    calibration_gain_df.to_csv(CALIB_DIR / "phase3_calibration_gains.csv", index=False)

    target_method_table = (
        metrics_df.groupby(["track", "primary_backbone", "target", "method"], as_index=False)
        .agg(
            avg_primary=("primary_metric", "mean"),
            worst_primary=("primary_metric", "min"),
            avg_coverage_68=("coverage_68", "mean"),
            avg_coverage_90=("coverage_90", "mean"),
            avg_z_std=("z_std", "mean"),
        )
        .sort_values(["track", "target", "avg_primary"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    best_system_table = (
        system_df.sort_values(["scheme", "primary_metric"], ascending=[True, False])
        .groupby("scheme", as_index=False)
        .first()
    )

    strongest_overall_sigma = (
        target_method_table[target_method_table["track"].eq("abundance")]
        .groupby("method", as_index=False)["avg_primary"]
        .mean()
        .sort_values("avg_primary", ascending=False)
        .iloc[0]["method"]
    )

    uncertainty_report = f"""# Phase 3 Uncertainty Engineering

## Fixed mean context
- Temperature backbone: `temp_rf_meta` is the primary mean model. `temp_et_meta` is retained only as a sigma comparison and disagreement signal.
- Abundance backbone: `abun_sep_et_spec_noise_meta_deriv` remains the primary mean family.
- Diversity-only abundance backups used for disagreement features: `{', '.join(ABUND_BACKUPS)}`.

## Available uncertainty signals
- Labels from OOF evidence: residual, absolute residual, squared residual.
- Primary model outputs: `mu_primary`, `tree_std_primary`, `tree_iqr_primary`.
- Cross-model disagreement:
  - temperature: RF vs ET disagreement
  - abundance: disagreement across derivative, spectrum+noise+metadata, and spectrum+noise specialists
- Difficulty proxies from inputs: metadata logs, spectral summaries, noise summaries, SNR summaries, derivative-aware compressed spectra, `edge_score`, `regime_group`, and `is_edge_regime`.
- Hard-regime marker used for shift-aware inflation: `edge_score`.

## Signal inventory (top correlations with absolute residual)
```
{format_frame(inventory_df.groupby(['backbone', 'target']).head(6), decimals=4)}
```

## Methods tested
- Constant sigma baseline.
- Raw tree spread and raw disagreement spread.
- Variance-scaled spread.
- Residual-model sigma on summary features.
- Residual-model sigma on compressed raw spectrum/noise/derivative features for abundances.
- Blended sigma from spread plus residual-model components.
- Post-hoc isotonic calibration on blended sigma.
- Hardness-aware sigma inflation on top of the abundance blend.

## What changed in belief
- Raw spread is useful as a signal but not calibrated enough to trust directly.
- Target-specific sigma choice matters materially, especially for `log_CO`.
- Shift-aware calibration matters more than extra mean diversity at this stage.
"""

    calibration_summary_report = f"""# Phase 3 Calibration Summary

## Best sigma method per target
```
{format_frame(best_method_by_target, decimals=4)}
```

## Temperature sigma ranking
```
{format_frame(temp_summary[['primary_backbone', 'target', 'method', 'avg_primary', 'worst_primary', 'avg_reliability_gap', 'avg_z_std']].head(10), decimals=4)}
```

## Abundance sigma ranking
```
{format_frame(abundance_summary[['target', 'method', 'avg_primary', 'worst_primary', 'avg_reliability_gap', 'avg_z_std']].head(20), decimals=4)}
```

## Strongest overall sigma family
- Overall winner by average abundance-target score: `{strongest_overall_sigma}`

## Calibration gains versus constant sigma
```
{format_frame(calibration_gain_df, decimals=4)}
```
"""

    tournament_report = f"""# Phase 3 Mu+Sigma Tournament

## Final system tournament
```
{format_frame(system_df.sort_values(['scheme', 'primary_metric'], ascending=[True, False]), decimals=4)}
```

## Best system by validation scheme
```
{format_frame(best_system_table, decimals=4)}
```

## System rank stability
```
{format_frame(system_stability, decimals=4)}
```

## Best track-level decisions
- Strongest temperature backbone/method: `{best_temp_backbone} + {best_temp_method}`
- Strongest abundance target-method map: `{', '.join(f'{target}:{method}' for target, method in best_method_map.items())}`
- `log_CO` special treatment selected: `{best_method_map['log_CO']}`

## Phase 4 shortlist recommendation
- Primary finalist: `target_specific_best`
- Conservative backup: `blend_uniform`
- Temperature-backup check: `temp_et_backup`
"""

    save_text(REPORT_DIR / "phase3_uncertainty.md", uncertainty_report)
    save_text(REPORT_DIR / "phase3_calibration_summary.md", calibration_summary_report)
    save_text(REPORT_DIR / "phase3_mu_sigma_tournament.md", tournament_report)


if __name__ == "__main__":
    main()
