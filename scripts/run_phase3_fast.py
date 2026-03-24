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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.calibration import (
    apply_log_linear_blend,
    apply_scale_floor,
    crossfit_residual_sigma,
    fit_constant_sigma,
    fit_log_linear_blend,
    fit_scale_floor_from_raw_sigma,
    target_crps,
    target_score,
)
from src.data_loading import load_test_data, load_training_data
from src.features import build_phase2_feature_bundle
from src.metrics import TARGET_COLS, array_to_submission, score_numpy
from src.models.catalog import phase2_candidates
from src.train import load_phase1_schemes


DATA_DIR = ROOT / "Hackathon_training"
OOF_DIR = ROOT / "outputs" / "oof"
REPORT_DIR = ROOT / "outputs" / "reports"
OUT_DIR = ROOT / "outputs" / "calibration"

TEMP_PRIMARY = "temp_rf_meta"
TEMP_BACKUP = "temp_et_meta"
ABUND_PRIMARY = "abun_sep_et_spec_noise_meta_deriv"
ABUND_BACKUP = "abun_sep_et_spec_noise"
TEMP_TARGET = "planet_temp"
ABUNDANCE_TARGETS = ["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
FAST_METHODS = ["constant", "residual_model", "disagreement_scaled", "blend_simple"]


@dataclass(frozen=True)
class MethodEval:
    track: str
    target: str
    scheme: str
    method: str
    primary_metric: float
    mean_crps: float


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


def load_oof(candidate_name: str, scheme_name: str) -> pd.DataFrame:
    return pd.read_csv(OOF_DIR / f"{candidate_name}__{scheme_name}.csv.gz")


def compute_edge_score_proxy(train_frame: pd.DataFrame, other_frame: pd.DataFrame) -> np.ndarray:
    cols = [
        "meta_log_star_temperature",
        "meta_log_planet_orbital_period",
        "meta_log_planet_radius_m",
        "meta_log_planet_surface_gravity",
        "log_spec_mean",
        "log_spec_std",
        "spectral_slope",
        "band_contrast",
        "log_noise_mean",
        "log_snr_mean",
        "log_snr_std",
    ]
    robust_z = []
    for col in cols:
        train_values = train_frame[col].to_numpy(dtype=np.float64)
        values = other_frame[col].to_numpy(dtype=np.float64)
        median = float(np.median(train_values))
        iqr = float(np.subtract(*np.percentile(train_values, [75, 25])))
        scale = iqr if iqr > 0 else float(train_values.std() + 1e-9)
        robust_z.append(np.abs((values - median) / (scale + 1e-9)))
    return np.sqrt(np.mean(np.square(np.column_stack(robust_z)), axis=1))


def build_base_frames():
    train_supp, train_targets, train_spectra = load_training_data(DATA_DIR)
    test_supp, test_spectra = load_test_data(DATA_DIR)
    bundle = build_phase2_feature_bundle(train_supp, train_targets, train_spectra, test_supp, test_spectra)
    schemes = load_phase1_schemes(ROOT / "outputs" / "folds")

    train_frame = bundle.train_frame.copy()
    test_frame = bundle.test_frame.copy()
    train_frame = train_frame.merge(schemes["edge_holdout_15pct"], on="planet_ID", how="left", validate="one_to_one")
    train_frame["edge_score_proxy"] = compute_edge_score_proxy(train_frame, train_frame)
    test_frame["edge_score_proxy"] = compute_edge_score_proxy(train_frame, test_frame)
    return bundle, schemes, train_frame, test_frame


def build_temp_frame(
    train_frame: pd.DataFrame,
    scheme_name: str,
    primary_name: str = TEMP_PRIMARY,
    backup_name: str = TEMP_BACKUP,
) -> pd.DataFrame:
    primary = load_oof(primary_name, scheme_name)
    backup = load_oof(backup_name, scheme_name)
    frame = train_frame.merge(
        primary[["planet_ID", f"mu_{TEMP_TARGET}", "is_valid"] + ([col for col in ["fold"] if col in primary.columns])],
        on="planet_ID",
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        backup[["planet_ID", f"mu_{TEMP_TARGET}"]],
        on="planet_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_backup"),
    )
    frame = frame.rename(columns={f"mu_{TEMP_TARGET}": "mu_primary", f"mu_{TEMP_TARGET}_backup": "mu_backup"})
    frame["disagreement"] = np.abs(frame["mu_primary"] - frame["mu_backup"]) / np.sqrt(2.0)
    frame["residual"] = frame[TEMP_TARGET] - frame["mu_primary"]
    frame["abs_residual"] = np.abs(frame["residual"])
    return frame


def build_abundance_frame(
    train_frame: pd.DataFrame,
    scheme_name: str,
    target_col: str,
    primary_name: str = ABUND_PRIMARY,
    backup_name: str = ABUND_BACKUP,
) -> pd.DataFrame:
    primary = load_oof(primary_name, scheme_name)
    backup = load_oof(backup_name, scheme_name)
    frame = train_frame.merge(
        primary[["planet_ID", f"mu_{target_col}", "is_valid"] + ([col for col in ["fold"] if col in primary.columns])],
        on="planet_ID",
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        backup[["planet_ID", f"mu_{target_col}"]],
        on="planet_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_backup"),
    )
    frame = frame.rename(columns={f"mu_{target_col}": "mu_primary", f"mu_{target_col}_backup": "mu_backup"})
    frame["disagreement"] = np.abs(frame["mu_primary"] - frame["mu_backup"]) / np.sqrt(2.0)
    frame["residual"] = frame[target_col] - frame["mu_primary"]
    frame["abs_residual"] = np.abs(frame["residual"])
    return frame


def sigma_feature_cols() -> list[str]:
    return [
        "mu_primary",
        "disagreement",
        "edge_score_proxy",
        "meta_log_star_distance",
        "meta_log_star_mass_kg",
        "meta_log_star_radius_m",
        "meta_log_star_temperature",
        "meta_log_planet_mass_kg",
        "meta_log_planet_orbital_period",
        "meta_log_planet_distance",
        "meta_log_planet_radius_m",
        "meta_log_planet_surface_gravity",
        "meta_eqtemp_proxy",
        "meta_flux_proxy",
        "meta_density_proxy",
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


def fit_predict_methods(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    mu_train = train_df["mu_primary"].to_numpy(dtype=np.float64)
    disagreement_train = train_df["disagreement"].to_numpy(dtype=np.float64)
    disagreement_val = val_df["disagreement"].to_numpy(dtype=np.float64)

    predictions: dict[str, np.ndarray] = {}
    predictions["constant"] = np.full(len(val_df), fit_constant_sigma(y_train, mu_train, target_col), dtype=np.float64)

    _, residual_sigma_val = crossfit_residual_sigma(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=target_col,
        builder="summary",
        use_isotonic=False,
    )
    predictions["residual_model"] = residual_sigma_val

    disagreement_model = fit_scale_floor_from_raw_sigma(y_train, mu_train, disagreement_train, target_col)
    disagreement_sigma_train = apply_scale_floor(disagreement_train, disagreement_model)
    predictions["disagreement_scaled"] = apply_scale_floor(disagreement_val, disagreement_model)

    _, residual_sigma_train = crossfit_residual_sigma(
        train_df=train_df,
        val_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        builder="summary",
        use_isotonic=False,
    )
    blend_components_train = np.column_stack([residual_sigma_train, disagreement_sigma_train])
    blend_components_val = np.column_stack([residual_sigma_val, predictions["disagreement_scaled"]])
    blend_model = fit_log_linear_blend(y_train, mu_train, blend_components_train, target_col)
    predictions["blend_simple"] = apply_log_linear_blend(blend_components_val, blend_model)
    return predictions


def evaluate_edge_then_random(track: str, target_col: str, random_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    feature_cols = sigma_feature_cols()
    results = []
    stored_predictions: dict[str, pd.DataFrame] = {}

    edge_train = random_frame.loc[~random_frame["is_holdout"] & random_frame["is_valid"]].copy().reset_index(drop=True)
    edge_val = edge_frame.loc[edge_frame["is_valid"]].copy().reset_index(drop=True)
    edge_predictions = fit_predict_methods(edge_train, edge_val, target_col, feature_cols)
    y_edge = edge_val[target_col].to_numpy(dtype=np.float64)
    mu_edge = edge_val["mu_primary"].to_numpy(dtype=np.float64)
    for method_name, sigma in edge_predictions.items():
        results.append(
            {
                "track": track,
                "target": target_col,
                "scheme": "edge_holdout_15pct",
                "method": method_name,
                "primary_metric": target_score(y_edge, mu_edge, sigma, target_col),
                "mean_crps": target_crps(y_edge, mu_edge, sigma, target_col),
            }
        )
        stored_predictions[f"edge_holdout_15pct::{method_name}"] = edge_val[["planet_ID", target_col, "mu_primary"]].assign(sigma=sigma)

    working = random_frame.loc[random_frame["is_valid"]].copy().reset_index(drop=True)
    random_sigma = {method: np.full(len(working), np.nan, dtype=np.float64) for method in FAST_METHODS}
    for fold in sorted(working["fold"].unique().tolist()):
        train_df = working.loc[working["fold"] != fold].reset_index(drop=True)
        val_df = working.loc[working["fold"] == fold].reset_index(drop=True)
        fold_preds = fit_predict_methods(train_df, val_df, target_col, feature_cols)
        val_idx = working.index[working["fold"] == fold].to_numpy()
        for method_name, sigma in fold_preds.items():
            random_sigma[method_name][val_idx] = sigma

    y_random = working[target_col].to_numpy(dtype=np.float64)
    mu_random = working["mu_primary"].to_numpy(dtype=np.float64)
    for method_name, sigma in random_sigma.items():
        results.append(
            {
                "track": track,
                "target": target_col,
                "scheme": "random_5fold",
                "method": method_name,
                "primary_metric": target_score(y_random, mu_random, sigma, target_col),
                "mean_crps": target_crps(y_random, mu_random, sigma, target_col),
            }
        )
        stored_predictions[f"random_5fold::{method_name}"] = working[["planet_ID", target_col, "mu_primary"]].assign(sigma=sigma)

    return pd.DataFrame(results), stored_predictions


def fit_full_mean_predictions(candidate_name: str, target_cols: list[str], bundle, output_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    candidates = candidate_map()
    candidate = candidates[candidate_name]
    train_frame = bundle.train_frame.copy()
    preds = {}
    for target_col in target_cols:
        model = candidate.build_pipeline(bundle.feature_groups)
        model.fit(train_frame, train_frame[target_col].to_numpy(dtype=np.float64))
        preds[target_col] = np.asarray(model.predict(output_frame), dtype=np.float64)
    return preds


def fit_final_sigma_method(
    method_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> np.ndarray:
    feature_cols = sigma_feature_cols()
    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    mu_train = train_df["mu_primary"].to_numpy(dtype=np.float64)

    if method_name == "constant":
        sigma = fit_constant_sigma(y_train, mu_train, target_col)
        return np.full(len(test_df), sigma, dtype=np.float64)
    if method_name == "residual_model":
        _, sigma_test = crossfit_residual_sigma(
            train_df=train_df,
            val_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            builder="summary",
            use_isotonic=False,
        )
        return sigma_test
    if method_name == "disagreement_scaled":
        model = fit_scale_floor_from_raw_sigma(y_train, mu_train, train_df["disagreement"].to_numpy(dtype=np.float64), target_col)
        return apply_scale_floor(test_df["disagreement"].to_numpy(dtype=np.float64), model)
    if method_name == "blend_simple":
        _, residual_sigma_train = crossfit_residual_sigma(
            train_df=train_df,
            val_df=train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            builder="summary",
            use_isotonic=False,
        )
        _, residual_sigma_test = crossfit_residual_sigma(
            train_df=train_df,
            val_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            builder="summary",
            use_isotonic=False,
        )
        disagreement_model = fit_scale_floor_from_raw_sigma(
            y_train,
            mu_train,
            train_df["disagreement"].to_numpy(dtype=np.float64),
            target_col,
        )
        disagreement_sigma_train = apply_scale_floor(train_df["disagreement"].to_numpy(dtype=np.float64), disagreement_model)
        disagreement_sigma_test = apply_scale_floor(test_df["disagreement"].to_numpy(dtype=np.float64), disagreement_model)
        blend_model = fit_log_linear_blend(
            y_train,
            mu_train,
            np.column_stack([residual_sigma_train, disagreement_sigma_train]),
            target_col,
        )
        return apply_log_linear_blend(np.column_stack([residual_sigma_test, disagreement_sigma_test]), blend_model)
    raise ValueError(method_name)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle, schemes, train_frame, test_frame = build_base_frames()

    temp_random = build_temp_frame(train_frame, "random_5fold")
    temp_edge = build_temp_frame(train_frame, "edge_holdout_15pct")
    temp_metrics, temp_preds = evaluate_edge_then_random("temperature", TEMP_TARGET, temp_random, temp_edge)

    abundance_metric_frames = []
    abundance_preds_by_target = {}
    for target_col in ABUNDANCE_TARGETS:
        random_frame = build_abundance_frame(train_frame, "random_5fold", target_col)
        edge_frame = build_abundance_frame(train_frame, "edge_holdout_15pct", target_col)
        metric_frame, preds = evaluate_edge_then_random("abundance", target_col, random_frame, edge_frame)
        abundance_metric_frames.append(metric_frame)
        abundance_preds_by_target[target_col] = preds

    metrics_df = pd.concat([temp_metrics] + abundance_metric_frames, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "phase3_fast_metrics.csv", index=False)

    edge_rank = (
        metrics_df[metrics_df["scheme"].eq("edge_holdout_15pct")]
        .sort_values(["track", "target", "primary_metric"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    random_rank = (
        metrics_df[metrics_df["scheme"].eq("random_5fold")]
        .sort_values(["track", "target", "primary_metric"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    temp_choice = (
        metrics_df[metrics_df["target"].eq(TEMP_TARGET)]
        .pivot(index="method", columns="scheme", values="primary_metric")
        .assign(score=lambda df: df["edge_holdout_15pct"] * 2.0 + df["random_5fold"])
        .sort_values(["score", "edge_holdout_15pct"], ascending=False)
        .reset_index()
    )
    best_temp_method = str(temp_choice.iloc[0]["method"])
    fallback_temp_method = "disagreement_scaled" if "disagreement_scaled" in temp_choice["method"].tolist() else "constant"

    best_abundance_methods = {}
    fallback_abundance_methods = {}
    for target_col in ABUNDANCE_TARGETS:
        target_choice = (
            metrics_df[metrics_df["target"].eq(target_col)]
            .pivot(index="method", columns="scheme", values="primary_metric")
            .assign(score=lambda df: df["edge_holdout_15pct"] * 2.0 + df["random_5fold"])
            .sort_values(["score", "edge_holdout_15pct"], ascending=False)
            .reset_index()
        )
        best_abundance_methods[target_col] = str(target_choice.iloc[0]["method"])
        fallback_abundance_methods[target_col] = "disagreement_scaled" if "disagreement_scaled" in target_choice["method"].tolist() else "constant"

    def build_system_eval(name: str, temp_method: str, abundance_methods: dict[str, str]) -> list[dict]:
        rows = []
        for scheme_name in ["edge_holdout_15pct", "random_5fold"]:
            temp_pred = temp_preds[f"{scheme_name}::{temp_method}"][["planet_ID", TEMP_TARGET, "mu_primary", "sigma"]].copy()
            temp_pred = temp_pred.rename(columns={"mu_primary": f"mu_{TEMP_TARGET}", "sigma": f"sigma_{TEMP_TARGET}"})
            frame = temp_pred
            for target_col in ABUNDANCE_TARGETS:
                pred = abundance_preds_by_target[target_col][f"{scheme_name}::{abundance_methods[target_col]}"][
                    ["planet_ID", target_col, "mu_primary", "sigma"]
                ].copy()
                pred = pred.rename(columns={"mu_primary": f"mu_{target_col}", "sigma": f"sigma_{target_col}"})
                frame = frame.merge(pred, on="planet_ID", how="inner", validate="one_to_one")
            y = frame[TARGET_COLS].to_numpy(dtype=np.float64)
            mu = frame[[f"mu_{target}" for target in TARGET_COLS]].to_numpy(dtype=np.float64)
            sigma = frame[[f"sigma_{target}" for target in TARGET_COLS]].to_numpy(dtype=np.float64)
            score = score_numpy(y, mu, sigma, target_cols=TARGET_COLS)
            rows.append(
                {
                    "system": name,
                    "scheme": scheme_name,
                    "primary_metric": float(score["score"]),
                    "mean_crps": float(score["mean_crps"]),
                    "temp_method": temp_method,
                    "abundance_signature": "|".join(f"{target}:{abundance_methods[target]}" for target in ABUNDANCE_TARGETS),
                }
            )
        return rows

    system_rows = []
    system_rows.extend(build_system_eval("baseline_constant", "constant", {target: "constant" for target in ABUNDANCE_TARGETS}))
    system_rows.extend(build_system_eval("residual_only", "residual_model", {target: "residual_model" for target in ABUNDANCE_TARGETS}))
    system_rows.extend(build_system_eval("disagreement_only", fallback_temp_method, fallback_abundance_methods))
    system_rows.extend(build_system_eval("blend_uniform", "blend_simple", {target: "blend_simple" for target in ABUNDANCE_TARGETS}))
    system_rows.extend(build_system_eval("target_specific_best", best_temp_method, best_abundance_methods))
    system_df = pd.DataFrame(system_rows).sort_values(["scheme", "primary_metric"], ascending=[True, False]).reset_index(drop=True)
    system_df.to_csv(OUT_DIR / "phase3_fast_systems.csv", index=False)

    best_system = "target_specific_best"
    fallback_system = "disagreement_only"

    train_mu_temp = fit_full_mean_predictions(TEMP_PRIMARY, [TEMP_TARGET], bundle, bundle.test_frame)[TEMP_TARGET]
    train_mu_temp_backup = fit_full_mean_predictions(TEMP_BACKUP, [TEMP_TARGET], bundle, bundle.test_frame)[TEMP_TARGET]

    final_mu = np.zeros((len(bundle.test_frame), len(TARGET_COLS)), dtype=np.float64)
    final_std = np.zeros((len(bundle.test_frame), len(TARGET_COLS)), dtype=np.float64)

    test_sigma_frame = test_frame.copy()
    test_sigma_frame["mu_primary"] = train_mu_temp
    test_sigma_frame["mu_backup"] = train_mu_temp_backup
    test_sigma_frame["disagreement"] = np.abs(test_sigma_frame["mu_primary"] - test_sigma_frame["mu_backup"]) / np.sqrt(2.0)
    final_mu[:, TARGET_COLS.index(TEMP_TARGET)] = train_mu_temp
    final_std[:, TARGET_COLS.index(TEMP_TARGET)] = fit_final_sigma_method(best_temp_method, temp_random.loc[temp_random["is_valid"]].copy(), test_sigma_frame, TEMP_TARGET)

    primary_test_preds = fit_full_mean_predictions(ABUND_PRIMARY, ABUNDANCE_TARGETS, bundle, bundle.test_frame)
    backup_test_preds = fit_full_mean_predictions(ABUND_BACKUP, ABUNDANCE_TARGETS, bundle, bundle.test_frame)

    for target_col in ABUNDANCE_TARGETS:
        idx = TARGET_COLS.index(target_col)
        sigma_train_frame = build_abundance_frame(train_frame, "random_5fold", target_col)
        sigma_test_frame = test_frame.copy()
        sigma_test_frame["mu_primary"] = primary_test_preds[target_col]
        sigma_test_frame["mu_backup"] = backup_test_preds[target_col]
        sigma_test_frame["disagreement"] = np.abs(sigma_test_frame["mu_primary"] - sigma_test_frame["mu_backup"]) / np.sqrt(2.0)
        final_mu[:, idx] = primary_test_preds[target_col]
        final_std[:, idx] = fit_final_sigma_method(best_abundance_methods[target_col], sigma_train_frame.loc[sigma_train_frame["is_valid"]].copy(), sigma_test_frame, target_col)

    final_mu_df = array_to_submission(final_mu, planet_ids=bundle.test_frame["planet_ID"].to_numpy(dtype=np.int64))
    final_std_df = array_to_submission(np.maximum(final_std, 1e-6), planet_ids=bundle.test_frame["planet_ID"].to_numpy(dtype=np.int64))
    final_mu_df.to_csv(ROOT / "final_mu.csv", index=False)
    final_std_df.to_csv(ROOT / "final_std.csv", index=False)

    summary = f"""# Phase 3 Fast-Track Winner

## Best system
- Winner: `{best_system}`
- Conservative fallback: `{fallback_system}`

## Temperature
- Backbone: `{TEMP_PRIMARY}`
- Chosen sigma method: `{best_temp_method}`
- Fallback sigma method: `{fallback_temp_method}`

## Abundances
- Backbone: `{ABUND_PRIMARY}`
- Runner-up used for disagreement only: `{ABUND_BACKUP}`
- Chosen sigma methods: `{', '.join(f'{target}:{method}' for target, method in best_abundance_methods.items())}`
- Conservative fallback methods: `{', '.join(f'{target}:{method}' for target, method in fallback_abundance_methods.items())}`

## Method ranking
### Edge first
```
{format_frame(edge_rank, decimals=4)}
```

### Random confirmation
```
{format_frame(random_rank, decimals=4)}
```

### System ranking
```
{format_frame(system_df, decimals=4)}
```

## Files
- Mu: [final_mu.csv]({(ROOT / 'final_mu.csv').resolve()})
- Std: [final_std.csv]({(ROOT / 'final_std.csv').resolve()})
"""
    save_text(REPORT_DIR / "phase3_fasttrack_summary.md", summary)


if __name__ == "__main__":
    main()
