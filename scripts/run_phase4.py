#!/usr/bin/env python
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.calibration import target_score
from src.metrics import TARGET_COLS
from scripts.run_phase3_fast import (
    ABUNDANCE_TARGETS,
    TEMP_TARGET,
    build_abundance_frame,
    build_base_frames,
    fit_predict_methods,
)


MU_PATH = ROOT / "final_mu.csv"
STD_PATH = ROOT / "final_std.csv"
MERGED_PATH = ROOT / "final_submission_ready.csv"
SUMMARY_PATH = ROOT / "final_summary.md"
METRICS_PATH = ROOT / "outputs" / "calibration" / "phase3_fast_metrics.csv"

MAINLINE_TEMP_METHOD = "constant"
MAINLINE_ABUNDANCE_METHODS = {
    "log_H2O": "residual_model",
    "log_CO2": "residual_model",
    "log_CH4": "disagreement_scaled",
    "log_CO": "residual_model",
    "log_NH3": "residual_model",
}
FALLBACK_TEMP_METHOD = "disagreement_scaled"
FALLBACK_ABUNDANCE_METHODS = {target: "disagreement_scaled" for target in ABUNDANCE_TARGETS}
LOG_CO_FACTORS = [1.1, 1.2, 1.3]


def validate_submission_files(mu: pd.DataFrame, std: pd.DataFrame) -> dict[str, object]:
    expected_cols = ["planet_ID"] + TARGET_COLS
    issues: list[str] = []

    if list(mu.columns) != expected_cols:
        issues.append(f"final_mu.csv columns mismatch: {list(mu.columns)}")
    if list(std.columns) != expected_cols:
        issues.append(f"final_std.csv columns mismatch: {list(std.columns)}")
    if mu.shape[1] != 7 or std.shape[1] != 7:
        issues.append(f"expected 7 columns per file, got mu={mu.shape[1]}, std={std.shape[1]}")
    if len(mu) != len(std):
        issues.append(f"row count mismatch: mu={len(mu)}, std={len(std)}")
    if not mu["planet_ID"].equals(std["planet_ID"]):
        issues.append("planet_ID ordering mismatch between mu and std")

    mu_values = mu[TARGET_COLS].to_numpy(dtype=np.float64)
    std_values = std[TARGET_COLS].to_numpy(dtype=np.float64)

    if not np.isfinite(mu_values).all():
        issues.append("final_mu.csv contains NaN or infinity")
    if not np.isfinite(std_values).all():
        issues.append("final_std.csv contains NaN or infinity")
    if not (std_values > 0).all():
        issues.append(f"final_std.csv contains {(std_values <= 0).sum()} non-positive std entries")

    sigma_summary = std[TARGET_COLS].agg(["min", "max", "mean", "median", "std"]).T.reset_index().rename(columns={"index": "target"})
    sigma_summary["p99"] = std[TARGET_COLS].quantile(0.99).values
    sigma_summary["p999"] = std[TARGET_COLS].quantile(0.999).values

    extreme_rules = []
    for target in TARGET_COLS:
        col = std[target]
        q1 = float(col.quantile(0.25))
        q3 = float(col.quantile(0.75))
        iqr = q3 - q1
        high_fence = q3 + 10.0 * iqr
        extreme_count = int((col > high_fence).sum()) if iqr > 0 else 0
        extreme_rules.append(
            {
                "target": target,
                "high_fence": high_fence,
                "extreme_count": extreme_count,
            }
        )
    extreme_frame = pd.DataFrame(extreme_rules)

    return {
        "issues": issues,
        "sigma_summary": sigma_summary,
        "extreme_frame": extreme_frame,
    }


def load_target_metric_table() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


def system_score_from_metrics(
    metrics_df: pd.DataFrame,
    temp_method: str,
    abundance_methods: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for scheme in ["edge_holdout_15pct", "random_5fold"]:
        temp_row = metrics_df[
            (metrics_df["target"].eq(TEMP_TARGET))
            & (metrics_df["scheme"].eq(scheme))
            & (metrics_df["method"].eq(temp_method))
        ].copy()
        if temp_row.empty:
            raise ValueError(f"missing metric row for {TEMP_TARGET} / {scheme} / {temp_method}")
        rows.append(temp_row)
        for target, method in abundance_methods.items():
            target_row = metrics_df[
                (metrics_df["target"].eq(target))
                & (metrics_df["scheme"].eq(scheme))
                & (metrics_df["method"].eq(method))
            ].copy()
            if target_row.empty:
                raise ValueError(f"missing metric row for {target} / {scheme} / {method}")
            rows.append(target_row)

    detail = pd.concat(rows, ignore_index=True)
    system = (
        detail.groupby("scheme", as_index=False)
        .agg(total_score=("primary_metric", "mean"), mean_crps=("mean_crps", "mean"))
        .sort_values("scheme")
        .reset_index(drop=True)
    )
    return system, detail.sort_values(["scheme", "target"]).reset_index(drop=True)


def _feature_cols() -> list[str]:
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


def compute_log_co_residual_sigma() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, train_frame, _ = build_base_frames()
    random_frame = build_abundance_frame(train_frame, "random_5fold", "log_CO")
    edge_frame = build_abundance_frame(train_frame, "edge_holdout_15pct", "log_CO")
    feature_cols = _feature_cols()

    edge_train = random_frame.loc[~random_frame["is_holdout"] & random_frame["is_valid"]].copy().reset_index(drop=True)
    edge_val = edge_frame.loc[edge_frame["is_valid"]].copy().reset_index(drop=True)
    edge_preds = fit_predict_methods(edge_train, edge_val, "log_CO", feature_cols)

    working = random_frame.loc[random_frame["is_valid"]].copy().reset_index(drop=True)
    sigma_random = np.full(len(working), np.nan, dtype=np.float64)
    for fold in sorted(working["fold"].dropna().unique().tolist()):
        train_df = working.loc[working["fold"] != fold].reset_index(drop=True)
        val_mask = working["fold"].eq(fold)
        val_df = working.loc[val_mask].reset_index(drop=True)
        fold_preds = fit_predict_methods(train_df, val_df, "log_CO", feature_cols)
        sigma_random[np.flatnonzero(val_mask.to_numpy())] = fold_preds["residual_model"]

    edge_output = edge_val[["planet_ID", "log_CO", "mu_primary"]].copy()
    edge_output["sigma_base"] = edge_preds["residual_model"]
    random_output = working[["planet_ID", "log_CO", "mu_primary"]].copy()
    random_output["sigma_base"] = sigma_random
    return edge_output, random_output


def compute_log_co_scaled_scores(edge_output: pd.DataFrame, random_output: pd.DataFrame, factor: float) -> pd.DataFrame:
    edge_score = target_score(
        edge_output["log_CO"].to_numpy(dtype=np.float64),
        edge_output["mu_primary"].to_numpy(dtype=np.float64),
        edge_output["sigma_base"].to_numpy(dtype=np.float64) * factor,
        "log_CO",
    )
    random_score = target_score(
        random_output["log_CO"].to_numpy(dtype=np.float64),
        random_output["mu_primary"].to_numpy(dtype=np.float64),
        random_output["sigma_base"].to_numpy(dtype=np.float64) * factor,
        "log_CO",
    )
    return pd.DataFrame(
        [
            {"scheme": "edge_holdout_15pct", "factor": factor, "log_CO_score": edge_score},
            {"scheme": "random_5fold", "factor": factor, "log_CO_score": random_score},
        ]
    )


def choose_log_co_tweak(metrics_df: pd.DataFrame) -> tuple[float | None, pd.DataFrame]:
    baseline = (
        metrics_df[
            (metrics_df["target"].eq("log_CO"))
            & (metrics_df["method"].eq("residual_model"))
            & (metrics_df["scheme"].isin(["edge_holdout_15pct", "random_5fold"]))
        ][["scheme", "primary_metric"]]
        .rename(columns={"primary_metric": "baseline_score"})
        .sort_values("scheme")
        .reset_index(drop=True)
    )
    trials = []
    edge_output, random_output = compute_log_co_residual_sigma()
    for factor in LOG_CO_FACTORS:
        trial = compute_log_co_scaled_scores(edge_output, random_output, factor)
        trial = trial.merge(baseline, on="scheme", how="left", validate="one_to_one")
        trial["delta_score"] = trial["log_CO_score"] - trial["baseline_score"]
        trials.append(trial)
    results = pd.concat(trials, ignore_index=True)

    pivot = results.pivot(index="factor", columns="scheme", values="delta_score").reset_index()
    pivot["priority_score"] = pivot["edge_holdout_15pct"] * 2.0 + pivot["random_5fold"]
    pivot = pivot.sort_values(["priority_score", "edge_holdout_15pct"], ascending=False).reset_index(drop=True)
    best = pivot.iloc[0]
    if float(best["priority_score"]) > 0 and float(best["edge_holdout_15pct"]) > 0:
        return float(best["factor"]), results
    return None, results


def apply_log_co_scale_if_needed(scale_factor: float | None) -> bool:
    if scale_factor is None:
        return False
    std = pd.read_csv(STD_PATH)
    std["log_CO"] = np.maximum(std["log_CO"].to_numpy(dtype=np.float64) * scale_factor, 1e-6)
    std.to_csv(STD_PATH, index=False)
    return True


def apply_log_co_trial_to_scores(
    mainline_system: pd.DataFrame,
    mainline_detail: pd.DataFrame,
    log_co_scale: float | None,
    log_co_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if log_co_scale is None:
        return mainline_system.copy(), mainline_detail.copy()

    updated_detail = mainline_detail.copy()
    chosen = log_co_trials[log_co_trials["factor"].eq(log_co_scale)].copy()
    for _, row in chosen.iterrows():
        scheme = row["scheme"]
        new_score = float(row["log_CO_score"])
        mask = updated_detail["scheme"].eq(scheme) & updated_detail["target"].eq("log_CO")
        updated_detail.loc[mask, "primary_metric"] = new_score
    updated_system = (
        updated_detail.groupby("scheme", as_index=False)
        .agg(total_score=("primary_metric", "mean"), mean_crps=("mean_crps", "mean"))
        .sort_values("scheme")
        .reset_index(drop=True)
    )
    return updated_system, updated_detail.sort_values(["scheme", "target"]).reset_index(drop=True)


def write_merged_submission(mu: pd.DataFrame, std: pd.DataFrame) -> None:
    merged = pd.DataFrame({"planet_ID": mu["planet_ID"].to_numpy(dtype=np.int64)})
    for target in TARGET_COLS:
        merged[f"{target}_mu"] = mu[target].to_numpy(dtype=np.float64)
        merged[f"{target}_std"] = std[target].to_numpy(dtype=np.float64)
    merged.to_csv(MERGED_PATH, index=False)


def write_summary(
    validation: dict[str, object],
    mainline_system: pd.DataFrame,
    mainline_detail: pd.DataFrame,
    fallback_system: pd.DataFrame,
    fallback_detail: pd.DataFrame,
    log_co_scale: float | None,
    log_co_trials: pd.DataFrame,
) -> None:
    sigma_summary = validation["sigma_summary"].copy()
    extreme_frame = validation["extreme_frame"].copy()
    issues = validation["issues"]

    lines = [
        "# Final Submission Summary",
        "",
        "## Validation",
        f"- `final_mu.csv` rows: {int(pd.read_csv(MU_PATH).shape[0])}",
        f"- `final_std.csv` rows: {int(pd.read_csv(STD_PATH).shape[0])}",
        f"- Identical `planet_ID` ordering: {'yes' if not issues or 'planet_ID ordering mismatch between mu and std' not in issues else 'no'}",
        f"- NaN / inf issues: {'none' if not any('NaN or infinity' in issue for issue in issues) else '; '.join(issue for issue in issues if 'NaN or infinity' in issue)}",
        f"- Non-positive std issues: {'none' if not any('non-positive' in issue for issue in issues) else '; '.join(issue for issue in issues if 'non-positive' in issue)}",
        "",
        "## Sigma summary",
        "```text",
        sigma_summary.round(6).to_string(index=False),
        "```",
        "",
        "## Extreme sigma check",
        "```text",
        extreme_frame.round(6).to_string(index=False),
        "```",
        "",
        "## Approximate OOF score: mainline",
        "```text",
        mainline_system.round(6).to_string(index=False),
        "```",
        "",
        "### Mainline per-target contributions",
        "```text",
        mainline_detail[["scheme", "target", "method", "primary_metric", "mean_crps"]].round(6).to_string(index=False),
        "```",
        "",
        "## Approximate OOF score: fallback",
        "```text",
        fallback_system.round(6).to_string(index=False),
        "```",
        "",
        "### Fallback per-target contributions",
        "```text",
        fallback_detail[["scheme", "target", "method", "primary_metric", "mean_crps"]].round(6).to_string(index=False),
        "```",
        "",
        "## One final tweak",
        "- Tested option A only: multiplicative `log_CO` sigma inflation on the mainline residual-model sigma.",
        "```text",
        log_co_trials.round(6).to_string(index=False),
        "```",
        f"- Adopted tweak: {'yes, scale log_CO sigma by ' + str(log_co_scale) if log_co_scale is not None else 'no'}",
        "",
        "## Final winner",
        f"- Final system: {'mainline + log_CO inflation' if log_co_scale is not None else 'mainline'}",
        "- Temperature: `temp_rf_meta` mean + constant sigma",
        "- Abundances: `abun_sep_et_spec_noise_meta_deriv` mean + target-specific sigma",
        "- Actual competition upload uses the separate files `final_mu.csv` and `final_std.csv`.",
        "",
        "## Files",
        f"- [final_mu.csv]({MU_PATH})",
        f"- [final_std.csv]({STD_PATH})",
        f"- [final_submission_ready.csv]({MERGED_PATH})",
    ]
    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    mu = pd.read_csv(MU_PATH)
    std = pd.read_csv(STD_PATH)
    validation = validate_submission_files(mu, std)
    if validation["issues"]:
        raise ValueError("Validation failed: " + "; ".join(validation["issues"]))

    metrics_df = load_target_metric_table()
    mainline_system, mainline_detail = system_score_from_metrics(
        metrics_df,
        temp_method=MAINLINE_TEMP_METHOD,
        abundance_methods=MAINLINE_ABUNDANCE_METHODS,
    )
    fallback_system, fallback_detail = system_score_from_metrics(
        metrics_df,
        temp_method=FALLBACK_TEMP_METHOD,
        abundance_methods=FALLBACK_ABUNDANCE_METHODS,
    )

    log_co_scale, log_co_trials = choose_log_co_tweak(metrics_df)
    mainline_system, mainline_detail = apply_log_co_trial_to_scores(
        mainline_system=mainline_system,
        mainline_detail=mainline_detail,
        log_co_scale=log_co_scale,
        log_co_trials=log_co_trials,
    )
    if apply_log_co_scale_if_needed(log_co_scale):
        std = pd.read_csv(STD_PATH)
        validation = validate_submission_files(mu, std)
        if validation["issues"]:
            raise ValueError("Validation failed after log_CO tweak: " + "; ".join(validation["issues"]))

    write_merged_submission(mu, pd.read_csv(STD_PATH))
    write_summary(
        validation=validation,
        mainline_system=mainline_system,
        mainline_detail=mainline_detail,
        fallback_system=fallback_system,
        fallback_detail=fallback_detail,
        log_co_scale=log_co_scale,
        log_co_trials=log_co_trials,
    )

    print("Phase 4 complete")
    print(f"mainline_edge={mainline_system.loc[mainline_system['scheme'].eq('edge_holdout_15pct'), 'total_score'].iloc[0]:.6f}")
    print(f"mainline_random={mainline_system.loc[mainline_system['scheme'].eq('random_5fold'), 'total_score'].iloc[0]:.6f}")
    print(f"fallback_edge={fallback_system.loc[fallback_system['scheme'].eq('edge_holdout_15pct'), 'total_score'].iloc[0]:.6f}")
    print(f"fallback_random={fallback_system.loc[fallback_system['scheme'].eq('random_5fold'), 'total_score'].iloc[0]:.6f}")
    print(f"log_co_scale={log_co_scale}")


if __name__ == "__main__":
    main()
