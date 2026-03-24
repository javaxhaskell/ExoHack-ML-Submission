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

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_phase3_fast import build_base_frames, fit_final_sigma_method
from src.models.builders import make_mlp, svd_with_side_features


ABUNDANCE_TARGETS = ["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
SINGLE_TARGETS = ["log_CO2", "log_CH4", "log_CO", "log_NH3"]
SEEDS = [42, 43, 44]

CURRENT_BEST_MU_PATH = ROOT / "final_mu_variant_single_target_hybrid.csv"
CURRENT_BEST_STD_PATH = ROOT / "final_std_variant_single_target_hybrid.csv"
SHARED_OOF_PATH = ROOT / "outputs" / "oof" / "abun_shared_mlp_svd_spec_noise_meta__random_5fold.csv.gz"
MU_OUT_PATH = ROOT / "final_mu_variant_seed_ensemble.csv"
STD_OUT_PATH = ROOT / "final_std_variant_seed_ensemble.csv"


def build_neural_pipeline(feature_groups: dict[str, list[str]], seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                svd_with_side_features(
                    feature_groups["spectrum_noise"],
                    passthrough_cols=feature_groups["metadata"],
                    n_components=24,
                ),
            ),
            ("scale", StandardScaler()),
            ("model", make_mlp(random_state=seed)),
        ]
    )


def fit_shared_h2o_seed(bundle, working: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(working), dtype=np.float64)
    folds = sorted(working["fold"].unique().tolist())
    for fold in folds:
        train_df = working.loc[working["fold"] != fold].reset_index(drop=True)
        val_idx = np.flatnonzero(working["fold"].to_numpy() == fold)
        val_df = working.loc[working["fold"] == fold].reset_index(drop=True)
        model = build_neural_pipeline(bundle.feature_groups, seed)
        model.fit(train_df, train_df[ABUNDANCE_TARGETS].to_numpy(dtype=np.float64))
        pred = np.asarray(model.predict(val_df), dtype=np.float64)
        oof[val_idx] = pred[:, ABUNDANCE_TARGETS.index("log_H2O")]

    full_model = build_neural_pipeline(bundle.feature_groups, seed)
    full_model.fit(bundle.train_frame, bundle.train_frame[ABUNDANCE_TARGETS].to_numpy(dtype=np.float64))
    test_pred = np.asarray(full_model.predict(bundle.test_frame), dtype=np.float64)[:, ABUNDANCE_TARGETS.index("log_H2O")]
    return oof, test_pred


def fit_single_target_seed(
    bundle,
    working: pd.DataFrame,
    target: str,
    seed: int,
    need_full_test: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    oof = np.zeros(len(working), dtype=np.float64)
    folds = sorted(working["fold"].unique().tolist())
    for fold in folds:
        train_df = working.loc[working["fold"] != fold].reset_index(drop=True)
        val_idx = np.flatnonzero(working["fold"].to_numpy() == fold)
        val_df = working.loc[working["fold"] == fold].reset_index(drop=True)
        model = build_neural_pipeline(bundle.feature_groups, seed)
        model.fit(train_df, train_df[target].to_numpy(dtype=np.float64))
        oof[val_idx] = np.asarray(model.predict(val_df), dtype=np.float64)

    if not need_full_test:
        return oof, None

    full_model = build_neural_pipeline(bundle.feature_groups, seed)
    full_model.fit(bundle.train_frame, bundle.train_frame[target].to_numpy(dtype=np.float64))
    test_pred = np.asarray(full_model.predict(bundle.test_frame), dtype=np.float64)
    return oof, test_pred


def main() -> None:
    bundle, schemes, train_frame, test_frame = build_base_frames()
    current_best_mu = pd.read_csv(CURRENT_BEST_MU_PATH)
    current_best_std = pd.read_csv(CURRENT_BEST_STD_PATH)

    random_scheme = schemes["random_5fold"].copy()
    working = train_frame.merge(random_scheme, on="planet_ID", how="inner", validate="one_to_one")
    working = working.sort_values("planet_ID").reset_index(drop=True)

    # Seed 42 for the shared H2O head is already present in the current winning artifacts.
    shared_seed42_oof = pd.read_csv(SHARED_OOF_PATH).sort_values("planet_ID").reset_index(drop=True)
    shared_seed42_oof = shared_seed42_oof["mu_log_H2O"].to_numpy(dtype=np.float64)
    shared_seed42_test = current_best_mu["log_H2O"].to_numpy(dtype=np.float64)

    shared_h2o_oof = [shared_seed42_oof]
    shared_h2o_test = [shared_seed42_test]
    for seed in [43, 44]:
        oof, test_pred = fit_shared_h2o_seed(bundle, working, seed)
        shared_h2o_oof.append(oof)
        shared_h2o_test.append(test_pred)
    shared_h2o_oof = np.column_stack(shared_h2o_oof)
    shared_h2o_test = np.column_stack(shared_h2o_test)

    # Seed 42 single-target test predictions are already present in the current winning file,
    # but their OOF predictions were not cached, so only those OOF fits are rerun.
    single_oof: dict[str, np.ndarray] = {}
    single_test: dict[str, np.ndarray] = {}
    for target in SINGLE_TARGETS:
        seed_oof = []
        seed_test = [current_best_mu[target].to_numpy(dtype=np.float64)]

        oof_42, _ = fit_single_target_seed(bundle, working, target, 42, need_full_test=False)
        seed_oof.append(oof_42)

        for seed in [43, 44]:
            oof_seed, test_seed = fit_single_target_seed(bundle, working, target, seed, need_full_test=True)
            seed_oof.append(oof_seed)
            seed_test.append(test_seed)

        single_oof[target] = np.column_stack(seed_oof)
        single_test[target] = np.column_stack(seed_test)

    mu_variant = current_best_mu.copy()
    mu_variant["log_H2O"] = shared_h2o_test.mean(axis=1)
    for target in SINGLE_TARGETS:
        mu_variant[target] = single_test[target].mean(axis=1)
    mu_variant.to_csv(MU_OUT_PATH, index=False)

    std_variant = current_best_std.copy()

    h2o_train = working.copy()
    h2o_train["mu_primary"] = shared_h2o_oof.mean(axis=1)
    h2o_train["disagreement"] = shared_h2o_oof.std(axis=1, ddof=0)
    h2o_train["residual"] = h2o_train["log_H2O"].to_numpy(dtype=np.float64) - h2o_train["mu_primary"].to_numpy(dtype=np.float64)
    h2o_train["abs_residual"] = np.abs(h2o_train["residual"].to_numpy(dtype=np.float64))
    h2o_test = test_frame.copy()
    h2o_test["mu_primary"] = mu_variant["log_H2O"].to_numpy(dtype=np.float64)
    h2o_test["disagreement"] = shared_h2o_test.std(axis=1, ddof=0)
    std_variant["log_H2O"] = np.maximum(
        fit_final_sigma_method("blend_simple", h2o_train, h2o_test, "log_H2O"),
        1e-6,
    )

    for target in SINGLE_TARGETS:
        train_sigma = working.copy()
        train_sigma["mu_primary"] = single_oof[target].mean(axis=1)
        train_sigma["disagreement"] = single_oof[target].std(axis=1, ddof=0)
        train_sigma["residual"] = train_sigma[target].to_numpy(dtype=np.float64) - train_sigma["mu_primary"].to_numpy(dtype=np.float64)
        train_sigma["abs_residual"] = np.abs(train_sigma["residual"].to_numpy(dtype=np.float64))
        test_sigma = test_frame.copy()
        test_sigma["mu_primary"] = mu_variant[target].to_numpy(dtype=np.float64)
        test_sigma["disagreement"] = single_test[target].std(axis=1, ddof=0)
        std_variant[target] = np.maximum(
            fit_final_sigma_method("blend_simple", train_sigma, test_sigma, target),
            1e-6,
        )

    std_variant.to_csv(STD_OUT_PATH, index=False)

    issues: list[str] = []
    if len(mu_variant) != len(std_variant):
        issues.append("row_count_mismatch")
    if not mu_variant["planet_ID"].equals(std_variant["planet_ID"]):
        issues.append("planet_id_order_mismatch")
    if mu_variant.isna().any().any() or std_variant.isna().any().any():
        issues.append("nan_found")
    if np.isinf(mu_variant.drop(columns=["planet_ID"]).to_numpy(dtype=np.float64)).any():
        issues.append("mu_inf_found")
    if np.isinf(std_variant.drop(columns=["planet_ID"]).to_numpy(dtype=np.float64)).any():
        issues.append("std_inf_found")
    if not (std_variant.drop(columns=["planet_ID"]).to_numpy(dtype=np.float64) > 0).all():
        issues.append("nonpositive_std_found")
    if not np.allclose(
        mu_variant["planet_temp"].to_numpy(dtype=np.float64),
        current_best_mu["planet_temp"].to_numpy(dtype=np.float64),
    ):
        issues.append("planet_temp_mean_changed")
    if not np.allclose(
        std_variant["planet_temp"].to_numpy(dtype=np.float64),
        current_best_std["planet_temp"].to_numpy(dtype=np.float64),
    ):
        issues.append("planet_temp_sigma_changed")

    print(
        json.dumps(
            {
                "three_seeds_each": True,
                "full_train_refits_needed": True,
                "sigma_construction": "target-specific log-linear blend of residual-model sigma and 3-seed disagreement on random_5fold OOF ensemble means",
                "planet_temp_mean_unchanged": "planet_temp_mean_changed" not in issues,
                "planet_temp_sigma_unchanged": "planet_temp_sigma_changed" not in issues,
                "validation_issues": issues,
                "mu_file": str(MU_OUT_PATH.resolve()),
                "std_file": str(STD_OUT_PATH.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
