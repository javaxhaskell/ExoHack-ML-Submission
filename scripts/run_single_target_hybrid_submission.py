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

from scripts.run_phase3_fast import build_base_frames, fit_final_sigma_method
from src.calibration import fit_constant_sigma, target_score
from src.models.catalog import phase2_candidates
from src.train import load_phase1_schemes


TARGETS = ["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
SHARED_NAME = "abun_shared_mlp_svd_spec_noise_meta"
CURRENT_BEST_MU_PATH = ROOT / "final_mu_variant_win_or_die.csv"
CURRENT_BEST_STD_PATH = ROOT / "final_std_variant_sharedmlp_randomsigma.csv"
SHARED_OOF_PATH = ROOT / "outputs" / "oof" / f"{SHARED_NAME}__random_5fold.csv.gz"
MU_OUT_PATH = ROOT / "final_mu_variant_single_target_hybrid.csv"
STD_OUT_PATH = ROOT / "final_std_variant_single_target_hybrid.csv"

SIGMA_FEATURE_COLS = [
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


def _shared_candidate():
    return {candidate.name: candidate for candidate in phase2_candidates()}[SHARED_NAME]


def _build_single_target_pipeline(feature_groups: dict[str, list[str]]):
    candidate = _shared_candidate()
    return candidate.build_pipeline(feature_groups)


def _load_shared_oof(train_frame: pd.DataFrame) -> pd.DataFrame:
    shared = pd.read_csv(SHARED_OOF_PATH)
    keep = ["planet_ID", "fold", "is_valid"] + [f"mu_{target}" for target in TARGETS]
    frame = train_frame.merge(shared[keep], on="planet_ID", how="inner", validate="one_to_one")
    return frame.loc[frame["is_valid"]].copy().reset_index(drop=True)


def _fit_single_target_oof(bundle, schemes, train_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    scheme = schemes["random_5fold"].copy()
    working = train_frame.merge(scheme, on="planet_ID", how="inner", validate="one_to_one")
    working = working.sort_values("planet_ID").reset_index(drop=True)

    oof = working[["planet_ID", "fold"]].copy()
    test_preds: dict[str, np.ndarray] = {}

    for target in TARGETS:
        oof[f"mu_{target}"] = np.nan

    for fold in sorted(working["fold"].unique().tolist()):
        train_df = working.loc[working["fold"] != fold].reset_index(drop=True)
        val_idx = np.flatnonzero(working["fold"].to_numpy() == fold)
        val_df = working.loc[working["fold"] == fold].reset_index(drop=True)
        for target in TARGETS:
            model = _build_single_target_pipeline(bundle.feature_groups)
            model.fit(train_df, train_df[target].to_numpy(dtype=np.float64))
            oof.loc[val_idx, f"mu_{target}"] = np.asarray(model.predict(val_df), dtype=np.float64)

    for target in TARGETS:
        model = _build_single_target_pipeline(bundle.feature_groups)
        model.fit(bundle.train_frame, bundle.train_frame[target].to_numpy(dtype=np.float64))
        test_preds[target] = np.asarray(model.predict(bundle.test_frame), dtype=np.float64)

    return oof, test_preds


def _shared_sigma_oof(shared_frame: pd.DataFrame, target: str) -> tuple[np.ndarray, float]:
    sigma_train = shared_frame.copy()
    sigma_train["mu_primary"] = sigma_train[f"mu_{target}"].to_numpy(dtype=np.float64)
    sigma_train["disagreement"] = np.zeros(len(sigma_train), dtype=np.float64)
    sigma_train["residual"] = sigma_train[target].to_numpy(dtype=np.float64) - sigma_train["mu_primary"].to_numpy(dtype=np.float64)
    sigma_train["abs_residual"] = np.abs(sigma_train["residual"].to_numpy(dtype=np.float64))

    sigma_oof = fit_final_sigma_method("residual_model", sigma_train, sigma_train, target)
    score = target_score(
        sigma_train[target].to_numpy(dtype=np.float64),
        sigma_train["mu_primary"].to_numpy(dtype=np.float64),
        sigma_oof,
        target,
    )
    return sigma_oof, float(score)


def _single_sigma_oof(shared_frame: pd.DataFrame, single_oof: pd.DataFrame, target: str) -> tuple[np.ndarray, float]:
    sigma_train = shared_frame.copy()
    sigma_train["mu_primary"] = single_oof[f"mu_{target}"].to_numpy(dtype=np.float64)
    sigma_train["disagreement"] = np.abs(
        single_oof[f"mu_{target}"].to_numpy(dtype=np.float64)
        - shared_frame[f"mu_{target}"].to_numpy(dtype=np.float64)
    ) / np.sqrt(2.0)
    sigma_train["residual"] = sigma_train[target].to_numpy(dtype=np.float64) - sigma_train["mu_primary"].to_numpy(dtype=np.float64)
    sigma_train["abs_residual"] = np.abs(sigma_train["residual"].to_numpy(dtype=np.float64))

    sigma_oof = fit_final_sigma_method("residual_model", sigma_train, sigma_train, target)
    score = target_score(
        sigma_train[target].to_numpy(dtype=np.float64),
        sigma_train["mu_primary"].to_numpy(dtype=np.float64),
        sigma_oof,
        target,
    )
    return sigma_oof, float(score)


def main() -> None:
    bundle, schemes, train_frame, test_frame = build_base_frames()
    shared_oof = _load_shared_oof(train_frame)
    single_oof, single_test_preds = _fit_single_target_oof(bundle, schemes, train_frame)

    current_best_mu = pd.read_csv(CURRENT_BEST_MU_PATH)
    current_best_std = pd.read_csv(CURRENT_BEST_STD_PATH)

    source_choice: dict[str, str] = {}
    sigma_source_choice: dict[str, str] = {}
    shared_test_preds = {target: current_best_mu[target].to_numpy(dtype=np.float64) for target in TARGETS}

    mu_variant = current_best_mu.copy()
    std_variant = current_best_std.copy()

    for target in TARGETS:
        shared_sigma_oof, shared_score = _shared_sigma_oof(shared_oof, target)
        single_sigma_oof, single_score = _single_sigma_oof(shared_oof, single_oof, target)

        if single_score > shared_score:
            source_choice[target] = "single-target MLP"
            sigma_source_choice[target] = "single-target residual-model sigma"
            mu_variant[target] = single_test_preds[target]

            sigma_train = shared_oof.copy()
            sigma_train["mu_primary"] = single_oof[f"mu_{target}"].to_numpy(dtype=np.float64)
            sigma_train["disagreement"] = np.abs(
                single_oof[f"mu_{target}"].to_numpy(dtype=np.float64)
                - shared_oof[f"mu_{target}"].to_numpy(dtype=np.float64)
            ) / np.sqrt(2.0)
            sigma_train["residual"] = sigma_train[target].to_numpy(dtype=np.float64) - sigma_train["mu_primary"].to_numpy(dtype=np.float64)
            sigma_train["abs_residual"] = np.abs(sigma_train["residual"].to_numpy(dtype=np.float64))

            sigma_test = test_frame.copy()
            sigma_test["mu_primary"] = single_test_preds[target]
            sigma_test["disagreement"] = np.abs(
                single_test_preds[target] - shared_test_preds[target]
            ) / np.sqrt(2.0)
            std_variant[target] = np.maximum(
                fit_final_sigma_method("residual_model", sigma_train, sigma_test, target),
                1e-6,
            )
        else:
            source_choice[target] = "shared MLP"
            sigma_source_choice[target] = "current best shared-MLP residual sigma"
            mu_variant[target] = shared_test_preds[target]
            std_variant[target] = current_best_std[target].to_numpy(dtype=np.float64)

    mu_variant.to_csv(MU_OUT_PATH, index=False)
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
                "source_choice": source_choice,
                "sigma_source_choice": sigma_source_choice,
                "full_train_refits_needed": True,
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
