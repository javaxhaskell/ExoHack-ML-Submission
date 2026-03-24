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
from sklearn.linear_model import LinearRegression, Ridge

from scripts.run_phase3_fast import build_base_frames, fit_final_sigma_method
from src.calibration import fit_constant_sigma, target_score


TARGETS = ["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
SHARED_NAME = "abun_shared_mlp_svd_spec_noise_meta"
ET_NAME = "abun_sep_et_spec_noise_meta_deriv"

CURRENT_BEST_MU_PATH = ROOT / "final_mu_variant_win_or_die.csv"
CURRENT_BEST_STD_PATH = ROOT / "final_std_variant_sharedmlp_randomsigma.csv"
ET_MU_PATH = ROOT / "final_mu.csv"

SHARED_OOF_PATH = ROOT / "outputs" / "oof" / f"{SHARED_NAME}__random_5fold.csv.gz"
ET_OOF_PATH = ROOT / "outputs" / "oof" / f"{ET_NAME}__random_5fold.csv.gz"

MU_OUT_PATH = ROOT / "final_mu_variant_stacked_abundance.csv"
STD_OUT_PATH = ROOT / "final_std_variant_stacked_abundance.csv"

META_CANDIDATES = [
    ("linear_regression", 0, lambda: LinearRegression()),
    ("non_negative_linear", 1, lambda: LinearRegression(positive=True)),
    ("ridge_regression", 2, lambda: Ridge(alpha=1.0)),
]


def _validate_frame_columns(frame: pd.DataFrame) -> None:
    expected = ["planet_ID", "planet_temp"] + TARGETS
    if frame.columns.tolist() != expected:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")


def _load_train_stack_frame(train_frame: pd.DataFrame) -> pd.DataFrame:
    shared = pd.read_csv(SHARED_OOF_PATH)
    et = pd.read_csv(ET_OOF_PATH)

    if not shared["planet_ID"].equals(et["planet_ID"]):
        raise ValueError("Shared and ET OOF exports are not aligned on planet_ID")

    keep_shared = ["planet_ID", "is_valid", "fold"] + [f"mu_{target}" for target in TARGETS]
    keep_et = ["planet_ID"] + [f"mu_{target}" for target in TARGETS]

    merged = train_frame.merge(shared[keep_shared], on="planet_ID", how="inner", validate="one_to_one")
    merged = merged.merge(et[keep_et], on="planet_ID", how="inner", validate="one_to_one", suffixes=("_shared", "_et"))
    merged = merged.loc[merged["is_valid"]].copy().reset_index(drop=True)
    return merged


def _candidate_rows(target: str, target_frame: pd.DataFrame) -> list[dict[str, object]]:
    x = target_frame[[f"mu_{target}_shared", f"mu_{target}_et"]].to_numpy(dtype=np.float64)
    y = target_frame[target].to_numpy(dtype=np.float64)
    folds = target_frame["fold"].to_numpy(dtype=np.int64)

    rows: list[dict[str, object]] = []
    unique_folds = sorted(np.unique(folds).tolist())
    for name, simplicity_rank, builder in META_CANDIDATES:
        oof_pred = np.zeros(len(target_frame), dtype=np.float64)
        for fold in unique_folds:
            train_mask = folds != fold
            val_mask = folds == fold
            model = builder()
            model.fit(x[train_mask], y[train_mask])
            oof_pred[val_mask] = np.asarray(model.predict(x[val_mask]), dtype=np.float64)

        sigma_const = fit_constant_sigma(y, oof_pred, target)
        score = target_score(y, oof_pred, np.full(len(oof_pred), sigma_const, dtype=np.float64), target)
        rows.append(
            {
                "target": target,
                "meta_learner": name,
                "simplicity_rank": simplicity_rank,
                "score": float(score),
                "sigma_const": float(sigma_const),
                "oof_pred": oof_pred,
            }
        )
    return rows


def _choose_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    ranked = sorted(rows, key=lambda row: (-float(row["score"]), int(row["simplicity_rank"])))
    return ranked[0]


def _fit_full_stack_predictions(
    target: str,
    target_frame: pd.DataFrame,
    shared_test: pd.Series,
    et_test: pd.Series,
    learner_name: str,
) -> np.ndarray:
    x_train = target_frame[[f"mu_{target}_shared", f"mu_{target}_et"]].to_numpy(dtype=np.float64)
    y_train = target_frame[target].to_numpy(dtype=np.float64)
    x_test = np.column_stack(
        [
            shared_test.to_numpy(dtype=np.float64),
            et_test.to_numpy(dtype=np.float64),
        ]
    )
    builder = {name: factory for name, _, factory in META_CANDIDATES}[learner_name]
    model = builder()
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float64)


def main() -> None:
    bundle, _, train_frame, test_frame = build_base_frames()
    current_best_mu = pd.read_csv(CURRENT_BEST_MU_PATH)
    current_best_std = pd.read_csv(CURRENT_BEST_STD_PATH)
    et_mu = pd.read_csv(ET_MU_PATH)

    _validate_frame_columns(current_best_mu)
    _validate_frame_columns(et_mu)
    _validate_frame_columns(current_best_std)

    train_stack = _load_train_stack_frame(train_frame)

    meta_choice: dict[str, str] = {}
    stacked_oof: dict[str, np.ndarray] = {}
    stacked_test: dict[str, np.ndarray] = {}

    for target in TARGETS:
        rows = _candidate_rows(target, train_stack)
        best = _choose_candidate(rows)
        learner_name = str(best["meta_learner"])
        meta_choice[target] = learner_name
        stacked_oof[target] = np.asarray(best["oof_pred"], dtype=np.float64)
        stacked_test[target] = _fit_full_stack_predictions(
            target=target,
            target_frame=train_stack,
            shared_test=current_best_mu[target],
            et_test=et_mu[target],
            learner_name=learner_name,
        )

    mu_variant = current_best_mu.copy()
    for target in TARGETS:
        mu_variant[target] = stacked_test[target]
    mu_variant.to_csv(MU_OUT_PATH, index=False)

    std_variant = current_best_std.copy()
    for target in TARGETS:
        sigma_train = train_stack.copy()
        sigma_train["mu_primary"] = stacked_oof[target]
        sigma_train["disagreement"] = np.abs(
            sigma_train[f"mu_{target}_shared"].to_numpy(dtype=np.float64)
            - sigma_train[f"mu_{target}_et"].to_numpy(dtype=np.float64)
        ) / np.sqrt(2.0)
        sigma_train["residual"] = sigma_train[target].to_numpy(dtype=np.float64) - sigma_train["mu_primary"].to_numpy(dtype=np.float64)
        sigma_train["abs_residual"] = np.abs(sigma_train["residual"].to_numpy(dtype=np.float64))

        sigma_test = test_frame.copy()
        sigma_test["mu_primary"] = stacked_test[target]
        sigma_test["disagreement"] = np.abs(
            current_best_mu[target].to_numpy(dtype=np.float64) - et_mu[target].to_numpy(dtype=np.float64)
        ) / np.sqrt(2.0)
        std_variant[target] = np.maximum(
            fit_final_sigma_method("residual_model", sigma_train, sigma_test, target),
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

    summary = {
        "meta_choice": meta_choice,
        "full_train_refits_needed": False,
        "planet_temp_mean_unchanged": "planet_temp_mean_changed" not in issues,
        "planet_temp_sigma_unchanged": "planet_temp_sigma_changed" not in issues,
        "stacked_abundance_sigma_method": "target-specific residual-model sigma on stacked random_5fold OOF predictions",
        "validation_issues": issues,
        "mu_file": str(MU_OUT_PATH.resolve()),
        "std_file": str(STD_OUT_PATH.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
