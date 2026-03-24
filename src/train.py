from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import EvaluationResult, summarize_predictions
from src.features import FeatureBundle
from src.models.catalog import CandidateSpec


def load_phase1_schemes(fold_dir: str | Path) -> dict[str, pd.DataFrame]:
    fold_dir = Path(fold_dir)
    return {
        "random_5fold": pd.read_csv(fold_dir / "phase1_random_5fold.csv"),
        "regime_group_5fold": pd.read_csv(fold_dir / "phase1_regime_group_5fold.csv"),
        "edge_holdout_15pct": pd.read_csv(fold_dir / "phase1_edge_holdout.csv"),
    }


def _align_scheme(train_frame: pd.DataFrame, scheme_df: pd.DataFrame) -> pd.DataFrame:
    merged = train_frame[["planet_ID"]].merge(scheme_df, on="planet_ID", how="left", validate="one_to_one")
    if merged.isna().any().any():
        missing = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"Scheme alignment missing values in columns: {missing}")
    return merged


def _fit_predict_independent(
    candidate: CandidateSpec,
    feature_groups: dict[str, list[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> np.ndarray:
    preds = np.zeros((len(val_df), len(candidate.target_cols)), dtype=np.float64)
    for idx, target in enumerate(candidate.target_cols):
        model = candidate.build_pipeline(feature_groups)
        model.fit(train_df, train_df[target].to_numpy(dtype=np.float64))
        preds[:, idx] = model.predict(val_df)
    return preds


def _fit_predict_multioutput(
    candidate: CandidateSpec,
    feature_groups: dict[str, list[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> np.ndarray:
    model = candidate.build_pipeline(feature_groups)
    model.fit(train_df, train_df[candidate.target_cols].to_numpy(dtype=np.float64))
    return np.asarray(model.predict(val_df), dtype=np.float64)


def _predict_candidate(
    candidate: CandidateSpec,
    feature_groups: dict[str, list[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> np.ndarray:
    if candidate.training_mode == "independent":
        return _fit_predict_independent(candidate, feature_groups, train_df, val_df)
    if candidate.training_mode == "multioutput":
        return _fit_predict_multioutput(candidate, feature_groups, train_df, val_df)
    raise ValueError(f"Unsupported training_mode: {candidate.training_mode}")


def _oof_template(train_frame: pd.DataFrame, candidate: CandidateSpec) -> pd.DataFrame:
    oof = train_frame[["planet_ID"]].copy()
    for target in candidate.target_cols:
        oof[f"mu_{target}"] = np.nan
    oof["is_valid"] = False
    return oof


def evaluate_oof_export(
    oof_export: pd.DataFrame,
    candidate: CandidateSpec,
    feature_bundle: FeatureBundle,
    scheme_name: str,
) -> EvaluationResult:
    train_frame = feature_bundle.train_frame
    merged = train_frame[["planet_ID"] + candidate.target_cols].merge(
        oof_export[["planet_ID", "is_valid"] + [f"mu_{target}" for target in candidate.target_cols]],
        on="planet_ID",
        how="inner",
        validate="one_to_one",
    )
    valid_mask = merged["is_valid"].to_numpy(dtype=bool)
    if not valid_mask.any():
        raise ValueError(f"No validation rows found in cached OOF export for {candidate.name} on {scheme_name}")

    mu = merged.loc[valid_mask, [f"mu_{target}" for target in candidate.target_cols]].to_numpy(dtype=np.float64)
    y_true = merged.loc[valid_mask, candidate.target_cols].to_numpy(dtype=np.float64)
    return summarize_predictions(
        y_true=y_true,
        mu=mu,
        target_cols=candidate.target_cols,
        candidate_name=candidate.name,
        scheme_name=scheme_name,
    )


def run_candidate_on_scheme(
    candidate: CandidateSpec,
    feature_bundle: FeatureBundle,
    scheme_name: str,
    scheme_df: pd.DataFrame,
    oof_dir: str | Path,
    reuse_existing: bool = False,
) -> tuple[pd.DataFrame, EvaluationResult]:
    out_path = Path(oof_dir) / f"{candidate.name}__{scheme_name}.csv.gz"
    if reuse_existing and out_path.exists():
        cached = pd.read_csv(out_path)
        evaluation = evaluate_oof_export(cached, candidate, feature_bundle, scheme_name)
        return cached, evaluation

    train_frame = feature_bundle.train_frame.copy()
    aligned_scheme = _align_scheme(train_frame, scheme_df)
    oof = _oof_template(train_frame, candidate)

    if "fold" in aligned_scheme.columns:
        unique_folds = sorted(aligned_scheme["fold"].unique().tolist())
        for fold in unique_folds:
            val_mask = aligned_scheme["fold"].to_numpy() == fold
            train_mask = ~val_mask
            preds = _predict_candidate(
                candidate,
                feature_bundle.feature_groups,
                train_frame.loc[train_mask].reset_index(drop=True),
                train_frame.loc[val_mask].reset_index(drop=True),
            )
            for idx, target in enumerate(candidate.target_cols):
                oof.loc[val_mask, f"mu_{target}"] = preds[:, idx]
            oof.loc[val_mask, "is_valid"] = True
    elif "is_holdout" in aligned_scheme.columns:
        val_mask = aligned_scheme["is_holdout"].to_numpy(dtype=bool)
        train_mask = ~val_mask
        preds = _predict_candidate(
            candidate,
            feature_bundle.feature_groups,
            train_frame.loc[train_mask].reset_index(drop=True),
            train_frame.loc[val_mask].reset_index(drop=True),
        )
        for idx, target in enumerate(candidate.target_cols):
            oof.loc[val_mask, f"mu_{target}"] = preds[:, idx]
        oof.loc[val_mask, "is_valid"] = True
    else:
        raise ValueError(f"Unsupported scheme frame columns for {scheme_name}: {aligned_scheme.columns.tolist()}")

    valid_mask = oof["is_valid"].to_numpy(dtype=bool)
    mu = oof.loc[valid_mask, [f"mu_{target}" for target in candidate.target_cols]].to_numpy(dtype=np.float64)
    y_true = train_frame.loc[valid_mask, candidate.target_cols].to_numpy(dtype=np.float64)
    evaluation = summarize_predictions(
        y_true=y_true,
        mu=mu,
        target_cols=candidate.target_cols,
        candidate_name=candidate.name,
        scheme_name=scheme_name,
    )

    scheme_export = oof.merge(aligned_scheme, on="planet_ID", how="left", validate="one_to_one")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scheme_export.to_csv(out_path, index=False, compression="gzip")
    return scheme_export, evaluation
