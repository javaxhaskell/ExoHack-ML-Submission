from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.builders import (
    make_extra_trees,
    make_adaboost,
    make_mlp,
    make_random_forest,
    passthrough_columns,
    svd_with_side_features,
)


TEMP_TARGET = ["planet_temp"]
ABUNDANCE_TARGETS = ["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]
ALL_TARGETS = ["planet_temp", "log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    track: str
    target_cols: list[str]
    feature_family: str
    model_family: str
    training_mode: str
    serious: bool
    notes: str
    build_pipeline: Callable[[dict[str, list[str]]], object]


def _passthrough_tree(feature_family: str, model_family: str):
    def builder(groups: dict[str, list[str]]) -> object:
        cols = groups[feature_family]
        if model_family == "extra_trees":
            estimator = make_extra_trees()
        elif model_family == "random_forest":
            estimator = make_random_forest()
        elif model_family == "adaboost":
            estimator = make_adaboost()
        else:
            raise ValueError(model_family)
        return Pipeline([("select", passthrough_columns(cols)), ("model", estimator)])

    return builder


def _svd_tree(svd_family: str, side_family: str | None, model_family: str, n_components: int = 24):
    def builder(groups: dict[str, list[str]]) -> object:
        svd_cols = groups[svd_family]
        side_cols = groups[side_family] if side_family else []
        if model_family == "extra_trees":
            estimator = make_extra_trees()
        elif model_family == "adaboost":
            estimator = make_adaboost()
        else:
            raise ValueError(model_family)
        return Pipeline(
            [
                ("features", svd_with_side_features(svd_cols, passthrough_cols=side_cols, n_components=n_components)),
                ("model", estimator),
            ]
        )

    return builder


def _svd_mlp(svd_family: str, side_family: str | None, n_components: int = 24):
    def builder(groups: dict[str, list[str]]) -> object:
        svd_cols = groups[svd_family]
        side_cols = groups[side_family] if side_family else []
        return Pipeline(
            [
                ("features", svd_with_side_features(svd_cols, passthrough_cols=side_cols, n_components=n_components)),
                ("scale", StandardScaler()),
                ("model", make_mlp()),
            ]
        )

    return builder


def _scaled_mlp(feature_family: str):
    def builder(groups: dict[str, list[str]]) -> object:
        cols = groups[feature_family]
        return Pipeline(
            [
                ("select", passthrough_columns(cols)),
                ("scale", StandardScaler()),
                ("model", make_mlp(hidden_layer_sizes=(128, 64), max_iter=260)),
            ]
        )

    return builder


def phase2_candidates() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="temp_et_meta",
            track="temperature",
            target_cols=TEMP_TARGET,
            feature_family="metadata",
            model_family="extra_trees",
            training_mode="independent",
            serious=True,
            notes="Temperature specialist on metadata only.",
            build_pipeline=_passthrough_tree("metadata", "extra_trees"),
        ),
        CandidateSpec(
            name="temp_rf_meta",
            track="temperature",
            target_cols=TEMP_TARGET,
            feature_family="metadata",
            model_family="random_forest",
            training_mode="independent",
            serious=True,
            notes="Random forest metadata baseline for temperature.",
            build_pipeline=_passthrough_tree("metadata", "random_forest"),
        ),
        CandidateSpec(
            name="temp_adaboost_meta",
            track="temperature",
            target_cols=TEMP_TARGET,
            feature_family="metadata",
            model_family="adaboost",
            training_mode="independent",
            serious=True,
            notes="Boosting baseline for temperature on metadata.",
            build_pipeline=_passthrough_tree("metadata", "adaboost"),
        ),
        CandidateSpec(
            name="temp_mlp_meta",
            track="temperature",
            target_cols=TEMP_TARGET,
            feature_family="metadata",
            model_family="mlp",
            training_mode="independent",
            serious=True,
            notes="Small metadata MLP temperature model.",
            build_pipeline=_scaled_mlp("metadata"),
        ),
        CandidateSpec(
            name="temp_et_meta_summary",
            track="temperature",
            target_cols=TEMP_TARGET,
            feature_family="metadata_summaries",
            model_family="extra_trees",
            training_mode="independent",
            serious=True,
            notes="Checks whether summary spectrum features help temperature.",
            build_pipeline=_passthrough_tree("metadata_summaries", "extra_trees"),
        ),
        CandidateSpec(
            name="temp_et_meta_svd",
            track="temperature",
            target_cols=TEMP_TARGET,
            feature_family="spectrum_noise+metadata",
            model_family="extra_trees",
            training_mode="independent",
            serious=True,
            notes="Checks whether compressed spectra help temperature beyond metadata.",
            build_pipeline=_svd_tree("spectrum_noise", "metadata", "extra_trees", n_components=16),
        ),
        CandidateSpec(
            name="abun_sep_et_summary",
            track="abundance",
            target_cols=ABUNDANCE_TARGETS,
            feature_family="summary_only",
            model_family="extra_trees",
            training_mode="independent",
            serious=False,
            notes="Summary-only abundance baseline.",
            build_pipeline=_passthrough_tree("summary_only", "extra_trees"),
        ),
        CandidateSpec(
            name="abun_sep_et_spec",
            track="abundance",
            target_cols=ABUNDANCE_TARGETS,
            feature_family="spectrum",
            model_family="extra_trees",
            training_mode="independent",
            serious=False,
            notes="Spectrum-only abundance ablation.",
            build_pipeline=_passthrough_tree("spectrum", "extra_trees"),
        ),
        CandidateSpec(
            name="abun_sep_et_spec_noise",
            track="abundance",
            target_cols=ABUNDANCE_TARGETS,
            feature_family="spectrum_noise",
            model_family="extra_trees",
            training_mode="independent",
            serious=True,
            notes="Per-target tree using spectrum and noise.",
            build_pipeline=_passthrough_tree("spectrum_noise", "extra_trees"),
        ),
        CandidateSpec(
            name="abun_sep_et_spec_noise_meta",
            track="abundance",
            target_cols=ABUNDANCE_TARGETS,
            feature_family="spectrum_noise_metadata",
            model_family="extra_trees",
            training_mode="independent",
            serious=True,
            notes="Per-target tree using spectrum, noise, and metadata.",
            build_pipeline=_passthrough_tree("spectrum_noise_metadata", "extra_trees"),
        ),
        CandidateSpec(
            name="abun_sep_et_spec_noise_meta_deriv",
            track="abundance",
            target_cols=ABUNDANCE_TARGETS,
            feature_family="spectrum_noise_metadata_derivatives",
            model_family="extra_trees",
            training_mode="independent",
            serious=True,
            notes="Per-target tree with derivative features.",
            build_pipeline=_passthrough_tree("spectrum_noise_metadata_derivatives", "extra_trees"),
        ),
        CandidateSpec(
            name="abun_shared_et_spec_noise_meta",
            track="abundance",
            target_cols=ABUNDANCE_TARGETS,
            feature_family="spectrum_noise_metadata",
            model_family="extra_trees",
            training_mode="multioutput",
            serious=True,
            notes="Shared tree abundance model benchmark.",
            build_pipeline=_passthrough_tree("spectrum_noise_metadata", "extra_trees"),
        ),
        CandidateSpec(
            name="abun_shared_mlp_svd_spec_noise_meta",
            track="abundance",
            target_cols=ABUNDANCE_TARGETS,
            feature_family="spectrum_noise+metadata",
            model_family="mlp",
            training_mode="multioutput",
            serious=True,
            notes="Compact neural-style abundance model using compressed spectra.",
            build_pipeline=_svd_mlp("spectrum_noise", "metadata", n_components=24),
        ),
        CandidateSpec(
            name="alltarget_et_spec_noise_meta",
            track="all_targets",
            target_cols=ALL_TARGETS,
            feature_family="spectrum_noise_metadata",
            model_family="extra_trees",
            training_mode="multioutput",
            serious=True,
            notes="All-target benchmark only.",
            build_pipeline=_passthrough_tree("spectrum_noise_metadata", "extra_trees"),
        ),
    ]
