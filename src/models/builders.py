from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def passthrough_columns(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("select", "passthrough", feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def svd_with_side_features(
    svd_cols: list[str],
    passthrough_cols: list[str] | None = None,
    n_components: int = 24,
    scale_before_svd: bool = True,
) -> ColumnTransformer:
    svd_steps = []
    if scale_before_svd:
        svd_steps.append(("scale", StandardScaler()))
    svd_steps.append(("svd", TruncatedSVD(n_components=n_components, random_state=42)))
    transformers = [("svd", Pipeline(svd_steps), svd_cols)]

    passthrough_cols = passthrough_cols or []
    if passthrough_cols:
        transformers.append(("side", "passthrough", passthrough_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_extra_trees(**kwargs) -> ExtraTreesRegressor:
    params = {
        "n_estimators": 64,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": 4,
    }
    params.update(kwargs)
    return ExtraTreesRegressor(**params)


def make_random_forest(**kwargs) -> RandomForestRegressor:
    params = {
        "n_estimators": 128,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": 4,
    }
    params.update(kwargs)
    return RandomForestRegressor(**params)


def make_adaboost(**kwargs) -> AdaBoostRegressor:
    params = {
        "estimator": DecisionTreeRegressor(max_depth=3, random_state=42),
        "n_estimators": 48,
        "learning_rate": 0.05,
        "random_state": 42,
    }
    params.update(kwargs)
    return AdaBoostRegressor(**params)


def make_mlp(hidden_layer_sizes=(160, 80), max_iter: int = 180, **kwargs) -> MLPRegressor:
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 1024,
        "learning_rate_init": 1e-3,
        "max_iter": max_iter,
        "early_stopping": True,
        "n_iter_no_change": 12,
        "random_state": 42,
        "verbose": False,
    }
    params.update(kwargs)
    return MLPRegressor(**params)
