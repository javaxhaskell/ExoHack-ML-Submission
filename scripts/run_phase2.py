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
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data_loading import load_test_data, load_training_data
from src.features import build_phase2_feature_bundle
from src.models.catalog import ABUNDANCE_TARGETS, TEMP_TARGET, phase2_candidates
from src.train import load_phase1_schemes, run_candidate_on_scheme


warnings.filterwarnings("ignore", category=ConvergenceWarning)


DATA_DIR = ROOT / "Hackathon_training"
FOLD_DIR = ROOT / "outputs" / "folds"
REPORT_DIR = ROOT / "outputs" / "reports"
OOF_DIR = ROOT / "outputs" / "oof"
ALL_SCHEMES = ["random_5fold", "regime_group_5fold", "edge_holdout_15pct"]


def format_frame(df: pd.DataFrame, decimals: int = 4) -> str:
    display = df.copy()
    numeric_cols = display.select_dtypes(include=["number", "bool"]).columns
    display[numeric_cols] = display[numeric_cols].round(decimals)
    return display.to_string(index=False)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run_scheme(candidate, scheme_name, scheme_df, bundle, reuse_existing: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    started_at = time.time()
    out_path = OOF_DIR / f"{candidate.name}__{scheme_name}.csv.gz"
    reused = reuse_existing and out_path.exists()
    _, evaluation = run_candidate_on_scheme(
        candidate,
        bundle,
        scheme_name,
        scheme_df,
        OOF_DIR,
        reuse_existing=reuse_existing,
    )
    finished_at = time.time()
    log_row = {
        "candidate": candidate.name,
        "track": candidate.track,
        "scheme": scheme_name,
        "reused_oof": reused,
        "elapsed_seconds": finished_at - started_at,
    }
    return evaluation.aggregate, evaluation.per_target, evaluation.sigma_by_target, log_row


def _abundance_scope_table(aggregate_df: pd.DataFrame, per_target_df: pd.DataFrame, scheme_name: str) -> pd.DataFrame:
    rows = []

    abundance_candidates = aggregate_df[
        aggregate_df["scheme"].eq(scheme_name)
        & aggregate_df["candidate"].str.startswith("abun_")
    ][["candidate", "primary_metric", "mean_crps", "mean_mae", "mean_rmse", "mean_r2", "mean_nll_norm"]]
    rows.append(abundance_candidates)

    alltarget_rows = per_target_df[
        per_target_df["scheme"].eq(scheme_name)
        & per_target_df["candidate"].eq("alltarget_et_spec_noise_meta")
        & per_target_df["target"].isin(ABUNDANCE_TARGETS)
    ]
    if not alltarget_rows.empty:
        rows.append(
            pd.DataFrame(
                [
                    {
                        "candidate": "alltarget_et_spec_noise_meta",
                        "primary_metric": alltarget_rows["primary_metric"].mean(),
                        "mean_crps": alltarget_rows["crps"].mean(),
                        "mean_mae": alltarget_rows["mae"].mean(),
                        "mean_rmse": alltarget_rows["rmse"].mean(),
                        "mean_r2": alltarget_rows["r2"].mean(),
                        "mean_nll_norm": alltarget_rows["nll_norm"].mean(),
                    }
                ]
            )
        )

    out = pd.concat(rows, ignore_index=True).sort_values("primary_metric", ascending=False).reset_index(drop=True)
    return out


def _temperature_scope_table(aggregate_df: pd.DataFrame, scheme_name: str) -> pd.DataFrame:
    return (
        aggregate_df[
            aggregate_df["scheme"].eq(scheme_name)
            & aggregate_df["candidate"].str.startswith("temp_")
        ][["candidate", "primary_metric", "mean_crps", "mean_mae", "mean_rmse", "mean_r2", "mean_nll_norm"]]
        .sort_values("primary_metric", ascending=False)
        .reset_index(drop=True)
    )


def _rank_stability(table: pd.DataFrame, group_label: str) -> pd.DataFrame:
    rank_rows = []
    for scheme_name, scheme_rows in table.groupby("scheme"):
        ranked = scheme_rows.sort_values("primary_metric", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        rank_rows.append(ranked[["candidate", "scheme", "rank", "primary_metric"]])

    ranks = pd.concat(rank_rows, ignore_index=True)
    summary = (
        ranks.groupby("candidate", as_index=False)
        .agg(avg_rank=("rank", "mean"), std_rank=("rank", "std"), avg_primary=("primary_metric", "mean"))
        .fillna(0.0)
        .sort_values(["avg_rank", "avg_primary"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "group", group_label)
    return summary


def _persist_phase2_outputs(
    aggregate_rows: list[pd.DataFrame],
    per_target_rows: list[pd.DataFrame],
    sigma_rows: list[pd.DataFrame],
    execution_log_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_df = pd.concat(aggregate_rows, ignore_index=True)
    per_target_df = pd.concat(per_target_rows, ignore_index=True)
    sigma_df = pd.concat(sigma_rows, ignore_index=True)
    execution_log = pd.DataFrame(execution_log_rows)

    aggregate_df.to_csv(REPORT_DIR / "phase2_candidate_aggregate_metrics.csv", index=False)
    per_target_df.to_csv(REPORT_DIR / "phase2_candidate_per_target_metrics.csv", index=False)
    sigma_df.to_csv(REPORT_DIR / "phase2_candidate_sigma_metrics.csv", index=False)
    execution_log.to_csv(REPORT_DIR / "phase2_execution_log.csv", index=False)
    return aggregate_df, per_target_df, sigma_df


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    train_supp, train_targets, train_spectra = load_training_data(DATA_DIR)
    test_supp, test_spectra = load_test_data(DATA_DIR)
    bundle = build_phase2_feature_bundle(train_supp, train_targets, train_spectra, test_supp, test_spectra)
    schemes = load_phase1_schemes(FOLD_DIR)
    candidates = phase2_candidates()
    catalog = pd.DataFrame([candidate.__dict__ | {"build_pipeline": None} for candidate in candidates])
    serious_candidate_names = set(catalog.loc[catalog["serious"], "name"])
    serious_candidates = [candidate for candidate in candidates if candidate.name in serious_candidate_names]

    aggregate_rows = []
    per_target_rows = []
    sigma_rows = []
    execution_log_rows = []

    for candidate in candidates:
        agg, per_target, sigma, log_row = _run_scheme(
            candidate,
            "random_5fold",
            schemes["random_5fold"],
            bundle,
            reuse_existing=True,
        )
        aggregate_rows.append(agg)
        per_target_rows.append(per_target)
        sigma_rows.append(sigma)
        execution_log_rows.append(log_row)
        aggregate_df, per_target_df, sigma_df = _persist_phase2_outputs(
            aggregate_rows,
            per_target_rows,
            sigma_rows,
            execution_log_rows,
        )

    random_temp = _temperature_scope_table(aggregate_df, "random_5fold")
    random_abundance = _abundance_scope_table(aggregate_df, per_target_df, "random_5fold")

    for scheme_name in [name for name in ALL_SCHEMES if name != "random_5fold"]:
        for candidate in serious_candidates:
            agg, per_target, sigma, log_row = _run_scheme(
                candidate,
                scheme_name,
                schemes[scheme_name],
                bundle,
                reuse_existing=True,
            )
            aggregate_df = pd.concat([aggregate_df, agg], ignore_index=True)
            per_target_df = pd.concat([per_target_df, per_target], ignore_index=True)
            sigma_df = pd.concat([sigma_df, sigma], ignore_index=True)
            execution_log_rows.append(log_row)
            aggregate_df, per_target_df, sigma_df = _persist_phase2_outputs(
                aggregate_rows=[aggregate_df],
                per_target_rows=[per_target_df],
                sigma_rows=[sigma_df],
                execution_log_rows=execution_log_rows,
            )

    serious_temp = catalog[
        catalog["name"].isin(serious_candidate_names) & catalog["track"].eq("temperature")
    ]["name"].tolist()
    serious_abundance = catalog[
        catalog["name"].isin(serious_candidate_names) & catalog["track"].isin(["abundance", "all_targets"])
    ]["name"].tolist()
    pd.DataFrame({"candidate": sorted(serious_candidate_names)}).to_csv(REPORT_DIR / "phase2_serious_candidates.csv", index=False)

    temp_tournament = (
        aggregate_df[
            aggregate_df["candidate"].isin(serious_temp)
            & aggregate_df["scheme"].isin(ALL_SCHEMES)
        ][["candidate", "scheme", "primary_metric", "mean_crps", "mean_mae", "mean_rmse", "mean_r2", "mean_nll_norm"]]
        .sort_values(["scheme", "primary_metric"], ascending=[True, False])
        .reset_index(drop=True)
    )

    abundance_rows = []
    for scheme_name in ALL_SCHEMES:
        table = _abundance_scope_table(aggregate_df, per_target_df, scheme_name)
        table = table[table["candidate"].isin(serious_abundance)].copy()
        table.insert(1, "scheme", scheme_name)
        abundance_rows.append(table)
    abundance_tournament = pd.concat(abundance_rows, ignore_index=True)

    temp_stability = _rank_stability(temp_tournament[["candidate", "scheme", "primary_metric"]], "temperature")
    abundance_stability = _rank_stability(
        abundance_tournament[["candidate", "scheme", "primary_metric"]], "abundance"
    )

    temp_ablation = random_temp.merge(
        catalog[["name", "feature_family", "model_family", "notes"]].rename(columns={"name": "candidate"}),
        on="candidate",
        how="left",
    )
    abundance_ablation = random_abundance.merge(
        catalog[["name", "feature_family", "model_family", "training_mode", "notes"]].rename(columns={"name": "candidate"}),
        on="candidate",
        how="left",
    )

    best_temp_name = temp_stability.iloc[0]["candidate"]
    best_abundance_name = abundance_stability.iloc[0]["candidate"]
    best_independent_abundance_name = (
        abundance_stability[abundance_stability["candidate"].str.startswith("abun_sep_")].iloc[0]["candidate"]
    )

    best_temp_rows = temp_tournament[temp_tournament["candidate"].eq(best_temp_name)].copy()
    best_abundance_rows = per_target_df[
        per_target_df["candidate"].eq(best_abundance_name)
        & per_target_df["scheme"].isin(ALL_SCHEMES)
        & per_target_df["target"].isin(ABUNDANCE_TARGETS)
    ]
    weakest_target = (
        best_abundance_rows.groupby("target", as_index=False)["primary_metric"].mean().sort_values("primary_metric").iloc[0]["target"]
    )

    brittle_rows = []
    for table in [temp_tournament, abundance_tournament]:
        pivot = table.pivot(index="candidate", columns="scheme", values="primary_metric").reset_index()
        if "random_5fold" in pivot and "regime_group_5fold" in pivot:
            pivot["drop_regime"] = pivot["random_5fold"] - pivot["regime_group_5fold"]
        if "edge_holdout_15pct" in pivot:
            pivot["drop_edge"] = pivot["random_5fold"] - pivot["edge_holdout_15pct"]
        brittle_rows.append(pivot)
    brittle_df = pd.concat(brittle_rows, ignore_index=True).fillna(0.0)
    brittle_df["max_drop"] = brittle_df[[col for col in ["drop_regime", "drop_edge"] if col in brittle_df.columns]].max(axis=1)
    brittle_df = brittle_df.sort_values("max_drop", ascending=False)

    targeted_vs_shared = pd.DataFrame(
        [
            {
                "comparison": "best_independent_et",
                "candidate": best_independent_abundance_name,
                "avg_primary": float(
                    abundance_tournament[abundance_tournament["candidate"].eq(best_independent_abundance_name)][
                        "primary_metric"
                    ].mean()
                ),
            },
            {
                "comparison": "shared_tree",
                "candidate": "abun_shared_et_spec_noise_meta",
                "avg_primary": float(
                    abundance_tournament[abundance_tournament["candidate"].eq("abun_shared_et_spec_noise_meta")][
                        "primary_metric"
                    ].mean()
                ),
            },
            {
                "comparison": "shared_neural",
                "candidate": "abun_shared_mlp_svd_spec_noise_meta",
                "avg_primary": float(
                    abundance_tournament[abundance_tournament["candidate"].eq("abun_shared_mlp_svd_spec_noise_meta")][
                        "primary_metric"
                    ].mean()
                ),
            },
            {
                "comparison": "all_target_benchmark",
                "candidate": "alltarget_et_spec_noise_meta",
                "avg_primary": float(
                    abundance_tournament[abundance_tournament["candidate"].eq("alltarget_et_spec_noise_meta")][
                        "primary_metric"
                    ].mean()
                ),
            },
        ]
    )

    ablation_report = f"""# Phase 2 Ablation

## Experiment framework
- Canonical Phase 2 feature bundle is generated by [src/features.py]({(ROOT / 'src/features.py').resolve()}).
- Candidate catalog lives in [src/models/catalog.py]({(ROOT / 'src/models/catalog.py').resolve()}).
- Fold execution and OOF writing live in [src/train.py]({(ROOT / 'src/train.py').resolve()}).
- Primary ranking metric is CRPS skill using a plug-in constant sigma fitted on OOF residuals per target. This is a mean-model proxy, not the final sigma system.
- Every serious candidate is evaluated across `random_5fold`, `regime_group_5fold`, and `edge_holdout_15pct`. Non-serious ablations stay on `random_5fold` only.

## Temperature ablation on random 5-fold
```
{format_frame(temp_ablation[['candidate', 'feature_family', 'model_family', 'primary_metric', 'mean_rmse', 'mean_r2']], decimals=4)}
```

What changed:
- Metadata-only models dominate temperature.
- Adding full spectral compression to metadata is not required unless it improves robust validation later.
- The temperature track remains justified as a specialist problem.

## Abundance ablation on random 5-fold
```
{format_frame(abundance_ablation[['candidate', 'feature_family', 'model_family', 'training_mode', 'primary_metric', 'mean_rmse', 'mean_r2']], decimals=4)}
```

Feature family implication:
- Summary-only features are too lossy for abundance retrieval.
- Spectrum-only beats metadata-only by a wide margin.
- Adding noise and then metadata can help, but the benefit is target/model dependent.
- Derivative features are useful enough to merit a finalist when they improve the extra-trees family.
- SVD-compressed spectra are a reasonable input for boosting and MLP models.

## Serious-candidate tournament scope
- Temperature serious candidates: `{', '.join(serious_temp)}`
- Abundance serious candidates: `{', '.join(serious_abundance)}`
- Execution log: [outputs/reports/phase2_execution_log.csv]({(ROOT / 'outputs/reports/phase2_execution_log.csv').resolve()})
"""

    tournament_report = f"""# Phase 2 Model Tournament

## Serious candidates
- Temperature: `{', '.join(serious_temp)}`
- Abundance: `{', '.join(serious_abundance)}`
- Neural constraint: the environment has no `torch`/`tensorflow`, so the neural branch here is a compact multi-output `MLPRegressor` on SVD-compressed spectral features plus metadata.

## Temperature tournament
```
{format_frame(temp_tournament, decimals=4)}
```

## Temperature rank stability
```
{format_frame(temp_stability, decimals=4)}
```

## Abundance tournament
```
{format_frame(abundance_tournament, decimals=4)}
```

## Abundance rank stability
```
{format_frame(abundance_stability, decimals=4)}
```

## Target-specialisation evidence
```
{format_frame(targeted_vs_shared, decimals=4)}
```

Interpretation:
- Temperature is clearly a specialist metadata task.
- Abundances do not behave like one clean shared target family; independent tree models remain strong contenders.
- Shared models and the all-target benchmark are useful sanity checks, but they must win on score, not elegance.

## Best-so-far diagnostics
### Best temperature model: `{best_temp_name}`
```
{format_frame(best_temp_rows[['scheme', 'primary_metric', 'mean_rmse', 'mean_r2', 'mean_nll_norm']], decimals=4)}
```

### Best abundance model family: `{best_abundance_name}`
```
{format_frame(best_abundance_rows[['scheme', 'target', 'primary_metric', 'rmse', 'r2', 'sigma_plugin']], decimals=4)}
```

### Brittleness under shift
```
{format_frame(brittle_df[['candidate', 'random_5fold', 'regime_group_5fold', 'edge_holdout_15pct', 'drop_regime', 'drop_edge', 'max_drop']].head(12), decimals=4)}
```

What to believe now:
- The model that wins random folds but collapses on regime or edge splits is not competition-ready.
- The better finalist is the one with strong average CRPS proxy and smaller shift-induced drop, not just the best IID score.
- `{weakest_target}` is currently the weakest abundance target under the best abundance family.
"""

    save_text(REPORT_DIR / "phase2_ablation.md", ablation_report)
    save_text(REPORT_DIR / "phase2_model_tournament.md", tournament_report)


if __name__ == "__main__":
    main()
