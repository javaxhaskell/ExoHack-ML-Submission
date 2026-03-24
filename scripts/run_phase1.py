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

import h5py
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading import SUPPLEMENTARY_COLS, TARGET_COLS, load_test_data, load_training_data
from src.data_validation import run_phase1_audit, top_abs_spearman_correlations
from src.eda import (
    plot_metadata_distributions,
    plot_metadata_target_relationships,
    plot_regime_scatter,
    plot_representative_regime_spectra,
    plot_snr_diagnostics,
    plot_spectrum_noise_summary,
    plot_target_correlation,
    plot_target_distributions,
)
from src.features import (
    build_flat_test_table,
    build_flat_training_table,
    log10_metadata_features,
    log10_spectrum_features,
    regime_feature_table,
    signal_to_noise_matrix,
)
from src.probes import evaluate_multioutput_cv, evaluate_single_target_cv, evaluate_single_target_holdout
from src.validation import make_edge_regime_holdout, make_random_folds, make_regime_group_folds, summarize_fold_balance


DATA_DIR = ROOT / "Hackathon_training"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
ASSET_DIR = REPORT_DIR / "assets"
FOLD_DIR = OUTPUT_DIR / "folds"


def format_frame(df: pd.DataFrame, decimals: int = 4) -> str:
    display = df.copy()
    numeric_cols = display.select_dtypes(include=["number", "bool"]).columns
    display[numeric_cols] = display[numeric_cols].round(decimals)
    return display.to_string(index=False)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_image_md(path: Path, alt: str) -> str:
    return f"![{alt}]({path.resolve()})"


def model_factory_probe() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=40,
        max_depth=28,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )


def model_factory_starter() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=10,
        random_state=42,
        n_jobs=1,
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    train_supp, train_targets, train_spectra = load_training_data(DATA_DIR)
    test_supp, test_spectra = load_test_data(DATA_DIR)

    train_flat = build_flat_training_table(train_supp, train_targets, train_spectra)
    test_flat = build_flat_test_table(test_supp, test_spectra)
    regime_table = regime_feature_table(train_supp, train_spectra)

    audit = run_phase1_audit(DATA_DIR)

    random_folds = make_random_folds(train_targets["planet_ID"].to_numpy(), n_splits=5, random_state=42)
    regime_folds = make_regime_group_folds(train_supp, train_spectra, n_splits=5, n_bins=4)
    edge_holdout = make_edge_regime_holdout(train_supp, train_spectra, holdout_fraction=0.15)

    random_folds.to_csv(FOLD_DIR / "phase1_random_5fold.csv", index=False)
    regime_folds.to_csv(FOLD_DIR / "phase1_regime_group_5fold.csv", index=False)
    edge_holdout.to_csv(FOLD_DIR / "phase1_edge_holdout.csv", index=False)

    target_dist_path = ASSET_DIR / "phase1_target_distributions.png"
    target_corr_path = ASSET_DIR / "phase1_target_correlation.png"
    metadata_dist_path = ASSET_DIR / "phase1_metadata_distributions.png"
    metadata_target_path = ASSET_DIR / "phase1_metadata_target_relationships.png"
    spectrum_noise_path = ASSET_DIR / "phase1_spectrum_noise_summary.png"
    snr_path = ASSET_DIR / "phase1_snr_diagnostics.png"
    regime_scatter_path = ASSET_DIR / "phase1_regime_scatter.png"
    representative_spectra_path = ASSET_DIR / "phase1_representative_regimes.png"

    plot_target_distributions(train_targets, target_dist_path)
    plot_target_correlation(train_targets, target_corr_path)
    plot_metadata_distributions(train_supp, metadata_dist_path)
    plot_metadata_target_relationships(train_flat, metadata_target_path)
    plot_spectrum_noise_summary(train_spectra, spectrum_noise_path)
    plot_snr_diagnostics(train_spectra, snr_path)
    plot_regime_scatter(train_supp, train_spectra, train_targets, regime_folds, regime_scatter_path)
    plot_representative_regime_spectra(
        train_supp,
        train_spectra,
        train_targets,
        regime_folds,
        representative_spectra_path,
    )

    with h5py.File(DATA_DIR / "Training_SpectralData.hdf5", "r") as handle:
        training_raw_keys = list(handle.keys())[:10]
    with h5py.File(DATA_DIR / "Test_SpectralData.hdf5", "r") as handle:
        test_raw_keys = list(handle.keys())[:10]

    snr = signal_to_noise_matrix(train_spectra)
    snr_stats = pd.DataFrame(
        {
            "metric": [
                "mean_snr_p0",
                "mean_snr_p5",
                "mean_snr_p25",
                "mean_snr_p50",
                "mean_snr_p75",
                "mean_snr_p95",
                "mean_snr_p100",
            ],
            "value": np.percentile(snr.mean(axis=1), [0, 5, 25, 50, 75, 95, 100]),
        }
    )

    metadata_summary = (
        train_supp[SUPPLEMENTARY_COLS]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .loc[["min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"]]
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    target_summary = (
        train_targets[TARGET_COLS]
        .describe()
        .loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        .T
        .reset_index()
        .rename(columns={"index": "target"})
    )

    target_corr = train_targets[TARGET_COLS].corr().reset_index().rename(columns={"index": "target"})

    metadata_corr_rows = []
    top_corr = top_abs_spearman_correlations(train_flat)
    for target in TARGET_COLS:
        for row in top_corr[target]:
            metadata_corr_rows.append({"target": target, **row})
    metadata_corr = pd.DataFrame(metadata_corr_rows)

    regime_counts = regime_folds["regime_group"].value_counts().rename_axis("regime_group").reset_index(name="n_planets")
    top_regimes = regime_counts.head(10).merge(
        train_targets[["planet_ID", "planet_temp"]].merge(regime_folds, on="planet_ID"),
        on="regime_group",
        how="left",
    ).groupby(["regime_group", "n_planets"], as_index=False)["planet_temp"].mean()

    X_meta = log10_metadata_features(train_supp).to_numpy(dtype=np.float64)
    X_spec = log10_spectrum_features(train_spectra)

    probe_random_meta = evaluate_single_target_cv(
        model_factory_probe,
        X_meta,
        train_targets,
        random_folds,
        TARGET_COLS,
        model_name="metadata_only_et40",
        scheme_name="random_5fold",
    )
    probe_random_spec = evaluate_single_target_cv(
        model_factory_probe,
        X_spec,
        train_targets,
        random_folds,
        TARGET_COLS,
        model_name="spectrum_only_et40",
        scheme_name="random_5fold",
    )
    probe_starter_random = evaluate_multioutput_cv(
        model_factory_starter,
        train_spectra.spectrum,
        train_targets,
        random_folds,
        TARGET_COLS,
        model_name="starter_like_et10_multioutput",
        scheme_name="random_5fold",
    )
    probe_temp_meta_regime = evaluate_single_target_cv(
        model_factory_probe,
        X_meta,
        train_targets,
        regime_folds[["planet_ID", "fold"]],
        ["planet_temp"],
        model_name="metadata_only_et40",
        scheme_name="regime_group_5fold",
    )
    probe_temp_meta_edge = evaluate_single_target_holdout(
        model_factory_probe,
        X_meta,
        train_targets,
        train_mask=~edge_holdout["is_holdout"].to_numpy(),
        val_mask=edge_holdout["is_holdout"].to_numpy(),
        target_cols=["planet_temp"],
        model_name="metadata_only_et40",
        scheme_name="edge_holdout_15pct",
    )
    probe_starter_regime = evaluate_multioutput_cv(
        model_factory_starter,
        train_spectra.spectrum,
        train_targets,
        regime_folds[["planet_ID", "fold"]],
        TARGET_COLS,
        model_name="starter_like_et10_multioutput",
        scheme_name="regime_group_5fold",
    )

    probe_results = pd.concat(
        [
            probe_random_meta,
            probe_random_spec,
            probe_starter_random,
            probe_temp_meta_regime,
            probe_temp_meta_edge,
            probe_starter_regime,
        ],
        ignore_index=True,
    )
    probe_results.to_csv(REPORT_DIR / "phase1_probe_metrics.csv", index=False)

    abundance_probe = (
        probe_results[
            probe_results["target"].isin(["log_H2O", "log_CO2", "log_CH4", "log_CO", "log_NH3"])
            & probe_results["scheme"].eq("random_5fold")
            & probe_results["model"].isin(["metadata_only_et40", "spectrum_only_et40"])
        ]
        .pivot(index="target", columns="model", values="r2")
        .reset_index()
    )
    temp_probe = probe_results[
        probe_results["target"].eq("planet_temp")
        & probe_results["model"].isin(["metadata_only_et40", "spectrum_only_et40", "starter_like_et10_multioutput"])
    ][["model", "scheme", "r2", "rmse", "mae"]]

    random_balance = summarize_fold_balance(random_folds)
    regime_balance = summarize_fold_balance(regime_folds)

    edge_holdout_rate = float(edge_holdout["is_holdout"].mean())
    edge_join = edge_holdout.merge(regime_table, on="planet_ID", how="left").merge(
        train_targets[["planet_ID", "planet_temp"]],
        on="planet_ID",
        how="left",
    )
    edge_summary = pd.DataFrame(
        {
            "split": ["train_core", "edge_holdout"],
            "n_planets": [
                int((~edge_join["is_holdout"]).sum()),
                int(edge_join["is_holdout"].sum()),
            ],
            "planet_temp_mean": [
                edge_join.loc[~edge_join["is_holdout"], "planet_temp"].mean(),
                edge_join.loc[edge_join["is_holdout"], "planet_temp"].mean(),
            ],
            "log_snr_mean_mean": [
                edge_join.loc[~edge_join["is_holdout"], "log_snr_mean"].mean(),
                edge_join.loc[edge_join["is_holdout"], "log_snr_mean"].mean(),
            ],
            "spectral_slope_mean": [
                edge_join.loc[~edge_join["is_holdout"], "spectral_slope"].mean(),
                edge_join.loc[edge_join["is_holdout"], "spectral_slope"].mean(),
            ],
            "log_spec_mean_mean": [
                edge_join.loc[~edge_join["is_holdout"], "log_spec_mean"].mean(),
                edge_join.loc[edge_join["is_holdout"], "log_spec_mean"].mean(),
            ],
        }
    )

    audit_report = f"""# Phase 1 Audit

## Relevant repository files
- `{ROOT / 'hackathon_starter_solution.ipynb'}`
- `{ROOT / 'utils.py'}`
- `{DATA_DIR / 'Training_targets.csv'}`
- `{DATA_DIR / 'Training_supplementary_data.csv'}`
- `{DATA_DIR / 'Training_SpectralData.hdf5'}`
- `{DATA_DIR / 'Test_supplementary_data.csv'}`
- `{DATA_DIR / 'Test_SpectralData.hdf5'}`

## Starter notebook reconstruction
1. Load training spectra with `utils.load_spectral_data()`.
2. Load training targets and supplementary CSVs with `index_col=0`.
3. Ignore the supplementary metadata entirely.
4. Use `X = spectrum_stack`, `y = training_targets[TARGET_COLS]`.
5. Make a single random 90/10 split with `train_test_split(..., random_state=42)`.
6. Run 5-fold CV only inside the 90% development block.
7. Fit `ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1)` once per fold.
8. Predict the 10% holdout with the fold ensemble mean; use ensemble spread plus `1e-6` as `std`.
9. Predict the public test HDF5 the same way and post two CSVs to the challenge API.

## What the local score really does
- `utils.compute_participant_score()` extracts the 6 physical targets and normalizes truth, predicted means, and predicted standard deviations by fixed training-set means/stds.
- It scores each target with Gaussian CRPS:
  `CRPS(N(mu, sigma), y) = sigma * [z * (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)]`, where `z = (y - mu) / sigma`.
- It compares that CRPS to a reference forecast with normalized mean `0` and normalized standard deviation `1`.
- Final score = average skill across the 6 targets: `1 - CRPS_model / CRPS_reference`.
- Any non-positive predicted `std` raises an error.

## Submission format
- `utils.array_to_submission()` creates a dataframe with columns:
  `planet_ID, planet_temp, log_H2O, log_CO2, log_CH4, log_CO, log_NH3`
- The starter notebook serializes submission files with `to_csv(index=True)`, which reproduces the leading unnamed index-column style seen in the provided CSVs.
- Internally, that leading unnamed column is junk and should be dropped for all analysis.

## HDF5 loading and alignment
- `utils.load_spectral_data()` stacks `instrument_spectrum` and `instrument_noise` into arrays of shape `(n_planets, 52)` and reads shared `instrument_wlgrid` and `instrument_width` from the first planet group.
- Training HDF5 groups happen to align with CSV order because all training IDs have the same digit width:
  `{training_raw_keys}`
- Test HDF5 raw iteration is lexicographic, not numeric:
  `{test_raw_keys}`
- Canonical rule: join by `planet_ID` after numerically sorting HDF5 group names, never by raw HDF5 iteration order.

## Shapes and canonical tables
- Training targets shape: `{train_targets.shape}`
- Training supplementary shape: `{train_supp.shape}`
- Training spectra shape: `{train_spectra.spectrum.shape}`
- Training noise shape: `{train_spectra.noise.shape}`
- Test supplementary shape: `{test_supp.shape}`
- Test spectra shape: `{test_spectra.spectrum.shape}`
- Shared wavelength grid across train/test: `{audit['shared_wavelength_grid_train_test']}`
- Canonical flat training table shape: `{train_flat.shape}`
- Canonical flat test table shape: `{test_flat.shape}`
- Shared arrays stored once globally: `wavelength[52]`, `width[52]`

## Data audit findings
- No missing values, duplicate rows, duplicate `planet_ID`s, or constant columns were found in the provided CSVs.
- `Unnamed: 0` equals `planet_ID` exactly in all provided CSVs, so it is a pure junk/index column.
- `planet_ID` is also not a usable feature: train IDs are `21988..91391`, test IDs are `0..21987`, so using it would encode split mechanics, not physics.
- `planet_surface_gravity` is highly redundant with `planet_mass_kg` and `planet_radius_m`; it is a derived feature, not leakage.
- Target columns look almost independent from one another, so a one-size multioutput model should not expect much help from label correlations.

## Canonical usable columns
- Identifiers / junk:
  - `Unnamed: 0`
  - `planet_ID`
- Usable metadata features:
  - `{', '.join(SUPPLEMENTARY_COLS)}`
- Usable spectral features:
  - `spectrum_bin_00` ... `spectrum_bin_51`
  - `noise_bin_00` ... `noise_bin_51`
- Targets:
  - `{', '.join(TARGET_COLS)}`

## Key implication
- The notebook is a demonstrator, not a reliable competition pipeline: it ignores metadata, assumes one generic multioutput model, uses uncalibrated fold spread for uncertainty, and its raw HDF5 test loading order is unsafe.
"""

    eda_report = f"""# Phase 1 EDA

## Target distributions
{make_image_md(target_dist_path, 'Target distributions')}

```
{format_frame(target_summary, decimals=4)}
```

Modelling implication:
- `planet_temp` is broad and right-skewed.
- The log-abundance targets are centered near their allowed mid-ranges with hard-looking bounds close to the prior limits.
- Clip or otherwise guard predictions to plausible support during inference and calibration.

## Target correlation matrix
{make_image_md(target_corr_path, 'Target correlation matrix')}

```
{format_frame(target_corr, decimals=3)}
```

Modelling implication:
- Cross-target correlations are effectively zero.
- Multioutput elegance is unlikely to buy much; target-specific models are more plausible.

## Metadata distributions
{make_image_md(metadata_dist_path, 'Metadata distributions')}

```
{format_frame(metadata_summary, decimals=4)}
```

Modelling implication:
- Most metadata spans orders of magnitude, especially masses, radii, orbital period, and distance.
- Log transforms should be the default for metadata models.

## Metadata vs target relationships
{make_image_md(metadata_target_path, 'Metadata versus targets')}

```
{format_frame(metadata_corr.groupby('target').head(3)[['target', 'feature', 'spearman']], decimals=4)}
```

Modelling implication:
- `planet_temp` is strongly metadata-dominated, especially by orbital period, orbital distance, and stellar/planet size proxies.
- Abundance targets show near-zero monotonic relationships with metadata, so metadata-only chemistry models are weak candidates.

## Spectrum and noise over wavelength
{make_image_md(spectrum_noise_path, 'Spectrum and noise summary')}

Modelling implication:
- Spectrum amplitude and variance change dramatically with wavelength.
- Noise is wavelength-dependent, so not all bins should be trusted equally.
- Log-transformed spectral inputs and explicit use of noise are justified.

## Signal-to-noise diagnostics
{make_image_md(snr_path, 'SNR diagnostics')}

```
{format_frame(snr_stats, decimals=3)}
```

Modelling implication:
- Mean SNR is extremely heavy-tailed: median is moderate, but the top tail is enormous.
- Robust scaling, clipping, or calibration by regime will matter more than naive Gaussian assumptions.

## Spectral regimes
{make_image_md(regime_scatter_path, 'Regime scatter')}

{make_image_md(representative_spectra_path, 'Representative regime spectra')}

```
{format_frame(top_regimes, decimals=3)}
```

Modelling implication:
- The spectra are not one homogeneous cloud. There are clear regimes in overall depth and slope, and those regimes carry different average temperatures.
- Validation should include regime-aware splits, not just random folds.

## Minimal hypothesis probes
These are intentionally point-prediction probes only. They are not final probabilistic model selection because uncertainty calibration is still missing.

### Temperature probe
```
{format_frame(temp_probe, decimals=4)}
```

Implication:
- Metadata-only temperature prediction is very strong.
- The starter-like spectrum-only multioutput baseline is materially weaker for temperature.

### Abundance probe
```
{format_frame(abundance_probe, decimals=4)}
```

Implication:
- Metadata-only models are weak for abundances.
- Spectrum-only models contain real abundance signal even in this minimal probe.
- This supports a specialist strategy: metadata-heavy temperature model, spectrum-driven abundance models.
"""

    validation_report = f"""# Phase 1 Validation Design

## Implemented validation schemes

### 1. Random 5-fold baseline
- File: `{FOLD_DIR / 'phase1_random_5fold.csv'}`
- Construction: shuffled `KFold(n_splits=5, random_state=42)`.
- Failure mode targeted: optimistic IID interpolation only.

```
{format_frame(random_balance, decimals=4)}
```

### 2. Regime-group 5-fold split
- File: `{FOLD_DIR / 'phase1_regime_group_5fold.csv'}`
- Construction: assign each planet to a regime group using quantile bins of:
  - `band_mid_mean`
  - `spectral_slope`
  - `log10(planet_orbital_period)`
- Then apply `GroupKFold(n_splits=5)` on those regime groups.
- Failure mode targeted: overestimating performance when train/validation share the same spectral+physical regime mix.

```
{format_frame(regime_balance, decimals=4)}
```

### 3. Edge-regime holdout
- File: `{FOLD_DIR / 'phase1_edge_holdout.csv'}`
- Construction: compute a robust edge score from metadata extremes, spectral summary extremes, and SNR extremes, then hold out the top 15%.
- Holdout fraction: `{edge_holdout_rate:.4f}`
- Failure mode targeted: extrapolation to rare planets and calibration under distribution shift.

```
{format_frame(edge_summary, decimals=4)}
```

## Why these three splits
- Random folds estimate the easy interpolation case.
- Regime-group folds punish models that only memorize common parts of feature space.
- Edge holdout tests whether a model breaks on rare, high-risk planets that are most likely to damage hidden-test score.

## Probe results under shift
### Metadata-only temperature model
```
{format_frame(probe_results[(probe_results['target'] == 'planet_temp') & (probe_results['model'] == 'metadata_only_et40')][['scheme', 'r2', 'rmse', 'mae']], decimals=4)}
```

### Starter-like spectrum-only multioutput baseline
```
{format_frame(probe_results[(probe_results['target'] == 'planet_temp') & (probe_results['model'] == 'starter_like_et10_multioutput')][['scheme', 'r2', 'rmse', 'mae']], decimals=4)}
```

Interpretation:
- Random folds alone would overstate comfort.
- Regime-aware and edge-regime evaluation expose whether a candidate is robust or merely average-case good.
- Future model selection should use random plus at least one harder split before anything is trusted.
"""

    save_text(REPORT_DIR / "phase1_audit.md", audit_report)
    save_text(REPORT_DIR / "phase1_eda.md", eda_report)
    save_text(REPORT_DIR / "phase1_validation_design.md", validation_report)


if __name__ == "__main__":
    main()
