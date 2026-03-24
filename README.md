# ExoHack Atmospheric Retrieval Research Log

ExoHack is a research-driven machine learning competition centred on exoplanet atmospheric retrieval. The task is to infer six planetary parameters from spectral data while also producing calibrated uncertainty under a probabilistic leaderboard metric. This research log records the modelling decisions, validation insights, and score-driven experiments behind that effort, and serves both as a reproducible account of the competition pipeline and as a broader study of how hidden probabilistic objectives shape high-performance scientific modelling.


## (3/100) Verified Leaderboard Score

**0.8707883387590796**



This repository documents a competition effort on probabilistic exoplanet atmospheric retrieval: predicting six planetary parameters together with calibrated uncertainty, under a CRPS-style leaderboard metric. The work was treated not as notebook decoration, but as an iterative research program in which every modelling change had to justify itself against the score actually used by the challenge.

## Abstract

This project studies a deceptively simple question: *what kind of inductive bias survives contact with a hidden probabilistic leaderboard?* The local evidence initially suggested that robust, target-specialised tree ensembles were the safest route, especially under regime-shifted validation. Yet the leaderboard ultimately revealed a different structure. The strongest hidden-test performance emerged from a neural, target-specialised abundance system that was initially discounted as too brittle under a hard edge split.

The final research conclusion is therefore methodological as much as predictive: **when the competition objective is probabilistic and the hidden test distribution is closer to IID than to an adversarial shift split, model family choice and uncertainty alignment can dominate conventional robustness intuitions**.

## Research Questions

This repository is organized around a set of explicit questions:

1. What does the official score actually reward?
2. Which parts of the problem are metadata-dominated and which are spectrum-dominated?
3. Should temperature and molecular abundances be modelled together or separately?
4. Is the best hidden-test inductive bias tree-based, neural, shared, or target-specialised?
5. How much of the final score comes from mean prediction and how much from uncertainty engineering?
6. What does the leaderboard teach us when it disagrees with our hardest local validation split?

## Headline Result

The best verified leaderboard system had the following structure:

| Component | Final choice |
| --- | --- |
| `planet_temp` mean | Fixed specialist temperature path |
| `planet_temp` sigma | 70% scaled version of the winning temperature sigma path |
| `log_H2O` mean | Shared MLP abundance model |
| `log_CO2` mean | Single-target MLP |
| `log_CH4` mean | Single-target MLP |
| `log_CO` mean | Single-target MLP |
| `log_NH3` mean | Single-target MLP |
| `log_H2O` sigma | Shared-MLP residual sigma |
| Remaining abundance sigmas | Single-target residual-model sigma |

This configuration achieved the best verified score:

> **0.8707883387590796**

## Why This Is Interesting

Most notable findings:

- Local robust validation initially favored an ExtraTrees abundance family.
- The hidden leaderboard repeatedly rejected sigma-only surgeries on that tree family.
- A deliberately risky switch to a shared MLP abundance backbone produced a large score jump.
- A further move to a target-wise neural hybrid produced the strongest verified score.

In other words, the leaderboard exposed a *distributional mismatch between our harshest local split and the actual hidden-test regime*. That mismatch became the central object of study.

## Experimental Arc

### Phase 1: Audit Before Modelling

The first phase reconstructed the exact data and metric pipeline before any serious modelling:

- exact CRPS-skill scoring logic
- submission formatting requirements
- HDF5 spectral loading and alignment
- metadata/spectrum role separation
- validation design beyond a naive random split

Key early finding:

- `planet_temp` was largely metadata-dominated
- molecular abundances were fundamentally spectrum-driven

### Phase 2: Mean-Prediction Tournament

Phase 2 asked a classical model-selection question:

> *Which mean predictors survive both local validation and distributional stress?*

This phase showed:

- temperature should be treated as its own specialist problem
- per-target abundance models beat monolithic all-target baselines
- derivative-enriched ExtraTrees were the strongest robust abundance family
- a shared MLP looked brilliant on IID folds but alarming on the hardest edge split

At this point, the conservative conclusion would have been to stay with trees.

### Phase 3: Uncertainty as a First-Class Modelling Problem

Phase 3 shifted focus from means to probabilistic performance:

- constant sigma baselines
- residual-model sigma
- disagreement-scaled sigma
- blended sigma systems

This phase reinforced an essential competition truth:

> good mean models can still underperform if their uncertainty is misaligned with the actual scoring rule.

### Leaderboard Pivot

The leaderboard then forced a conceptual pivot.

The robust local favourite was not the leaderboard favourite.
The hidden test behaved more like the IID/random-fold regime than the edge-stress regime.

That changed the research program:

- the shared MLP abundance family became a serious contender
- abundance sigma-only modifications were mostly dead ends
- target-specialised neural hybrids became the high-upside path

### Final Best System

The final best verified submission emerged from this revised hypothesis:

> use the neural family, but let target structure break the multioutput compromise.

That produced the winning target allocation:

- `log_H2O`: shared MLP
- `log_CO2`: single-target MLP
- `log_CH4`: single-target MLP
- `log_CO`: single-target MLP
- `log_NH3`: single-target MLP

## Repository Guide

### Core Code

- `src/`: data loading, features, metrics, calibration, and models
- `scripts/`: phase runners and submission helpers
- `outputs/reports/`: audit, EDA, tournament, and calibration reports

### Recommended Reading Order

1. `outputs/reports/phase1_audit.md`
2. `outputs/reports/phase1_eda.md`
3. `outputs/reports/phase2_model_tournament.md`
4. `outputs/reports/phase2_ablation.md`
5. `outputs/reports/phase3_fasttrack_summary.md`
6. `docs/research_report.md`

## Figures

The repository includes phase-1 figures that remain useful for understanding the modelling story:

- `outputs/reports/assets/phase1_target_distributions.png`
- `outputs/reports/assets/phase1_target_correlation.png`
- `outputs/reports/assets/phase1_metadata_target_relationships.png`
- `outputs/reports/assets/phase1_spectrum_noise_summary.png`
- `outputs/reports/assets/phase1_snr_diagnostics.png`
- `outputs/reports/assets/phase1_regime_scatter.png`

## Framing

This project should be read less as a polished “final pipeline” and more as a compact case study in empirical scientific reasoning under leaderboard pressure:

- audit the metric before optimising it
- mistrust elegant monoliths until they win
- treat uncertainty as part of the prediction, not an accessory
- let the leaderboard update your beliefs when it contradicts your preferred local story

## Repository Scope

This GitHub-facing version intentionally excludes:

- competition datasets
- generated submission files
- large experiment outputs
- cached artifacts
- secrets and team credentials

The goal is a clean research/code repository rather than a data dump.
