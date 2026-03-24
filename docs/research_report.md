# From Robust Trees To Neural Target Specialisation

## A Research Note On Hidden-Test Inference In Probabilistic Exoplanet Retrieval

### Best Verified Score

**0.8707883387590796**

## 1. Problem Setting

The challenge was not standard regression. Each planet required, for six atmospheric targets:

- a predictive mean
- a predictive standard deviation

The official metric was a Gaussian CRPS-style skill score, not a point-estimate metric. This had two immediate implications:

1. mean accuracy alone was insufficient
2. uncertainty calibration was part of the core modelling problem

The six targets were:

- `planet_temp`
- `log_H2O`
- `log_CO2`
- `log_CH4`
- `log_CO`
- `log_NH3`

## 2. Initial Scientific Hypotheses

The early working hypotheses were:

- temperature might be largely metadata-driven
- abundances would be spectrum-driven and likely target-specific
- uncertainty engineering could move score materially
- shared posterior-style elegance was less important than hidden-test utility

These hypotheses were not accepted as truths. They were treated as falsifiable claims.

## 3. Metric-First Audit

The first phase was entirely diagnostic.

The repository audit established:

- the exact local score logic
- the requirement that all predicted standard deviations be strictly positive
- the correct submission format
- the true HDF5 loading behavior
- the possibility of silent misalignment if spectral groups were read lexicographically rather than numerically

One of the most practically important discoveries was not a model at all:

> the public-test HDF5 file could be misaligned with `planet_ID` if loaded naively.

This kind of infrastructure error is often more damaging than a modest model misspecification.

## 4. What The Data Suggested

EDA was conducted only insofar as it informed modelling.

The clearest conclusions were:

- `planet_temp` behaved like a metadata specialist target
- molecular abundances did not behave like a single clean shared family
- the spectral space had visible regimes and strong heterogeneity
- signal-to-noise structure mattered

In practical terms, this argued against a monolithic retrieval model and in favor of explicit target-wise thinking.

## 5. Phase-2 Mean Tournament

The phase-2 tournament compared serious model families under multiple validation schemes:

- `random_5fold`
- `regime_group_5fold`
- `edge_holdout_15pct`

The robust conclusion at that stage was:

- `temp_rf_meta` was the strongest temperature model
- `abun_sep_et_spec_noise_meta_deriv` was the strongest robust abundance family

This was a sensible result. Tree ensembles were stable, interpretable, and hard to embarrass on the harder local splits.

But they were not the end of the story.

## 6. The Critical Contradiction

A shared MLP abundance model,

`abun_shared_mlp_svd_spec_noise_meta`,

looked extraordinary on `random_5fold` and deeply suspect on the hardest edge split.

This created the central tension of the project:

- **robust-local view**: reject the MLP as too brittle
- **IID-hidden-test view**: treat the MLP as a high-upside candidate whose local “failure” may reflect a mismatch between our hardest split and the actual hidden distribution

The leaderboard later made this tension decisive.

## 7. Uncertainty Engineering

The next phase focused on sigma rather than mu:

- constant sigma baselines
- residual-model sigma
- disagreement-scaled sigma
- blended sigma systems

The important methodological lesson here was that uncertainty design could not be postponed. The score directly rewarded calibrated predictive distributions.

The strongest recurring pattern was:

- naive spread was not enough
- residual-model sigma was usually stronger
- temperature required special handling because ensemble disagreement could collapse on the full-test fit

## 8. What The Leaderboard Taught

The leaderboard did something local validation alone could not do:

it revealed that the hidden test was much more IID-like than the most pessimistic local regime-shift proxy.

This changed the scientific interpretation of earlier experiments.

What had looked like “brittleness” under the edge split became, under the leaderboard lens, evidence of a different inductive bias that was simply better matched to the hidden distribution.

That shift justified a more aggressive neural strategy.

## 9. The Winning Path

The best verified solution did not end at the shared MLP. It refined it.

The final winning system used a target-wise neural hybrid:

- `log_H2O`: shared MLP
- `log_CO2`: single-target MLP
- `log_CH4`: single-target MLP
- `log_CO`: single-target MLP
- `log_NH3`: single-target MLP

Temperature remained on the established specialist path, with its sigma held at the previously winning 70%-scaled setting.

This structure says something scientifically interesting:

> the neural family was right, but complete sharing across chemistry targets was still too strong an assumption.

The best hidden-test performance emerged when the model family stayed neural while the target structure became more selective.

## 10. Why The Final System Likely Worked

Several interacting explanations are plausible.

### 10.1 The hidden set rewarded IID-style inductive bias

The shared/single-target MLP family appears to have extracted signal that tree models did not capture as effectively in the hidden regime.

### 10.2 Multioutput sharing helped some targets, but not all

`log_H2O` appears to benefit from the shared multioutput representation.
Other molecules improved when the network was allowed to specialize.

### 10.3 Sigma had to be aligned to the chosen mean source

Using a mismatched uncertainty system for a new mean backbone was repeatedly punished.
Once the abundance means became neural and target-specialised, the sigma system also had to become source-aware.

## 11. Best Verified Leaderboard Progression

The largest jumps came from structural changes, not cosmetic tweaks.

Approximate trajectory:

- baseline-final tree-era submission: `0.6523547402202179`
- temperature sigma improvement: `0.6526698642474705`
- neural abundance “win-or-die” pivot: `0.7869490306408244`
- shared-MLP random-fold sigma rebuild: `0.8505132253309213`
- target-wise neural hybrid: `0.8707883387590796`

This progression matters because it shows that the main gains did **not** come from small local smoothing. They came from updating the model family and target structure in response to hidden-test evidence.

## 12. Methodological Lessons

The broader lessons of this repository are transferable beyond this specific competition.

### 12.1 Hard local robustness splits are not automatically the truth

They are hypotheses about distribution shift.
Useful hypotheses, but still hypotheses.

### 12.2 Probabilistic competitions punish incoherent pipelines

Prediction and uncertainty must be developed together.

### 12.3 “Elegant” shared models are not automatically superior

Sharing should survive empirical challenge, not aesthetic preference.

### 12.4 The leaderboard can be epistemically useful

When used carefully, the leaderboard is not just a scoreboard. It is an instrument for learning about the hidden distribution.

## 13. Open Questions

Several questions remain genuinely interesting:

- Would a larger neural seed ensemble have improved further, or merely reduced useful variance?
- Could a better stacked sigma system have pushed the target-wise neural hybrid further?
- Is the hidden-test regime merely IID-like, or does it privilege a more specific spectral manifold that the neural family captured?

These remain open because the goal here was competitive performance under deadline, not exhaustive scientific closure.

## 14. Final Position

The strongest interpretation of this work is not “we found the best architecture.”

It is this:

> the winning solution emerged by repeatedly revising our beliefs about the hidden distribution, then letting those revised beliefs change both model family and uncertainty strategy.

That is the core research story of this repository.
