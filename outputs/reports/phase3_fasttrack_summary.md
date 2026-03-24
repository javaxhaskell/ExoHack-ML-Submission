# Phase 3 Fast-Track Summary

## OOF winner
- Edge-first winner: `target_specific_best`
- Conservative fallback: `disagreement_only`
- OOF edge score: `0.490429`
- OOF random score: `0.641360`

## Deployed final artifact
- `final_mu.csv` uses the locked mean backbones only:
  - `planet_temp`: `temp_rf_meta`
  - abundances: `abun_sep_et_spec_noise_meta_deriv`
- `final_std.csv` uses a deployable hybrid sigma system:
  - `planet_temp`: edge-calibrated constant sigma = `10.016041`
  - `log_H2O`: `residual_model`
  - `log_CO2`: `residual_model`
  - `log_CH4`: `disagreement_scaled`
  - `log_CO`: `residual_model`
  - `log_NH3`: `residual_model`

## Why temperature changed at deployment
- The raw OOF winner used `disagreement_scaled` for temperature.
- On the full-data test fit, `temp_rf_meta` vs `temp_et_meta` disagreement collapsed toward zero, producing unsafe near-zero `planet_temp` standard deviations.
- The residual-model sigma for temperature also collapsed on the full-data test fit.
- The shipped fix is the edge-holdout calibrated constant temperature sigma, which is materially safer under shift and keeps `std > 0` with reasonable magnitude.

## Fast-track ranking
### Edge holdout
```text
target_specific_best  0.490429
residual_only         0.488343
disagreement_only     0.478824
blend_uniform         0.474607
baseline_constant     0.472515
```

### Random 5-fold
```text
residual_only         0.643348
target_specific_best  0.641360
blend_uniform         0.635664
disagreement_only     0.629351
baseline_constant     0.628519
```

## Files
- Mu: [final_mu.csv](/Users/arhamshuaib/Desktop/ExoHack/final_mu.csv)
- Std: [final_std.csv](/Users/arhamshuaib/Desktop/ExoHack/final_std.csv)
- Metrics: [phase3_fast_metrics.csv](/Users/arhamshuaib/Desktop/ExoHack/outputs/calibration/phase3_fast_metrics.csv)
- System ranking: [phase3_fast_systems.csv](/Users/arhamshuaib/Desktop/ExoHack/outputs/calibration/phase3_fast_systems.csv)
