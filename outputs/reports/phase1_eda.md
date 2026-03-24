# Phase 1 EDA

## Target distributions
![Target distributions](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_target_distributions.png)

```
     target      mean      std      min      25%       50%       75%       max
planet_temp 1203.4022 683.3461 114.2213 710.1348 1066.8588 1534.0772 5476.0183
    log_H2O   -6.0000   1.7335  -9.0000  -7.4954   -5.9996   -4.4998   -3.0000
    log_CO2   -6.5067   1.4448  -8.9999  -7.7600   -6.5125   -5.2533   -4.0000
    log_CH4   -5.9995   1.7410  -9.0000  -7.5154   -6.0026   -4.4837   -3.0000
     log_CO   -4.4931   0.8633  -6.0000  -5.2367   -4.4921   -3.7442   -3.0000
    log_NH3   -6.4903   1.4404  -8.9999  -7.7388   -6.4792   -5.2420   -4.0000
```

Modelling implication:
- `planet_temp` is broad and right-skewed.
- The log-abundance targets are centered near their allowed mid-ranges with hard-looking bounds close to the prior limits.
- Clip or otherwise guard predictions to plausible support during inference and calibration.

## Target correlation matrix
![Target correlation matrix](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_target_correlation.png)

```
     target  planet_temp  log_H2O  log_CO2  log_CH4  log_CO  log_NH3
planet_temp        1.000   -0.001   -0.002    0.007   0.001    0.001
    log_H2O       -0.001    1.000   -0.004   -0.010   0.003   -0.002
    log_CO2       -0.002   -0.004    1.000   -0.003   0.001    0.003
    log_CH4        0.007   -0.010   -0.003    1.000  -0.005    0.005
     log_CO        0.001    0.003    0.001   -0.005   1.000   -0.004
    log_NH3        0.001   -0.002    0.003    0.005  -0.004    1.000
```

Modelling implication:
- Cross-target correlations are effectively zero.
- Multioutput elegance is unlikely to buy much; target-specific models are more plausible.

## Metadata distributions
![Metadata distributions](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_metadata_distributions.png)

```
               feature          min           1%           5%          25%          50%          75%          95%          99%          max
         star_distance 6.531300e+00 2.799560e+01 6.423080e+01 2.237570e+02 4.632230e+02 8.123670e+02 1.393850e+03 2.011410e+03 7.596910e+03
          star_mass_kg 3.181456e+29 8.104759e+29 1.193046e+30 1.667520e+30 1.948642e+30 2.286671e+30 3.168556e+30 5.010793e+30 6.228813e+30
         star_radius_m 1.175733e+08 2.621224e+08 3.965490e+08 5.913450e+08 7.443990e+08 1.015722e+09 1.579239e+09 2.678445e+09 6.226515e+09
      star_temperature 2.940000e+03 3.414000e+03 3.957000e+03 5.253000e+03 5.731000e+03 6.109000e+03 7.069000e+03 8.828700e+03 1.017000e+04
        planet_mass_kg 1.157780e+25 1.793610e+25 2.135902e+25 4.118660e+25 1.155217e+26 6.518284e+26 4.251520e+27 8.180380e+27 2.733120e+29
 planet_orbital_period 2.242000e-01 6.536000e-01 1.346100e+00 3.352700e+00 6.264300e+00 1.553190e+01 8.457330e+01 3.743644e+02 3.650000e+03
       planet_distance 5.600000e-03 1.420000e-02 2.350000e-02 4.420000e-02 6.630000e-02 1.191000e-01 3.736000e-01 9.960000e-01 4.478700e+00
       planet_radius_m 9.363231e+06 9.577807e+06 1.055656e+07 1.535807e+07 2.678689e+07 7.598476e+07 1.112915e+08 1.335300e+08 1.558913e+08
planet_surface_gravity 5.940000e-01 4.783900e+00 6.515600e+00 8.014400e+00 1.040780e+01 1.184190e+01 2.693200e+01 1.237090e+02 2.431812e+03
```

Modelling implication:
- Most metadata spans orders of magnitude, especially masses, radii, orbital period, and distance.
- Log transforms should be the default for metadata models.

## Metadata vs target relationships
![Metadata versus targets](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_metadata_target_relationships.png)

```
     target                feature  spearman
planet_temp  planet_orbital_period   -0.8210
planet_temp        planet_distance   -0.7437
planet_temp          star_radius_m    0.6821
    log_H2O         planet_mass_kg   -0.0052
    log_H2O        planet_radius_m   -0.0050
    log_H2O          star_distance    0.0034
    log_CO2  planet_orbital_period    0.0057
    log_CO2        planet_distance    0.0054
    log_CO2 planet_surface_gravity   -0.0025
    log_CH4        planet_radius_m    0.0080
    log_CH4           star_mass_kg    0.0073
    log_CH4       star_temperature    0.0068
     log_CO       star_temperature    0.0032
     log_CO          star_radius_m    0.0027
     log_CO          star_distance    0.0026
    log_NH3         planet_mass_kg   -0.0022
    log_NH3        planet_radius_m   -0.0015
    log_NH3          star_distance    0.0013
```

Modelling implication:
- `planet_temp` is strongly metadata-dominated, especially by orbital period, orbital distance, and stellar/planet size proxies.
- Abundance targets show near-zero monotonic relationships with metadata, so metadata-only chemistry models are weak candidates.

## Spectrum and noise over wavelength
![Spectrum and noise summary](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_spectrum_noise_summary.png)

Modelling implication:
- Spectrum amplitude and variance change dramatically with wavelength.
- Noise is wavelength-dependent, so not all bins should be trusted equally.
- Log-transformed spectral inputs and explicit use of noise are justified.

## Signal-to-noise diagnostics
![SNR diagnostics](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_snr_diagnostics.png)

```
       metric        value
  mean_snr_p0       31.413
  mean_snr_p5       57.301
 mean_snr_p25       83.709
 mean_snr_p50      122.829
 mean_snr_p75      199.615
 mean_snr_p95      602.096
mean_snr_p100 12078926.607
```

Modelling implication:
- Mean SNR is extremely heavy-tailed: median is moderate, but the top tail is enormous.
- Robust scaling, clipping, or calibration by regime will matter more than naive Gaussian assumptions.

## Spectral regimes
![Regime scatter](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_regime_scatter.png)

![Representative regime spectra](/Users/arhamshuaib/Desktop/ExoHack/outputs/reports/assets/phase1_representative_regimes.png)

```
 regime_group  n_planets  planet_temp
            3       2304      515.597
            7       2057      642.987
           13       2112     1325.852
           14       2960     1071.198
           19       3139      452.226
           28       2368     2001.793
           44       2797     2319.054
           48       2515     1622.786
           52       2794     1702.657
           53       2560     1323.536
```

Modelling implication:
- The spectra are not one homogeneous cloud. There are clear regimes in overall depth and slope, and those regimes carry different average temperatures.
- Validation should include regime-aware splits, not just random folds.

## Minimal hypothesis probes
These are intentionally point-prediction probes only. They are not final probabilistic model selection because uncertainty calibration is still missing.

### Temperature probe
```
                        model             scheme     r2     rmse      mae
           metadata_only_et40       random_5fold 1.0000   1.6520   0.0179
           spectrum_only_et40       random_5fold 0.8239 286.7453 164.9500
starter_like_et10_multioutput       random_5fold 0.8274 283.8668 162.5281
           metadata_only_et40 regime_group_5fold 0.9971  36.6659  10.4853
           metadata_only_et40 edge_holdout_15pct 0.9702 160.2837  62.5979
starter_like_et10_multioutput regime_group_5fold 0.4931 486.5375 301.3485
```

Implication:
- Metadata-only temperature prediction is very strong.
- The starter-like spectrum-only multioutput baseline is materially weaker for temperature.

### Abundance probe
```
 target  metadata_only_et40  spectrum_only_et40
log_CH4             -0.0782              0.8720
 log_CO             -0.0710              0.3562
log_CO2             -0.0695              0.8114
log_H2O             -0.0672              0.6653
log_NH3             -0.0643              0.5778
```

Implication:
- Metadata-only models are weak for abundances.
- Spectrum-only models contain real abundance signal even in this minimal probe.
- This supports a specialist strategy: metadata-heavy temperature model, spectrum-driven abundance models.
