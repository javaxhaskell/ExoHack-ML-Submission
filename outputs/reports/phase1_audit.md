# Phase 1 Audit

## Relevant repository files
- `/Users/arhamshuaib/Desktop/ExoHack/hackathon_starter_solution.ipynb`
- `/Users/arhamshuaib/Desktop/ExoHack/utils.py`
- `/Users/arhamshuaib/Desktop/ExoHack/Hackathon_training/Training_targets.csv`
- `/Users/arhamshuaib/Desktop/ExoHack/Hackathon_training/Training_supplementary_data.csv`
- `/Users/arhamshuaib/Desktop/ExoHack/Hackathon_training/Training_SpectralData.hdf5`
- `/Users/arhamshuaib/Desktop/ExoHack/Hackathon_training/Test_supplementary_data.csv`
- `/Users/arhamshuaib/Desktop/ExoHack/Hackathon_training/Test_SpectralData.hdf5`

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
  `['Planet_21988', 'Planet_21989', 'Planet_21990', 'Planet_21991', 'Planet_21992', 'Planet_21993', 'Planet_21994', 'Planet_21995', 'Planet_21996', 'Planet_21997']`
- Test HDF5 raw iteration is lexicographic, not numeric:
  `['Planet_0', 'Planet_1', 'Planet_10', 'Planet_100', 'Planet_1000', 'Planet_10000', 'Planet_10001', 'Planet_10002', 'Planet_10003', 'Planet_10004']`
- Canonical rule: join by `planet_ID` after numerically sorting HDF5 group names, never by raw HDF5 iteration order.

## Shapes and canonical tables
- Training targets shape: `(69404, 7)`
- Training supplementary shape: `(69404, 10)`
- Training spectra shape: `(69404, 52)`
- Training noise shape: `(69404, 52)`
- Test supplementary shape: `(21988, 10)`
- Test spectra shape: `(21988, 52)`
- Shared wavelength grid across train/test: `True`
- Canonical flat training table shape: `(69404, 120)`
- Canonical flat test table shape: `(21988, 114)`
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
  - `star_distance, star_mass_kg, star_radius_m, star_temperature, planet_mass_kg, planet_orbital_period, planet_distance, planet_radius_m, planet_surface_gravity`
- Usable spectral features:
  - `spectrum_bin_00` ... `spectrum_bin_51`
  - `noise_bin_00` ... `noise_bin_51`
- Targets:
  - `planet_temp, log_H2O, log_CO2, log_CH4, log_CO, log_NH3`

## Key implication
- The notebook is a demonstrator, not a reliable competition pipeline: it ignores metadata, assumes one generic multioutput model, uses uncalibrated fold spread for uncertainty, and its raw HDF5 test loading order is unsafe.
