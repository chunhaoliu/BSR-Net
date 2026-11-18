# Test-Set Protocol

The test set is generated once, independently of the development data, and is used only for the final evaluation.
`scripts/generate_confirmatory_holdout.py` binds this document by SHA-256 into the dataset manifest.

## Data contract

- Protocol identifier: `hgv_multiregime_confirmatory_holdout`
- Generator seed: `20260831`
- Complete trajectories: `360`, trajectory IDs `1800`-`2159`
- Strata: quasi-equilibrium / skip glide crossed with longitudinal / turning / weaving, exactly `60` trajectories each
- Samples per trajectory: `1000` at `1 s`
- Input and prediction length: `256` each; evaluation window stride `5` (`98` windows per trajectory, `35,280` in total)
- Integration: DOP853, relative tolerance `1e-7`, absolute tolerance `1e-9`, maximum step `4 s`
- Simulator, control sampling, physical-envelope filters, and acceptance criteria: identical to the development set
  (`scripts/generate_multiregime_dataset.py --formal`)
- Output: `data_generation/data/processed/hgv_confirmatory_holdout.npz` (generation never overwrites an existing file)

## Density-mismatch sets

`--density-mismatch A` regenerates the same 360 trajectories with the true atmospheric density multiplied by
`1 + A sin(2 pi h / 8 km)`, while identification, prior, and training data keep the U.S. Standard Atmosphere 1976.
The paper uses `A = 0.15` and `A = 0.30` (`hgv_confirmatory_holdout_mismatch15.npz`, `..._mismatch30.npz`).

## Aerodynamic-mismatch sets

`--aero-scale K_L K_D` regenerates the same 360 trajectories with the true CAV-H lift and drag coefficients scaled by
`K_L` and `K_D` (the vehicle's own guidance uses the scaled lift), while all predictors keep the nominal training data.
The paper uses `(1.1, 0.9)` and `(0.9, 1.1)` (`hgv_confirmatory_holdout_mismatchaero110_90.npz`, `..._mismatchaero90_110.npz`).

## Evaluation contract

- Models are trained and selected on the development set only; no model, checkpoint, normalization statistic, or
  hyperparameter is changed after the test set is generated.
- Seeds `42`, `123`, `456`; noise levels `0`, `50`, `100`, `200 m` with fixed noise seeds.
- Metrics: ADE and FDE in kilometers in the local east-north-up frame at horizons `32`, `64`, `128`, `256 s`;
  significance by two-sided Wilcoxon signed-rank tests on per-trajectory seed-mean ADE.

The test set comes from the same simulator family as the development data and supports simulation-based conclusions only.
