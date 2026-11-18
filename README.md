# BSR-Net

Official code for **Maneuver-Aware Hypersonic Glide Vehicle Trajectory Prediction from Noisy Radar Tracks via Differentiable Bank-Schedule Rollout** (Aerospace Science and Technology, under review).

BSR-Net predicts the position of a hypersonic glide vehicle (HGV) 256 s ahead from a noisy position-only radar track. A batch Levenberg–Marquardt fit identifies the state and the equivalent lift coefficient, drag coefficient, and bank angle, for the newest window and as a sliding history. A Transformer reads the track and this parameter history and predicts a piecewise-constant bank schedule (32 blocks of 8 s). The network reads the identified C_L and C_D only relative to the newest fit, while the prior and the rollout use their absolute values. The schedule is integrated through a differentiable rotating-Earth three-degree-of-freedom model as a shift relative to the held-bank physics prior, and a residual decoder corrects the remaining error. All learned outputs are zero-initialized, so training starts exactly at the physics prior.

On 360 independent test trajectories with 0–200 m position noise, BSR-Net reduces the 256-s average displacement error of the identified physics prior by 45%–48%, and under noise that of the strongest learner given the same physics inputs by 9.7%–10.6% (14%–20% when the true lift and drag coefficients deviate by 10% from the training vehicle).

## Contents

| Path | Role |
|---|---|
| `models/bsrnet.py` | BSR-Net (`ManeuverAST`) and the residual-on-prior / direct ablation models (`ResidualAST`) |
| `models/dynamics.py` | Rotating-Earth 3-DOF point-mass model and RK4 propagation |
| `utils/filter_prior.py` | Batch LM identification from positions, held-bank rollout, sliding parameter series, VARMA forecast |
| `models/hgv_baselines.py`, `models/public_baselines.py` | PIT and AF-CILN reimplementations, prior-augmented inputs, adapter for the official Time-Series-Library models |
| `scripts/generate_multiregime_dataset.py`, `scripts/generate_confirmatory_holdout.py`, `data_generation/` | Simulator and the development, test, density-mismatch, and aerodynamic-mismatch datasets |
| `scripts/pipeline.py` | Identification cache, training, evaluation, stress tests, per-window errors, latency |
| `scripts/analysis.py`, `scripts/plot_figures.py` | Significance tests, label-free schedule analysis, error split, AUROC, GFLOPs; paper figures |
| `checkpoints/` | Trained BSR-Net weights (3 seeds), BSR-Net without bank supervision (λ = 0), BSR-Net without rollout |
| `results/` | Result tables behind the paper (see below) |
| `docs/CONFIRMATORY_HOLDOUT_PROTOCOL.md` | Test-set and density-mismatch protocol (the aerodynamic-mismatch sets use the same seeds, `--aero-scale`) |

Variant names used by the code: `ours2s` = BSR-Net; `ours2s_nb` = w/o bank-schedule rollout; `ours2` = w/o identified-parameter sequence; `ours` = w/o both (residual on prior); `direct` = w/o physics prior. `--aux` is the bank-term weight λ; `--blocks` is the number K of schedule blocks; `--aero rel|abs` overrides the default aerodynamic input representation (relative for BSR-Net and its ablations, absolute for baselines), stored in the checkpoint as `rel_aero`. Method labels in the result tables: `Ours-MS` = BSR-Net, `Ours-MS w/o bank rollout`, `Ours-M` = w/o identified-parameter sequence, `Ours` = w/o both, `Direct` = w/o physics prior, `A_fit` = identified physics prior; `NoAux` is a residual-on-prior variant without the auxiliary term that is not reported in the paper.

## Installation

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or later and a CUDA GPU (the rollout is captured as a CUDA graph). The generic forecasters use the official [Time-Series-Library](https://github.com/thuml/Time-Series-Library) at commit `4e938a1767106324dd753b2a44832bf870a0252e`; clone it to `third_party/Time-Series-Library` or set `HGV_TSLIB_ROOT`. BSR-Net itself does not need it.

## Reproducing the paper

```bash
# 1. Data: 1,800 development trajectories, 360 test trajectories, two density-mismatch and two aerodynamic-mismatch sets
python scripts/generate_multiregime_dataset.py --formal
python scripts/generate_confirmatory_holdout.py
python scripts/generate_confirmatory_holdout.py --density-mismatch 0.15
python scripts/generate_confirmatory_holdout.py --density-mismatch 0.30
python scripts/generate_confirmatory_holdout.py --aero-scale 1.1 0.9
python scripts/generate_confirmatory_holdout.py --aero-scale 0.9 1.1

# 2. Identification: physics prior (GPU) and sliding parameter series (CPU; slow for the training split)
for s in train val holdout mismatch15 mismatch30 mismatchaero110_90 mismatchaero90_110; do
  python scripts/pipeline.py cache --split $s
  python scripts/pipeline.py analytic --split $s --iterations 5
done

# 3. BSR-Net (seeds 42, 123, 456), the lambda = 0 control, and the ablations
python scripts/pipeline.py train --variant ours2s --aux 1 --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant ours2s --aux 0 --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant ours2s_nb --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant ours2 --aux 1 --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant ours --aux 0.1 --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant direct --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant ours2s --aux 1 --lr 1e-3 --seed 42 --blocks 16 --tag _K16   # also K = 64

# 4. Baselines (learning rates as in Table 3; DLinear and AF-CILN with the 100-epoch budget)
python scripts/pipeline.py train --variant itransformer_priors --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant itransformer_priors --lr 1e-3 --seed 42 --aero rel --tag _rel   # input-representation control
python scripts/pipeline.py train --variant dlinear --lr 1e-3 --epochs 100 --patience 20 --tag _ep100p20 --seed 42
python scripts/pipeline.py analytic --split val && python scripts/pipeline.py analytic --split holdout   # VARMA-PM series
python scripts/pipeline.py analytic --select && python scripts/pipeline.py analytic --evaluate

# 5. Evaluation and analyses
python scripts/pipeline.py stress --split holdout      # Tables 4-7 -> outputs/results/holdout_summary.csv
for s in mismatch15 mismatch30 mismatchaero110_90 mismatchaero90_110; do python scripts/pipeline.py stress --split $s; done   # Fig. 7
python scripts/pipeline.py stress --split holdout --tag _K16 --blocks 16             # Table 8 (also _K64 --blocks 64)
python scripts/pipeline.py stress --split holdout --variant itransformer_priors --tag _rel   # relative-input control (all five splits)
python scripts/pipeline.py errors                      # per-window errors for Wilcoxon tests and Fig. 3
python scripts/pipeline.py latency --fit-iterations 5  # Table 9 (identification timed with 5 LM iterations)
python scripts/analysis.py wilcoxon
python scripts/analysis.py lambda0
python scripts/analysis.py extras
python scripts/analysis.py flops
python scripts/plot_figures.py --out figures
```

Repeat the training commands for seeds 123 and 456. Outputs go to `outputs/` (override with `HGV_OUT`). To evaluate the released weights without training, copy `checkpoints/*.pt` to `outputs/checkpoints/` after steps 1 and 2.

Learning rates of the remaining baselines: `transformer`, `itransformer`, `af_ciln` 1e-3; `patchtst`, `pit`, `pit_prior`, `itransformer_prior`, `itransformer_priorf` 3e-4.

## Results

| File | Paper |
|---|---|
| `results/holdout_summary.csv` | Tables 4–8 and Fig. 3: ADE/FDE per method, noise level, group (overall, bank change, strata), and horizon; mean and standard deviation over seeds |
| `results/varma_summary.csv` | VARMA-PM rows of Table 4 |
| `results/mismatch15_summary.csv`, `results/mismatch30_summary.csv`, `results/mismatchaero110_90_summary.csv`, `results/mismatchaero90_110_summary.csv` | Fig. 7 (density and aerodynamic mismatch) |
| `results/holdout_K16_summary.csv`, `results/holdout_K64_summary.csv` | Schedule resolution (Table 8) |
| `results/*_rel_summary.csv` | iTransformer+prior+series with relative aerodynamic inputs (Section 5.5) |
| `results/latency.csv` | Table 9 |
| `results/tuning.csv` | Learning-rate selection (Table 3) |
| `results/lam0_holdout.txt`, `results/holdout_extras.txt` | Label-free schedule (Section 5.4); percentiles, error split, AUROC, GFLOPs (Section 5.3, Table 9) |

## Tests

```bash
pytest
```

## Citation

See [`CITATION.cff`](CITATION.cff); the bibliographic record will be updated on publication.

## License

MIT; see [`LICENSE`](LICENSE).
