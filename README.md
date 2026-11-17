# BSR-Net

Official code for **Maneuver-Aware Hypersonic Glide Vehicle Trajectory Prediction from Noisy Radar Tracks via Differentiable Bank-Schedule Rollout** (Aerospace Science and Technology, under review).

BSR-Net predicts the position of a hypersonic glide vehicle (HGV) 256 s ahead from a noisy position-only radar track. A batch Levenberg–Marquardt fit identifies the state and the equivalent lift coefficient, drag coefficient, and bank angle, for the newest window and as a sliding history. A Transformer reads the track and this parameter history and predicts a piecewise-constant bank schedule (32 blocks of 8 s). The schedule is integrated through a differentiable rotating-Earth three-degree-of-freedom model as a shift relative to the held-bank physics prior, and a residual decoder corrects the remaining error. All learned outputs are zero-initialized, so training starts exactly at the physics prior.

On 360 independent test trajectories with 0–200 m position noise, BSR-Net reduces the 256-s average displacement error of the identified physics prior by 43%–48%, and under noise that of the strongest learner given the same physics inputs by 9.7%–10.4%.

## Contents

| Path | Role |
|---|---|
| `models/bsrnet.py` | BSR-Net (`ManeuverAST`) and the residual-on-prior / direct ablation models (`ResidualAST`) |
| `models/dynamics.py` | Rotating-Earth 3-DOF point-mass model and RK4 propagation |
| `utils/filter_prior.py` | Batch LM identification from positions, held-bank rollout, sliding parameter series, VARMA forecast |
| `models/hgv_baselines.py`, `models/public_baselines.py` | PIT and AF-CILN reimplementations, prior-augmented inputs, adapter for the official Time-Series-Library models |
| `scripts/generate_multiregime_dataset.py`, `scripts/generate_confirmatory_holdout.py`, `data_generation/` | Simulator and the development, test, and density-mismatch datasets |
| `scripts/pipeline.py` | Identification cache, training, evaluation, stress tests, per-window errors, latency |
| `scripts/analysis.py`, `scripts/plot_figures.py` | Significance tests, label-free schedule analysis, error split, AUROC, GFLOPs; paper figures |
| `checkpoints/` | Trained BSR-Net weights (3 seeds), BSR-Net without bank supervision (λ = 0), BSR-Net without rollout |
| `results/` | Result tables behind the paper (see below) |
| `docs/CONFIRMATORY_HOLDOUT_PROTOCOL.md` | Test-set and density-mismatch protocol |

Variant names used by the code: `ours2s` = BSR-Net; `ours2s_nb` = w/o bank-schedule rollout; `ours2` = w/o identified-parameter sequence; `ours` = w/o both (residual on prior); `direct` = w/o physics prior. `--aux` is the bank-term weight λ. Method labels in the result tables: `Ours-MS` = BSR-Net, `Ours-MS w/o bank rollout`, `Ours-M` = w/o identified-parameter sequence, `Ours` = w/o both, `Direct` = w/o physics prior, `A_fit` = identified physics prior; `NoAux` is a residual-on-prior variant without the auxiliary term that is not reported in the paper.

## Installation

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or later and a CUDA GPU (the rollout is captured as a CUDA graph). The generic forecasters use the official [Time-Series-Library](https://github.com/thuml/Time-Series-Library) at commit `4e938a1767106324dd753b2a44832bf870a0252e`; clone it to `third_party/Time-Series-Library` or set `HGV_TSLIB_ROOT`. BSR-Net itself does not need it.

## Reproducing the paper

```bash
# 1. Data: 1,800 development trajectories, 360 test trajectories, and the two density-mismatch sets
python scripts/generate_multiregime_dataset.py --formal
python scripts/generate_confirmatory_holdout.py
python scripts/generate_confirmatory_holdout.py --density-mismatch 0.15
python scripts/generate_confirmatory_holdout.py --density-mismatch 0.30

# 2. Identification: physics prior (GPU) and sliding parameter series (CPU; slow for the training split)
for s in train val holdout mismatch15 mismatch30; do
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

# 4. Baselines (learning rates as in Table 2; DLinear and AF-CILN with the 100-epoch budget)
python scripts/pipeline.py train --variant itransformer_priors --lr 1e-3 --seed 42
python scripts/pipeline.py train --variant dlinear --lr 1e-3 --epochs 100 --patience 20 --tag _ep100p20 --seed 42
python scripts/pipeline.py analytic --split val && python scripts/pipeline.py analytic --split holdout   # VARMA-PM series
python scripts/pipeline.py analytic --select && python scripts/pipeline.py analytic --evaluate

# 5. Evaluation and analyses
python scripts/pipeline.py stress --split holdout      # Tables 3-7 -> outputs/results/holdout_summary.csv
python scripts/pipeline.py stress --split mismatch15   # Fig. 7
python scripts/pipeline.py stress --split mismatch30
python scripts/pipeline.py errors                      # per-window errors for Wilcoxon tests and Fig. 3
python scripts/pipeline.py latency                     # Table 8
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
| `results/holdout_summary.csv` | Tables 3–7 and Fig. 3: ADE/FDE per method, noise level, group (overall, bank change, strata), and horizon; mean and standard deviation over seeds |
| `results/varma_summary.csv` | VARMA-PM rows of Table 3 |
| `results/mismatch15_summary.csv`, `results/mismatch30_summary.csv` | Fig. 7 (density mismatch) |
| `results/latency.csv` | Table 8 |
| `results/tuning.csv` | Learning-rate selection (Table 2) |
| `results/lam0_holdout.txt`, `results/holdout_extras.txt` | Label-free schedule (Section 5.4); percentiles, error split, AUROC, GFLOPs (Section 5.3, Table 8) |

## Tests

```bash
pytest
```

## Citation

See [`CITATION.cff`](CITATION.cff); the bibliographic record will be updated on publication.

## License

MIT; see [`LICENSE`](LICENSE).
