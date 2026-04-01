# Preferential Feedback Thompson Sampling (PF-TS)

Code accompanying the paper:

> **A Finite Time Analysis of Thompson Sampling for Bayesian Optimization with Preferential Feedback**
> Joseph Lazzaro, Davide Buffelli, Da-shan Shiu, Sattar Vakili
> *Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS), 2026*

## Overview

We propose **PF-TS**, a Thompson Sampling approach to Bayesian Optimization with preferential (pairwise comparison) feedback. The method models comparisons via a monotone link on latent utility differences, leverages the dueling kernel induced by a base kernel, and selects both elements of each queried pair through independent posterior samples — yielding a fully sequential and symmetric algorithm.

**Key contributions:**

1. **Anchor independence** (Proposition 1): A Thompson sample of the pairwise-difference function decomposes as a difference of latent utilities, so pair selection reduces to two independent single-argument maximisations.
2. **Finite-time regret guarantee** (Theorem 1): PF-TS achieves cumulative regret Õ(β_T √(TΓ(T))), matching the rate of standard GP-TS under scalar feedback.
3. **Empirical validation** on synthetic benchmarks and a real catalyst design task using the OCx24 dataset.

## Repository Structure

```
.
├── Ackley/                         # 1-D Ackley synthetic experiments
│   ├── ackley_experiments.py       # Main script — reproduces Figures 2a, 2b
│   ├── Thompson_Sampling_cpu.py    # PF-TS implementation (Algorithm 1)
│   ├── POPBO_cpu.py                # POP-BO baseline (Xu et al., 2024)
│   ├── Algos_direct_feedback_cpu.py  # GP-UCB / GP-TS scalar baselines
│   └── clean_test_algos_synthetic_cpu.py  # Shared utilities, MR-LPF, MaxMinLCB
│
├── Catalyst/                       # OCx24 catalyst design experiments
│   ├── cat_experiments.py          # Main script — reproduces Figures 2c, 2d
│   ├── Thompson_cat.py             # PF-TS and POP-BO (high-dimensional, GPU)
│   ├── Algos_high_dim_cpu.py       # GP-TS / GP-UCB scalar baselines (high-dim)
│   ├── clean_test_algos_yelp_full_gpu_optim_fix.py  # Shared utilities (GPU)
│   └── CatData_final.csv           # OCx24 hydrogen-yield data (Ag/Au/Zn)
│
└── requirements.txt
```

## Reproducing the Main Results

### Prerequisites

```bash
pip install -r requirements.txt
```

The Ackley experiments run on CPU only. The Catalyst experiments benefit from a CUDA-capable GPU but will fall back to CPU automatically.

### Ackley Experiments (Figures 2a, 2b)

```bash
cd Ackley
python ackley_experiments.py
```

This runs all five methods (PF-TS, MR-LPF, MaxMinLCB, POP-BO, GP-TS) for 30 independent runs with T = 300 iterations on the 1-D Ackley function discretised into 40 actions. Results are saved to `ackley40/` as CSV files and PDF plots.

**Expected runtime:** ~5–10 hours per preference-feedback algorithm on a modern CPU.

### Catalyst Experiments (Figures 2c, 2d)

```bash
cd Catalyst
python cat_experiments.py
```

This runs all five methods for 30 runs with T = 800 iterations on the OCx24 catalyst composition dataset (63 Ag/Au/Zn compositions). Results are saved to `cat_sims/`.

**Expected runtime:** ~10–20 hours per preference-feedback algorithm (GPU recommended).

### Hyperparameters

All experiments use the following shared hyperparameters (matching the paper):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Kernel | Matérn-5/2 | Base kernel for GP surrogate |
| Length scale | 0.1 | Kernel length scale |
| λ (regularisation) | 0.05 | Logistic regression regulariser |
| β (confidence width) | 1.0 | Confidence interval coefficient (CI-based methods) |
| α_GP | 0.05 | GP noise parameter |
| Learning rate | 0.001 | Gradient descent for logistic regression |

## Algorithms Compared

| Method | Reference | Feedback |
|--------|-----------|----------|
| **PF-TS** (ours) | This paper | Pairwise preferences |
| MR-LPF | Kayal et al. (2025) | Pairwise preferences |
| MaxMinLCB | Pásztor et al. (2024) | Pairwise preferences |
| POP-BO | Xu et al. (2024) | Pairwise preferences |
| GP-TS | Chowdhury & Gopalan (2017) | Direct scalar |

## Data

The catalyst data (`CatData_final.csv`) is extracted from the [Open Catalyst Experiments 2024 (OCx24)](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/applications/ocx/data/experimental_data) dataset released by FAIR Chemistry (Meta AI). It contains hydrogen yields for 63 catalyst compositions of silver (Ag), gold (Au), and zinc (Zn) under the CO₂R reaction at 300-amp current.

## Citation

```bibtex
@inproceedings{lazzaro2026finite,
  title={A Finite Time Analysis of Thompson Sampling for Bayesian Optimization with Preferential Feedback},
  author={Lazzaro, Joseph and Buffelli, Davide and Shiu, Da-shan and Vakili, Sattar},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026}
}
```

## Acknowledgements

This work was completed while Joseph Lazzaro was an intern at MediaTek Research. The MR-LPF and MaxMinLCB baseline implementations are adapted from the codebase of [Kayal et al. (2025)](https://arxiv.org/abs/2402.01316).
