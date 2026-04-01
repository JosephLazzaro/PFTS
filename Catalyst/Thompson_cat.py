"""
PF-TS and POP-BO for the Catalyst (OCx24) experiments.

Adapts the 1-D algorithms to handle multi-dimensional action features
(3-D catalyst compositions) using GPyTorch's dueling kernel on the GPU.
"""

import numpy as np
import time
from itertools import product

import torch
import gpytorch
from botorch.models import SingleTaskGP

from clean_test_algos_yelp_full_gpu_optim_fix import (
    DuelingKernel2, predict_f, sigmoid, sigmoid_torch,
    find_most_preferred_action, device
)

# Re-export for use in the Ackley-side Thompson_Sampling_cpu
# (the 1-D code imports get_cov_scaling from Thompson_Sampling_cpu,
#  and the Catalyst code re-uses the same function via Thompson_cat)
import sys
sys.path.insert(0, "../Ackley")
try:
    from Thompson_Sampling_cpu import get_cov_scaling
except ImportError:
    # Fallback: define locally
    def get_cov_scaling(t, args, delta=0.05):
        return np.sqrt(2.0 * (t + 1 + np.log(2.0 / delta)))


# ---------------------------------------------------------------------------
#  Covariance helpers (GPU-based)
# ---------------------------------------------------------------------------

def update_cov_D_cat(dataset, kernel, alpha_gp, values, arm_features):
    """
    Compute the full posterior covariance over all action pairs
    using BoTorch's SingleTaskGP with the dueling kernel.
    """
    X = torch.tensor(dataset[:, :2], dtype=torch.float32, device=device)
    y = torch.tensor(dataset[:, 2], dtype=torch.float32, device=device)
    af = torch.tensor(arm_features, dtype=torch.float32, device=device)

    X_feat = torch.stack([af[int(x)] for x in X.flatten()]).view(X.shape[0], -1)

    model = SingleTaskGP(X_feat, y.unsqueeze(-1), covar_module=kernel)
    model.likelihood.noise = torch.tensor([alpha_gp], dtype=torch.float32,
                                           device=device)
    model.likelihood.raw_noise.requires_grad = False
    model.eval()

    X_grid = torch.tensor(list(product(values, repeat=2)),
                           dtype=torch.float32, device=device)
    X_grid_feat = torch.stack([af[int(x)] for x in X_grid.flatten()])\
                        .view(X_grid.shape[0], -1)

    with torch.no_grad():
        cov = model(X_grid_feat).covariance_matrix.cpu()

    return cov


def get_cov_strip(full_cov, x0, values):
    """Extract Cov(h(·, x0), h(·, x0)) from the full covariance."""
    n = len(values)
    cov_strip = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_strip[i, j] = full_cov[i * n + x0, j * n + x0]
    return cov_strip


def compute_regret_cat(x_index, x_prime_index, f, Reward_function):
    """Dueling regret for index-based action selection."""
    i_star = np.argmax(Reward_function)
    return (sigmoid(f[i_star, x_index]) +
            sigmoid(f[i_star, x_prime_index]) - 1.0) / 2.0


# ---------------------------------------------------------------------------
#  PF-TS for the Catalyst setting
# ---------------------------------------------------------------------------

def TS_cat(args, values, Reward_function, f, timestamp, seed=None):
    """
    Preferential Feedback Thompson Sampling (PF-TS) for multi-dimensional
    action spaces with feature-based kernel evaluation.

    Parameters
    ----------
    values : ndarray, shape (n_actions, d)
        Feature vectors for each action (catalyst compositions).
    """
    if seed is not None:
        np.random.seed(seed)

    fn_start = time.time()
    dataset = np.empty((0, 3))
    regret_list = []

    # Build dueling kernel
    base_kernel = (gpytorch.kernels.MaternKernel(nu=args.smoothness)
                   if args.kernel == "Matern"
                   else gpytorch.kernels.RBFKernel())
    dk = DuelingKernel2(kernel=base_kernel, len_scale=args.length_scale).to(device)

    log_fn = (f"TS_{args.kernel}_{args.learning_rate}_"
              f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_fn, "a") if log_fn else None

    try:
        for t in range(args.n_iterations):
            if len(dataset) == 0:
                # Random pair when no data available
                i, j = np.random.randint(len(values), size=2)
            else:
                # Estimate h_t on the grid (using action indices)
                h_t = predict_f(
                    dataset, np.arange(len(values)), args.grid_size, dk,
                    lambda_reg=args.lambda_reg, learning_rate=args.learning_rate,
                    n_iterations_GD=args.n_iterations_GD, lr_decay=args.lr_decay,
                    filename=log_fn, arm_features=values
                )[0]

                # Full posterior covariance
                cov_D = update_cov_D_cat(dataset, dk, args.alpha_gp,
                                          np.arange(len(values)), values)

                # Anchor-independent: use x0 = 0
                x0 = 0
                cov_scaling = get_cov_scaling(t, args)
                cov_strip = get_cov_strip(cov_D, x0, values)

                # Double Thompson sampling
                samples = np.random.multivariate_normal(
                    h_t[:, x0], cov_scaling * cov_strip, size=2
                )
                i = np.argmax(samples[0])
                j = np.argmax(samples[1])

            if file:
                file.write(f"Iteration {t}, Pair indices: ({i}, {j})\n")

            # Generate preference feedback
            p = sigmoid(f[i, j])
            y = np.random.binomial(1, p)
            dataset = np.vstack((dataset, [i, j, y]))

            regret = compute_regret_cat(i, j, f, Reward_function)
            regret_list.append(regret)

            if t % 5 == 0:
                print(f"t={t:4d}  elapsed={time.time()-fn_start:.1f}s  "
                      f"regret={regret:.4f}")
    finally:
        if file:
            file.close()

    return regret_list


# ---------------------------------------------------------------------------
#  POP-BO for the Catalyst setting
# ---------------------------------------------------------------------------

def _get_UCB_argmax_index(index_t_prime, h_estimate, sigma_D, beta):
    """Return index maximising UCB(x, x_{t'})."""
    return np.argmax(h_estimate[:, index_t_prime] +
                     beta * sigma_D[:, index_t_prime])


def POP_BO_cat(args, values, Reward_function, f, timestamp, arm_features):
    """POP-BO (Xu et al., 2024) adapted for multi-dimensional actions."""
    fn_start = time.time()
    dataset = np.empty((0, 3))
    regret_list = []

    base_kernel = (gpytorch.kernels.MaternKernel(nu=args.smoothness)
                   if args.kernel == "Matern"
                   else gpytorch.kernels.RBFKernel())
    dk = DuelingKernel2(kernel=base_kernel, len_scale=args.length_scale).to(device)

    log_fn = (f"POP_BO_{args.kernel}_{args.learning_rate}_"
              f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_fn, "a") if log_fn else None

    index_t = np.random.randint(0, args.grid_size)

    try:
        for t in range(args.n_iterations):
            if len(dataset) == 0:
                fv = np.zeros((args.grid_size, args.grid_size))
                sd = np.zeros((args.grid_size, args.grid_size))
            else:
                from clean_test_algos_yelp_full_gpu_optim_fix import update_sigma_D
                fv, _ = predict_f(
                    dataset, values, args.grid_size, dk,
                    lambda_reg=args.lambda_reg, learning_rate=args.learning_rate,
                    n_iterations_GD=args.n_iterations_GD, lr_decay=args.lr_decay,
                    filename=log_fn, arm_features=arm_features
                )
                sd = update_sigma_D(dataset, dk, args.alpha_gp, values,
                                    arm_features).numpy()

            index_t_prime = index_t
            index_t = _get_UCB_argmax_index(index_t_prime, fv, sd, args.beta)

            if file:
                file.write(f"Iteration {t}, Pair: ({index_t}, {index_t_prime})\n")

            p = sigmoid(f[index_t, index_t_prime])
            y = np.random.binomial(1, p)
            dataset = np.vstack((dataset, [index_t, index_t_prime, y]))

            regret = compute_regret_cat(index_t, index_t_prime, f, Reward_function)
            regret_list.append(regret)

            if t % 20 == 0:
                print(f"t={t:4d}  elapsed={time.time()-fn_start:.1f}s  "
                      f"regret={regret:.4f}")
    finally:
        if file:
            file.close()

    return regret_list
