"""
Core utilities and baseline algorithms for the Catalyst (OCx24) experiments.

Uses PyTorch/GPyTorch/BoTorch for GPU-accelerated GP inference in higher
dimensions (3-D catalyst compositions: Ag, Au, Zn).

Includes:
  - Dueling kernel (GPyTorch-compatible)
  - Preference prediction via regularised logistic regression (GPU)
  - Uncertainty estimation via BoTorch's SingleTaskGP
  - Baseline algorithms: MaxMinLCB and MR-LPF
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import datetime
import os
import argparse

import torch
import gpytorch
from botorch.models import SingleTaskGP

# Auto-select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[clean_test_algos] Using device: {device}")


# ---------------------------------------------------------------------------
#  Basic helpers
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Element-wise logistic sigmoid (NumPy)."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_torch(x):
    """Element-wise logistic sigmoid (PyTorch)."""
    return torch.sigmoid(x)


# ---------------------------------------------------------------------------
#  Dueling kernel (GPyTorch-compatible)
# ---------------------------------------------------------------------------

class DuelingKernel2(gpytorch.kernels.Kernel):
    """
    Dueling (difference) kernel for pairs of d-dimensional actions.

    Given a base kernel k on R^d, the dueling kernel on R^{2d} is:
        k^Δ((x,x'), (u,u')) = k(x,u) + k(x',u') - k(x,u') - k(x',u)

    Input tensors are assumed to be [n, 2d] where the first d columns
    represent the first action and the last d columns the second.
    """

    def __init__(self, kernel, len_scale=0.1):
        super().__init__(has_lengthscale=True)
        self.kernel = kernel
        self.kernel.lengthscale = torch.tensor([len_scale], device=device)
        self.kernel.raw_lengthscale.requires_grad = False

    def forward(self, X, Y=None, diag=False, last_dim_is_batch=False,
                chunk_size=10000):
        if Y is None:
            Y = X

        n_features = X.shape[1] // 2
        assert n_features * 2 == X.shape[1], "Input must have even dimensionality."

        K = torch.zeros(X.shape[0], Y.shape[0], dtype=torch.float32, device=device)

        # Process in chunks to manage GPU memory
        for i in range(0, X.shape[0], chunk_size):
            for j in range(0, Y.shape[0], chunk_size):
                Xc = X[i:i + chunk_size]
                Yc = Y[j:j + chunk_size]
                X1, X2 = Xc[:, :n_features], Xc[:, n_features:]
                Y1, Y2 = Yc[:, :n_features], Yc[:, n_features:]

                K[i:i + chunk_size, j:j + chunk_size] = (
                    self.kernel(X1, Y1).to_dense()
                    + self.kernel(X2, Y2).to_dense()
                    - self.kernel(X1, Y2).to_dense()
                    - self.kernel(X2, Y1).to_dense()
                )

        # Jitter for numerical stability
        if X.shape == Y.shape:
            K += torch.eye(X.shape[0], dtype=torch.float32, device=device) * 1e-6

        return K.diag() if diag else K


# ---------------------------------------------------------------------------
#  Preference prediction (GPU-accelerated logistic regression)
# ---------------------------------------------------------------------------

def _loss(alpha, K, y, lambda_reg):
    """Regularised cross-entropy loss (PyTorch)."""
    alpha = alpha.view(-1, 1)
    sig = torch.clamp(sigmoid_torch(K @ alpha), 1e-10, 1 - 1e-10)
    y_col = y.view(-1, 1)
    N = y.size(0)
    return (1.0 / N) * (-y_col * torch.log(sig) - (1 - y_col) * torch.log(1 - sig)).sum() \
           + (lambda_reg / 2.0) * (alpha ** 2).sum()


def _gradient(alpha, K, y, lambda_reg):
    """Gradient of the regularised cross-entropy loss (PyTorch)."""
    alpha = alpha.view(-1, 1)
    sig = torch.clamp(sigmoid_torch(K @ alpha), 1e-10, 1 - 1e-10)
    N = y.size(0)
    return ((1.0 / N) * K.T @ (sig - y.view(-1, 1)) + lambda_reg * alpha).view(-1)


def _f_t_batch(X_train_features, optimal_alpha, kernel, arm_features,
               grid_points):
    """Evaluate the learned h_t on a batch of grid points."""
    grid_np = np.array([[arm_features[int(idx)] for idx in pt]
                        for pt in grid_points])
    grid_feat = torch.tensor(grid_np, dtype=torch.float32, device=device)\
                      .view(grid_np.shape[0], -1)
    K = kernel(X_train_features, grid_feat).evaluate()
    return torch.matmul(optimal_alpha, K)


def predict_f(dataset, values, grid_size, kernel, lambda_reg=0.05,
              learning_rate=0.1, n_iterations_GD=200000, lr_decay=0,
              filename=None, arm_features=None):
    """
    Fit the preference model and evaluate h_t on the full action-pair grid.

    Parameters
    ----------
    arm_features : ndarray, shape (n_actions, d)
        Feature vectors for each action (catalyst compositions).

    Returns
    -------
    f_values : ndarray, shape (grid_size, grid_size)
    current_loss : float
    """
    file = open(filename, "a") if filename else None

    X_train = dataset[:, :2]
    y_train = dataset[:, 2]

    # Map action indices to feature vectors
    X_np = np.array([[arm_features[int(x)] for x in row] for row in X_train])
    X_feat = torch.tensor(X_np, dtype=torch.float32, device=device)\
                   .view(X_train.shape[0], -1)
    K = kernel(X_feat).to_dense().to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    # Gradient descent
    alpha = torch.zeros(X_train.shape[0], dtype=torch.float32, device=device)
    init_lr = learning_rate
    prev_loss = float("inf")

    for it in range(n_iterations_GD):
        lr = init_lr / (1 + 0.1 * it) if lr_decay != 0 else init_lr
        grad = _gradient(alpha, K, y_t, lambda_reg)
        alpha -= lr * grad
        cur_loss = _loss(alpha, K, y_t, lambda_reg).item()
        if abs(prev_loss - cur_loss) < 1e-6 or grad.norm().item() < 1e-6:
            break
        prev_loss = cur_loss

    # Evaluate on grid
    grid_pts = [(x1, x2) for x1 in values for x2 in values]
    f_vals = _f_t_batch(X_feat, alpha, kernel, arm_features, grid_pts)
    f_values = f_vals.view(grid_size, grid_size).cpu().numpy()

    if file:
        file.write(f"f_values: {f_values}\n")
        file.close()

    return f_values, cur_loss


# ---------------------------------------------------------------------------
#  Uncertainty estimation
# ---------------------------------------------------------------------------

def update_sigma_D(dataset, kernel, alpha_gp, values, arm_features):
    """
    GP posterior standard deviation for every action pair, computed
    via BoTorch's SingleTaskGP with the dueling kernel.
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
    batch_size = 10000
    all_std = []

    for i in range(0, X_grid.shape[0], batch_size):
        batch = X_grid[i:i + batch_size]
        batch_feat = torch.stack([af[int(x)] for x in batch.flatten()])\
                          .view(batch.shape[0], -1)
        with torch.no_grad():
            post = model(batch_feat)
            all_std.append(torch.sqrt(post.variance).cpu())
        del batch, batch_feat, post
        torch.cuda.empty_cache()

    return torch.cat(all_std).reshape(len(values), len(values))


# ---------------------------------------------------------------------------
#  Active-set helpers (shared with 1-D code)
# ---------------------------------------------------------------------------

def update_M_t(f_values, sigma_D, beta, values, threshold=0.5):
    M_t = []
    for i, x in enumerate(values):
        if all(sigmoid(f_values[i, j]) + beta * sigma_D[i, j] >= threshold
               for j in range(len(values))):
            M_t.append(x)
    return M_t


def update_M_t_previous(f_values, sigma_D, beta, values, Mt_prev, threshold=0.5):
    prev_idx = [np.where(values == x)[0][0] for x in Mt_prev]
    return [values[i] for i in prev_idx
            if all(sigmoid(f_values[i, j]) + beta * sigma_D[i, j] >= threshold
                   for j in prev_idx)]


def select_pair(values, f_values, sigma_D, beta, M_t):
    best_pair, max_val = None, -np.inf
    for x1 in M_t:
        i = np.searchsorted(values, x1)
        best_xp, min_f = None, np.inf
        for x2 in M_t:
            j = np.searchsorted(values, x2)
            v = sigmoid(f_values[i, j]) - beta * sigma_D[i, j]
            if v < min_f:
                min_f, best_xp = v, [x2]
        if best_xp is not None:
            j = np.searchsorted(values, best_xp[0])
            lv = sigmoid(f_values[i, j]) - beta * sigma_D[i, j]
            if lv > max_val:
                max_val, best_pair = lv, ([x1], best_xp)
    return best_pair


def find_most_preferred_action(Reward_function, values):
    return values[np.argmax(Reward_function)]


def compute_regret(x_star, x, x_prime, f, values):
    i_star = np.searchsorted(values, x_star)
    i = np.searchsorted(values, x)
    j = np.searchsorted(values, x_prime)
    return (sigmoid(f[i_star, i]) + sigmoid(f[i_star, j]) - 1.0) / 2.0


# ---------------------------------------------------------------------------
#  MaxMinLCB (Pásztor et al., 2024) — high-dimensional version
# ---------------------------------------------------------------------------

def Max_Min_LCB(args, values, Reward_function, f, timestamp, arm_features):
    dataset = np.empty((0, 3))
    M_t = values.tolist()
    regret_list = []

    base_kernel = (gpytorch.kernels.MaternKernel(nu=args.smoothness)
                   if args.kernel == "Matern"
                   else gpytorch.kernels.RBFKernel())
    dk = DuelingKernel2(kernel=base_kernel, len_scale=args.length_scale).to(device)

    log_fn = (f"MaxMinLCB_{args.kernel}_{args.learning_rate}_"
              f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_fn, "a") if log_fn else None

    try:
        for t in range(args.n_iterations):
            if len(dataset) == 0:
                fv = np.zeros((args.grid_size, args.grid_size))
                sd = np.zeros((args.grid_size, args.grid_size))
            else:
                fv, _ = predict_f(dataset, values, args.grid_size, dk,
                                  lambda_reg=args.lambda_reg,
                                  learning_rate=args.learning_rate,
                                  n_iterations_GD=args.n_iterations_GD,
                                  lr_decay=args.lr_decay, filename=log_fn,
                                  arm_features=arm_features)
                sd = update_sigma_D(dataset, dk, args.alpha_gp, values,
                                    arm_features)
                M_t = update_M_t(fv, sd, args.beta, values)

            pair = select_pair(values, fv, sd, args.beta, M_t)
            if file:
                file.write(f"Iteration {t}, Pair: {pair}\n")

            if pair:
                x, xp = pair
                i = np.searchsorted(values, x[0])
                j = np.searchsorted(values, xp[0])
                p = sigmoid(f[i, j])
                y = np.random.binomial(1, p)
                dataset = np.vstack((dataset, [x[0], xp[0], y])).astype(np.float32)
                x_star = find_most_preferred_action(Reward_function, values)
                regret_list.append(compute_regret(x_star, x[0], xp[0], f, values))
    finally:
        if file:
            file.close()

    return regret_list, M_t


# ---------------------------------------------------------------------------
#  MR-LPF (Kayal et al., 2025) — high-dimensional version
# ---------------------------------------------------------------------------

def BOHF(args, values, Reward_function, f, timestamp, arm_features):
    M_t = values.tolist()
    regret_list = []

    base_kernel = (gpytorch.kernels.MaternKernel(nu=args.smoothness)
                   if args.kernel == "Matern"
                   else gpytorch.kernels.RBFKernel())
    dk = DuelingKernel2(kernel=base_kernel, len_scale=args.length_scale).to(device)

    N, t, T = 1, 0, args.n_iterations

    log_fn = (f"BOHF_{args.kernel}_{args.learning_rate}_"
              f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_fn, "a") if log_fn else None

    try:
        while True:
            N = int(np.ceil(np.sqrt(T * N)))
            ds_round = np.empty((0, 3))

            if args.append_identical_pairs:
                for x in values:
                    i = np.where(values == x)[0][0]
                    for _ in range(args.repeat_identical_pairs):
                        p = sigmoid(f[i, i])
                        y = np.random.binomial(1, p)
                        ds_round = np.vstack((ds_round, [x, x, y]))

            for n in range(N):
                if len(ds_round) == 0:
                    sd = np.zeros((args.grid_size, args.grid_size))
                else:
                    sd = update_sigma_D(ds_round, dk, args.alpha_gp, values,
                                        arm_features)

                M_idx = [np.where(values == x)[0][0] for x in M_t]
                sub = sd[np.ix_(M_idx, M_idx)]
                mi, mj = np.unravel_index(np.argmax(sub), sub.shape)
                i, j = M_idx[mi], M_idx[mj]
                x, xp = values[i], values[j]

                p = sigmoid(f[i, j])
                y = np.random.binomial(1, p)
                ds_round = np.vstack((ds_round, [x, xp, y]))

                x_star = find_most_preferred_action(Reward_function, values)
                regret_list.append(compute_regret(x_star, x, xp, f, values))

                t += 1
                if t >= T:
                    return regret_list, M_t

            sd = update_sigma_D(ds_round, dk, args.alpha_gp, values, arm_features)
            fv, _ = predict_f(ds_round, values, args.grid_size, dk,
                              lambda_reg=args.lambda_reg,
                              learning_rate=args.learning_rate,
                              n_iterations_GD=args.n_iterations_GD,
                              lr_decay=args.lr_decay, filename=log_fn,
                              arm_features=arm_features)
            M_t = update_M_t_previous(fv, sd, args.beta, values, M_t)
    finally:
        if file:
            file.close()

    return regret_list, M_t
