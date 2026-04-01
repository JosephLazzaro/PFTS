"""
Core utilities and baseline algorithms for 1D synthetic experiments.

Includes:
  - Utility function generation (Ackley, RKHS)
  - Dueling kernel definition (sklearn-compatible)
  - Preference prediction via regularised logistic regression
  - Uncertainty estimation via GP posterior standard deviation
  - Baseline algorithms: MaxMinLCB (Pásztor et al., 2024) and MR-LPF (Kayal et al., 2025)
  - Regret computation helpers
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, Matern
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
import datetime
import os
import argparse


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def scale_to_range(data, new_min, new_max):
    """Linearly rescale *data* to [new_min, new_max]."""
    old_min, old_max = np.min(data), np.max(data)
    return new_min + (data - old_min) * (new_max - new_min) / (old_max - old_min)


def sigmoid(z):
    """Element-wise logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
#  Utility-function generators
# ---------------------------------------------------------------------------

def ackley_function_1d(x, a=20, b=0.2, c=2 * np.pi):
    """Negated 1-D Ackley function (we maximise utility)."""
    term1 = -a * np.exp(-b * np.sqrt(x ** 2))
    term2 = -np.exp(np.cos(c * x))
    return -1.0 * (a + np.exp(1) + term1 + term2)


def generate_ackley(grid_size=100):
    """
    Generate a 1-D Ackley utility landscape and the corresponding
    pairwise-difference matrix used for preference feedback.

    Returns
    -------
    values : ndarray, shape (grid_size,)
        Evenly spaced actions in [-5, 5].
    Reward_function : ndarray, shape (grid_size,)
        Utility at each action (negated Ackley).
    f : ndarray, shape (grid_size, grid_size)
        Pairwise difference matrix rescaled to [-3, 3].
    """
    values = np.linspace(-5, 5, grid_size)
    Reward_function = ackley_function_1d(values)

    # Build pairwise difference matrix f[i, j] = R[i] - R[j]
    f = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            f[i, j] = Reward_function[i] - Reward_function[j]
    f = scale_to_range(f, -3, 3)

    return values, Reward_function, f


def generate_preference_RKHS(grid_size=100, alpha_gp=0.05, length_scale=0.1,
                              n_samples=10, base_kernel=None, smoothness=1.5,
                              random_state=None):
    """
    Generate a utility function sampled from an RKHS (GP posterior mean)
    and the corresponding pairwise-difference matrix.

    Returns
    -------
    values, Reward_function, f  (same semantics as generate_ackley)
    """
    values = np.linspace(0, 1, grid_size)

    if base_kernel == "Matern":
        kernel = Matern(length_scale=length_scale, nu=smoothness,
                        length_scale_bounds="fixed")
    elif base_kernel == "RBF":
        kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")
    else:
        raise ValueError(f"Unknown kernel: {base_kernel}")

    # Fixed sample points used to define the RKHS function
    X_gp = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = np.array([-0.5, 0, 0.5, 1, 0.7, 0.2, 0.2, 0.5, 0.95, 0.7])

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                    alpha=alpha_gp, random_state=random_state)
    gpr.fit(X_gp, y)

    X_full = values.reshape(-1, 1)
    Reward_function = gpr.predict(X_full)

    # Pairwise difference matrix
    f = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            f[i, j] = Reward_function[i] - Reward_function[j]
    f = f * 5

    return values, Reward_function, f


# ---------------------------------------------------------------------------
#  Dueling kernel (sklearn-compatible)
# ---------------------------------------------------------------------------

class DuelingKernel2(Kernel):
    """
    Dueling (difference) kernel induced by a base kernel k:

        k^Δ((x,x'), (u,u')) = k(x,u) + k(x',u') - k(x,u') - k(x',u)

    Compatible with sklearn's GaussianProcessRegressor.
    """

    def __init__(self, base_kernel=None, length_scale=0.1, smoothness=1.5):
        self.length_scale = length_scale
        self.length_scale_bounds = "fixed"
        self.base_kernel = base_kernel
        self.smoothness = smoothness

        if self.base_kernel == "Matern":
            self.kernel = Matern(length_scale=length_scale, nu=smoothness)
        elif self.base_kernel == "RBF":
            self.kernel = RBF(length_scale=length_scale)
        else:
            raise ValueError(f"Unknown base_kernel: {base_kernel}")

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.dueling_kernel(X, Y, eval_gradient=eval_gradient)

    def dueling_kernel(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Split each row into (x, x') components
        X1, X2 = X[:, 0:1], X[:, 1:2]
        Y1, Y2 = Y[:, 0:1], Y[:, 1:2]

        K_x1_x1p = self.kernel(X1, Y1)
        K_x2_x2p = self.kernel(X2, Y2)
        K_x1_x2p = self.kernel(X1, Y2)
        K_x2_x1p = self.kernel(X2, Y1)

        K = K_x1_x1p + K_x2_x2p - K_x1_x2p - K_x2_x1p

        # Small jitter for numerical stability
        if X.shape == Y.shape:
            K += np.eye(X.shape[0]) * 1e-6

        if eval_gradient:
            _, g1 = self.kernel(X1, Y1, eval_gradient=True)
            _, g2 = self.kernel(X2, Y2, eval_gradient=True)
            _, g3 = self.kernel(X1, Y2, eval_gradient=True)
            _, g4 = self.kernel(X2, Y1, eval_gradient=True)
            return K, g1 + g2 - g3 - g4
        return K

    def diag(self, X):
        return np.diag(self.__call__(X))

    def is_stationary(self):
        return True


# ---------------------------------------------------------------------------
#  Preference prediction (regularised logistic regression in the RKHS)
# ---------------------------------------------------------------------------

def loss_function(alpha, K, y, lambda_reg):
    """Regularised cross-entropy loss in parameter space."""
    alpha = alpha.reshape(-1, 1)
    K_alpha = K @ alpha
    sig = np.clip(sigmoid(K_alpha), 1e-10, 1 - 1e-10)
    y_col = y.reshape(-1, 1)
    N = y.shape[0]
    loss = (1.0 / N) * np.sum(-y_col * np.log(sig) - (1 - y_col) * np.log(1 - sig))
    loss += (lambda_reg / 2.0) * np.sum(alpha ** 2)
    return loss


def gradient_loss_function(alpha, K, y, lambda_reg):
    """Gradient of the regularised cross-entropy loss."""
    alpha = alpha.reshape(-1, 1)
    K_alpha = K @ alpha
    sig = np.clip(sigmoid(K_alpha), 1e-10, 1 - 1e-10)
    N = y.shape[0]
    grad = (1.0 / N) * K.T @ (sig - y.reshape(-1, 1)) + lambda_reg * alpha
    return grad.flatten()


def f_t(x, X_train, optimal_alpha, kernel):
    """Evaluate the learned difference function at a single pair x."""
    k_t = kernel(X_train, np.array([x])).flatten()
    return np.dot(optimal_alpha, k_t)


def predict_f(dataset, values, grid_size, kernel, lambda_reg=0.05,
              learning_rate=0.1, n_iterations_GD=200000, lr_decay=0,
              filename=None):
    """
    Fit the regularised logistic model on *dataset* and evaluate the
    estimated difference function h_t on the full grid of action pairs.

    Returns
    -------
    f_values : ndarray, shape (grid_size, grid_size)
        Estimated h_t(x_i, x_j) for all pairs.
    current_loss : float
        Final training loss.
    """
    file = open(filename, "a") if filename else None

    X_train = dataset[:, :2]
    y_train = dataset[:, 2]
    K = kernel(X_train)

    alpha = np.zeros(X_train.shape[0])
    initial_lr = learning_rate
    tolerance = 1e-6
    previous_loss = float('inf')

    # Gradient descent
    for iteration in range(n_iterations_GD):
        lr = initial_lr / (1 + 0.1 * iteration) if lr_decay != 0 else initial_lr
        grad = gradient_loss_function(alpha, K, y_train, lambda_reg)
        alpha -= lr * grad

        current_loss = loss_function(alpha, K, y_train, lambda_reg)
        if abs(previous_loss - current_loss) < tolerance or np.linalg.norm(grad) < tolerance:
            break
        previous_loss = current_loss

    # Evaluate on the full grid
    f_values = np.zeros((grid_size, grid_size))
    for i, x1 in enumerate(values):
        for j, x2 in enumerate(values):
            f_values[i, j] = f_t([x1, x2], X_train, alpha, kernel)

    if file is not None:
        file.write(f"f_values: {f_values}\n")
        file.close()

    return f_values, current_loss


# ---------------------------------------------------------------------------
#  Uncertainty estimation
# ---------------------------------------------------------------------------

def update_sigma_D(dataset, kernel, alpha_gp, values):
    """
    Compute the GP posterior standard deviation for every action pair
    on the grid, using sklearn's GaussianProcessRegressor with the
    dueling kernel.
    """
    X = dataset[:, :2]
    y = dataset[:, 2]
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha_gp)
    gp.fit(X, y)
    X_grid = np.array(list(product(values, repeat=2)))
    _, y_std = gp.predict(X_grid, return_std=True)
    return y_std.reshape(len(values), len(values))


# ---------------------------------------------------------------------------
#  Active-set maintenance helpers (used by MaxMinLCB and MR-LPF)
# ---------------------------------------------------------------------------

def update_M_t(f_values, sigma_D, beta, values, threshold=0.5):
    """Return actions that cannot yet be ruled sub-optimal."""
    M_t = []
    for i, x in enumerate(values):
        if all(sigmoid(f_values[i, j]) + beta * sigma_D[i, j] >= threshold
               for j in range(len(values))):
            M_t.append(x)
    return M_t


def update_M_t_previous(f_values, sigma_D, beta, values, Mt_prev, threshold=0.5):
    """Refine the active set from the previous round."""
    prev_indices = [np.where(values == x)[0][0] for x in Mt_prev]
    M_t = []
    for i in prev_indices:
        if all(sigmoid(f_values[i, j]) + beta * sigma_D[i, j] >= threshold
               for j in prev_indices):
            M_t.append(values[i])
    return M_t


def select_pair(values, f_values, sigma_D, beta, M_t):
    """
    Stackelberg pair selection for MaxMinLCB:
    leader maximises LCB, follower minimises LCB.
    """
    best_pair = None
    max_value = -np.inf

    for x1 in M_t:
        i = np.searchsorted(values, x1)
        best_x_prime, min_follower = None, np.inf
        for x2 in M_t:
            j = np.searchsorted(values, x2)
            val = sigmoid(f_values[i, j]) - beta * sigma_D[i, j]
            if val < min_follower:
                min_follower = val
                best_x_prime = [x2]

        if best_x_prime is not None:
            j = np.searchsorted(values, best_x_prime[0])
            leader_val = sigmoid(f_values[i, j]) - beta * sigma_D[i, j]
            if leader_val > max_value:
                max_value = leader_val
                best_pair = ([x1], best_x_prime)

    return best_pair


# ---------------------------------------------------------------------------
#  Regret computation
# ---------------------------------------------------------------------------

def find_most_preferred_action(Reward_function, values):
    """Return the action with the highest utility."""
    return values[np.argmax(Reward_function)]


def find_x_star_predicted(f_values, values):
    """Return the Copeland winner under the estimated preference matrix."""
    best_x, max_wins = None, -1
    for i in range(len(values)):
        wins = sum(1 for j in range(len(values)) if i != j and sigmoid(f_values[i, j]) > 0.5)
        if wins > max_wins:
            max_wins = wins
            best_x = values[i]
    return best_x


def compute_regret(x_star, x, x_prime, f, values):
    """
    Instantaneous dueling regret:
        r_t = (μ(f(x*, x)) + μ(f(x*, x')) - 1) / 2
    """
    i_star = np.searchsorted(values, x_star)
    i = np.searchsorted(values, x)
    j = np.searchsorted(values, x_prime)
    return (sigmoid(f[i_star, i]) + sigmoid(f[i_star, j]) - 1.0) / 2.0


# ---------------------------------------------------------------------------
#  Baseline algorithm: MaxMinLCB (Pásztor et al., 2024)
# ---------------------------------------------------------------------------

def Max_Min_LCB(args, values, Reward_function, f, timestamp):
    """
    MaxMinLCB algorithm for BOHF.

    Selects pairs via a Stackelberg game on the lower confidence bound.
    """
    dataset = np.empty((0, 3))
    M_t = values.tolist()
    regret_list = []

    dueling_kernel = DuelingKernel2(base_kernel=args.kernel,
                                     length_scale=args.length_scale,
                                     smoothness=args.smoothness)
    log_filename = (f"MaxMinLCB_{args.kernel}_{args.learning_rate}_"
                    f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_filename, "a") if log_filename else None

    try:
        for t in range(args.n_iterations):
            if len(dataset) == 0:
                f_values = np.zeros((args.grid_size, args.grid_size))
                sigma_D = np.zeros((args.grid_size, args.grid_size))
            else:
                f_values, _ = predict_f(dataset, values, args.grid_size,
                                        dueling_kernel, lambda_reg=args.lambda_reg,
                                        learning_rate=args.learning_rate,
                                        n_iterations_GD=args.n_iterations_GD,
                                        lr_decay=args.lr_decay, filename=log_filename)
                sigma_D = update_sigma_D(dataset, dueling_kernel,
                                         alpha_gp=args.alpha_gp, values=values)
                M_t = update_M_t(f_values, sigma_D, beta=args.beta, values=values)

            pair = select_pair(values, f_values, sigma_D, beta=args.beta, M_t=M_t)
            if file:
                file.write(f"Iteration {t}, Selected Pair: {pair}\n")

            if pair:
                x, x_prime = pair
                i = np.searchsorted(values, x[0])
                j = np.searchsorted(values, x_prime[0])
                p = sigmoid(f[i, j])
                y = np.random.binomial(1, p)
                dataset = np.vstack((dataset, [x[0], x_prime[0], y]))
                x_star = find_most_preferred_action(Reward_function, values)
                regret_list.append(compute_regret(x_star, x[0], x_prime[0], f, values))
    finally:
        if file:
            file.close()

    return regret_list, M_t


# ---------------------------------------------------------------------------
#  Baseline algorithm: MR-LPF (Kayal et al., 2025)
# ---------------------------------------------------------------------------

def BOHF(args, values, Reward_function, f, timestamp):
    """
    MR-LPF (Multi-Round Log-Partition Function) algorithm for BOHF.

    Uses batched exploration with geometrically growing batch sizes
    and active-set refinement between rounds.
    """
    M_t = values.tolist()
    regret_list = []
    dueling_kernel = DuelingKernel2(base_kernel=args.kernel,
                                     length_scale=args.length_scale,
                                     smoothness=args.smoothness)
    N = 1
    t = 0
    T = args.n_iterations

    log_filename = (f"BOHF_{args.kernel}_{args.learning_rate}_"
                    f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_filename, "a") if log_filename else None

    try:
        while True:
            N = int(np.ceil(np.sqrt(T * N)))
            dataset_round = np.empty((0, 3))

            # Optionally seed with self-comparisons
            if args.append_identical_pairs:
                for x in values:
                    i = np.where(values == x)[0][0]
                    for _ in range(args.repeat_identical_pairs):
                        p = sigmoid(f[i, i])
                        y = np.random.binomial(1, p)
                        dataset_round = np.vstack((dataset_round, [x, x, y]))

            for n in range(N):
                if len(dataset_round) == 0:
                    sigma_D = np.zeros((args.grid_size, args.grid_size))
                else:
                    sigma_D = update_sigma_D(dataset_round, dueling_kernel,
                                             alpha_gp=args.alpha_gp, values=values)

                # Pick pair with maximum uncertainty within active set
                M_t_idx = [np.where(values == x)[0][0] for x in M_t]
                sigma_sub = sigma_D[np.ix_(M_t_idx, M_t_idx)]
                mi, mj = np.unravel_index(np.argmax(sigma_sub), sigma_sub.shape)
                i, j = M_t_idx[mi], M_t_idx[mj]
                pair = (values[i], values[j])

                if t % 20 == 0:
                    print(f"Iteration {t}, Selected Pair: {pair}, |M_t|={len(M_t)}")

                x, x_prime = pair
                p = sigmoid(f[i, j])
                y = np.random.binomial(1, p)
                dataset_round = np.vstack((dataset_round, [x, x_prime, y]))

                x_star = find_most_preferred_action(Reward_function, values)
                regret_list.append(compute_regret(x_star, x, x_prime, f, values))

                t += 1
                if t >= T:
                    return regret_list, M_t

            # Refine active set between rounds
            sigma_D = update_sigma_D(dataset_round, dueling_kernel,
                                     alpha_gp=args.alpha_gp, values=values)
            f_values, _ = predict_f(dataset_round, values, args.grid_size,
                                    dueling_kernel, lambda_reg=args.lambda_reg,
                                    learning_rate=args.learning_rate,
                                    n_iterations_GD=args.n_iterations_GD,
                                    lr_decay=args.lr_decay, filename=log_filename)
            M_t = update_M_t_previous(f_values, sigma_D, beta=args.beta,
                                      values=values, Mt_prev=M_t)
    finally:
        if file:
            file.close()

    return regret_list, M_t


# ---------------------------------------------------------------------------
#  CLI argument parser (for standalone use)
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run dueling-bandits experiments.")
    parser.add_argument("--alpha_gp", type=float, default=0.05)
    parser.add_argument("--length_scale", type=float, default=0.1)
    parser.add_argument("--grid_size", type=int, default=40)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_iterations_GD", type=int, default=200000)
    parser.add_argument("--n_iterations", type=int, default=300)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--algo", type=str, default="BOHF")
    parser.add_argument("--lr_decay", type=int, default=0)
    parser.add_argument("--kernel", type=str, default="Matern")
    parser.add_argument("--smoothness", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preference_function", type=str, default="RKHS")
    parser.add_argument("--enable_logging", type=int, default=0)
    parser.add_argument("--append_identical_pairs", type=int, default=0)
    parser.add_argument("--repeat_identical_pairs", type=int, default=1)
    return parser.parse_args()
