"""
Preferential Feedback Thompson Sampling (PF-TS) for 1-D synthetic experiments.

Implements Algorithm 1 from the paper:
  "A Finite Time Analysis of Thompson Sampling for Bayesian Optimization
   with Preferential Feedback" (Lazzaro et al., AISTATS 2026).

Key idea: draw two independent posterior samples of the pairwise-difference
function h, maximise each with respect to a fixed anchor (anchor-independent
by Proposition 1), and play the resulting pair.
"""

import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from itertools import product

from clean_test_algos_synthetic_cpu import (
    DuelingKernel2, predict_f, sigmoid, find_most_preferred_action, compute_regret
)


# ---------------------------------------------------------------------------
#  Covariance helpers
# ---------------------------------------------------------------------------

def get_cov_strip(full_cov, x0, values):
    """
    Extract the covariance sub-matrix for all actions paired with
    a fixed anchor index *x0* from the full (grid_size^2 × grid_size^2)
    covariance matrix.

    Returns
    -------
    cov_strip : ndarray, shape (len(values), len(values))
        cov_strip[i, j] = Cov(h(x_i, x0), h(x_j, x0))
    """
    n = len(values)
    cov_strip = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_strip[i, j] = full_cov[i * n + x0, j * n + x0]
    return cov_strip


def get_cov_strip_directly(dataset, kernel, alpha_gp, values, x0):
    """
    Fit a GP on the dueling kernel, compute the full posterior covariance
    over the action-pair grid, and extract the strip for anchor *x0*.
    """
    X = dataset[:, :2]
    y = dataset[:, 2]
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=alpha_gp)
    gp.fit(X, y)

    X_grid = np.array(list(product(values, repeat=2)))
    _, full_cov = gp.predict(X_grid, return_cov=True)

    return get_cov_strip(full_cov, x0, values)


def get_cov_scaling(t, args, delta=0.05):
    """
    Exploration scale v_t^2 for Thompson sampling.

    Uses a practical approximation: v_t = sqrt(2 * (t + 1 + log(2/δ))).
    This is a minor simplification of the theoretical β_t(δ) from Eq. (12),
    replacing the information gain with its upper bound t.
    """
    info_gain = 1.0 * t
    return np.sqrt(2.0 * (info_gain + 1 + np.log(2.0 / delta)))


# ---------------------------------------------------------------------------
#  PF-TS algorithm
# ---------------------------------------------------------------------------

def TS(args, values, Reward_function, f, timestamp, seed=None):
    """
    Preferential Feedback Thompson Sampling (PF-TS) — Algorithm 1.

    Parameters
    ----------
    args : namespace
        Hyperparameters (grid_size, kernel, length_scale, smoothness,
        lambda_reg, learning_rate, n_iterations_GD, lr_decay, alpha_gp,
        n_iterations, enable_logging).
    values : ndarray
        Discrete action set.
    Reward_function : ndarray
        Utility at each action (used only for regret computation).
    f : ndarray
        Pairwise difference matrix f[i,j] = utility(i) - utility(j).
    timestamp : str
        Identifier for log files.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    regret_list : list of float
        Instantaneous regret at each round.
    """
    if seed is not None:
        np.random.seed(seed)

    fn_start_time = time.time()

    dataset = np.empty((0, 3))
    regret_list = []

    dueling_kernel = DuelingKernel2(base_kernel=args.kernel,
                                     length_scale=args.length_scale,
                                     smoothness=args.smoothness)

    log_filename = (f"TS_{args.kernel}_{args.learning_rate}_"
                    f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_filename, "a") if log_filename else None

    x_star = find_most_preferred_action(Reward_function, values)

    try:
        for t in range(args.n_iterations):
            if len(dataset) == 0:
                # Random initialisation when no data is available
                i, j = np.random.randint(len(values), size=2)
            else:
                # --- Fit the preference model ---
                h_t_values = predict_f(
                    dataset, values, args.grid_size, dueling_kernel,
                    lambda_reg=args.lambda_reg, learning_rate=args.learning_rate,
                    n_iterations_GD=args.n_iterations_GD, lr_decay=args.lr_decay,
                    filename=log_filename
                )[0]

                # Anchor-independent: use x0 = 0 (Proposition 1)
                x0 = 0

                # Exploration scale
                cov_scaling = get_cov_scaling(t, args)

                # Posterior covariance strip for anchor x0
                cov_strip = get_cov_strip_directly(
                    dataset, dueling_kernel, args.alpha_gp, values, x0
                )

                # Draw two independent posterior samples (Double-TS)
                samples = np.random.multivariate_normal(
                    h_t_values[:, x0], cov_scaling * cov_strip, size=2
                )

                # Select actions by maximising each sample
                i = np.argmax(samples[0])
                j = np.argmax(samples[1])

            x, x_prime = values[i], values[j]

            if file:
                file.write(f"Iteration {t}, Selected Pair: ({x}, {x_prime})\n")

            # Generate preference feedback
            p = sigmoid(f[i, j])
            y = np.random.binomial(1, p)
            dataset = np.vstack((dataset, [x, x_prime, y]))

            # Record regret
            regret = compute_regret(x_star, x, x_prime, f, values)
            regret_list.append(regret)

            if t % 20 == 0:
                elapsed = time.time() - fn_start_time
                print(f"t={t:4d}  elapsed={elapsed:.1f}s  regret={regret:.4f}")

    finally:
        if file:
            file.close()

    return regret_list
