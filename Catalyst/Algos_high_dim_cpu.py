"""
Vanilla GP-TS and GP-UCB with direct scalar feedback for multi-dimensional
action spaces (Catalyst experiments).

These methods receive noisy scalar observations and serve as oracle baselines
that upper-bound the performance achievable with preference-only feedback.
"""

import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

from clean_test_algos_yelp_full_gpu_optim_fix import sigmoid

# Import get_cov_scaling (used for the exploration bonus in GP-TS)
import sys
sys.path.insert(0, "../Ackley")
try:
    from Thompson_Sampling_cpu import get_cov_scaling
except ImportError:
    def get_cov_scaling(t, args, delta=0.05):
        return np.sqrt(2.0 * (t + 1 + np.log(2.0 / delta)))


def compute_regret_high_dim(x_index, x_prime_index, f, Reward_function):
    """Dueling regret for index-based action selection."""
    i_star = np.argmax(Reward_function)
    return (sigmoid(f[i_star, x_index]) +
            sigmoid(f[i_star, x_prime_index]) - 1.0) / 2.0


def gp_ts_high_dim(args, values, Reward_function, f):
    """
    GP-TS with direct scalar feedback for multi-dimensional actions.

    Parameters
    ----------
    values : ndarray, shape (n_actions, d)
        Feature vectors for each action.
    """
    kernel = (Matern(length_scale=args.length_scale, nu=args.smoothness)
              if args.kernel == "Matern"
              else RBF(length_scale=args.length_scale))

    fn_start = time.time()
    dataset = np.empty((0, values.shape[1] + 1))  # [features..., y]
    regret_list = []

    for t in range(args.n_iterations):
        if len(dataset) == 0:
            x_index = np.random.randint(0, args.grid_size)
        else:
            d = values.shape[1]
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                           alpha=args.alpha_gp)
            gp.fit(dataset[:, :d], dataset[:, d])
            f_est, cov = gp.predict(values, return_cov=True)
            cov_scaling = get_cov_scaling(t, args)
            sample = np.random.multivariate_normal(f_est, cov_scaling * cov)
            x_index = np.argmax(sample)

        x = values[x_index]
        y = Reward_function[x_index] + np.random.normal(0, 1)

        regret = compute_regret_high_dim(x_index, x_index, f, Reward_function)
        regret_list.append(regret)
        dataset = np.vstack((dataset, np.hstack((x, [y]))))

        if t % 20 == 0:
            print(f"GP-TS   t={t:4d}  elapsed={time.time()-fn_start:.1f}s  "
                  f"regret={regret:.4f}")

    return regret_list


def gp_ucb_high_dim(args, values, Reward_function, f):
    """GP-UCB with direct scalar feedback for multi-dimensional actions."""
    kernel = (Matern(length_scale=args.length_scale, nu=args.smoothness)
              if args.kernel == "Matern"
              else RBF(length_scale=args.length_scale))

    fn_start = time.time()
    dataset = np.empty((0, values.shape[1] + 1))
    regret_list = []

    for t in range(args.n_iterations):
        if len(dataset) == 0:
            x_index = np.random.randint(0, args.grid_size)
        else:
            d = values.shape[1]
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                           alpha=args.alpha_gp)
            gp.fit(dataset[:, :d], dataset[:, d])
            f_est, std = gp.predict(values, return_std=True)
            x_index = np.argmax(f_est + args.beta * std)

        x = values[x_index]
        y = Reward_function[x_index] + np.random.normal(0, 1)

        regret = compute_regret_high_dim(x_index, x_index, f, Reward_function)
        regret_list.append(regret)
        dataset = np.vstack((dataset, np.hstack((x, [y]))))

        if t % 50 == 0:
            print(f"GP-UCB  t={t:4d}  elapsed={time.time()-fn_start:.1f}s  "
                  f"regret={regret:.4f}")

    return regret_list
