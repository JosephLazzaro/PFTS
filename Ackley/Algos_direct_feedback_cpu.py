"""
Vanilla GP-UCB and GP-TS baselines with direct scalar feedback.

These methods receive noisy scalar observations y_t = f(x_t) + ε_t
(with ε_t ~ N(0,1)) rather than pairwise preferences.  They serve
as an oracle baseline: since scalar feedback is strictly more
informative than preference feedback, these methods establish a
performance ceiling for BOHF algorithms at the same iteration count.

To place all methods on the same regret scale, we evaluate regret
using the dueling definition (Eq. 2) with x'_t = x_t.
"""

import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

from Thompson_Sampling_cpu import get_cov_scaling
from clean_test_algos_synthetic_cpu import compute_regret


def gp_ucb(args, values, Reward_function, f):
    """
    GP-UCB with direct scalar feedback (Srinivas et al., 2010).

    Selects x_t = argmax_x  μ_{t-1}(x) + β · σ_{t-1}(x).
    """
    max_index = np.argmax(Reward_function)
    x_star = values[max_index]

    kernel = (Matern(length_scale=args.length_scale, nu=args.smoothness)
              if args.kernel == "Matern"
              else RBF(length_scale=args.length_scale))

    fn_start_time = time.time()
    dataset = np.empty((0, 2))
    regret_list = []

    for t in range(args.n_iterations):
        if len(dataset) == 0:
            x_index = np.random.randint(0, args.grid_size)
        else:
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                           alpha=args.alpha_gp)
            gp.fit(dataset[:, 0].reshape(-1, 1), dataset[:, 1])
            f_est, sigma = gp.predict(values.reshape(-1, 1), return_std=True)
            x_index = np.argmax(f_est + args.beta * sigma)

        x = values[x_index]
        y = Reward_function[x_index] + np.random.normal(0, 1)

        # Dueling regret with x'_t = x_t
        regret = compute_regret(x_star, x, x, f, values)
        regret_list.append(regret)
        dataset = np.vstack((dataset, [x, y]))

        if t % 20 == 0:
            print(f"GP-UCB  t={t:4d}  elapsed={time.time()-fn_start_time:.1f}s  "
                  f"regret={regret:.4f}")

    return regret_list


def gp_ts(args, values, Reward_function, f):
    """
    GP-TS with direct scalar feedback (Chowdhury & Gopalan, 2017).

    Draws a posterior sample of f and selects x_t = argmax_x f̃(x).
    """
    max_index = np.argmax(Reward_function)
    x_star = values[max_index]

    kernel = (Matern(length_scale=args.length_scale, nu=args.smoothness)
              if args.kernel == "Matern"
              else RBF(length_scale=args.length_scale))

    fn_start_time = time.time()
    dataset = np.empty((0, 2))
    regret_list = []

    for t in range(args.n_iterations):
        if len(dataset) == 0:
            x_index = np.random.randint(0, args.grid_size)
        else:
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                           alpha=args.alpha_gp)
            gp.fit(dataset[:, 0].reshape(-1, 1), dataset[:, 1])
            f_est, cov = gp.predict(values.reshape(-1, 1), return_cov=True)
            cov_scaling = get_cov_scaling(t, args)
            sample = np.random.multivariate_normal(f_est, cov_scaling * cov)
            x_index = np.argmax(sample)

        x = values[x_index]
        y = Reward_function[x_index] + np.random.normal(0, 1)

        regret = compute_regret(x_star, x, x, f, values)
        regret_list.append(regret)
        dataset = np.vstack((dataset, [x, y]))

        if t % 20 == 0:
            print(f"GP-TS   t={t:4d}  elapsed={time.time()-fn_start_time:.1f}s  "
                  f"regret={regret:.4f}")

    return regret_list
