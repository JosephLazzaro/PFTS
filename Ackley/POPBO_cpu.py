"""
POP-BO (Xu et al., 2024) baseline for 1-D synthetic experiments.

POP-BO is an optimistic, UCB-based method that fixes one action to
the point chosen in the previous round and selects the other by
maximising the UCB acquisition function.
"""

import numpy as np
import time

from clean_test_algos_synthetic_cpu import (
    DuelingKernel2, predict_f, update_sigma_D, sigmoid,
    find_most_preferred_action, compute_regret
)


def _get_UCB_argmax_index(index_t_prime, h_estimate, sigma_D, beta):
    """Return the index maximising UCB(x, x_{t'}) over x."""
    ucb_values = h_estimate[:, index_t_prime] + beta * sigma_D[:, index_t_prime]
    return np.argmax(ucb_values)


def POP_BO(args, values, Reward_function, f, timestamp, seed=None):
    """
    POP-BO algorithm (Xu et al., 2024).

    At each round:
      1. Set x'_t = x_{t-1}  (carry forward the previous action).
      2. Set x_t = argmax_x UCB(x, x'_t).
      3. Play (x_t, x'_t) and observe preference.

    Parameters & returns follow the same convention as TS().
    """
    if seed is not None:
        np.random.seed(seed)

    fn_start_time = time.time()

    dataset = np.empty((0, 3))
    regret_list = []

    dueling_kernel = DuelingKernel2(base_kernel=args.kernel,
                                     length_scale=args.length_scale,
                                     smoothness=args.smoothness)

    log_filename = (f"POP_BO_{args.kernel}_{args.learning_rate}_"
                    f"{args.lr_decay}_{timestamp}.txt") if args.enable_logging else None
    file = open(log_filename, "a") if log_filename else None

    # Randomly initialise the carried-forward action
    index_t = np.random.randint(0, args.grid_size)

    try:
        for t in range(args.n_iterations):
            if len(dataset) == 0:
                h_estimate = np.zeros((args.grid_size, args.grid_size))
                sigma_D = np.zeros((args.grid_size, args.grid_size))
            else:
                h_estimate = predict_f(
                    dataset, values, args.grid_size, dueling_kernel,
                    lambda_reg=args.lambda_reg, learning_rate=args.learning_rate,
                    n_iterations_GD=args.n_iterations_GD, lr_decay=args.lr_decay,
                    filename=log_filename
                )[0]
                sigma_D = update_sigma_D(dataset, dueling_kernel,
                                         alpha_gp=args.alpha_gp, values=values)

            # Carry forward the previous action
            index_t_prime = index_t
            # Select new action via UCB
            index_t = _get_UCB_argmax_index(index_t_prime, h_estimate,
                                             sigma_D, args.beta)

            x, x_prime = values[index_t], values[index_t_prime]

            if file:
                file.write(f"Iteration {t}, Selected Pair: ({x}, {x_prime})\n")

            # Generate preference feedback
            p = sigmoid(f[index_t, index_t_prime])
            y = np.random.binomial(1, p)
            dataset = np.vstack((dataset, [x, x_prime, y]))

            # Record regret
            x_star = find_most_preferred_action(Reward_function, values)
            regret = compute_regret(x_star, x, x_prime, f, values)
            regret_list.append(regret)

            if t % 20 == 0:
                elapsed = time.time() - fn_start_time
                print(f"t={t:4d}  elapsed={elapsed:.1f}s  regret={regret:.4f}")

    finally:
        if file:
            file.close()

    return regret_list
