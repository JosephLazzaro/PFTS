"""
Catalyst experiments — reproduce Figures 2c and 2d from the paper.

Runs PF-TS, MR-LPF, MaxMinLCB, POP-BO, and GP-TS (scalar baseline)
on the OCx24 catalyst hydrogen-yield dataset (Ag/Au/Zn compositions),
with T = 800 iterations and 30 independent runs.

Usage
-----
    python cat_experiments.py

Outputs are saved to the ``cat_sims/`` directory as CSV files and
PDF plots of instantaneous / cumulative regret.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from Algos_high_dim_cpu import gp_ts_high_dim
from Thompson_cat import TS_cat, POP_BO_cat
from clean_test_algos_yelp_full_gpu_optim_fix import (
    sigmoid, BOHF, Max_Min_LCB
)


# ---------------------------------------------------------------------------
#  Hyperparameters
# ---------------------------------------------------------------------------

class DefaultArgs:
    """Default hyperparameters (same as the Ackley experiments)."""
    alpha_gp = 0.05
    length_scale = 0.1
    grid_size = 63             # Number of catalyst compositions
    lambda_reg = 0.05
    beta = 1
    learning_rate = 0.001
    n_samples = 10
    n_iterations_GD = 200000
    n_iterations = 800         # Optimisation horizon T
    lr_decay = 0
    kernel = "Matern"
    smoothness = 2.5
    enable_logging = 0
    append_identical_pairs = 0
    repeat_identical_pairs = 1


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def plot_instant(data, n_iterations, n_runs, color, label):
    avg = np.mean(data, axis=1)
    se = np.std(data, axis=1) / np.sqrt(n_runs)
    t = np.arange(n_iterations)
    plt.plot(t, avg, color=color, label=label)
    plt.fill_between(t, avg - se, avg + se, color=color, alpha=0.2)


def plot_cum(data, n_iterations, n_runs, color, label):
    plot_instant(np.cumsum(data, axis=0), n_iterations, n_runs, color, label)


# ---------------------------------------------------------------------------
#  Load OCx24 dataset
# ---------------------------------------------------------------------------

def load_catalyst_data(path="CatData_final.csv"):
    """
    Load the OCx24 catalyst dataset.

    Returns
    -------
    values : ndarray, shape (63, 3)
        Catalyst compositions (Ag, Au, Zn).
    Reward_function : ndarray, shape (63,)
        Hydrogen yield rescaled to [0, 10].
    f : ndarray, shape (63, 63)
        Pairwise difference matrix.
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    values = data[:, :3]
    Reward_function = data[:, 3] / 10.0  # Rescale [0,100] → [0,10]

    n = len(Reward_function)
    f = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            f[i, j] = Reward_function[i] - Reward_function[j]

    return values, Reward_function, f


# ---------------------------------------------------------------------------
#  Main experiment loop
# ---------------------------------------------------------------------------

def run_simulations():
    args = DefaultArgs()
    num_runs = 30

    os.makedirs("cat_sims", exist_ok=True)

    values, Reward_function, f = load_catalyst_data()
    args.grid_size = len(Reward_function)

    # ------------------------------------------------------------------
    #  PF-TS (ours)
    # ------------------------------------------------------------------
    print("\n=== Running PF-TS ===")
    data_ts = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_ts[:, i] = TS_cat(args, values, Reward_function, f, "0")
        np.savetxt("cat_sims/data_cat_TS_longer.csv", data_ts, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  GP-TS (scalar feedback baseline)
    # ------------------------------------------------------------------
    print("\n=== Running GP-TS (scalar feedback) ===")
    data_gpts = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_gpts[:, i] = gp_ts_high_dim(args, values, Reward_function, f)
        np.savetxt("cat_sims/data_cat_GP_TS_longer.csv", data_gpts, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  MR-LPF (Kayal et al., 2025)
    # ------------------------------------------------------------------
    print("\n=== Running MR-LPF ===")
    data_bohf = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_bohf[:, i] = BOHF(args, np.arange(63), Reward_function, f,
                                 "0", arm_features=values)[0]
        np.savetxt("cat_sims/data_cat_BOHF_longer.csv", data_bohf, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  MaxMinLCB (Pásztor et al., 2024)
    # ------------------------------------------------------------------
    print("\n=== Running MaxMinLCB ===")
    data_lcb = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_lcb[:, i] = Max_Min_LCB(args, np.arange(63), Reward_function, f,
                                       "0", arm_features=values)[0]
        np.savetxt("cat_sims/data_cat_Max_Min_LCB_longer.csv", data_lcb,
                   delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  POP-BO (Xu et al., 2024)
    # ------------------------------------------------------------------
    print("\n=== Running POP-BO ===")
    data_pop = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_pop[:, i] = POP_BO_cat(args, np.arange(63), Reward_function, f,
                                      "0", arm_features=values)
        np.savetxt("cat_sims/data_cat_POP_BO_longer.csv", data_pop, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  Generate plots (Figures 2c and 2d)
    # ------------------------------------------------------------------
    print("\n=== Generating plots ===")

    # Instantaneous regret (Figure 2c)
    plt.figure(figsize=(7, 5))
    plot_instant(data_ts,   args.n_iterations, num_runs, "b",      "PF-TS")
    plot_instant(data_gpts, args.n_iterations, num_runs, "purple", "GP-TS")
    plot_instant(data_bohf, args.n_iterations, num_runs, "r",      "MR-LPF")
    plot_instant(data_lcb,  args.n_iterations, num_runs, "g",      "MaxMinLCB")
    plot_instant(data_pop,  args.n_iterations, num_runs, "k",      "POP-BO")
    plt.ylim([0, 0.5])
    plt.xlabel("t")
    plt.ylabel("Instantaneous Regret")
    plt.legend()
    plt.savefig("cat_sims/fig_catalyst_instant.pdf", bbox_inches="tight")
    plt.close()

    # Cumulative regret (Figure 2d)
    plt.figure(figsize=(7, 5))
    plot_cum(data_ts,   args.n_iterations, num_runs, "b",      "PF-TS")
    plot_cum(data_gpts, args.n_iterations, num_runs, "purple", "GP-TS")
    plot_cum(data_bohf, args.n_iterations, num_runs, "r",      "MR-LPF")
    plot_cum(data_lcb,  args.n_iterations, num_runs, "g",      "MaxMinLCB")
    plot_cum(data_pop,  args.n_iterations, num_runs, "k",      "POP-BO")
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.savefig("cat_sims/fig_catalyst_cumulative.pdf", bbox_inches="tight")
    plt.close()

    print("Done. Plots saved to cat_sims/")


if __name__ == "__main__":
    run_simulations()
