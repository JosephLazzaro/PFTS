"""
Ackley experiments — reproduce Figures 2a and 2b from the paper.

Runs PF-TS, MR-LPF, MaxMinLCB, POP-BO, and GP-TS (scalar baseline)
on a 1-D Ackley utility function discretised into 40 actions over
[-5, 5], with T = 300 iterations and 30 independent runs.

Usage
-----
    python ackley_experiments.py

Outputs are saved to the ``ackley40/`` directory as CSV files and
PDF plots of instantaneous / cumulative regret.
"""

import os
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt

from Thompson_Sampling_cpu import TS
from clean_test_algos_synthetic_cpu import (
    generate_ackley, BOHF, Max_Min_LCB
)
from POPBO_cpu import POP_BO
from Algos_direct_feedback_cpu import gp_ts


# ---------------------------------------------------------------------------
#  Hyperparameters (shared across all methods)
# ---------------------------------------------------------------------------

class DefaultArgs:
    """Default hyperparameters for all algorithms."""
    alpha_gp = 0.05           # GP noise parameter
    length_scale = 0.1        # Matérn kernel length scale
    grid_size = 40            # Number of discrete actions
    lambda_reg = 0.05         # Regularisation for logistic regression
    beta = 1                  # Confidence width coefficient (CI-based methods)
    learning_rate = 0.001     # Learning rate for gradient descent
    n_samples = 10
    n_iterations_GD = 200000  # Max gradient descent steps
    n_iterations = 300        # Optimisation horizon T
    lr_decay = 0              # 0 = constant learning rate
    kernel = "Matern"
    smoothness = 2.5          # Matérn-5/2
    enable_logging = 0
    append_identical_pairs = 0
    repeat_identical_pairs = 1


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def plot_instant(data, n_iterations, color, label, num_runs=30):
    """Plot mean instantaneous regret with ±1 standard-error bands."""
    avg = np.mean(data, axis=1)
    se = np.std(data, axis=1) / np.sqrt(num_runs)
    t = np.arange(n_iterations)
    plt.plot(t, avg, color=color, label=label)
    plt.fill_between(t, avg - se, avg + se, color=color, alpha=0.2)


def plot_cum(data, n_iterations, color, label, num_runs=30):
    """Plot mean cumulative regret with ±1 standard-error bands."""
    plot_instant(np.cumsum(data, axis=0), n_iterations, color, label, num_runs)


# ---------------------------------------------------------------------------
#  Main experiment loop
# ---------------------------------------------------------------------------

def run_simulations():
    args = DefaultArgs()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    num_runs = 30

    os.makedirs("ackley40", exist_ok=True)

    # Generate utility landscape
    values, Reward_function, f = generate_ackley(grid_size=args.grid_size)

    # ------------------------------------------------------------------
    #  PF-TS (ours)
    # ------------------------------------------------------------------
    print("\n=== Running PF-TS ===")
    data_ts = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_ts[:, i] = TS(args, values, Reward_function, f, timestamp, seed=i)
        np.savetxt("ackley40/data_TS_final.csv", data_ts, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  MR-LPF (Kayal et al., 2025)
    # ------------------------------------------------------------------
    print("\n=== Running MR-LPF ===")
    data_bohf = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_bohf[:, i] = BOHF(args, values, Reward_function, f, timestamp)[0]
        np.savetxt("ackley40/data_BOHF_final.csv", data_bohf, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  MaxMinLCB (Pásztor et al., 2024)
    # ------------------------------------------------------------------
    print("\n=== Running MaxMinLCB ===")
    data_lcb = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_lcb[:, i] = Max_Min_LCB(args, values, Reward_function, f, timestamp)[0]
        np.savetxt("ackley40/data_Max_Min_LCB_final.csv", data_lcb, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  POP-BO (Xu et al., 2024)
    # ------------------------------------------------------------------
    print("\n=== Running POP-BO ===")
    data_pop = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_pop[:, i] = POP_BO(args, values, Reward_function, f, timestamp)
        np.savetxt("ackley40/data_POP_BO_final.csv", data_pop, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  GP-TS — scalar feedback baseline
    # ------------------------------------------------------------------
    print("\n=== Running GP-TS (scalar feedback) ===")
    data_gpts = np.zeros((args.n_iterations, num_runs))
    t0 = time.time()
    for i in range(num_runs):
        data_gpts[:, i] = gp_ts(args, values, Reward_function, f)
        np.savetxt("ackley40/data_gpts_final.csv", data_gpts, delimiter=",")
        print(f"  Run {i+1}/{num_runs} done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    #  Generate plots (Figures 2a and 2b)
    # ------------------------------------------------------------------
    print("\n=== Generating plots ===")

    # --- Instantaneous regret (Figure 2a) ---
    plt.figure(figsize=(7, 5))
    plot_instant(data_ts,   args.n_iterations, "b",      "PF-TS",     num_runs)
    plot_instant(data_bohf, args.n_iterations, "r",      "MR-LPF",    num_runs)
    plot_instant(data_lcb,  args.n_iterations, "g",      "MaxMinLCB", num_runs)
    plot_instant(data_pop,  args.n_iterations, "k",      "POP-BO",    num_runs)
    plot_instant(data_gpts, args.n_iterations, "purple", "GP-TS",     num_runs)
    plt.ylim([0, 0.5])
    plt.xlabel("t")
    plt.ylabel("Instantaneous Regret")
    plt.legend()
    plt.savefig("ackley40/fig_ackley_instant.pdf", bbox_inches="tight")
    plt.close()

    # --- Cumulative regret (Figure 2b) ---
    plt.figure(figsize=(7, 5))
    plot_cum(data_ts,   args.n_iterations, "b",      "PF-TS",     num_runs)
    plot_cum(data_bohf, args.n_iterations, "r",      "MR-LPF",    num_runs)
    plot_cum(data_lcb,  args.n_iterations, "g",      "MaxMinLCB", num_runs)
    plot_cum(data_pop,  args.n_iterations, "k",      "POP-BO",    num_runs)
    plot_cum(data_gpts, args.n_iterations, "purple", "GP-TS",     num_runs)
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.savefig("ackley40/fig_ackley_cumulative.pdf", bbox_inches="tight")
    plt.close()

    print("Done. Plots saved to ackley40/")


if __name__ == "__main__":
    run_simulations()
