#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

def main():
    # ---- load datasets config ----
    with open("../../datasets/config.json") as f:
        config = json.load(f)
    datasets = config["datasets"]
    dataset_names = [d["name"] for d in datasets]

    # ---- read dataset-level beta/gamma/alpha (data_percentage=100) ----
    beta  = pd.read_csv("dataset_beta_values_data_percentage_100.csv", index_col=0)
    gamma = pd.read_csv("dataset_gamma_values_data_percentage_100.csv", index_col=0)
    alpha = pd.read_csv("dataset_alpha_values_data_percentage_100.csv", index_col=0)
    # normalize column names if needed
    if "beta" not in beta.columns:
        beta.columns = ["beta"]
    if "gamma" not in gamma.columns:
        gamma.columns = ["gamma"]
    if "alpha" not in alpha.columns:
        alpha.columns = ["alpha"]

    # ---- build a stats frame from the actual train sets ----
    stats_rows = []
    data_path = "../../data"
    for dataset_name in dataset_names:
        # Load data
        train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
        n_feat = train_set.shape[1] - 1   # assuming last column is target
        n_rows = train_set.shape[0]
        pos_ratio = train_set.iloc[:, -1].mean()  # assuming binary target in last column

        stats_rows.append(
            {"Dataset": dataset_name, "n_features": n_feat, "n_rows": n_rows, "pos_ratio": pos_ratio}
        )
    stats = pd.DataFrame(stats_rows).set_index("Dataset")

    # ---- join on dataset name, drop rows with missing any of beta/gamma/alpha ----
    df = stats.join(beta, how="inner").join(gamma, how="inner").join(alpha, how="inner")
    df = df.dropna(subset=["beta", "gamma", "alpha"])

    if df.empty:
        raise RuntimeError("No datasets with complete (stats, beta, gamma, alpha). "
                           "Check your config keys and CSV filenames.")

    # ---- plotting: 3x3 grid ----
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # rows: x variables (name, label, use_logscale)
    specs = [
        ("n_features", "Number of features", True),
        ("n_rows",     "Number of rows",     True),
        ("pos_ratio",  "Positive class ratio", False),
    ]
    # cols: y variables
    targets = [
        ("beta",  r"$\beta$ (sensitivity to $k$)"),
        ("gamma", r"$\gamma$ (available-features indicator)"),
        ("alpha", r"$\alpha$ (intercept)"),
    ]

    for r, (xcol, xlabel, logx) in enumerate(specs):
        for c, (ycol, ylabel) in enumerate(targets):
            ax = axes[r, c]
            x = df[xcol].values
            y = df[ycol].values
            ax.scatter(x, y, s=40)

            # annotate points with dataset names (lightly)
            for name, xv, yv in zip(df.index, x, y):
                ax.annotate(name, (xv, yv), xytext=(3, 3),
                            textcoords="offset points", fontsize=8, alpha=0.8)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # ax.set_ylim(0, 0.08)  # keep same y-range as your current plots

            if logx:
                pos = x > 0
                if pos.any():
                    ax.set_xscale("log")
            ax.grid(alpha=0.25)

    fig.suptitle("Dataset-level sensitivity (β), regime effect (γ), and intercept (α) vs. dataset characteristics", y=0.99)
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])
    out = "datasets_beta_gamma_alpha_vs_stats.png"
    fig.savefig(out, dpi=200)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
