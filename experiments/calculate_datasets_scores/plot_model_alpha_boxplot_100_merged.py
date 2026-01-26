#!/usr/bin/env python3
"""
Make 2 subplots by *merging* the original 2x2:
- Left  subplot: Homophilic Models  × {Homophilic Datasets, Non-Homophilic Datasets}
- Right subplot: Non-Homophilic Models × {Homophilic Datasets, Non-Homophilic Datasets}

For each subplot:
  • y-axis lists models (one row per model)
  • x-axis is Beta (sensitivity to k)
  • For each model, draw TWO HORIZONTAL boxplots stacked vertically:
      - top  : Homophilic Datasets
      - bottom: Non-Homophilic Datasets
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from my_models import ModelFactory


def load_panel_data(models_homophility, datasets_homophility, sort):
    """
    Load all CSVs once; return:
      results[(model_type, dataset_type)] = (model_names, per_model_values)
      all_values = flat list of all beta values (for global x-limits)
    """
    results = {}
    all_values = []

    for model_type in models_homophility:
        for dataset_type in datasets_homophility:
            key = (model_type, dataset_type)
            csv_path = f"model_dataset_{parameter}_values{model_type}{dataset_type}_data_percentage_100.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            df = pd.read_csv(csv_path, index_col=0)
            model_names = df.index.tolist()
            per_model_values = [df.loc[m].dropna().values.astype(float) for m in model_names]

            # Keep only models with at least one value
            keep = [len(v) > 0 for v in per_model_values]
            model_names = [m for m, k in zip(model_names, keep) if k]
            per_model_values = [v for v, k in zip(per_model_values, keep) if k]

            if len(per_model_values) == 0:
                raise ValueError(f"No valid {parameter} values in {csv_path}")

            # Optional sorting (within this panel)
            if sort != "none":
                if sort == "median":
                    keys = [np.nanmedian(v) for v in per_model_values]
                else:
                    keys = [np.nanmean(v) for v in per_model_values]
                order = np.argsort(keys)
                model_names = [model_names[i] for i in order]
                per_model_values = [per_model_values[i] for i in order]

            results[key] = (model_names, per_model_values)
            for v in per_model_values:
                all_values.extend(v)

    return results, all_values


def draw_merged_panel(ax, results, model_type, datasets_homophility, xlim,
                      colors=("#3182bd", "#e6550d")):
    """
    Draw ONE subplot for a fixed model_type, merging the two dataset types as stacked
    horizontal boxplots per model.

    colors[0]: homophilic datasets
    colors[1]: non-homophilic datasets
    """
    key_homo = (model_type, '_datasets_homophility')
    key_non  = (model_type, '_datasets_non_homophility')

    model_names_h, vals_h = results[key_homo]
    model_names_n, vals_n = results[key_non]

    # Use the intersection so each model has *both* boxes
    common = [m for m in model_names_h if m in model_names_n]
    if not common:
        raise ValueError(f"No common models for {model_type}")

    # Align values to the common model order
    map_h = {m: v for m, v in zip(model_names_h, vals_h)}
    map_n = {m: v for m, v in zip(model_names_n, vals_n)}
    vals_h = [map_h[m] for m in common]
    vals_n = [map_n[m] for m in common]

    # Positions: for each model i, place two boxes stacked vertically
    # base positions separated by 2 to leave room
    base = np.arange(len(common)) * 2.0
    pos_h = base + 0.6  # top (Homophilic Datasets)
    pos_n = base + 0.2  # bottom (Non-Homophilic Datasets)
    yticks = base + 0.4

    # Horizontal boxplots
    bp_h = ax.boxplot(
        vals_h, vert=False, positions=pos_h, patch_artist=True,
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5, facecolor=colors[0], alpha=0.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.7)
    )
    bp_n = ax.boxplot(
        vals_n, vert=False, positions=pos_n, patch_artist=True,
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5, facecolor=colors[1], alpha=0.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.7)
    )

    # Overlay means (horizontal points)
    means_h = [float(np.nanmean(v)) for v in vals_h]
    means_n = [float(np.nanmean(v)) for v in vals_n]
    ax.plot(means_h, pos_h, marker="o", linestyle="none", color=colors[0])
    ax.plot(means_n, pos_n, marker="o", linestyle="none", color=colors[1])

    # Labels & ticks
    model_display_names = [ModelFactory.display_name(m) for m in common]
    ax.set_yticks(yticks)
    ax.set_yticklabels(model_display_names, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Model")
    ax.set_xlim(xlim)
    ax.grid(axis="x", alpha=0.25)

    # Legend
    legend_patches = [
        Patch(facecolor=colors[0], alpha=0.5, label="Homophilic Datasets"),
        Patch(facecolor=colors[1], alpha=0.5, label="Non-Homophilic Datasets")
    ]
    ax.legend(handles=legend_patches, loc="lower right", frameon=False)


def main():
    parser = argparse.ArgumentParser(description=f"Merged per-model {parameter} boxplots (100%) as 1x2 grid.")
    parser.add_argument("--sort", choices=["none", "median", "mean"], default="none",
                        help="Optionally sort models within panels by median/mean beta.")
    parser.add_argument("--width", type=float, default=16.0, help="Figure width (inches).")
    parser.add_argument("--height", type=float, default=8.0, help="Figure height (inches).")
    parser.add_argument("--out", default=f"model_dataset_{parameter}_values_boxplots_100.png",
                        help="Output PNG filename.")
    args = parser.parse_args()

    models_homophility = ['_models_homophility', '_models_non_homophility']
    datasets_homophility = ['_datasets_homophility', '_datasets_non_homophility']

    MODEL_TITLE = {
        '_models_homophility': 'Homophilic Models',
        '_models_non_homophility': 'Non-Homophilic Models',
    }

    # Load all data, compute global x-limits
    results, all_values = load_panel_data(models_homophility, datasets_homophility, args.sort)
    xmin, xmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    xlim = (xmin - pad, xmax + pad)

    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height))

    # Left  panel: Homophilic Models (merge datasets)
    draw_merged_panel(
        axes[0], results, model_type='_models_homophility',
        datasets_homophility=datasets_homophility, xlim=xlim,
        colors=("#3182bd", "#e6550d")  # blue vs orange
    )
    axes[0].set_title(MODEL_TITLE['_models_homophility'], fontsize=12)

    # Right panel: Non-Homophilic Models (merge datasets)
    draw_merged_panel(
        axes[1], results, model_type='_models_non_homophility',
        datasets_homophility=datasets_homophility, xlim=xlim,
        colors=("#3182bd", "#e6550d")
    )
    axes[1].set_title(MODEL_TITLE['_models_non_homophility'], fontsize=12)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[OK] Saved {args.out}")


if __name__ == "__main__":
    beta = 'beta'
    gamma = 'gamma'
    alpha = 'alpha'

    parameter = alpha

    if parameter == beta:
        xlabel = "Beta (sensitivity to k)"
    elif parameter == gamma:
        xlabel = "Gamma (available-features indicator)"
    else:
        xlabel = "Alpha (intercept)"
    main()
