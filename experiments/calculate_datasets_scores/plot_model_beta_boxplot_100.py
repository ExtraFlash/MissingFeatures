#!/usr/bin/env python3
"""
Generate a single figure with 4 subplots (2x2 grid), each showing
per-model beta boxplots for different combinations of model/dataset homophily.

- Subplot titles adapt to the combination (correctly).
- All subplots share the same y-axis limits for consistent scaling.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_models import ModelFactory


def main():
    parser = argparse.ArgumentParser(description="Plot per-model beta boxplots (100%) as a 2x2 subplot grid.")
    parser.add_argument(
        "--sort",
        choices=["none", "median", "mean"],
        default="none",
        help="Optionally sort models by their median/mean beta."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=16.0,
        help="Figure width in inches."
    )
    parser.add_argument(
        "--height",
        type=float,
        default=12.0,
        help="Figure height in inches."
    )
    parser.add_argument(
        "--out",
        default="model_dataset_beta_values_boxplots_grid_100.png",
        help="Output PNG filename."
    )
    args = parser.parse_args()

    models_homophility = ['_models_homophility', '_models_non_homophility']
    datasets_homophility = ['_datasets_homophility', '_datasets_non_homophility']

    # Title maps (avoid substring ambiguities)
    MODEL_TITLE = {
        '_models_homophility': 'Homophilic Models',
        '_models_non_homophility': 'Non-Homophilic Models',
    }
    DATASET_TITLE = {
        '_datasets_homophility': 'Homophilic Datasets',
        '_datasets_non_homophility': 'Non-Homophilic Datasets',
    }

    # Prepare storage for y-limits
    all_values = []

    # Preload all per_model_values to determine global y-axis limits
    results = {}
    for model_type in models_homophility:
        for dataset_type in datasets_homophility:
            key = (model_type, dataset_type)
            csv_path = f"model_dataset_beta_values{model_type}{dataset_type}_data_percentage_100.csv"
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            df = pd.read_csv(csv_path, index_col=0)
            model_names = df.index.tolist()
            per_model_values = [df.loc[m].dropna().values.astype(float) for m in model_names]

            # Filter out models with no data
            keep = [len(v) > 0 for v in per_model_values]
            model_names = [m for m, k in zip(model_names, keep) if k]
            per_model_values = [v for v, k in zip(per_model_values, keep) if k]

            if len(per_model_values) == 0:
                raise ValueError(f"No valid beta values in {csv_path}")

            # Optional sorting
            if args.sort != "none":
                if args.sort == "median":
                    keys = [np.nanmedian(v) for v in per_model_values]
                else:
                    keys = [np.nanmean(v) for v in per_model_values]
                order = np.argsort(keys)
                model_names = [model_names[i] for i in order]
                per_model_values = [per_model_values[i] for i in order]

            results[key] = (model_names, per_model_values)

            for v in per_model_values:
                all_values.extend(v)

    # Determine global y-limits (add small padding)
    ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ymin, ymax = ymin - pad, ymax + pad

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(args.width, args.height))
    axes = axes.flatten()

    # Ensure a stable order of keys (same as loops above)
    ordered_keys = [(m, d) for m in models_homophility for d in datasets_homophility]

    for ax, (model_type, dataset_type) in zip(axes, ordered_keys):
        model_names, per_model_values = results[(model_type, dataset_type)]

        model_display_names = [ModelFactory.display_name(model_name) for model_name in model_names]

        bp = ax.boxplot(
            per_model_values,
            labels=model_display_names,
            showfliers=True,
            patch_artist=True,
            medianprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.5)
        )
        for patch in bp['boxes']:
            patch.set_alpha(0.5)

        # Overlay per-model means
        means = [float(np.nanmean(v)) for v in per_model_values]
        ax.plot(range(1, len(means) + 1), means, marker="o", linestyle="none")

        # Titles (use exact-map, not substring)
        title_model = MODEL_TITLE[model_type]
        title_dataset = DATASET_TITLE[dataset_type]
        ax.set_title(f"{title_model} × {title_dataset}", fontsize=12)

        ax.set_ylabel("Beta (sensitivity to k)")
        ax.set_xlabel("Model")
        ax.set_xticklabels(model_display_names, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[OK] Saved {args.out}")


if __name__ == "__main__":
    main()
