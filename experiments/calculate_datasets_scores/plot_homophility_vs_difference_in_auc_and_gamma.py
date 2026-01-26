#!/usr/bin/env python3
import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from experiments.calculate_datasets_scores import plot_style

from my_models import ModelFactory
try:
    from plots_utils import utils
    HAVE_UTILS = True
except Exception:
    HAVE_UTILS = False

CFG_PATH = "../../datasets/config.json"
HOMO_CSV = "../check_homophily/homophily_scores.csv"
DP = 100
CVS = 5
METRIC = "auc"
METRIC_TITLE = "AUC"

# ------------ helpers ------------

def mean_auc_over_folds(csv_path, index_key):
    """Read a single auc/auc_available csv and return the AUC at index_key."""
    df = pd.read_csv(csv_path, index_col=0)
    return float(df.loc[index_key, "auc"])

def mean_auc_for_model_dataset_k(model, dataset, k, j=0):
    """
    Mean AUC across CV folds for (model, dataset, k, j).
    j=0 => auc.csv (dropout-at-inference)
    j=1 => auc_available.csv (train on available)
    """
    fn = "auc_available.csv" if j == 1 else "auc.csv"
    vals = []
    for cv in range(CVS):
        p = f"data_percentage_{DP}/{dataset}/{model}/cv_{cv}/{fn}"
        if not os.path.exists(p):
            return np.nan
        df = pd.read_csv(p, index_col=0)
        if (k not in df.index) or ("auc" not in df.columns):
            return np.nan
        v = df.loc[k, "auc"]
        vals.append(float(v.item() if hasattr(v, "item") else v))
    return float(np.mean(vals)) if vals else np.nan

def get_features_list(dataset, placeholder_model):
    """Return [total_features] + [k for k in rest if k%5==1]."""
    p0 = f"data_percentage_{DP}/{dataset}/{placeholder_model}/cv_0/{METRIC}.csv"
    df0 = pd.read_csv(p0)
    feats = df0["num_features"].to_list()
    total = feats[0]
    filtered = [total] + [k for k in feats[1:] if k % 5 == 1]
    return filtered

def get_best_baseline_model(dataset):
    """Best model by average AUC across CVS (excluding the robust trio)."""
    candidates = [m for m in ModelFactory.MODELS
                  if m not in (ModelFactory.DAE_NAME,
                               ModelFactory.GAT_NAME,
                               ModelFactory.GCN_NAME)]
    best_m, best_auc = None, -math.inf
    for m in candidates:
        aucs = []
        for cv in range(CVS):
            p = f"data_percentage_{DP}/{dataset}/{m}/cv_{cv}/auc.csv"
            if not os.path.exists(p):
                aucs = []
                break
            df = pd.read_csv(p, index_col=0)
            aucs.append(float(df["auc"].mean()))
        if aucs:
            a = float(np.mean(aucs))
            if a > best_auc:
                best_auc, best_m = a, m
    return best_m

def linear_fit_and_line(x_pts, y_pts, color, ax):
    """Fit y~x and draw dashed line."""
    if not x_pts:
        return
    x_arr = np.array(x_pts).reshape(-1, 1)
    y_arr = np.array(y_pts)
    if np.all(np.isnan(y_arr)):
        return
    ok = ~(np.isnan(x_arr.squeeze()) | np.isnan(y_arr))
    x_arr, y_arr = x_arr[ok], y_arr[ok]
    if x_arr.size < 2:
        return
    reg = LinearRegression()
    reg.fit(x_arr, y_arr)
    x_range = np.linspace(np.min(x_arr), np.max(x_arr), 100).reshape(-1, 1)
    y_pred = reg.predict(x_range)
    ax.plot(x_range, y_pred, color=color, linewidth=2, linestyle="--")

# ------------ main plotting ------------

def main():
    plot_style.set_plot_style()

    # Load config & datasets
    with open(CFG_PATH) as f:
        cfg = json.load(f)
    datasets = cfg["datasets"]
    dataset_names = [d["name"] for d in datasets]

    # Homophily scores
    hom_df = pd.read_csv(HOMO_CSV, index_col=0)
    hom_map = hom_df["scores"].to_dict()

    # Robust models to plot + their styles
    models = [ModelFactory.DAE_NAME, ModelFactory.GAT_NAME, ModelFactory.GCN_NAME]
    markers = {ModelFactory.DAE_NAME: "o",
               ModelFactory.GAT_NAME: "s",
               ModelFactory.GCN_NAME: "^"}
    offsets = {ModelFactory.DAE_NAME: -0.015,
               ModelFactory.GAT_NAME: -0.005,
               ModelFactory.GCN_NAME:  0.005,
               "Best": 0.015}

    # Pairwise gamma (rows=models, cols=datasets)
    G_md = pd.read_csv(f"model_dataset_gamma_values_data_percentage_{DP}.csv", index_col=0)

    # Placeholder model for reading the feature-count list
    placeholder_model = ModelFactory.MODELS[0]

    if HAVE_UTILS:
        utils.make_style(plt)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6), sharex=False)

    # -------------------- LEFT SUBPLOT: AUC difference vs Homophily --------------------
    for m in models:
        xs, ys = [], []
        color = ModelFactory.color_for(m)
        for ds in dataset_names:
            if ds not in hom_map:
                continue
            # pick feature counts
            k_list = get_features_list(ds, placeholder_model)

            # best baseline model for this dataset
            best_m = get_best_baseline_model(ds)
            if best_m is None:
                continue

            # average difference across selected k (best - model), each AUC averaged over 5 CVs
            diffs = []
            for k in k_list:
                auc_m   = mean_auc_for_model_dataset_k(m,      ds, k, j=0)
                auc_best= mean_auc_for_model_dataset_k(best_m, ds, k, j=0)
                if np.isnan(auc_m) or np.isnan(auc_best):
                    continue
                diffs.append(auc_best - auc_m)
            if not diffs:
                continue

            xs.append(hom_map[ds] + offsets[m])
            ys.append(float(np.mean(diffs)))

        # scatter + regression
        ax_left.scatter(xs, ys, s=80, alpha=0.85, marker=markers[m], edgecolors="k",
                        label=ModelFactory.display_name(m), color=color)
        linear_fit_and_line([x - offsets[m] for x in xs], ys, color, ax_left)

    ax_left.set_xlabel("Homophily score", fontsize=12)
    ax_left.set_ylabel(f"Difference in {METRIC_TITLE} vs. best baseline", fontsize=12)
    ax_left.set_title(f"{METRIC_TITLE} gap to best model vs. homophily", fontsize=13)
    ax_left.grid(True, linestyle="--", alpha=0.4)
    ax_left.legend(title="Model", frameon=True)

    # -------------------- RIGHT SUBPLOT: gamma vs Homophily --------------------
    # Also add best baseline gamma curve per dataset (using the best model’s γ_{m,D})
    for key in models + ["Best"]:
        xs, ys = [], []
        color = (ModelFactory.color_for(key) if key in models else "green")
        marker = (markers.get(key, "D"))
        for ds in dataset_names:
            if ds not in hom_map:
                continue
            if key == "Best":
                best_m = get_best_baseline_model(ds)
                if best_m is None or best_m not in G_md.index or ds not in G_md.columns:
                    continue
                gamma_val = G_md.loc[best_m, ds]
            else:
                if key not in G_md.index or ds not in G_md.columns:
                    continue
                gamma_val = G_md.loc[key, ds]

            xs.append(hom_map[ds] + offsets.get(key, 0.0))
            ys.append(float(gamma_val))

        ax_right.scatter(xs, ys, s=80, alpha=0.85, marker=marker, edgecolors="k",
                         label=("Best baseline" if key == "Best" else ModelFactory.display_name(key)),
                         color=color)
        # regression on de-jittered x
        linear_fit_and_line([x - offsets.get(key, 0.0) for x in xs], ys, color, ax_right)

    ax_right.set_xlabel("Homophily score", fontsize=12)
    ax_right.set_ylabel(r"$\gamma$ (regime effect)", fontsize=12)
    ax_right.set_title(r"$\gamma$ vs. homophily per dataset & model", fontsize=13)
    ax_right.grid(True, linestyle="--", alpha=0.4)
    ax_right.legend(title="Model", frameon=True)

    fig.tight_layout()
    out = "homophility_vs_difference_in_auc_and_gamma.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
