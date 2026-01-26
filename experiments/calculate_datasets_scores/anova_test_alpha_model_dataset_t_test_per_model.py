#!/usr/bin/env python3
"""
Plots p-values only (no CIs, no stars):

1) One-way ANOVA across models for alpha_{m,D}:
   - Single panel that displays just the ANOVA p-value.

2) Uncorrected two-sided one-sample t-tests for gamma_{m,D} per model (H0: mean == 0):
   - Bar chart of p-values per model (y-axis = p-value in [0,1]).
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from my_models import ModelFactory  # only used for display names; optional

# ----------------- helpers -----------------

def read_matrix(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col=0)

def one_way_anova_pvalue(alpha_df: pd.DataFrame) -> float:
    """Return ANOVA p-value for α across models (rows=models, cols=datasets)."""
    groups = []
    for model in alpha_df.index:
        vals = alpha_df.loc[model].dropna().values.astype(float)
        if vals.size >= 2:
            groups.append(vals)
    if len(groups) < 2:
        return np.nan
    F, p = stats.f_oneway(*groups)
    return float(p)

def ttest_pvalues_gamma(gamma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uncorrected two-sided one-sample t-tests per model for gamma_{m,D} (H0: mean == 0).
    Returns DataFrame with columns: ['p_value', 'n'] indexed by model.
    """
    rows = []
    for model in gamma_df.index:
        vals = gamma_df.loc[model].dropna().values.astype(float)
        n = vals.size
        if n >= 2:
            t_stat, p_val = stats.ttest_1samp(vals, popmean=0.0)
            p = float(p_val)
        else:
            p = np.nan
        rows.append({"model": model, "p_value": p, "n": n})
    out = pd.DataFrame(rows).set_index("model")
    return out

# ----------------- plotting -----------------

def plot_anova_p(ax, p_value: float):
    ax.axis("off")
    if np.isnan(p_value):
        txt = "ANOVA p-value for $\\alpha$: n/a (insufficient data)"
    else:
        txt = f"ANOVA p-value for $\\alpha$: {p_value:.3e}"
    ax.text(0.5, 0.55, txt, ha="center", va="center", fontsize=16)
    ax.text(0.5, 0.40, "(One-way ANOVA across models for $\\alpha_{m,D}$)",
            ha="center", va="center", fontsize=11, alpha=0.8)

def plot_gamma_pbars(ax, pvals_df: pd.DataFrame):
    # Drop NA p-values, sort ascending for readability
    df = pvals_df.dropna(subset=["p_value"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "No valid t-tests for $\\gamma$", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return

    # Optional: use display names if available
    try:
        labels = [ModelFactory.display_name(m) for m in df.index]
    except Exception:
        labels = df.index.tolist()

    order = np.argsort(df["p_value"].values)
    labels = [labels[i] for i in order]
    pvals = df["p_value"].values[order]

    y = np.arange(len(labels))
    bars = ax.barh(y, pvals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("p-value")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()  # smallest p at top
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Uncorrected t-test p-values for $\\gamma_{m,D}$ (per model)")

    # Annotate exact p-values
    for yi, p in zip(y, pvals):
        ax.text(min(p + 0.02, 0.98), yi, f"{p:.2e}", va="center", fontsize=8)

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=int, default=100, help="data percentage (e.g., 100, 75, 50)")
    ap.add_argument("--alpha_csv", default=None,
                    help="Path to model_dataset_alpha_values_data_percentage_{dp}.csv")
    ap.add_argument("--gamma_csv", default=None,
                    help="Path to model_dataset_gamma_values_data_percentage_{dp}.csv")
    ap.add_argument("--out", default="statistical_tests.png", help="Optional path to save the combined figure")
    args = ap.parse_args()

    dp = args.dp
    alpha_path = args.alpha_csv or f"model_dataset_alpha_values_data_percentage_{dp}.csv"
    gamma_path = args.gamma_csv or f"model_dataset_gamma_values_data_percentage_{dp}.csv"

    # Load pairwise coefficient matrices (rows=models, cols=datasets)
    A_md = read_matrix(alpha_path)
    G_md = read_matrix(gamma_path)

    # Compute p-values
    p_anova = one_way_anova_pvalue(A_md)
    gamma_pvals = ttest_pvalues_gamma(G_md)

    # Build a 1x2 plot: (left) ANOVA p-value, (right) gamma p-values per model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_anova_p(ax1, p_anova)
    plot_gamma_pbars(ax2, gamma_pvals)

    fig.suptitle(f"P-values only (dp={dp})", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.out:
        plt.savefig(args.out, dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
