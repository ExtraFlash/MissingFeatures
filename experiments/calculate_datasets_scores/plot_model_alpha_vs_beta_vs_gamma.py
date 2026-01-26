#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_models import ModelFactory
from experiments.calculate_datasets_scores import plot_style

def _ensure_single_col(df: pd.DataFrame, want: str) -> pd.DataFrame:
    if want not in df.columns:
        if df.shape[1] == 1:
            df.columns = [want]
        else:
            raise ValueError(f"Expected column '{want}' in dataframe with columns {df.columns.tolist()}")
    return df[[want]]

def _annotate(ax, x, y, labels):
    for xv, yv, name in zip(x, y, labels):
        ax.annotate(name, (xv, yv), xytext=(3, 3),
                    textcoords="offset points", fontsize=8, alpha=0.8)

def main():
    plot_style.set_plot_style()
    # Load per-model coefficients (for data_percentage=100)
    beta_df  = pd.read_csv("model_beta_values_data_percentage_100.csv",  index_col=0)
    gamma_df = pd.read_csv("model_gamma_values_data_percentage_100.csv", index_col=0)

    # Normalize expected column names
    beta_df  = _ensure_single_col(beta_df,  "beta")
    gamma_df = _ensure_single_col(gamma_df, "gamma")

    # Join by model index
    df = beta_df.join(gamma_df, how="inner")
    if df.empty:
        raise RuntimeError("No overlapping models across beta/gamma CSVs.")

    # Display names for annotations
    try:
        df["display_name"] = [ModelFactory.display_name(m) for m in df.index]
    except Exception:
        df["display_name"] = df.index.astype(str)

    # Single scatter: gamma (y) vs beta (x)
    fig, ax = plt.subplots(figsize=(6, 5))
    x = df["beta"].values
    y = df["gamma"].values
    ax.scatter(x, y, s=40)
    _annotate(ax, x, y, df["display_name"].values)

    ax.set_xlabel("β (sensitivity to k)")
    ax.set_ylabel("γ (available-features indicator)")
    # Optional fixed limits:
    # ax.set_xlim(0, 0.08)
    # ax.set_ylim(0, 0.08)

    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("models_scatter_gamma_vs_beta.png", dpi=200)
    print("[OK] Saved models_scatter_gamma_vs_beta.png")


if __name__ == "__main__":
    main()
