#!/usr/bin/env python3
"""
Six-panel figure:

Left column = scatter (Actual AUC vs Predicted AUC), with one point per combination:
  (1) model–dataset–k–j points:
        predicted uses pairwise α_{m,D}, β_{m,D}, γ_{m,D}
        actual    is mean over CV folds for that (m,D,k,j)
  (2) model–k–j–dataset points:
        predicted uses dataset α_D and model β_m, γ_m
        actual    as above (per (m,D,k,j))
  (3) dataset–k–j–model points:
        predicted uses dataset α_D, β_D, γ_D
        actual    as above (per (m,D,k,j))

Right column = horizontal boxplots across datasets per model
  (4) α_{m,D} distributions (from pairwise file)
  (5) β_{m,D} distributions (from pairwise file)
  (6) γ_{m,D} distributions (from pairwise file)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
from experiments.calculate_datasets_scores import plot_style

from my_models import ModelFactory

CFG_PATH  = "../../datasets/config.json"
DATA_ROOT = "../../data"

# ---------- helpers ----------

def read_series_csv(path: str, expected_col: str) -> pd.Series:
    """Read a 1-col CSV (index is the label), normalize column name to expected_col."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    if expected_col not in df.columns:
        if df.shape[1] == 1:
            df.columns = [expected_col]
        else:
            raise ValueError(f"{path} must contain '{expected_col}' or a single column. Got {df.columns}.")
    return df[expected_col]

def read_k_list_for_dataset(dataset: str, dp: int, models: List[str]) -> List[int]:
    """Find a canonical k-list by scanning models for auc_available.csv (cv_0)."""
    for m in models:
        p = f"data_percentage_{dp}/{dataset}/{m}/cv_0/auc_available.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df.iloc[:, 0].tolist()
    return []

def load_num_features(dataset: str) -> int:
    df = pd.read_csv(os.path.join(DATA_ROOT, dataset, "train", "data.csv"))
    return df.shape[1] - 1  # last column assumed target

def load_cv_frames(model: str, dataset: str, dp: int, j: int, cv_folds: int) -> List[pd.DataFrame]:
    """Load CV dataframes for given (model, dataset, j). Return [] if any missing."""
    frames = []
    fn = "auc_available.csv" if j == 1 else "auc.csv"
    for cv in range(cv_folds):
        p = f"data_percentage_{dp}/{dataset}/{model}/cv_{cv}/{fn}"
        if not os.path.exists(p):
            return []
        frames.append(pd.read_csv(p, index_col=0))
    return frames

def mean_auc_for_k(frames: List[pd.DataFrame], k: int) -> float:
    """Mean AUC across folds for a given k. Returns np.nan if k missing."""
    vals = []
    for df in frames:
        if (k not in df.index) or ("auc" not in df.columns):
            return np.nan
        v = df.loc[k, "auc"]
        vals.append(float(v.item() if hasattr(v, "item") else v))
    return float(np.mean(vals)) if vals else np.nan

# ---------- main ----------


def main(dp: int = 100, cv_folds: int = 5, out_png: str = "six_panel_model_heterophilic_dataset_eval.png"):

    plot_style.set_plot_style()

    # config
    with open(CFG_PATH) as f:
        cfg = json.load(f)
    dataset_names = [d["name"] for d in cfg["datasets"]]
    model_names   = ModelFactory.MODELS

    # pairwise coefficients (rows=models, cols=datasets)
    A_md = pd.read_csv(f"model_dataset_heterophilic_alpha_values_data_percentage_{dp}.csv", index_col=0)
    B_md = pd.read_csv(f"model_dataset_heterophilic_beta_values_data_percentage_{dp}.csv",  index_col=0)
    G_md = pd.read_csv(f"model_dataset_heterophilic_gamma_values_data_percentage_{dp}.csv", index_col=0)
    A_md = A_md.reindex(index=model_names, columns=dataset_names)
    B_md = B_md.reindex(index=model_names, columns=dataset_names)
    G_md = G_md.reindex(index=model_names, columns=dataset_names)

    # dataset-level coefficients (Series indexed by dataset)
    beta_D  = read_series_csv(f"dataset_heterophilic_beta_values_data_percentage_{dp}.csv",  "beta").reindex(dataset_names)
    gamma_D = read_series_csv(f"dataset_heterophilic_gamma_values_data_percentage_{dp}.csv", "gamma").reindex(dataset_names)
    # per-(dataset, model) etas: rows=datasets, cols=models
    ETA_Dm = pd.read_csv(f"dataset_heterophilic_eta_values_data_percentage_{dp}.csv", index_col=0)
    ETA_Dm = ETA_Dm.reindex(index=dataset_names, columns=model_names)

    # model-level coefficients (Series indexed by model) — no model-level α
    beta_m  = read_series_csv(f"model_heterophilic_beta_values_data_percentage_{dp}.csv",  "beta").reindex(model_names)
    gamma_m = read_series_csv(f"model_heterophilic_gamma_values_data_percentage_{dp}.csv", "gamma").reindex(model_names)
    # per-(model, dataset) thetas: rows=models, cols=datasets
    THETA_mD = pd.read_csv(f"model_heterophilic_theta_values_data_percentage_{dp}.csv", index_col=0)
    THETA_mD = THETA_mD.reindex(index=model_names, columns=dataset_names)

    # precompute k-lists & feature counts
    k_lists = {D: read_k_list_for_dataset(D, dp, model_names) for D in dataset_names}
    d_feats = {D: load_num_features(D) for D in dataset_names}

    # cache CV frames to avoid repeated I/O
    cv_cache: Dict[Tuple[str, str, int], List[pd.DataFrame]] = {}

    def get_frames(m: str, D: str, j: int) -> List[pd.DataFrame]:
        key = (m, D, j)
        if key not in cv_cache:
            cv_cache[key] = load_cv_frames(m, D, dp, j, cv_folds)
        return cv_cache[key]

    # ------------------ Left column: scatters with one point per (m,D,k,j) ------------------
    md_actual, md_pred = [], []   # pairwise coefficients α_{m,D},β_{m,D},γ_{m,D}
    m_actual,  m_pred  = [], []   # model-level: θ_{m,D} + γ_m * j + β_m * (k/d)
    d_actual,  d_pred  = [], []   # dataset-level: η_{D,m} + γ_D * j + β_D * (k/d)

    knorm_mD_vals = defaultdict(list)  # (m, D) -> val
    knorm_m_vals = defaultdict(list)  # m -> val
    knorm_D_vals = defaultdict(list)  # D -> val

    # Iterate all combinations and add points
    for m in tqdm(model_names, desc=f"build points (dp={dp})"):
        for D in dataset_names:
            ks = k_lists.get(D, [])
            d = d_feats.get(D, 0)
            if not ks or d <= 0:
                continue
            for j in (0, 1):
                frames = get_frames(m, D, j)
                if not frames:
                    continue
                for k in ks:
                    # actual
                    auc = mean_auc_for_k(frames, k)
                    if np.isnan(auc):
                        continue
                    knorm = k / float(d)

                    # store knorm values for mean calculation
                    knorm_mD_vals[(m, D)].append(knorm)
                    knorm_m_vals[m].append(knorm)
                    knorm_D_vals[D].append(knorm)

    # Iterate all combinations and add points
    for m in tqdm(model_names, desc=f"build points (dp={dp})"):
        for D in dataset_names:
            ks = k_lists.get(D, [])
            d  = d_feats.get(D, 0)
            if not ks or d <= 0:
                continue

            # coefficients present?
            a_md = A_md.loc[m, D] if (m in A_md.index and D in A_md.columns) else np.nan
            b_md = B_md.loc[m, D] if (m in B_md.index and D in B_md.columns) else np.nan
            g_md = G_md.loc[m, D] if (m in G_md.index and D in G_md.columns) else np.nan

            b_D  = beta_D.get(D, np.nan)
            g_D  = gamma_D.get(D, np.nan)
            eta_Dm = ETA_Dm.loc[D, m] if (D in ETA_Dm.index and m in ETA_Dm.columns) else np.nan

            b_m_val = beta_m.get(m, np.nan)
            g_m_val = gamma_m.get(m, np.nan)
            theta_mD = THETA_mD.loc[m, D] if (m in THETA_mD.index and D in THETA_mD.columns) else np.nan

            for j in (0, 1):
                frames = get_frames(m, D, j)
                if not frames:
                    continue
                for k in ks:
                    # actual
                    auc = mean_auc_for_k(frames, k)
                    if np.isnan(auc):
                        continue
                    knorm = k / float(d)

                    knorm_mD_mean = np.mean(knorm_mD_vals[(m, D)])
                    knorm_m_mean = np.mean(knorm_m_vals[m])
                    knorm_D_mean = np.mean(knorm_D_vals[D])

                    # (1) pairwise prediction
                    if not (pd.isna(a_md) or pd.isna(b_md) or pd.isna(g_md)):
                        pred_md = float(a_md) + float(g_md) * j + float(b_md) * (knorm - knorm_mD_mean)
                        md_actual.append(auc)
                        md_pred.append(pred_md)

                    # (2) model-level prediction: θ_{m,D} + γ_m * j + β_m * (k/d)
                    if not (pd.isna(theta_mD) or pd.isna(b_m_val) or pd.isna(g_m_val)):
                        pred_m = float(theta_mD) + float(g_m_val) * j + float(b_m_val) * (knorm - knorm_m_mean)
                        m_actual.append(auc)
                        m_pred.append(pred_m)

                    # (3) dataset-level prediction: η_{D,m} + γ_D * j + β_D * (k/d)
                    if not (pd.isna(eta_Dm) or pd.isna(b_D) or pd.isna(g_D)):
                        pred_d = float(eta_Dm) + float(g_D) * j + float(b_D) * (knorm - knorm_D_mean)
                        d_actual.append(auc)
                        d_pred.append(pred_d)

    # ------------------ Right column: boxplots (pairwise coeffs across datasets per model) ------------------
    box_alpha = [A_md.loc[m].dropna().values.astype(float) for m in model_names]
    box_beta = [B_md.loc[m].dropna().values.astype(float) for m in model_names]
    box_gamma = [G_md.loc[m].dropna().values.astype(float) for m in model_names]
    model_labels = [ModelFactory.display_name(m) for m in model_names]

    # ------------------ Plot ------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_md, ax_m, ax_d), (ax_a, ax_b, ax_g) = axes

    def scatter_ax(ax, x, y, title):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = ~(np.isnan(x) | np.isnan(y))
        x, y = x[ok], y[ok]
        ax.scatter(x, y, s=12)
        if x.size and y.size:
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            pad = 0.02 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1, alpha=0.6)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Predicted AUC")
        ax.set_ylabel("Actual AUC")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    # left column scatters (now with per-(m,D,k,j) points)
    scatter_ax(ax_md, md_pred, md_actual, "Model–Dataset–k–j points (pairwise coeffs)")
    scatter_ax(ax_m,  m_pred,  m_actual,  "Model–k–j–Dataset points (α_D + β_m, γ_m)")
    scatter_ax(ax_d,  d_pred,  d_actual,  "Dataset–k–j–Model points (α_D, β_D, γ_D)")

    def hbox(ax, data, title, xlabel):
        bp = ax.boxplot(
            data, vert=False, labels=model_labels, showfliers=True, patch_artist=True,
            medianprops=dict(linewidth=2), whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5), boxprops=dict(linewidth=1.5)
        )
        for patch in bp["boxes"]:
            patch.set_alpha(0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticklabels(model_labels, fontsize=8)
        ax.grid(axis="x", alpha=0.25)

    hbox(ax_a, box_alpha, r"$\alpha_{m,D}$ across datasets (per model)", r"$\alpha$")
    hbox(ax_b, box_beta,  r"$\beta_{m,D}$ across datasets (per model)",  r"$\beta$ (sensitivity to $k/d$)")
    hbox(ax_g, box_gamma, r"$\gamma_{m,D}$ across datasets (per model)", r"$\gamma$ (regime effect)")

    fig.suptitle(f"Actual vs Predicted AUC points and Coefficient Distributions for Heterotrophic datasets", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    print(f"[OK] Saved {out_png}")


if __name__ == "__main__":
    # choose dp in {100, 75, 50}
    main(dp=100)
