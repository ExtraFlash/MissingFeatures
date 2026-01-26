#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from my_models import ModelFactory

# ---------- helper: find a canonical k-list for a dataset ----------
def read_k_list_for_dataset(dataset_name: str, data_percentage: int, models_names):
    """Return the num_features list by scanning models until we find auc_available.csv."""
    for try_model in models_names:
        p = f"data_percentage_{data_percentage}/{dataset_name}/{try_model}/cv_0/auc_available.csv"
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df.iloc[:, 0].tolist()  # first column = num_features
    return []

def main():
    # ---- load dataset names (keep order from config) ----
    with open("../../datasets/config.json") as f:
        config = json.load(f)
    datasets_cfg = config["datasets"]
    dataset_names = [d["name"] for d in datasets_cfg]

    models_names = ModelFactory.MODELS
    data_percentages = [100, 75, 50]
    cv_folds = 5

    # We’ll collect betas into a DataFrame: rows=datasets, cols=percentages
    beta_df = pd.DataFrame(index=dataset_names, columns=data_percentages, dtype=float)

    # ---- compute beta per (dataset, data_percentage) ----
    for dp in data_percentages:
        for dataset_name in tqdm(dataset_names, desc=f"data%={dp}"):
            # get canonical k list for this dataset@dp
            k_list = read_k_list_for_dataset(dataset_name, dp, models_names)
            if len(k_list) < 2:
                continue  # need ≥2 distinct k to estimate slope

            X_rows, y_vals = [], []
            for j in (0, 1):  # 0: auc.csv (dropout at inference), 1: auc_available.csv (train on available)
                # load CV files for each model; skip if any missing
                cvs_by_model = {}
                missing_any = False
                for m in models_names:
                    frames = []
                    for cv in range(cv_folds):
                        fn = ("auc_available.csv" if j == 1 else "auc.csv")
                        p = f"data_percentage_{dp}/{dataset_name}/{m}/cv_{cv}/{fn}"
                        if not os.path.exists(p):
                            missing_any = True
                            break
                        frames.append(pd.read_csv(p, index_col=0))
                    if missing_any:
                        break
                    cvs_by_model[m] = frames
                if missing_any:
                    continue

                # build (X, y)
                for mi, m in enumerate(models_names):
                    frames = cvs_by_model.get(m, [])
                    if not frames:
                        continue
                    for k in k_list:
                        # one-hot for model m
                        one_hot = [0] * len(models_names)
                        one_hot[mi] = 1
                        X_rows.append(one_hot + [j, k])

                        # mean AUC over CV folds
                        acc = 0.0
                        valid = True
                        for cv in range(cv_folds):
                            df_cv = frames[cv]
                            if k not in df_cv.index or "auc" not in df_cv.columns:
                                valid = False
                                break
                            v = df_cv.loc[k, "auc"]
                            acc += float(v.item() if hasattr(v, "item") else v)
                        if valid:
                            y_vals.append(acc / cv_folds)
                        else:
                            # drop the just-appended X if invalid
                            X_rows.pop()

            if not X_rows:
                continue

            X = np.asarray(X_rows, dtype=float)
            y = np.asarray(y_vals, dtype=float)

            # fit regression: y = (one-hot over models) + j + k
            reg = LinearRegression()
            reg.fit(X, y)
            beta = reg.coef_[len(models_names) + 1]  # coefficient for k
            beta_df.loc[dataset_name, dp] = float(beta)

    # ---- plot grouped bars (3 per dataset) ----
    # keep only datasets with at least one value
    beta_df = beta_df.dropna(how="all")

    x = np.arange(len(beta_df.index))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    bars_100 = ax.bar(x - width, beta_df[100].values, width, label="100%", color="#1f77b4")
    bars_75  = ax.bar(x,        beta_df[75].values,  width, label="75%",  color="#ff7f0e")
    bars_50  = ax.bar(x + width, beta_df[50].values, width, label="50%",  color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(beta_df.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Beta (sensitivity to k)")
    ax.set_xlabel("Dataset")
    ax.set_title("Per-dataset feature-sensitivity β for different data percentages")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.savefig("datasets_beta_values_grouped.png", dpi=200)
    print("[OK] Saved datasets_beta_values_grouped.png")

if __name__ == "__main__":
    main()
