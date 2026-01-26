#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from my_models import ModelFactory

DATA_DIR = "../../data"
CFG_PATH = "../../datasets/config.json"

def read_k_list_for_dataset(dataset_name: str, data_percentage: int, models_names):
    """
    Find a canonical k-list for a dataset by reading auc_available.csv
    from the first model that has files. Returns [] if none found.
    """
    for try_model in models_names:
        path = f"data_percentage_{data_percentage}/{dataset_name}/{try_model}/cv_0/auc_available.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df.iloc[:, 0].tolist()  # first column = num_features
    return []

def effects_code(d_idx: int, D: int) -> np.ndarray:
    """
    Effects (sum-to-zero) coding for D datasets into D-1 columns:
      - For datasets 0..D-2: a single 1 in its own position
      - For the last dataset (D-1): -1 in all positions
    This enforces sum(theta_d) = 0 and keeps the intercept as the grand mean.
    """
    v = np.zeros(D - 1, dtype=float)
    if d_idx < D - 1:
        v[d_idx] = 1.0
    else:
        v[:] = -1.0
    return v

def collect_rows_for_model(model_name: str,
                           dataset_names,
                           data_percentage: int,
                           k_lists: dict,
                           cv_folds: int = 5):
    """
    Build pooled design for a single model across *all* datasets using
    effects coding for dataset fixed effects:

        y = alpha
            + (effects-coded dataset contrasts) * theta   [D columns]
            + gamma * j                                   [j in {0,1}]
            + beta  * k_norm                              [k normalized by #features of dataset]

    Returns X, y where columns are:
        [ effects-coded (D-1), j, k_norm ]
    """
    X_rows, y_vals = [], []

    D = len(dataset_names)
    for d_idx, dataset_name in enumerate(dataset_names):

        one_hot = [0] * len(dataset_names)
        one_hot[d_idx] = 1

        k_list = k_lists.get(dataset_name, [])
        if len(k_list) < 2:
            continue

        # load train to normalize k by number of features (minus target col)
        train_csv = os.path.join(DATA_DIR, dataset_name, "train", "data.csv")
        if not os.path.exists(train_csv):
            continue
        train_df = pd.read_csv(train_csv)
        num_features = train_df.shape[1] - 1
        if num_features <= 0:
            continue

        for j in (0, 1):  # 0: auc.csv (dropout at inference), 1: auc_available.csv (train on available)
            # load all CV files for this (model, dataset, j)
            frames = []
            missing = False
            for cv in range(cv_folds):
                fn = "auc_available.csv" if j == 1 else "auc.csv"
                p = f"data_percentage_{data_percentage}/{dataset_name}/{model_name}/cv_{cv}/{fn}"
                if not os.path.exists(p):
                    missing = True
                    break
                frames.append(pd.read_csv(p, index_col=0))
            if missing or not frames:
                continue

            # rows for each k in dataset's canonical list
            for k in k_list:
                # mean AUC across folds
                acc, valid = 0.0, True
                for cv in range(cv_folds):
                    df_cv = frames[cv]
                    if k not in df_cv.index or "auc" not in df_cv.columns:
                        valid = False
                        break
                    v = df_cv.loc[k, "auc"]
                    acc += float(v.item() if hasattr(v, "item") else v)
                if not valid:
                    continue

                y_mean = acc / cv_folds
                k_norm = k / float(num_features)

                X_rows.append(one_hot + [j, k_norm])
                y_vals.append(y_mean)

    vals = []
    for row in X_rows:
        vals.append(row[-1])

    mean = np.mean(vals)
    for row in X_rows:
        row[-1] -= mean

    if not X_rows:
        # (D-1) effects-coded columns + j + k_norm
        return np.empty((0, (len(dataset_names) - 1) + 2), dtype=float), np.empty((0,), dtype=float)

    return np.asarray(X_rows, dtype=float), np.asarray(y_vals, dtype=float)

def main():
    """
    For each data_percentage in {100, 75, 50}:
      For each model m:
        Fit (pooled across all datasets D), no intercept:
            AUC = sum_{d=1..D} theta_d * 1{dataset=d} + gamma * j + beta * k_norm
        Extract:
            beta  (coef for k_norm),
            gamma (coef for j),
            theta (vector of length D; one per dataset).
      Save one CSV per coefficient per data_percentage:
        model_beta_values_data_percentage_{dp}.csv     (Series, index=models)
        model_gamma_values_data_percentage_{dp}.csv    (Series, index=models)
        model_theta_values_data_percentage_{dp}.csv    (DataFrame, rows=models, cols=datasets)
    """
    data_percentages = [100, 75, 50]
    cv_folds = 5
    os.makedirs(".", exist_ok=True)

    # config / lists
    with open(CFG_PATH) as f:
        cfg = json.load(f)
    datasets = cfg["datasets"]
    dataset_names = [d["name"] for d in datasets]
    models_names = ModelFactory.MODELS

    for dp in data_percentages:
        # precompute k-lists for all datasets at this dp
        k_lists = {ds: read_k_list_for_dataset(ds, dp, models_names) for ds in dataset_names}

        # outputs (indexed by model)
        beta_series  = pd.Series(np.nan, index=models_names, dtype=float)
        gamma_series = pd.Series(np.nan, index=models_names, dtype=float)
        # θ matrix: rows=models, cols=datasets
        theta_df = pd.DataFrame(np.nan, index=models_names, columns=dataset_names, dtype=float)

        for model_name in tqdm(models_names, desc=f"data%={dp} (per model)"):
            X, y = collect_rows_for_model(
                model_name=model_name,
                dataset_names=dataset_names,
                data_percentage=dp,
                k_lists=k_lists,
                cv_folds=cv_folds,
            )
            if X.shape[0] == 0:
                continue

            # Need at least 2 distinct k_norm values overall to estimate a slope
            D = len(dataset_names)
            # Columns are: [ one-hot(D datasets), j, k_norm ]
            k_col_idx = D + 1
            unique_k = np.unique(X[:, k_col_idx])
            if unique_k.size < 2:
                continue

            # No intercept so the first D coefs are exactly thetas for dataset dummies
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y)

            # Coeff order: [theta_1..theta_D, gamma, beta]
            theta_vec = reg.coef_[0:D]
            gamma     = reg.coef_[D]
            beta      = reg.coef_[D + 1]

            theta_df.loc[model_name, :] = theta_vec
            gamma_series.loc[model_name] = float(gamma)
            beta_series.loc[model_name]  = float(beta)

        # save per-model coefficients
        beta_out  = f"model_beta_values_data_percentage_{dp}.csv"
        gamma_out = f"model_gamma_values_data_percentage_{dp}.csv"
        theta_out = f"model_theta_values_data_percentage_{dp}.csv"

        beta_series.to_csv(beta_out, header=["beta"])
        gamma_series.to_csv(gamma_out, header=["gamma"])
        theta_df.to_csv(theta_out, index=True)

        print(f"[OK] Wrote {beta_out}")
        print(f"[OK] Wrote {gamma_out}")
        print(f"[OK] Wrote {theta_out}")


if __name__ == "__main__":
    main()
