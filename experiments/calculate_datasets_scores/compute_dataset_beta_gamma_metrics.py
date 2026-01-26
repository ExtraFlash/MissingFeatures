#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from my_models import ModelFactory


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


def effects_code_model(m_idx: int, M: int) -> np.ndarray:
    """
    Effects coding for M models into M-1 columns:
      models 0..M-2 -> standard basis; model M-1 -> all -1.
    Sum of model effects = 0; intercept stays meaningful.
    """
    v = np.zeros(M - 1, dtype=float)
    if m_idx < M - 1:
        v[m_idx] = 1.0
    else:
        v[:] = -1.0
    return v


def collect_rows_for_dataset(dataset_name: str,
                             data_percentage: int,
                             models_names,
                             k_list,
                             cv_folds: int = 5):
    """
    Build a dataset-level regression design:

      y = alpha + (one-hot over models) * eta + gamma * j + beta * k

    where
      - j in {0,1}: 0 => auc.csv (dropout-at-inference), 1 => auc_available.csv (train on available features)
      - k: number of available features
      - one-hot over models controls for baseline differences across models

    Returns (X, y) with columns: [ one-hot(models) ..., j, k ].
    """
    X_rows, y_vals = [], []

    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    num_features = train_set.shape[1] - 1  # minus target column

    for j in (0, 1):
        # Load all CV files per model for this j; skip j entirely if any are missing.
        cvs_by_model = {}
        missing_any = False
        for m in models_names:
            frames = []
            for cv in range(cv_folds):
                fn = "auc_available.csv" if j == 1 else "auc.csv"
                p = f"data_percentage_{data_percentage}/{dataset_name}/{m}/cv_{cv}/{fn}"
                if not os.path.exists(p):
                    missing_any = True
                    break
                frames.append(pd.read_csv(p, index_col=0))
            if missing_any:
                break
            cvs_by_model[m] = frames
        if missing_any:
            continue

        # Build rows across models and k values
        for mi, m in enumerate(models_names):
            frames = cvs_by_model.get(m, [])
            if not frames:
                continue
            for k in k_list:
                # One-hot for model m
                one_hot = [0] * len(models_names)
                one_hot[mi] = 1

                # Mean AUC over CV folds
                acc = 0.0
                valid = True
                for cv in range(cv_folds):
                    df_cv = frames[cv]
                    if k not in df_cv.index or "auc" not in df_cv.columns:
                        valid = False
                        break
                    v = df_cv.loc[k, "auc"]
                    acc += float(v.item() if hasattr(v, "item") else v)
                if not valid:
                    continue

                X_rows.append(one_hot + [j, k / num_features])  # normalize k by num_features
                y_vals.append(acc / cv_folds)

    vals = []
    for row in X_rows:
        vals.append(row[-1])

    mean = np.mean(vals)
    for row in X_rows:
        row[-1] -= mean

    if not X_rows:
        return np.empty((0, len(models_names) + 2), dtype=float), np.empty((0,), dtype=float)

    return np.asarray(X_rows, dtype=float), np.asarray(y_vals, dtype=float)


def main():
    """
    For each data_percentage in {100, 75, 50}:
      * For each dataset D:
          - Build X with one-hot(model) + j + k_norm (k normalized by #features)
          - Fit LinearRegression(fit_intercept=False)
          - Extract:
              η (coeffs for model one-hots; length = len(models))
              γ (coef for j)
              β (coef for k_norm)
      * Save:
          dataset_beta_values_data_percentage_{dp}.csv        (Series: index=datasets, col=beta)
          dataset_gamma_values_data_percentage_{dp}.csv       (Series: index=datasets, col=gamma)
          dataset_eta_values_data_percentage_{dp}.csv         (DataFrame: rows=datasets, cols=models; values=eta)
    """
    data_percentages = [100, 75, 50]
    cv_folds = 5

    os.makedirs(".", exist_ok=True)

    config_path = "../../datasets/config.json"
    with open(config_path) as f:
        config = json.load(f)

    datasets = config['datasets']
    dataset_names = [d['name'] for d in datasets]
    models_names = ModelFactory.MODELS

    for dp in data_percentages:
        # Precompute k-list per dataset
        k_lists = {ds: read_k_list_for_dataset(ds, dp, models_names) for ds in dataset_names}

        # Outputs (dataset-level)
        beta_series  = pd.Series(np.nan, index=dataset_names, dtype=float)
        gamma_series = pd.Series(np.nan, index=dataset_names, dtype=float)

        # η: per-dataset vector over models  → save as DataFrame (rows=datasets, cols=models)
        eta_df = pd.DataFrame(np.nan, index=dataset_names, columns=models_names, dtype=float)

        for dataset_name in tqdm(dataset_names, desc=f"data%={dp} (per dataset)"):
            k_list = k_lists.get(dataset_name, [])
            if len(k_list) < 2:
                continue  # need ≥2 distinct k to estimate a slope

            X, y = collect_rows_for_dataset(
                dataset_name=dataset_name,
                data_percentage=dp,
                models_names=models_names,
                k_list=k_list,
                cv_folds=cv_folds,
            )
            if X.shape[0] == 0:
                continue

            # Sanity: need at least 2 distinct k values
            # Columns: [ one-hot(models) ..., j, k_norm ] → k_norm at index len(models_names)+1
            unique_k = np.unique(X[:, len(models_names) + 1])
            if unique_k.size < 2:
                continue

            # Fit regression (no intercept; one-hot covers model baselines)
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y)

            # Coeffs order = [ η_1 ... η_M, γ, β ]
            M = len(models_names)
            eta_vec = reg.coef_[0:M]
            gamma   = reg.coef_[M]
            beta    = reg.coef_[M + 1]

            # Store
            eta_df.loc[dataset_name, :] = eta_vec
            gamma_series.loc[dataset_name] = float(gamma)
            beta_series.loc[dataset_name]  = float(beta)

        # Save dataset-level β, γ, and η (matrix)
        beta_out_csv  = f"dataset_beta_values_data_percentage_{dp}.csv"
        gamma_out_csv = f"dataset_gamma_values_data_percentage_{dp}.csv"
        eta_out_csv   = f"dataset_eta_values_data_percentage_{dp}.csv"

        beta_series.to_csv(beta_out_csv, header=["beta"])
        gamma_series.to_csv(gamma_out_csv, header=["gamma"])
        eta_df.to_csv(eta_out_csv, index=True)

        print(f"[OK] Wrote {beta_out_csv}")
        print(f"[OK] Wrote {gamma_out_csv}")
        print(f"[OK] Wrote {eta_out_csv}")



if __name__ == "__main__":
    main()
