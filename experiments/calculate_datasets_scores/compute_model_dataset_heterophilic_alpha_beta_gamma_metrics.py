import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt  # optional (kept for parity with your env)
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
            # First column is num_features
            return df.iloc[:, 0].tolist()
    return []


def collect_rows_for_pair(model_name: str, dataset_name: str, data_percentage: int, k_list, cv_folds=5):
    """
    For fixed (model, dataset), build rows of X=[j, k] and y=mean AUC over CV folds.
    j = 1 -> auc_available.csv ; j = 0 -> auc.csv
    Returns (X, y) as numpy arrays.
    """
    X_rows, y_vals = [], []

    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    num_features = train_set.shape[1] - 1  # minus target column

    for j in (0, 1):
        # Load CV files for this j
        cvs = []
        for cv in range(cv_folds):
            auc_path = (
                f"data_percentage_{data_percentage}/{dataset_name}/{model_name}/cv_{cv}/auc_available.csv"
                if j == 1 else
                f"data_percentage_{data_percentage}/{dataset_name}/{model_name}/cv_{cv}/auc.csv"
            )
            if not os.path.exists(auc_path):
                cvs = []
                break
            df_cv = pd.read_csv(auc_path, index_col=0)
            cvs.append(df_cv)

        if len(cvs) != cv_folds:
            # Missing files for this j; skip this j entirely
            continue

        for k in k_list:
            # Ensure k exists in all CV frames
            valid = True
            auc_sum = 0.0
            for cv in range(cv_folds):
                df_cv = cvs[cv]
                if k not in df_cv.index or "auc" not in df_cv.columns:
                    valid = False
                    break
                val = df_cv.loc[k, "auc"]
                try:
                    auc_sum += float(val.item() if hasattr(val, "item") else val)
                except Exception:
                    auc_sum += float(val)
            if not valid:
                continue

            y_mean = auc_sum / cv_folds
            X_rows.append([j, k / num_features])  # normalize k by num_features
            y_vals.append(y_mean)

    vals = []
    for row in X_rows:
        vals.append(row[-1])

    mean = np.mean(vals)
    for row in X_rows:
        row[-1] -= mean

    if not X_rows:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    return np.asarray(X_rows, dtype=float), np.asarray(y_vals, dtype=float)


def main():
    """
    For each data_percentage in {100, 75, 50}:
      For each model m:
        For each dataset D:
          Fit: AUC(j, k | m, D) = gamma * j + beta * k + alpha
          Extract beta and store at [m, D].
    Save one CSV per data_percentage:
      model_dataset_beta_values_data_percentage_{dp}.csv
    """
    # --- Config / inputs ---
    config_path = "../../datasets/config.json"
    data_percentages = [100, 75, 50]
    cv_folds = 5

    dataset_names = ModelFactory.DATASETS_NON_HOMOPHILITY

    models_names = ModelFactory.MODELS

    # # Models list
    # models_names = ModelFactory.MODELS

    os.makedirs(".", exist_ok=True)

    for dp in data_percentages:
        # Dataframe to accumulate beta per (model, dataset)
        beta_mat = pd.DataFrame(
            data=np.nan,
            index=models_names,
            columns=dataset_names
        )

        gamma_mat = pd.DataFrame(
            data=np.nan,
            index=models_names,
            columns=dataset_names
        )

        alpha_mat = pd.DataFrame(
            data=np.nan,
            index=models_names,
            columns=dataset_names
        )

        # Precompute k-list per dataset (same as your previous approach)
        k_lists = {
            ds: read_k_list_for_dataset(ds, dp, models_names) for ds in dataset_names
        }

        for model_name in tqdm(models_names, desc=f"data%={dp} (per model,dataset)"):
            for dataset_name in dataset_names:
                k_list = k_lists.get(dataset_name, [])
                if len(k_list) < 2:
                    # Not enough distinct k to estimate slope robustly
                    continue

                # Collect rows for this (model, dataset)
                X, y = collect_rows_for_pair(model_name, dataset_name, dp, k_list, cv_folds=cv_folds)
                if X.shape[0] == 0:
                    continue

                # Sanity: need at least 2 distinct k values or regression on k is ill-posed
                unique_k = np.unique(X[:, 1])
                if unique_k.size < 2:
                    continue

                # Fit linear regression: y = a*j + b*k  (no intercept implies bias? keep intercept=True)
                reg = LinearRegression()
                reg.fit(X, y)
                # Coeffs are [gamma (for j), beta (for k)]
                beta = reg.coef_[1]
                gamma = reg.coef_[0]
                alpha = reg.intercept_
                beta_mat.loc[model_name, dataset_name] = beta
                gamma_mat.loc[model_name, dataset_name] = gamma
                alpha_mat.loc[model_name, dataset_name] = alpha

        beta_out_csv = f"model_dataset_heterophilic_beta_values_data_percentage_{dp}.csv"
        beta_mat.to_csv(beta_out_csv, index=True)
        print(f"[OK] Wrote {beta_out_csv}")

        gamma_out_csv = f"model_dataset_heterophilic_gamma_values_data_percentage_{dp}.csv"
        gamma_mat.to_csv(gamma_out_csv, index=True)
        print(f"[OK] Wrote {gamma_out_csv}")

        alpha_out_csv = f"model_dataset_heterophilic_alpha_values_data_percentage_{dp}.csv"
        alpha_mat.to_csv(alpha_out_csv, index=True)
        print(f"[OK] Wrote {alpha_out_csv}")


if __name__ == "__main__":
    main()
