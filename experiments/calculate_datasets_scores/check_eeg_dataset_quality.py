#!/usr/bin/env python3
"""
EDA & Diagnostics for tabular binary classification datasets.

What it does:
- Summaries: shape, missingness, duplicates, class balance, constant features, conflicting labels
- Missingness map
- Correlation heatmap (numeric)
- Per-feature signal: univariate ROC-AUC (binary) and mutual information
- Train/val shift per feature: KS test + standardized mean difference
- Histograms per class for top-K features (fast, NumPy-based; shared bins; optional downsampling)
- PCA 2D scatter (standardized)
- Quick baselines: Logistic Regression + Random Forest
- Learning curve (RF)
- Permutation importance (RF)

Assumptions:
- CSV at: ../../data/<dataset_name>/train/data.csv
- Last column is the target (binary preferred).
- If your project has utils.preprocess_split, it's used; else a stratified split fallback is used.

Outputs:
- A folder eda_<dataset_name>/ with CSVs and PNGs.
"""

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
from scipy.stats import ks_2samp

# sklearn
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
try:
    from sklearn.feature_selection import mutual_info_classif
except Exception:
    mutual_info_classif = None

# try your project's splitter (optional)
try:
    import utils
    HAS_UTILS = hasattr(utils, "preprocess_split")
except Exception:
    HAS_UTILS = False


# ---------------- helpers ----------------
def stratified_split_fallback(df: pd.DataFrame, target_col: str, seed: int = 42):
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )
    return X_train, pd.Series(y_train, name=target_col), X_val, pd.Series(y_val, name=target_col)


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def safe_auc(y_true, scores) -> float:
    # Robust AUC (returns nan if not computable)
    if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return np.nan


def standardized_mean_diff(x_tr, x_va) -> float:
    # Cohen's d-like standardized mean difference
    a = np.asarray(x_tr, float)
    b = np.asarray(x_va, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    n1, n2 = a.size, b.size
    denom = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2)
    if denom <= 0:
        return np.nan
    sp = math.sqrt(denom / (n1 + n2 - 2))
    return np.nan if sp == 0 else (m1 - m2) / sp


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_binary_01(y: pd.Series) -> Tuple[np.ndarray, dict]:
    """
    Convert a 2-class target to {0,1} deterministically.
    Returns y_bin and mapping dict {original_label: 0/1}.
    If not binary, returns original values and empty mapping.
    """
    uniq = pd.unique(y)
    if len(uniq) != 2:
        return y.to_numpy(), {}
    uniq_sorted = sorted(list(uniq), key=lambda z: str(z))
    mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
    return y.map(mapping).to_numpy(), mapping


# ---------------- main ----------------
def run_eda(dataset_name: str, data_dir: str = "../../data", seed: int = 42, top_k: int = 16):
    out_dir = Path(f"eda_{dataset_name}")
    ensure_dir(out_dir)

    # ---- load ----
    train_csv = Path(data_dir) / dataset_name / "train" / "data.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing file: {train_csv}")

    df = pd.read_csv(train_csv)
    target_col = df.columns[-1]  # assume last col is target

    # ---- split ----
    if HAS_UTILS:
        X_train, y_train, X_val, y_val = utils.preprocess_split(df, seed=seed)  # type: ignore
        X = pd.concat([X_train, X_val], axis=0)
        y = pd.concat([y_train, y_val], axis=0)
    else:
        X_train, y_train, X_val, y_val = stratified_split_fallback(df, target_col=target_col, seed=seed)
        X = pd.concat([X_train, X_val], axis=0)
        y = pd.concat([y_train, y_val], axis=0)

    # Ensure aligned integer indices (avoid slow Pandas alignment later)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Binary mapping (for AUC & class plots)
    y_all_bin, mapping = to_binary_01(y)
    y_tr_bin, _ = to_binary_01(y_train)
    y_va_bin, _ = to_binary_01(y_val)

    # ---- basic info ----
    summary = {
        "n_rows_total": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "target_name": target_col,
        "binary_target": bool(len(pd.unique(y)) == 2),
        "class_ratio_pos_overall": float(np.nan if len(pd.unique(y)) != 2 else np.mean(y_all_bin)),
        "class_ratio_pos_train": float(np.nan if len(pd.unique(y_train)) != 2 else np.mean(y_tr_bin)),
        "class_ratio_pos_val": float(np.nan if len(pd.unique(y_val)) != 2 else np.mean(y_va_bin)),
        "missing_any": bool(df.isna().any().any()),
        "num_constant_features": int((X.nunique(dropna=False) <= 1).sum()),
        "num_duplicate_rows": int(df.duplicated().sum()),
    }

    # Conflicting labels for duplicate X (hashed by str-join; cheap & robust)
    Xy = df.copy()
    Xy["__dup_key__"] = Xy.drop(columns=[target_col]).astype(str).agg("|".join, axis=1)
    grp_nunique = Xy.groupby("__dup_key__")[target_col].nunique()
    summary["duplicate_X_conflicting_y_groups"] = int((grp_nunique > 1).sum())

    pd.Series(summary).to_csv(out_dir / "summary.csv")

    # ---- missingness map ----
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(df.isna().values, aspect="auto", interpolation="nearest")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_title("Missingness map")
        fig.tight_layout()
        fig.savefig(out_dir / "missingness_map.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # ---- class balance (overall) ----
    if len(pd.unique(y)) == 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = pd.Series(y_all_bin).value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title("Class balance (overall)")
        ax.set_xlabel("Class (0/1)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "class_balance_overall.png", dpi=200)
        plt.close(fig)

    # ---- correlation heatmap (numeric) ----
    num_cols = [c for c in X.columns if is_numeric_series(X[c])]
    if len(num_cols) > 1:
        try:
            corr = X[num_cols].corr(method="pearson")
            side = min(12, 0.5 + 0.25 * len(num_cols))
            fig, ax = plt.subplots(figsize=(side, side))
            cax = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
            ax.set_xticks(range(len(num_cols)))
            ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=90, fontsize=6)
            ax.set_yticklabels(num_cols, fontsize=6)
            ax.set_title("Feature correlation (Pearson)")
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_dir / "correlation_heatmap.png", dpi=200)
            plt.close(fig)
        except Exception:
            pass

    # ---- per-feature univariate AUC & MI ----
    per_feature = []
    y_bin = y_all_bin
    if len(pd.unique(y)) == 2:
        for c in num_cols:
            s = X[c].values
            per_feature.append((c, safe_auc(y_bin, s)))
    else:
        for c in num_cols:
            per_feature.append((c, np.nan))

    perf_df = pd.DataFrame(per_feature, columns=["feature", "auc_uni"]).set_index("feature")

    if mutual_info_classif is not None:
        try:
            Xnum = X[num_cols].copy()
            Xnum = Xnum.fillna(Xnum.median())
            # MI expects y as 1D array
            y_for_mi = y_bin if len(pd.unique(y)) == 2 else y.to_numpy()
            mi = mutual_info_classif(Xnum.values, y_for_mi, discrete_features=False, random_state=seed)
            mi_df = pd.DataFrame({"feature": num_cols, "mi": mi}).set_index("feature")
            perf_df = perf_df.join(mi_df, how="left")
        except Exception:
            pass

    perf_df.sort_values(["auc_uni"], ascending=False, na_position="last").to_csv(out_dir / "per_feature_signal.csv")

    # bar chart: top features by univariate AUC (if any)
    top_from_auc = perf_df.dropna(subset=["auc_uni"]).sort_values("auc_uni", ascending=False)
    if not top_from_auc.empty:
        top = top_from_auc.head(top_k)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top.index, top["auc_uni"].values)
        ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Univariate ROC-AUC")
        ax.set_title(f"Top {len(top)} features by univariate AUC")
        fig.tight_layout()
        fig.savefig(out_dir / "top_features_univariate_auc.png", dpi=200)
        plt.close(fig)

    # ---- train/val shift per feature (KS + std mean diff) ----
    drift_rows = []
    for c in num_cols:
        a = X_train[c].values
        b = X_val[c].values
        a_ = a[np.isfinite(a)]
        b_ = b[np.isfinite(b)]
        if a_.size > 1 and b_.size > 1:
            ks_stat, ks_p = ks_2samp(a_, b_)
        else:
            ks_stat, ks_p = np.nan, np.nan
        smd = standardized_mean_diff(a, b)
        drift_rows.append((c, ks_stat, ks_p, smd, np.nanmean(a), np.nanmean(b)))
    drift_df = pd.DataFrame(
        drift_rows,
        columns=["feature", "ks_stat", "ks_p", "std_mean_diff", "mean_train", "mean_val"]
    ).set_index("feature")
    drift_df.sort_values("ks_stat", ascending=False).to_csv(out_dir / "train_val_shift.csv")

    # ---- histograms per class for top features (fast; NumPy masks; shared bins) ----
    if len(pd.unique(y)) == 2 and not top_from_auc.empty:
        feats_to_plot = top_from_auc.index.tolist()[:min(8, len(top_from_auc))]

        # Prepare masks once
        y_np = y_bin.astype(int)
        mask0 = (y_np == 0)
        mask1 = (y_np == 1)

        n = len(feats_to_plot)
        cols = 4
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for i, c in enumerate(feats_to_plot):
            ax = axes[i // cols, i % cols]
            xvals = X[c].to_numpy()

            # determine shared bins via robust percentiles
            finite = np.isfinite(xvals)
            if not finite.any():
                ax.set_title(f"{c} (all NaN)")
                continue
            vmin, vmax = np.nanpercentile(xvals[finite], [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = np.nanmin(xvals[finite]), np.nanmax(xvals[finite])
                if vmin == vmax:
                    vmin, vmax = vmin - 0.5, vmax + 0.5
            bins = np.linspace(vmin, vmax, 30)

            # extract class-specific arrays (fast)
            v0 = xvals[mask0 & np.isfinite(xvals)]
            v1 = xvals[mask1 & np.isfinite(xvals)]

            # optional downsampling to keep plotting snappy
            MAX_POINTS = 200_000
            rng = np.random.default_rng(0)
            if v0.size > MAX_POINTS:
                v0 = rng.choice(v0, size=MAX_POINTS, replace=False)
            if v1.size > MAX_POINTS:
                v1 = rng.choice(v1, size=MAX_POINTS, replace=False)

            ax.hist(v0, bins=bins, alpha=0.6, density=True, label="y=0", rasterized=True)
            ax.hist(v1, bins=bins, alpha=0.6, density=True, label="y=1", rasterized=True)
            ax.set_title(c)
            ax.legend(fontsize=8)

        # remove any empty subplots
        total_axes = rows * cols
        for j in range(i + 1, total_axes):
            fig.delaxes(axes[j // cols, j % cols])

        fig.tight_layout()
        fig.savefig(out_dir / "class_histograms_top_features.png", dpi=200)
        plt.close(fig)

    # ---- PCA (2D) on standardized numeric features ----
    if len(num_cols) >= 2:
        try:
            scaler = StandardScaler()
            Xnum = X[num_cols].copy()
            Xnum = Xnum.fillna(Xnum.median())
            X_std = scaler.fit_transform(Xnum)
            pca = PCA(n_components=2, random_state=seed)
            Z = pca.fit_transform(X_std)
            fig, ax = plt.subplots(figsize=(5, 4))
            if len(pd.unique(y)) == 2:
                sc = ax.scatter(Z[:, 0], Z[:, 1], c=y_bin, s=10, cmap="coolwarm", alpha=0.8)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="class (0/1)")
            else:
                ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.8)
            ev = pca.explained_variance_ratio_.sum()
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA (2D), explained var: {ev:.2f}")
            fig.tight_layout()
            fig.savefig(out_dir / "pca_scatter.png", dpi=200)
            plt.close(fig)
        except Exception:
            pass

    # ---- quick baselines ----
    lr_auc, rf_auc = np.nan, np.nan
    try:
        # prepare numeric matrices
        Xtr_num = X_train[num_cols].copy()
        Xva_num = X_val[num_cols].copy()
        med = Xtr_num.median()
        Xtr_num = Xtr_num.fillna(med)
        Xva_num = Xva_num.fillna(med)

        # Logistic Regression (standardized)
        sclr = MinMaxScaler().fit(Xtr_num)
        Xtr_s = sclr.transform(Xtr_num)
        Xva_s = sclr.transform(Xva_num)

        if len(pd.unique(y_train)) == 2 and len(pd.unique(y_val)) == 2:
            lr = LogisticRegression(max_iter=2000)
            lr.fit(Xtr_s, y_tr_bin)
            lr_scores = lr.decision_function(Xva_s)
            lr_auc = safe_auc(y_va_bin, lr_scores)

        # Random Forest (unstandardized numeric)
        rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)

        # from my_models import ModelFactory
        # optimized_model_name = ModelFactory.get_optimized_model_name(ModelFactory.Random_Forest_NAME)
        # checkpoint_dir = f"../../optimized_models/{dataset_name}/{optimized_model_name}"
        # best_params_path = f"{checkpoint_dir}/best_params.json"
        # with open(best_params_path) as f:
        #     best_params = json.load(f)
        # if 'n_estimators' in best_params:
        #     best_params['n_estimators'] = min(best_params['n_estimators'], 100)
        # config_path = "../../datasets/config.json"
        # # get datasets config
        # with open(config_path) as f:
        #     config = json.load(f)
        # # get datasets from config
        # datasets: list = config['datasets']
        # dataset = [d for d in datasets if d['name'] == dataset_name][0]
        # types_list: List[str] = dataset['types_list']
        #
        # model, loaded = ModelFactory.get_model(optimized_model_name, X_train.shape[1], dataset_name=dataset_name,
        #                                        types_list=types_list, **best_params)

        rf.fit(Xtr_s, y_train.values if len(pd.unique(y_train)) == 2 else y_train)
        if len(pd.unique(y_val)) == 2:
            rf_scores = rf.predict_proba(Xva_s)[:, 1]
            rf_auc = safe_auc(y_va_bin, rf_scores)

        with open(out_dir / "quick_baselines.txt", "w") as f:
            f.write(f"LogReg val AUC: {lr_auc:.4f}\n")
            f.write(f"RandomForest val AUC: {rf_auc:.4f}\n")

        # Learning curve (RF) — clone-safe
        if len(pd.unique(y_train)) == 2:
            sizes, train_scores, val_scores = learning_curve(
                rf, Xtr_num, y_tr_bin, train_sizes=np.linspace(0.1, 1.0, 6),
                cv=5, scoring="roc_auc", n_jobs=-1, shuffle=True, random_state=seed
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(sizes, train_scores.mean(axis=1), marker="o", label="train AUC")
            ax.plot(sizes, val_scores.mean(axis=1), marker="o", label="CV AUC")
            ax.set_xlabel("Train size")
            ax.set_ylabel("ROC-AUC")
            ax.set_title("Learning curve (RF)")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "learning_curve_rf.png", dpi=200)
            plt.close(fig)

        # Permutation importance (RF) on validation
        if len(pd.unique(y_val)) == 2:
            pi = permutation_importance(
                rf, Xva_num, y_va_bin, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=-1
            )
            order = np.argsort(pi.importances_mean)[::-1]
            top_imp = min(20, len(num_cols))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(top_imp), pi.importances_mean[order][:top_imp][::-1])
            ax.set_yticks(range(top_imp))
            ax.set_yticklabels([num_cols[i] for i in order][:top_imp][::-1], fontsize=8)
            ax.set_xlabel("Permutation importance (val AUC drop)")
            ax.set_title("Top features (RF, validation)")
            fig.tight_layout()
            fig.savefig(out_dir / "permutation_importance_rf.png", dpi=200)
            plt.close(fig)

    except Exception as e:
        with open(out_dir / "quick_baselines.txt", "w") as f:
            f.write(f"[WARN] Baselines failed: {repr(e)}\n")

    print(f"[OK] Wrote EDA to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Change as needed:
    DATASET = "eeg-eye-state"
    run_eda(DATASET)
