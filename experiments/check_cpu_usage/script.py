import os
import json
import time
import threading
from statistics import mean, median
from typing import Callable, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# Optional: if psutil exists we get detailed sampling; otherwise we fall back.
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

from tqdm import tqdm

# Your project imports (unchanged)
from my_models import ModelFactory
from utils import utils

# -----------------------------
# CPU Monitoring Utilities
# -----------------------------

class CPUMonitor:
    """
    Monitors *process* CPU percent during a code block.
    If psutil is available: samples psutil.Process().cpu_percent(interval=sample_sec).
    Otherwise: estimates average CPU % from process_time()/wall_time.
    CPU percent can exceed 100 on multi-core (e.g., 400% means 4 cores saturated).
    """
    def __init__(self, sample_sec: float = 0.25):
        self.sample_sec = sample_sec
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_wall: float = 0.0
        self._end_wall: float = 0.0
        self._start_cpu: float = 0.0
        self._end_cpu: float = 0.0

        # psutil process handle (if available)
        self._proc = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None

    def _run_sampler(self):
        # First call establishes baseline; psutil recommends a first 0.0 to prime
        if self._proc is not None:
            self._proc.cpu_percent(None)  # prime
        while not self._stop.is_set():
            if self._proc is not None:
                pct = self._proc.cpu_percent(interval=self.sample_sec)  # blocks for sample_sec
                self._samples.append(float(pct))
            else:
                # When psutil is not available, we don't sample here. We compute aggregate later.
                time.sleep(self.sample_sec)

    def __enter__(self):
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        if PSUTIL_AVAILABLE:
            self._thread = threading.Thread(target=self._run_sampler, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end_cpu = time.process_time()
        self._end_wall = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def wall_seconds(self) -> float:
        return max(0.0, self._end_wall - self._start_wall)

    def summary(self) -> Tuple[float, float, float]:
        """
        Returns (avg_cpu_pct, max_cpu_pct, wall_seconds).
        If psutil present -> computed from samples.
        Else -> avg_cpu_pct = 100 * (proc_cpu_seconds / wall_seconds), max_cpu_pct = avg_cpu_pct.
        """
        w = self.wall_seconds
        if PSUTIL_AVAILABLE and self._samples:
            avg_cpu = float(mean(self._samples))
            max_cpu = float(max(self._samples))
            return avg_cpu, max_cpu, w
        else:
            # Fallback: estimate average CPU% = (CPU time / wall time) * 100
            cpu_sec = max(0.0, self._end_cpu - self._start_cpu)
            avg_cpu = 100.0 * (cpu_sec / w) if w > 0 else 0.0
            return avg_cpu, avg_cpu, w


# -----------------------------
# Training / Evaluation Helpers
# -----------------------------

def fit_model_with_monitor(model, model_name: str, X_train, y_train, X_val, y_val) -> Dict[str, float]:
    """
    Fit the model while monitoring CPU usage. Supports the ModelFactory training modes you already use.
    Returns dict with CPU/time stats for 'fit_' prefix.
    """
    with CPUMonitor(sample_sec=0.25) as mon:
        if ModelFactory.is_train_with_val(model_name):
            model.fit(X_train, y_train, X_val, y_val)
        elif ModelFactory.is_eval_set_format(model_name):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)
    avg_cpu, max_cpu, wall = mon.summary()
    return {
        "fit_avg_cpu": avg_cpu,
        "fit_max_cpu": max_cpu,
        "fit_wall_s": wall,
    }


def predict_with_monitor(model, model_name: str, X_val) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Predict labels (and probs if available) while monitoring CPU usage.
    Returns (y_pred, y_prob_or_None, stats_dict_with_predict_prefix).
    """
    with CPUMonitor(sample_sec=0.25) as mon:
        # Some models may not implement predict_proba; handle gracefully.
        y_pred = model.predict(X_val)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_val)
            except Exception:
                y_prob = None
    avg_cpu, max_cpu, wall = mon.summary()
    return y_pred, y_prob, {
        "predict_avg_cpu": avg_cpu,
        "predict_max_cpu": max_cpu,
        "predict_wall_s": wall,
    }


def run_single_model(
    model_name: str,
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    is_multy_class: bool,
    types_list
) -> Dict[str, object]:
    """
    Build model from ModelFactory using best_params (capped n_estimators to 100 as in your code),
    fit once, predict once, compute metrics, and capture CPU/time stats for fit and predict.
    """
    # Resolve checkpoint dir (same logic you had)
    checkpoint_dir = f"../../optimized_models/{dataset_name}/{model_name}"
    if model_name == ModelFactory.Complete_Random_Forest_NAME:
        checkpoint_dir = checkpoint_dir.replace(ModelFactory.Complete_Random_Forest_NAME,
                                                ModelFactory.Random_Forest_NAME)
    elif model_name == ModelFactory.Complete_Gradient_Boosting_Classifier_Name:
        checkpoint_dir = checkpoint_dir.replace(ModelFactory.Complete_Gradient_Boosting_Classifier_Name,
                                                ModelFactory.Gradient_Boosting_Classifier_Name)
    elif model_name == ModelFactory.Mean_Gradient_Boosting_Classifier_Name:
        checkpoint_dir = checkpoint_dir.replace(ModelFactory.Mean_Gradient_Boosting_Classifier_Name,
                                                ModelFactory.Gradient_Boosting_Classifier_Name)
    elif model_name == ModelFactory.Weighted_Gradient_Boosting_Classifier_Name:
        checkpoint_dir = checkpoint_dir.replace(ModelFactory.Weighted_Gradient_Boosting_Classifier_Name,
                                                ModelFactory.Gradient_Boosting_Classifier_Name)

    best_params_path = os.path.join(checkpoint_dir, "best_params.json")
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
    else:
        best_params = {}

    # Cap estimators to keep comparisons saner & faster (your original guard)
    if 'n_estimators' in best_params:
        best_params['n_estimators'] = min(best_params['n_estimators'], 100)

    # Build model
    input_size = X_train.shape[1]
    model, _loaded = ModelFactory.get_model(
        model_name,
        input_size,
        dataset_name=dataset_name,
        types_list=types_list,
        **best_params
    )

    # Fit with CPU monitor
    fit_stats = fit_model_with_monitor(model, model_name, X_train, y_train, X_val, y_val)

    # Predict with CPU monitor
    y_pred, y_prob, pred_stats = predict_with_monitor(model, model_name, X_val)

    # Metrics
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = float(accuracy_score(y_val, y_pred))
    auc = None
    if not is_multy_class and y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        try:
            auc = float(roc_auc_score(y_val, y_prob[:, 1]))
        except Exception:
            auc = None

    result = {
        "model": model_name,
        "accuracy": acc,
        "auc": auc,
        **fit_stats,
        **pred_stats,
    }
    return result


# -----------------------------
# Main Experiment
# -----------------------------

def save_cpu_profile_for_dataset(dataset_name: str, is_multy_class: bool, types_list):
    """
    Loads dataset, runs every model exactly once (train -> predict) and writes a CPU profile CSV.
    """
    # Load once (train split only; you were using train.csv)
    data_path = "../../data"
    train_path = f"{data_path}/{dataset_name}/train/data.csv"
    if not os.path.exists(train_path):
        print(f"[WARN] Train file not found for dataset '{dataset_name}': {train_path}")
        return

    train_df = pd.read_csv(train_path)

    # Consistent preprocessing (your utility)
    X_train, y_train, X_val, y_val = utils.preprocess_split(train_df)

    models_names = ModelFactory.MODELS

    rows = []
    for model_name in models_names:
        print(f"[{dataset_name}] Running model: {model_name}")
        try:
            res = run_single_model(
                model_name,
                dataset_name,
                X_train,
                y_train,
                X_val,
                y_val,
                is_multy_class,
                types_list
            )
        except Exception as e:
            # Don't break whole run if a single model fails
            print(f"[ERROR] {dataset_name} / {model_name} failed: {e}")
            res = {
                "model": model_name,
                "accuracy": None,
                "auc": None,
                "fit_avg_cpu": None,
                "fit_max_cpu": None,
                "fit_wall_s": None,
                "predict_avg_cpu": None,
                "predict_max_cpu": None,
                "predict_wall_s": None,
                "error": str(e),
            }
        rows.append(res)

    # Ensure output dir
    out_dir = f"{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Save one CSV with CPU + metrics
    df_out = pd.DataFrame(rows)
    df_out_cols = [
        "model",
        "fit_avg_cpu", "fit_max_cpu", "fit_wall_s",
        "predict_avg_cpu", "predict_max_cpu", "predict_wall_s",
        "accuracy", "auc", "error"
    ]
    # Add missing columns if error path skipped some
    for c in df_out_cols:
        if c not in df_out.columns:
            df_out[c] = None
    df_out = df_out[df_out_cols]
    df_out.to_csv(os.path.join(out_dir, "cpu_profile.csv"), index=False)
    print(f"[{dataset_name}] Saved CPU profile -> {out_dir}/cpu_profile.csv")


if __name__ == "__main__":
    # Optional: if you want to limit thread usage globally (uncomment as needed)
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"

    config_path = "../../datasets/config.json"
    with open(config_path) as f:
        config = json.load(f)

    datasets: list = config['datasets']

    for dataset in tqdm(datasets, desc="Datasets"):
        dataset_name_ = dataset['name']
        is_multy_class_ = dataset['is_multy_class']
        types_list = dataset.get('types_list', None)

        print(f"\n=== Dataset: {dataset_name_} ===")
        save_cpu_profile_for_dataset(dataset_name_, is_multy_class_, types_list)

    print("\nAll done. Check each <dataset>/cpu_profile.csv for per-model CPU usage.")
