"""
Kaggle Benchmark Evaluation Task for Predicting Smartphone Addiction (Kaggle S6E8).
Evaluates 10-fold CV OOF ROC-AUC, Gauss-Rank Logistic Stacking, KS-Drift stability,
and resource consumption in a deterministic, reproducible harness.
"""
import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.model.formulation import preprocess_and_engineer
from src.model.solver import CompetitionSolver, LogisticStacker, perform_ks_drift_screen, to_gauss_rank
from src.train import resolve_data_path

# Optional kbench decorator integration
try:
    import kbench
    task_decorator = kbench.task(name="s6e8_smartphone_addiction_benchmark")
except (ImportError, AttributeError):
    def task_decorator(func):
        return func


@task_decorator
def run_benchmark(n_splits: int = 10, proxy_sample: int = None, n_estimators: int = None) -> dict:
    """
    Executes the deterministic competition benchmark and returns structured performance metrics.
    """
    start_time = time.time()
    process = psutil.Process(os.getpid())
    init_mem_mb = process.memory_info().rss / (1024 * 1024)

    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)

    if proxy_sample and len(df) > proxy_sample:
        df = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)

    target_col = "addicted_label"
    X = df.drop(columns=["id", target_col], errors="ignore")
    y = df[target_col]

    # 1. Base Multi-Model Cross Validation
    solver = CompetitionSolver(
        n_splits=n_splits,
        random_state=42,
        use_neural_net=True,
        n_estimators=n_estimators
    )
    oof_preds_matrix, base_mean_auc = solver.cross_validate(X, y)

    # 2. Gauss-Rank Normal Percentiles
    rank_oof = np.zeros_like(oof_preds_matrix)
    for i in range(oof_preds_matrix.shape[1]):
        preds = oof_preds_matrix[:, i]
        percentiles = (scipy.stats.rankdata(preds) - 0.5) / len(preds)
        rank_oof[:, i] = to_gauss_rank(percentiles)

    # 3. Logistic Stacker with C=0.03
    stacker = LogisticStacker(C=0.03, random_state=42)
    stacker.fit(rank_oof, y.values)
    stacked_oof_preds = stacker.predict_proba(rank_oof)
    stacked_auc = roc_auc_score(y.values, stacked_oof_preds)

    # 4. Profiling and Stability Screening
    end_time = time.time()
    elapsed_sec = end_time - start_time
    peak_mem_mb = process.memory_info().rss / (1024 * 1024) - init_mem_mb

    results = {
        "dataset_rows": len(df),
        "n_splits": n_splits,
        "base_4way_oof_auc": float(base_mean_auc),
        "gauss_rank_stacked_auc": float(stacked_auc),
        "auc_uplift": float(stacked_auc - base_mean_auc),
        "stacker_coefficients": [float(c) for c in stacker.coef_],
        "stacker_intercept": float(stacker.intercept_),
        "elapsed_seconds": round(elapsed_sec, 2),
        "memory_delta_mb": round(peak_mem_mb, 2)
    }

    print("\n" + "=" * 60)
    print("🏆 KAGGLE BENCHMARK SUITE EXECUTION SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  • {k:25s}: {v}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    # Sanity benchmark on fast proxy sample
    run_benchmark(n_splits=3, proxy_sample=5000, n_estimators=10)
