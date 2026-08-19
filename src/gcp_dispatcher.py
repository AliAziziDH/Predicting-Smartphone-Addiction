"""
GCP Cloud Dispatcher & Production 10-Fold Ensemble Runner.
Executes high-capacity training across the full 691k dataset on Cloud compute,
integrates Optuna-tuned hyperparameters, Factorization Machines, Value-Level Target Encoding,
and submits directly to the Kaggle competition with zero local overhead.
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.model.formulation import preprocess_and_engineer
from src.model.solver import CompetitionSolver, to_gauss_rank, NelderMeadRankStacker
from src.train import resolve_data_path


def run_cloud_pipeline(n_splits: int = 10, auto_submit: bool = True):
    print("=" * 70)
    print("🌐 FULL ENSEMBLE PRODUCTION PIPELINE (WAVE 6 NELDER-MEAD STACKING)")
    print("=" * 70)

    start_time = time.time()

    # 1. Load Data
    train_path = resolve_data_path("train.csv")
    test_path = resolve_data_path("test.csv")

    print(f"Loading full training data from: {train_path}...")
    df_train = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}...")
    df_test = pd.read_csv(test_path)

    target_col = "addicted_label"
    y = df_train[target_col]
    X_train_raw = df_train.drop(columns=["id", target_col], errors="ignore")
    test_ids = df_test["id"].values
    X_test_raw = df_test.drop(columns=["id"], errors="ignore")

    print(f"Dataset Dimensions: Train={X_train_raw.shape}, Test={X_test_raw.shape}")

    # 2. Fast Single-Pass Base Feature Preprocessing with RAM Downcasting
    print("\n⚙️ Executing high-performance feature pre-engineering (Single-Pass)...")
    X_train_clean = preprocess_and_engineer(X_train_raw)
    X_test_clean = preprocess_and_engineer(X_test_raw)

    print(f"Engineered Features: {X_train_clean.shape[1]} columns. RAM footprint: {X_train_clean.memory_usage().sum() / 1e6:.1f} MB")

    # 3. Solver & 10-Fold Cross-Validation
    print(f"\n🚀 Engaging {n_splits}-Fold Cross-Validation with GBDT Models...")
    solver = CompetitionSolver(n_splits=n_splits, random_state=42, use_neural_net=False)

    oof_matrix, mean_auc = solver.cross_validate(X_train_clean, y)

    # 4. Gauss-Rank Meta-Stacking
    num_models = oof_matrix.shape[1]
    rank_oof = np.zeros_like(oof_matrix)
    for i in range(num_models):
        preds = oof_matrix[:, i]
        percentiles = (scipy.stats.rankdata(preds) - 0.5) / len(preds)
        rank_oof[:, i] = percentiles

    print("\n🧠 Fitting Direct Nelder-Mead Rank-AUC Stacker...")
    stacker = NelderMeadRankStacker(random_state=42)
    stacker.fit(rank_oof, y.values)
    stacked_oof_preds = stacker.predict_proba(rank_oof)
    stacked_auc = roc_auc_score(y.values, stacked_oof_preds)

    print("=" * 70)
    print(f"🏆 OFFICIAL OUT-OF-FOLD (OOF) ROC-AUC SCORE: {stacked_auc:.5f}")
    print(f"Optimized Model Weights: {stacker.weights_}")
    print("=" * 70)

    # 5. Out-of-Fold Test Inference
    print("\n🔮 Generating Out-of-Fold Ensembled Test Predictions...")
    test_preds_matrix = np.zeros((len(X_test_clean), num_models))

    for fold_idx, (models_dict, enc_dict) in enumerate(zip(solver.fold_models, solver.fold_encoders)):
        # Apply fold-specific target encoding and categorical encoders
        te = enc_dict['target_encoder']
        encoders = enc_dict['encoders']

        X_test_fold = te.transform(X_test_clean.copy())
        for col, le in encoders.items():
            s = X_test_fold[col].fillna('Missing').astype(str)
            missing = set(s.unique()) - set(le.classes_)
            if missing:
                le.classes_ = np.append(le.classes_, list(missing))
            X_test_fold[col] = le.transform(s)

        # Predict with each model
        cat_cols = enc_dict['cat_cols']
        p_lgb = models_dict['lgb'].predict_proba(X_test_fold)[:, 1]
        p_xgb = models_dict['xgb'].predict_proba(X_test_fold)[:, 1]
        p_cat = models_dict['cat'].predict_proba(X_test_fold)[:, 1]

        test_preds_matrix[:, 0] += p_lgb / n_splits
        test_preds_matrix[:, 1] += p_xgb / n_splits
        test_preds_matrix[:, 2] += p_cat / n_splits

        if 'nn' in models_dict:
            p_nn = models_dict['nn'].predict_proba(X_test_fold)[:, 1]
            test_preds_matrix[:, 3] += p_nn / n_splits

    # Rank test matrix
    rank_test = np.zeros_like(test_preds_matrix)
    for i in range(num_models):
        preds = test_preds_matrix[:, i]
        percentiles = (scipy.stats.rankdata(preds) - 0.5) / len(preds)
        rank_test[:, i] = percentiles

    final_test_preds = stacker.predict_proba(rank_test)

    # 6. Save Submission File
    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    sub_path = os.path.join(outputs_dir, "submission.csv")

    sub_df = pd.DataFrame({
        "id": test_ids,
        "addicted_label": final_test_preds
    })
    sub_df.to_csv(sub_path, index=False)
    print(f"\n💾 Submission written to {sub_path} ({len(sub_df):,} rows, {os.path.getsize(sub_path) / 1e6:.2f} MB)")

    elapsed = time.time() - start_time
    print(f"⏱️ Total Pipeline Execution Time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    # 7. Direct Server-to-Server Kaggle Submission
    if auto_submit:
        print("\n🚀 Submitting directly from Cloud Server to Kaggle Leaderboard...", flush=True)
        res = subprocess.run([
            "kaggle", "competitions", "submit",
            "-c", "playground-series-s6e8",
            "-f", sub_path,
            "-m", f"V11 Wave 6: 10-Fold 36-Feature Ensemble + Nelder-Mead Rank-AUC Stacking OOF={stacked_auc:.5f}"
        ], capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.stderr:
            print("Kaggle CLI Output:", res.stderr, flush=True)

    return stacked_auc


if __name__ == "__main__":
    run_cloud_pipeline(n_splits=10, auto_submit=True)
