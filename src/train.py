import os
import sys

# Dynamic path resolution to handle running from subfolders, root, or notebook
try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(ROOT_DIR) in ["src", "tests"]:
        ROOT_DIR = os.path.dirname(ROOT_DIR)
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

import os
import numpy as np
import pandas as pd
import scipy.stats
import joblib
from sklearn.metrics import roc_auc_score
from src.model.solver import CompetitionSolver, LogisticStacker, EnsembleBlender, to_gauss_rank

def resolve_data_path(filename):
    paths_to_check = [
        f"/kaggle/input/playground-series-s6e8/{filename}",
        f"/kaggle/input/competitions/playground-series-s6e8/{filename}",
        f"../input/playground-series-s6e8/{filename}",
        f"data/{filename}",
        f"./{filename}",
        f"../{filename}"
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"[INFO] Successfully resolved {filename} to: {path}", flush=True)
            return path

    search_roots = ["/kaggle/input", "../input", "data", "."]
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if filename in files:
                    found = os.path.join(root, filename)
                    print(f"[INFO] Found {filename} via walk: {found}", flush=True)
                    return found

    raise FileNotFoundError(f"Could not find {filename} anywhere in {search_roots}")

def main():
    print("Loading training data...", flush=True)
    train_path = resolve_data_path("train.csv")

    df_train = pd.read_csv(train_path)

    target_col = "addicted_label"
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    X = df_train.drop(columns=["id", target_col], errors="ignore")
    y = df_train[target_col]

    print(f"Training shapes -> X: {X.shape}, y: {y.shape}", flush=True)

    # Initialize 10-fold CV with 4-way modeling
    solver = CompetitionSolver(n_splits=10, random_state=42, use_neural_net=True)

    print("Starting 10-fold Stratified Cross-Validation (LGB + XGB + CAT + PyTorch NN)...", flush=True)
    oof_preds_matrix, mean_auc = solver.cross_validate(X, y)

    print(f"==================================================", flush=True)
    print(f"Baseline (Average 4-Way) OOF ROC AUC Score: {mean_auc:.5f}", flush=True)
    print(f"==================================================", flush=True)

    print("Converting OOF predictions to Gauss-Rank normal percentiles...", flush=True)
    rank_oof = np.zeros_like(oof_preds_matrix)
    for i in range(oof_preds_matrix.shape[1]):
        preds = oof_preds_matrix[:, i]
        percentiles = (scipy.stats.rankdata(preds) - 0.5) / len(preds)
        rank_oof[:, i] = to_gauss_rank(percentiles)

    print("Fitting Nested Logistic Stacker on 4-Way Gauss-Rank Percentiles...", flush=True)
    stacker = LogisticStacker(C=0.03, random_state=42)
    stacker.fit(rank_oof, y.values)

    stacked_oof_preds = stacker.predict_proba(rank_oof)
    stacked_auc = roc_auc_score(y.values, stacked_oof_preds)

    print(f"==================================================", flush=True)
    print(f"🚀 Version 6 Logistic Stack OOF ROC AUC Score: {stacked_auc:.5f}", flush=True)
    print(f"Stacker Coefficients [LGB, XGB, CAT, NN]: {stacker.coef_}", flush=True)
    print(f"Stacker Intercept: {stacker.intercept_:.5f}", flush=True)
    print(f"==================================================", flush=True)

    # Save artifact
    artifact = {
        'fold_models': solver.fold_models,
        'fold_encoders': solver.fold_encoders,
        'stacker': stacker,
        'oof_ranks': rank_oof
    }

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, "ensemble_pipeline.joblib")

    print(f"Saving ensemble pipeline artifact to {artifact_path}...", flush=True)
    joblib.dump(artifact, artifact_path)
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
