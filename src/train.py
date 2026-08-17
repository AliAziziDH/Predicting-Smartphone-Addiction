import os
import pandas as pd
import joblib
from src.model.solver import CompetitionSolver, EnsembleBlender

def resolve_data_path(filename):
    paths_to_check = [
        f"/kaggle/input/playground-series-s6e8/{filename}",
        f"../input/playground-series-s6e8/{filename}",
        f"data/{filename}"
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"[INFO] Successfully resolved {filename} to: {path}")
            return path
    raise FileNotFoundError(f"Could not find {filename} in any of the expected locations: {paths_to_check}")

def main():
    print("Loading training data...")
    train_path = resolve_data_path("train.csv")

    df_train = pd.read_csv(train_path)

    # The target column is addicted_label
    target_col = "addicted_label"
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    X = df_train.drop(columns=["id", target_col], errors="ignore")
    y = df_train[target_col]

    print(f"Training shapes -> X: {X.shape}, y: {y.shape}")

    # Initialize the solver
    solver = CompetitionSolver(n_splits=10, random_state=42)

    print("Starting 10-fold Stratified Cross-Validation...")
    oof_preds_matrix, mean_auc = solver.cross_validate(X, y)

    print(f"==================================================")
    print(f"Baseline (Average) OOF ROC AUC Score: {mean_auc:.4f}")
    print(f"==================================================")


    print("Converting OOF predictions to rank percentiles...")
    import scipy.stats

    # rankdata(preds) - 0.5 / len(preds) applied column-wise
    for i in range(oof_preds_matrix.shape[1]):
        preds = oof_preds_matrix[:, i]
        oof_preds_matrix[:, i] = (scipy.stats.rankdata(preds) - 0.5) / len(preds)

    print("Running Global SLSQP Optimization on OOF Predictions...")

    blender = EnsembleBlender()
    optimal_weights = blender.fit(oof_preds_matrix, y.values)

    # Calculate Optimized OOF AUC
    import numpy as np
    from sklearn.metrics import roc_auc_score
    optimized_oof_preds = np.dot(oof_preds_matrix, optimal_weights)
    optimized_auc = roc_auc_score(y.values, optimized_oof_preds)

    print(f"==================================================")
    print(f"Optimized Global OOF ROC AUC Score: {optimized_auc:.4f}")
    print(f"Optimal Weights [LGB, XGB, CAT]: {optimal_weights}")
    print(f"==================================================")

    # Save the pipeline artifact (Option B strategy)
    artifact = {
        'fold_models': solver.fold_models,
        'fold_encoders': solver.fold_encoders,
        'ensemble_weights': optimal_weights.tolist()
    }

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, "ensemble_pipeline.joblib")

    print(f"Saving ensemble pipeline artifact to {artifact_path}...")
    joblib.dump(artifact, artifact_path)
    print("Done!")

if __name__ == "__main__":
    main()
