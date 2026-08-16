import os
import pandas as pd
import joblib
from src.model.solver import CompetitionSolver

def main():
    print("Loading training data...")
    train_path = os.path.join("data", "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

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
    oof_preds, mean_auc = solver.cross_validate(X, y)

    print(f"==================================================")
    print(f"Global OOF ROC AUC Score: {mean_auc:.4f}")
    print(f"==================================================")

    # Save the pipeline artifact (Option B strategy)
    artifact = {
        'fold_models': solver.fold_models,
        'fold_encoders': solver.fold_encoders
    }

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, "ensemble_pipeline.joblib")

    print(f"Saving ensemble pipeline artifact to {artifact_path}...")
    joblib.dump(artifact, artifact_path)
    print("Done!")

if __name__ == "__main__":
    main()
