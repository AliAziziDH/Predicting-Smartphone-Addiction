import os
import sys

# Dynamic path resolution to handle running from subfolders or root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT_DIR) in ["src", "tests"]:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import os
import numpy as np
import pandas as pd
import joblib
from src.model.formulation import preprocess_and_engineer

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
    print("Loading test data...")
    test_path = resolve_data_path("test.csv")

    df_test = pd.read_csv(test_path)

    # Store IDs for submission
    test_ids = df_test["id"].copy()
    X_test = df_test.drop(columns=["id"], errors="ignore")

    print(f"Test shape: {X_test.shape}")

    models_dir = "models"
    artifact_path = os.path.join(models_dir, "ensemble_pipeline.joblib")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Pipeline artifact not found at {artifact_path}. Did you run train.py?")

    print("Loading ensemble pipeline artifact...")
    artifact = joblib.load(artifact_path)

    fold_models = artifact['fold_models']
    fold_encoders = artifact['fold_encoders']
    ensemble_weights = np.array(artifact.get('ensemble_weights', [1/3, 1/3, 1/3]))

    num_folds = len(fold_models)
    print(f"Loaded {num_folds} folds from artifact.")

    # Initialize matrices to store predictions from all folds for each model
    p_lgb_folds = np.zeros((len(X_test), num_folds))
    p_xgb_folds = np.zeros((len(X_test), num_folds))
    p_cat_folds = np.zeros((len(X_test), num_folds))

    # Process each fold
    for fold in range(num_folds):
        print(f"Processing Fold {fold + 1}/{num_folds}...")

        # 1. Feature Engineering (independent of train stats, safe to apply directly)
        X_test_clean = preprocess_and_engineer(X_test)

        # Extract fold-specific artifacts
        fold_artifacts = fold_encoders[fold]
        encoders = fold_artifacts['encoders']
        target_encoder = fold_artifacts['target_encoder']
        # No imputation logic needed

        X_test_encoded = target_encoder.transform(X_test_clean, cols=list(encoders.keys()))

        # 3. Categorical Encoding (apply fold encoders safely)
        for col, le in encoders.items():
            if col in X_test_encoded.columns:
                # To be absolutely safe from np.nan or pd.NA
                test_series = X_test_encoded[col].fillna('Missing').astype(str)

                # Handle unseen labels in test set safely
                test_classes = list(set(test_series.tolist()))
                missing_classes = set(test_classes) - set(le.classes_)
                if missing_classes:
                    # append unseen classes to the encoder classes to prevent ValueError
                    le.classes_ = np.append(le.classes_, list(missing_classes))
                X_test_encoded[col] = le.transform(test_series)

        # 4. Generate Predictions from fold models
        models = fold_models[fold]
        lgb = models['lgb']
        xgb = models['xgb']
        cat = models['cat']

        p_lgb_folds[:, fold] = lgb.predict_proba(X_test_encoded)[:, 1]
        p_xgb_folds[:, fold] = xgb.predict_proba(X_test_encoded)[:, 1]
        p_cat_folds[:, fold] = cat.predict_proba(X_test_encoded)[:, 1]

    # Average predictions across folds for each model
    p_lgb_mean = np.mean(p_lgb_folds, axis=1)
    p_xgb_mean = np.mean(p_xgb_folds, axis=1)
    p_cat_mean = np.mean(p_cat_folds, axis=1)


    # Combine into a single matrix
    test_preds_matrix = np.column_stack((p_lgb_mean, p_xgb_mean, p_cat_mean))

    print("Converting test predictions to rank percentiles...")
    import scipy.stats
    for i in range(test_preds_matrix.shape[1]):
        preds = test_preds_matrix[:, i]
        test_preds_matrix[:, i] = (scipy.stats.rankdata(preds) - 0.5) / len(preds)

    # Apply global optimized weights

    print(f"Applying global ensemble weights: {ensemble_weights}")
    final_preds = np.dot(test_preds_matrix, ensemble_weights)

    print("Generating submission file...")
    submission = pd.DataFrame({
        "id": test_ids,
        "addicted_label": final_preds
    })

    # Strict programmatic sanity checks
    assert submission.shape[0] == df_test.shape[0], "Shape mismatch: submission rows != test rows"
    assert submission.shape[1] == 2, "Submission must have exactly 2 columns"
    assert not submission.isnull().values.any(), "Submission contains NaN values"
    assert submission['addicted_label'].min() >= 0.0, "Probabilities < 0.0 found"
    assert submission['addicted_label'].max() <= 1.0, "Probabilities > 1.0 found"

    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    sub_path = os.path.join(outputs_dir, "submission.csv")

    submission.to_csv(sub_path, index=False)
    print(f"Sanity checks passed. Final submission saved to {sub_path}")

if __name__ == "__main__":
    main()
