import os
import numpy as np
import pandas as pd
import joblib
from src.model.formulation import preprocess_and_engineer

def main():
    print("Loading test data...")
    test_path = os.path.join("data", "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

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

    num_folds = len(fold_models)
    print(f"Loaded {num_folds} folds from artifact.")

    # Initialize array to store predictions from all folds
    final_preds = np.zeros(len(X_test))

    # Process each fold
    for fold in range(num_folds):
        print(f"Processing Fold {fold + 1}/{num_folds}...")

        # 1. Feature Engineering (independent of train stats, safe to apply directly)
        X_test_clean = preprocess_and_engineer(X_test)

        # Extract fold-specific artifacts
        fold_artifacts = fold_encoders[fold]
        encoders = fold_artifacts['encoders']
        imputation_medians = fold_artifacts['imputation_medians']
        imputation_modes = fold_artifacts['imputation_modes']

        # 2. Leak-Free Local Imputation (apply fold stats)
        for col, median_val in imputation_medians.items():
            if col in X_test_clean.columns:
                X_test_clean[col] = X_test_clean[col].fillna(median_val)

        for col, mode_val in imputation_modes.items():
            if col in X_test_clean.columns:
                X_test_clean[col] = X_test_clean[col].fillna(mode_val)

        # 3. Categorical Encoding (apply fold encoders safely)
        for col, le in encoders.items():
            if col in X_test_clean.columns:
                # Handle unseen labels in test set safely
                test_classes = np.unique(X_test_clean[col].astype(str))
                missing_classes = set(test_classes) - set(le.classes_)
                if missing_classes:
                    # append unseen classes to the encoder classes to prevent ValueError
                    le.classes_ = np.append(le.classes_, list(missing_classes))
                X_test_clean[col] = le.transform(X_test_clean[col].astype(str))

        # 4. Generate Predictions from fold models
        models = fold_models[fold]
        lgb = models['lgb']
        xgb = models['xgb']
        cat = models['cat']

        p_lgb = lgb.predict_proba(X_test_clean)[:, 1]
        p_xgb = xgb.predict_proba(X_test_clean)[:, 1]
        p_cat = cat.predict_proba(X_test_clean)[:, 1]

        # Blend (Average) for this fold
        fold_blend = (p_lgb + p_xgb + p_cat) / 3.0

        # Add to final predictions array
        final_preds += fold_blend

    # Average across all folds
    final_preds /= num_folds

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
