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
from src.model.formulation import preprocess_and_engineer
from src.model.solver import perform_ks_drift_screen

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
            print(f"[INFO] Successfully resolved {filename} to: {path}")
            return path

    search_roots = ["/kaggle/input", "../input", "data", "."]
    for root_dir in search_roots:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if filename in files:
                    found = os.path.join(root, filename)
                    print(f"[INFO] Found {filename} via walk: {found}")
                    return found

    raise FileNotFoundError(f"Could not find {filename} anywhere in {search_roots}")

def main():
    print("Loading test data...")
    test_path = resolve_data_path("test.csv")

    df_test = pd.read_csv(test_path)

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
    stacker = artifact.get('stacker')
    oof_ranks = artifact.get('oof_ranks')

    num_folds = len(fold_models)
    has_nn = 'nn' in fold_models[0]
    num_models = 4 if has_nn else 3
    print(f"Loaded {num_folds} folds ({num_models}-way modeling) from artifact.")

    p_lgb_folds = np.zeros((len(X_test), num_folds))
    p_xgb_folds = np.zeros((len(X_test), num_folds))
    p_cat_folds = np.zeros((len(X_test), num_folds))
    if has_nn:
        p_nn_folds = np.zeros((len(X_test), num_folds))

    for fold in range(num_folds):
        print(f"Processing Fold {fold + 1}/{num_folds}...")

        X_test_clean = preprocess_and_engineer(X_test)

        fold_artifacts = fold_encoders[fold]
        encoders = fold_artifacts['encoders']

        for col, le in encoders.items():
            if col in X_test_clean.columns:
                test_series = X_test_clean[col].fillna('Missing').astype(str)
                test_classes = list(set(test_series.tolist()))
                missing_classes = set(test_classes) - set(le.classes_)
                if missing_classes:
                    le.classes_ = np.append(le.classes_, list(missing_classes))
                X_test_clean[col] = le.transform(test_series)

        models = fold_models[fold]
        p_lgb_folds[:, fold] = models['lgb'].predict_proba(X_test_clean)[:, 1]
        p_xgb_folds[:, fold] = models['xgb'].predict_proba(X_test_clean)[:, 1]
        p_cat_folds[:, fold] = models['cat'].predict_proba(X_test_clean)[:, 1]
        if has_nn:
            p_nn_folds[:, fold] = models['nn'].predict_proba(X_test_clean)[:, 1]

    p_lgb_mean = np.mean(p_lgb_folds, axis=1)
    p_xgb_mean = np.mean(p_xgb_folds, axis=1)
    p_cat_mean = np.mean(p_cat_folds, axis=1)

    if has_nn:
        p_nn_mean = np.mean(p_nn_folds, axis=1)
        test_preds_matrix = np.column_stack((p_lgb_mean, p_xgb_mean, p_cat_mean, p_nn_mean))
    else:
        test_preds_matrix = np.column_stack((p_lgb_mean, p_xgb_mean, p_cat_mean))

    print("Converting test predictions to rank percentiles...")
    rank_test = np.zeros_like(test_preds_matrix)
    for i in range(test_preds_matrix.shape[1]):
        preds = test_preds_matrix[:, i]
        rank_test[:, i] = (scipy.stats.rankdata(preds) - 0.5) / len(preds)

    # Kolmogorov-Smirnov Drift Screening
    if oof_ranks is not None:
        model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'PyTorch NN'][:test_preds_matrix.shape[1]]
        for i, name in enumerate(model_names):
            passed, stat = perform_ks_drift_screen(oof_ranks[:, i], rank_test[:, i])
            status_str = "PASSED ✅" if passed else "WARNING ⚠️"
            print(f"[KS-Drift Screen] {name}: stat={stat:.4f} -> {status_str}")

    if stacker is not None:
        print("Applying Nested Logistic Stacker...")
        final_preds = stacker.predict_proba(rank_test)
    else:
        final_preds = np.mean(rank_test, axis=1)

    print("Generating submission file...")
    submission = pd.DataFrame({
        "id": test_ids,
        "addicted_label": final_preds
    })

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
