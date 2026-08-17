import os
import json
import re

def create_notebook_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
    }

def clean_code(filepath):
    with open(filepath, 'r') as f:
        code = f.read()

    # Strip out local relative imports
    code = re.sub(r'from src\.model\.formulation import .*\n', '', code)
    code = re.sub(r'from src\.model\.solver import .*\n', '', code)
    code = re.sub(r'from src\.train import .*\n', '', code)
    code = re.sub(r'from src\.predict import .*\n', '', code)
    code = re.sub(r'import src\.model\.formulation\n', '', code)
    code = re.sub(r'import src\.model\.solver\n', '', code)
    code = re.sub(r'import src\.train\n', '', code)
    code = re.sub(r'import src\.predict\n', '', code)
    return code.strip()

def compile_notebook(output_path):
    cells = []

    # 1. Header Markdown & Setup
    cells.append(create_notebook_cell("markdown", """# Smartphone Addiction Prediction - Elite Ensemble
## Deterministic AI Engineering

This notebook implements an automated pipeline designed for the Kaggle Playground Series s6e8 competition. It combines extreme gradient boosting (LightGBM, XGBoost, CatBoost) with a SciPy SLSQP-based Ensemble Blender to maximize Out-of-Fold (OOF) ROC AUC.

### Setup and Imports"""))

    setup_code = """import os
import gc
import warnings
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from pydantic import BaseModel, Field, ValidationError
from scipy.optimize import minimize
from scipy.stats import rankdata

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# Dynamic Kaggle vs Local Path Resolution
train_path = "/kaggle/input/playground-series-s6e8/train.csv"
if not os.path.exists(train_path):
    train_path = "../input/playground-series-s6e8/train.csv"
if not os.path.exists(train_path):
    train_path = "data/train.csv"  # Local fallback

test_path = "/kaggle/input/playground-series-s6e8/test.csv"
if not os.path.exists(test_path):
    test_path = "../input/playground-series-s6e8/test.csv"
if not os.path.exists(test_path):
    test_path = "data/test.csv"  # Local fallback

# Ensure reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)"""
    cells.append(create_notebook_cell("code", setup_code))

    # 2. Predictive Layer & Features
    cells.append(create_notebook_cell("markdown", r"""### Predictive Layer & Features

#### Separation of Concerns (SoC)
Our predictive layer applies transformations deterministically via strict schema validation (`Pydantic`). By separating raw data ingestion from engineered features, we encapsulate noisy estimations securely.

#### 6 Advanced Interaction Features Formulations
1. **Sleep Deficit**:
$$ Sleep\ Deficit = max(0, 8 - \text{Sleep Duration (hours)}) $$

2. **Distraction Ratio**:
$$ Distraction\ Ratio = \frac{\text{Social Media Usage (hours)} + \text{Gaming (hours)}}{\text{Total App Usage (hours)} + \epsilon} $$

3. **Notification Intensity**:
$$ Notification\ Intensity = \frac{\text{Notifications Received}}{\text{Total App Usage (hours)} + \epsilon} $$

4. **Productivity Balance**:
$$ Productivity\ Balance = \frac{\text{Productivity (hours)}}{\text{Social Media Usage (hours)} + \text{Gaming (hours)} + \epsilon} $$

5. **Screen Time Proportion**:
$$ Screen\ Time\ Proportion = \frac{\text{Total App Usage (hours)}}{24} $$

6. **Age-Screen Time Interaction**:
$$ Age\times Screen\ Time = \text{Age} \times \text{Total App Usage (hours)} $$"""))
    cells.append(create_notebook_cell("code", clean_code('src/model/formulation.py')))

    # 3. Preprocessing, Local-Fold Imputers, and SciPy SLSQP EnsembleBlender
    cells.append(create_notebook_cell("markdown", r"""### Preprocessing, Local-Fold Imputers & SLSQP Ensemble Blender

#### Local-Fold Preprocessing
To strictly prevent target leakage, all preprocessing steps—including median imputation for numericals, mode imputation for categoricals, and label encoding—are executed exclusively within local CV folds.

#### SciPy SLSQP Ensemble Blender
Rather than simple averaging, we optimize weights by minimizing the negative Out-of-Fold (OOF) ROC AUC using SciPy's Sequential Least Squares Programming (SLSQP).

**Objective**:
$$ \min_{w} -ROC\_AUC(y, \sum_{i} w_i p_i) $$

**Constraints**:
$$ \sum_{i} w_i = 1.0, \quad w_i \in [0.0, 1.0] $$

**Transformation**:
We apply intra-test ranking using `scipy.stats.rankdata` percentile ranking before averaging to normalize score distributions:
$$ \hat{p} = \frac{rankdata(p) - 0.5}{N} $$"""))
    cells.append(create_notebook_cell("code", clean_code('src/model/solver.py')))

    # 4. 10-Fold Stratified CV training logic
    cells.append(create_notebook_cell("markdown", """### Training Loop (10-Fold Stratified CV)"""))
    cells.append(create_notebook_cell("code", clean_code('src/train.py')))

    # 5. Inference, intra-test ranking, and submission formatting
    cells.append(create_notebook_cell("markdown", """### Inference and Submission Formatting"""))
    cells.append(create_notebook_cell("code", clean_code('src/predict.py')))

    # Add a final execution block if the user runs the notebook from start to finish
    cells.append(create_notebook_cell("code", """if __name__ == '__main__':
    print("Executing Kaggle Notebook Pipeline...")

    # Train
    # We call main from train.py logic (but without os.path dependencies and using the global train_path/test_path)
    import scipy.stats

    print("Loading training data...")
    df_train = pd.read_csv(train_path)

    # The target column is addicted_label
    target_col = "addicted_label"
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

    # rankdata(preds) - 0.5 / len(preds) applied column-wise
    for i in range(oof_preds_matrix.shape[1]):
        preds = oof_preds_matrix[:, i]
        oof_preds_matrix[:, i] = (scipy.stats.rankdata(preds) - 0.5) / len(preds)

    print("Running Global SLSQP Optimization on OOF Predictions...")

    blender = EnsembleBlender()
    optimal_weights = blender.fit(oof_preds_matrix, y.values)

    optimized_oof_preds = np.dot(oof_preds_matrix, optimal_weights)
    optimized_auc = roc_auc_score(y.values, optimized_oof_preds)

    print(f"==================================================")
    print(f"Optimized Global OOF ROC AUC Score: {optimized_auc:.4f}")
    print(f"Optimal Weights [LGB, XGB, CAT]: {optimal_weights}")
    print(f"==================================================")

    artifact = {
        'fold_models': solver.fold_models,
        'fold_encoders': solver.fold_encoders,
        'ensemble_weights': optimal_weights.tolist()
    }

    # Predict
    print("Loading test data...")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=["id"], errors="ignore")

    # We duplicate the predict.py logic here
    fold_models = artifact['fold_models']
    fold_encoders = artifact['fold_encoders']
    ensemble_weights = np.array(artifact['ensemble_weights'])

    n_folds = len(fold_models)

    # Pre-allocate array for all fold predictions
    all_fold_preds = np.zeros((len(X_test), n_folds))

    print(f"Generating predictions across {n_folds} folds...")

    for i in range(n_folds):
        models = fold_models[i]
        encoders_data = fold_encoders[i]

        # --- Local Preprocessing for this specific fold ---
        X_test_clean = preprocess_and_engineer(X_test.copy())

        # Imputation
        num_cols = X_test_clean.select_dtypes(include=[np.number]).columns
        cat_cols = X_test_clean.select_dtypes(exclude=[np.number]).columns

        for col in num_cols:
            median_val = encoders_data['imputation_medians'][col]
            X_test_clean[col] = X_test_clean[col].fillna(median_val)

        for col in cat_cols:
            mode_val = encoders_data['imputation_modes'][col]
            X_test_clean[col] = X_test_clean[col].fillna(mode_val)

        # Encoding
        encoders = encoders_data['encoders']
        for col in cat_cols:
            le = encoders[col]

            # Safe mapping for unseen classes in test set during inference
            val_classes = np.unique(X_test_clean[col].astype(str))
            missing_classes = set(val_classes) - set(le.classes_)
            if missing_classes:
                # Need a fallback for inference. We can't safely append to classes_ here without changing indices.
                # Standard practice is to map to a known category (like 'Unknown' mode).
                # But since our encoders safely mapped unseen during validation by expanding, let's expand.
                le.classes_ = np.append(le.classes_, list(missing_classes))

            X_test_clean[col] = le.transform(X_test_clean[col].astype(str))

        # --- Inference ---
        p_lgb = models['lgb'].predict_proba(X_test_clean)[:, 1]
        p_xgb = models['xgb'].predict_proba(X_test_clean)[:, 1]
        p_cat = models['cat'].predict_proba(X_test_clean)[:, 1]

        # Intra-test rank prediction logic for base models
        p_lgb = (scipy.stats.rankdata(p_lgb) - 0.5) / len(p_lgb)
        p_xgb = (scipy.stats.rankdata(p_xgb) - 0.5) / len(p_xgb)
        p_cat = (scipy.stats.rankdata(p_cat) - 0.5) / len(p_cat)

        preds_matrix = np.column_stack((p_lgb, p_xgb, p_cat))

        # Apply optimal ensemble weights
        blend_preds = np.dot(preds_matrix, ensemble_weights)

        all_fold_preds[:, i] = blend_preds

    print("Averaging across folds...")
    # Final fold averaging
    final_preds = np.mean(all_fold_preds, axis=1)

    # Final rank normalization (optional but good practice)
    final_preds = (scipy.stats.rankdata(final_preds) - 0.5) / len(final_preds)

    print("Formatting submission...")
    submission = pd.DataFrame({
        "id": df_test["id"],
        "addicted_label": final_preds
    })

    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Successfully generated {submission_path}!")
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == '__main__':
    compile_notebook('outputs/predicting-smartphone-addiction-elite.ipynb')
    print("Notebook compiled successfully at outputs/predicting-smartphone-addiction-elite.ipynb")
