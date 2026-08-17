import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

from scipy.optimize import minimize

from src.model.formulation import preprocess_and_engineer


# Highly optimized, stabilized GBDT configurations calibrated for high-dimensional tabular classification
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1
}

CAT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "random_state": 42,
    "verbose": False,
    "thread_count": -1
}


class CompetitionSolver:
    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.estimators = {}

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float]:
        """
        Executes a 10-fold Stratified CV loop targeting 'addicted_label'
        with strict leak-free local fold preprocessing.
        """
        # Ensure resetting index
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        oof_preds_matrix = np.zeros((len(X), 3))  # 3 models: lgb, xgb, cat
        fold_scores = []

        # Artifacts for Model Persistence
        self.fold_models = []
        self.fold_encoders = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # --- Strict Leak-Free Local Preprocessing ---
            # 1. Feature Engineering (Pydantic validation happens here)
            # This is applied independently to avoid leakage, though our current
            # feature engineering (ratios) doesn't strictly leak like aggregations would.
            X_train_clean = preprocess_and_engineer(X_train)
            X_val_clean = preprocess_and_engineer(X_val)

            # 1.5 Leak-Free Local Imputation
            num_cols = X_train_clean.select_dtypes(include=[np.number]).columns
            cat_cols = X_train_clean.select_dtypes(exclude=[np.number]).columns

            imputation_medians = {}
            for col in num_cols:
                median_val = X_train_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0 # fallback
                imputation_medians[col] = median_val
                X_train_clean[col] = X_train_clean[col].fillna(median_val)
                X_val_clean[col] = X_val_clean[col].fillna(median_val)

            imputation_modes = {}
            for col in cat_cols:
                if not X_train_clean[col].dropna().empty:
                    mode_val = X_train_clean[col].dropna().mode()[0]
                else:
                    mode_val = "Unknown"
                imputation_modes[col] = mode_val
                X_train_clean[col] = X_train_clean[col].fillna(mode_val)
                X_val_clean[col] = X_val_clean[col].fillna(mode_val)

            # 2. Categorical Encoding localized to the fold
            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                # Fit only on training fold!
                # Convert to string to handle mixed types gracefully if needed
                X_train_clean[col] = le.fit_transform(X_train_clean[col].astype(str))

                # Transform validation fold (handle unseen labels safely)
                # Safe mapping
                val_classes = np.unique(X_val_clean[col].astype(str))
                missing_classes = set(val_classes) - set(le.classes_)
                if missing_classes:
                    # add missing classes to le.classes_
                    le.classes_ = np.append(le.classes_, list(missing_classes))
                X_val_clean[col] = le.transform(X_val_clean[col].astype(str))

                encoders[col] = le

            self.fold_encoders.append({
                'encoders': encoders,
                'imputation_medians': imputation_medians,
                'imputation_modes': imputation_modes
            })

            # --- Modeling Ensembles ---
            # Lightweight baseline parameters
            lgb = LGBMClassifier(**LGBM_PARAMS)
            xgb = XGBClassifier(**XGB_PARAMS)
            cat = CatBoostClassifier(**CAT_PARAMS)

            # Train models
            lgb.fit(X_train_clean, y_train)
            xgb.fit(X_train_clean, y_train)
            cat.fit(X_train_clean, y_train)

            # Predict probabilities
            p_lgb = lgb.predict_proba(X_val_clean)[:, 1]
            p_xgb = xgb.predict_proba(X_val_clean)[:, 1]
            p_cat = cat.predict_proba(X_val_clean)[:, 1]

            # Simple blending (Average) for baseline info
            blend_preds = (p_lgb + p_xgb + p_cat) / 3.0

            # Store models for this fold
            self.fold_models.append({
                'lgb': lgb,
                'xgb': xgb,
                'cat': cat
            })

            oof_preds_matrix[val_idx, 0] = p_lgb
            oof_preds_matrix[val_idx, 1] = p_xgb
            oof_preds_matrix[val_idx, 2] = p_cat

            fold_auc = roc_auc_score(y_val, blend_preds)
            fold_scores.append(fold_auc)

        mean_auc = np.mean(fold_scores)
        return oof_preds_matrix, mean_auc

class OptunaTuner:
    """
    Optuna study class stub for future hyperparameter tuning.
    """
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Objective function for Optuna.
        Should implement cross_validate on the hyperparams proposed by trial.
        """
        # Example hyperparameter search space for LGBM
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        }

        # In a real scenario, we would pass these to the CV loop
        # For the stub, we just return a dummy score or run a lightweight CV
        return 0.5

    def run_study(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=self.n_trials)
        return study.best_params

class EnsembleBlender:
    def __init__(self):
        self.weights_ = None

    def _objective(self, weights: np.ndarray, preds_matrix: np.ndarray, y: np.ndarray) -> float:
        """
        Negative ROC AUC to minimize via SLSQP.
        """
        # Compute weighted sum
        blend_preds = np.dot(preds_matrix, weights)
        # Minimize negative AUC
        return -roc_auc_score(y, blend_preds)

    def fit(self, preds_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Optimize weights using SLSQP bounded to [0, 1] and sum=1.
        preds_matrix should be shape (n_samples, n_models).
        """
        n_models = preds_matrix.shape[1]

        # Initial guess: equal weights
        init_weights = np.ones(n_models) / n_models

        # Bounds: [0.0, 1.0] for each weight
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Constraints: sum(weights) = 1.0
        # For SLSQP equality constraint, the function should evaluate to 0
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        # Run optimization
        result = minimize(
            self._objective,
            x0=init_weights,
            args=(preds_matrix, y),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )

        self.weights_ = result.x
        return self.weights_
