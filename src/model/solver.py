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

from src.model.formulation import preprocess_and_engineer

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

        oof_preds = np.zeros(len(X))
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
            lgb = LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1)
            xgb = XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss', enable_categorical=False)
            cat = CatBoostClassifier(random_state=self.random_state, verbose=0)

            # Train models
            lgb.fit(X_train_clean, y_train)
            xgb.fit(X_train_clean, y_train)
            cat.fit(X_train_clean, y_train)

            # Predict probabilities
            p_lgb = lgb.predict_proba(X_val_clean)[:, 1]
            p_xgb = xgb.predict_proba(X_val_clean)[:, 1]
            p_cat = cat.predict_proba(X_val_clean)[:, 1]

            # Simple blending (Average)
            blend_preds = (p_lgb + p_xgb + p_cat) / 3.0

            # Store models for this fold
            self.fold_models.append({
                'lgb': lgb,
                'xgb': xgb,
                'cat': cat
            })

            oof_preds[val_idx] = blend_preds
            fold_auc = roc_auc_score(y_val, blend_preds)
            fold_scores.append(fold_auc)

        mean_auc = np.mean(fold_scores)
        return oof_preds, mean_auc

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
