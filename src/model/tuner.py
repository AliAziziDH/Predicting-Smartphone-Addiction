import json
import logging
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

from src.model.formulation import preprocess_and_engineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeakFreeOptunaTuner:
    """
    Optuna hyperparameter tuner implementing safe local fold preprocessing
    and proxy downsampling to prevent target leakage.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42, downsample_ratio: float = 0.3):
        self.n_splits = n_splits
        self.random_state = random_state
        self.downsample_ratio = downsample_ratio

    def _stratified_downsample(self, X: pd.DataFrame, y: pd.Series, ratio: float = 0.3) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extracts a representative proxy subset of data, preserving the exact target class distribution.
        """
        # train_test_split with stratify=y preserves the target class distribution
        # we take 'ratio' proportion for training (the proxy dataset)
        _, X_proxy, _, y_proxy = train_test_split(
            X, y,
            test_size=ratio,
            stratify=y,
            random_state=self.random_state
        )
        return X_proxy, y_proxy


    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, model_type: str) -> float:
        """
        Objective function for Optuna.
        Evaluates cross-validation on the hyperparams proposed by trial.
        Pre-processing is done locally on training folds to prevent Target Leakage.
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []

        params = self._get_search_space(trial, model_type)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # --- Strict Leak-Free Local Preprocessing ---
            # 1. Feature Engineering
            X_train_clean = preprocess_and_engineer(X_train)
            X_val_clean = preprocess_and_engineer(X_val)

            # 1.5 Leak-Free Local Imputation
            num_cols = X_train_clean.select_dtypes(include=[np.number]).columns
            cat_cols = X_train_clean.select_dtypes(exclude=[np.number]).columns

            for col in num_cols:
                median_val = X_train_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_train_clean[col] = X_train_clean[col].fillna(median_val)
                X_val_clean[col] = X_val_clean[col].fillna(median_val)

            for col in cat_cols:
                if not X_train_clean[col].dropna().empty:
                    mode_val = X_train_clean[col].dropna().mode()[0]
                else:
                    mode_val = "Unknown"
                X_train_clean[col] = X_train_clean[col].fillna(mode_val)
                X_val_clean[col] = X_val_clean[col].fillna(mode_val)

            # 2. Categorical Encoding localized to the fold
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            if len(cat_cols) > 0:
                X_train_clean[cat_cols] = oe.fit_transform(X_train_clean[cat_cols].astype(str))
                X_val_clean[cat_cols] = oe.transform(X_val_clean[cat_cols].astype(str))

            # --- Modeling ---
            if model_type == 'lgb':
                model = LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1, **params)
            elif model_type == 'xgb':
                model = XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss', enable_categorical=False, **params)
            elif model_type == 'cat':
                model = CatBoostClassifier(random_state=self.random_state, verbose=0, **params)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            model.fit(X_train_clean, y_train)

            p_val = model.predict_proba(X_val_clean)[:, 1]
            fold_auc = roc_auc_score(y_val, p_val)
            fold_scores.append(fold_auc)

        return np.mean(fold_scores)

    def _get_search_space(self, trial: optuna.Trial, model_type: str) -> dict:
        """
        Defines search spaces for GBDTs.
        """
        if model_type == 'lgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
        elif model_type == 'xgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'alpha': trial.suggest_float('alpha', 1e-5, 10.0),
                'lambda': trial.suggest_float('lambda', 1e-5, 10.0),
            }
        elif model_type == 'cat':
            return {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def run_study(self, X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int = 20) -> Dict[str, Any]:
        """
        Runs the Optuna optimization study on a proxy subset of data.
        """
        logger.info(f"Running Optuna study for {model_type} with {n_trials} trials...")
        X_proxy, y_proxy = self._stratified_downsample(X, y, ratio=self.downsample_ratio)
        logger.info(f"Using proxy dataset of size {len(X_proxy)} (ratio: {self.downsample_ratio})")

        study = optuna.create_study(direction='maximize', study_name=f"{model_type}_tuning")
        study.optimize(lambda trial: self.objective(trial, X_proxy, y_proxy, model_type), n_trials=n_trials)

        logger.info(f"Best trial for {model_type}: AUC={study.best_value:.4f}")
        return study.best_params
