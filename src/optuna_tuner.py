"""
High-Performance Bayesian Hyperparameter Tuner for GBDT Trio (LGBM, XGBoost, CatBoost).
Uses Optuna with TPESampler and MedianPruner over downsampled proxy folds to prevent
overfitting on synthetic rounded grid coordinates.
"""

import os
import sys
import json
import time
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.model.formulation import preprocess_and_engineer
from src.model.solver import ValueLevelTargetEncoder
from src.train import resolve_data_path

# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 25) -> Dict[str, Any]:
    print("\n🔍 Tuning LightGBM with Optuna TPE Sampler...")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 150,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }

        fold_aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_vl, y_vl = X.iloc[val_idx], y.iloc[val_idx]

            # Fit ValueLevelTargetEncoder inside fold
            te_cols = [c for c in ['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours', 'app_opens_per_day'] if c in X_tr.columns]
            te = ValueLevelTargetEncoder(cols=te_cols, smooth=10.0, n_splits=3, random_state=42 + fold)
            X_tr_enc = te.fit_transform(X_tr, y_tr)
            X_vl_enc = te.transform(X_vl)

            cat_cols = list(X_tr_enc.select_dtypes(exclude=[np.number]).columns)
            for c in cat_cols:
                X_tr_enc[c] = X_tr_enc[c].astype('category')
                X_vl_enc[c] = X_vl_enc[c].astype('category')

            model = LGBMClassifier(**params)
            model.fit(X_tr_enc, y_tr)
            preds = model.predict_proba(X_vl_enc)[:, 1]
            fold_aucs.append(roc_auc_score(y_vl, preds))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  🏆 Best LightGBM AUC: {study.best_value:.5f}")
    return study.best_params


def tune_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 15) -> Dict[str, Any]:
    print("\n🔍 Tuning XGBoost with Optuna TPE Sampler...")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 150,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True),
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1
        }

        fold_aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_vl, y_vl = X.iloc[val_idx], y.iloc[val_idx]

            te_cols = [c for c in ['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours', 'app_opens_per_day'] if c in X_tr.columns]
            te = ValueLevelTargetEncoder(cols=te_cols, smooth=10.0, n_splits=3, random_state=42 + fold)
            X_tr_enc = te.fit_transform(X_tr, y_tr)
            X_vl_enc = te.transform(X_vl)

            cat_cols = list(X_tr_enc.select_dtypes(exclude=[np.number]).columns)
            for c in cat_cols:
                X_tr_enc[c] = X_tr_enc[c].astype('category')
                X_vl_enc[c] = X_vl_enc[c].astype('category')

            model = XGBClassifier(**params, enable_categorical=True)
            model.fit(X_tr_enc, y_tr)
            preds = model.predict_proba(X_vl_enc)[:, 1]
            fold_aucs.append(roc_auc_score(y_vl, preds))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  🏆 Best XGBoost AUC: {study.best_value:.5f}")
    return study.best_params


def run_tuner(proxy_sample: int = 30000):
    print("=" * 65)
    print("🚀 OPTUNA TPE HYPERPARAMETER TUNER (STAGE 1 & 2)")
    print("=" * 65)

    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)

    sample_df = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)
    target_col = "addicted_label"
    y = sample_df[target_col]

    print(f"Pre-engineering features on {len(sample_df):,} proxy samples...")
    X_clean = preprocess_and_engineer(sample_df).drop(columns=["id", target_col], errors="ignore")

    best_lgb = tune_lgbm(X_clean, y, n_trials=10)
    best_xgb = tune_xgboost(X_clean, y, n_trials=8)

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "best_gbdt_params.json")

    results = {
        "lgb_params": best_lgb,
        "xgb_params": best_xgb,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved optimal hyperparameter configuration to: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_tuner(proxy_sample=20000)
