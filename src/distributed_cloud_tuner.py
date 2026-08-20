"""
Distributed Cloud Tuner with Warm-Start Recovery & Cloud SQL Ledger.
Enables multi-worker parallel Bayesian hyperparameter optimization for LightGBM, XGBoost, and CatBoost.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()
    sys.path.insert(0, ROOT_DIR)

from src.model.formulation import preprocess_and_engineer
from src.model.solver import ValueLevelTargetEncoder, get_or_create_cloud_study
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DistributedCloudTuner")


def run_distributed_tuning(
    study_name: str = "s6e8_elite_hyperopt",
    n_trials: int = 50,
    sample_size: int = 30000,
    n_splits: int = 3,
    data_dir: str = "data",
    random_state: int = 42
):
    """
    Executes distributed Optuna tuning with automatic Warm-Start recovery.
    """
    logger.info("=" * 75)
    logger.info("🚀 DISTRIBUTED CLOUD OPTUNA TUNER (WARM-START & CLOUD SQL RESILIENT)")
    logger.info("=" * 75)

    # 1. Connect or create Study
    study = get_or_create_cloud_study(study_name=study_name)
    if study is None:
        logger.error("Failed to initialize Optuna study.")
        return

    # 2. Load and downsample data for proxy hyperparameter search
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        train_path = "train.csv"

    logger.info(f"• Loading proxy dataset from: {train_path}...")
    df = pd.read_csv(train_path)

    target_col = "addicted_label"
    if target_col not in df.columns:
        for c in df.columns:
            if "addict" in c.lower() or "class" in c.lower() or "target" in c.lower():
                target_col = c
                break

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info(f"• Proxy sample downscaled to {len(df):,} rows for rapid tuning.")

    X_raw = df.drop(columns=[target_col, "id"], errors="ignore")
    y = df[target_col].values

    # Preprocessing with 41 features
    logger.info("• Preprocessing & Engineering 41 features on proxy dataset...")
    X_proc = preprocess_and_engineer(X_raw)

    def objective(trial):
        model_family = trial.suggest_categorical("model_family", ["lightgbm", "xgboost", "catboost"])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        val_scores = []

        for fold, (t_idx, v_idx) in enumerate(skf.split(X_proc, y)):
            X_tr, y_tr = X_proc.iloc[t_idx].copy(), y[t_idx]
            X_va, y_va = X_proc.iloc[v_idx].copy(), y[v_idx]

            # Target Encoding
            te_cols = [c for c in ['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours'] if c in X_tr.columns]
            te = ValueLevelTargetEncoder(cols=te_cols, smooth=10.0, n_splits=3, random_state=random_state + fold)
            X_tr = te.fit_transform(X_tr, pd.Series(y_tr))
            X_va = te.transform(X_va)

            # Categorical encoding
            for col in X_tr.select_dtypes(exclude=[np.number]).columns:
                le = LabelEncoder()
                X_tr[col] = le.fit_transform(X_tr[col].fillna('Missing').astype(str))
                val_s = X_va[col].fillna('Missing').astype(str)
                missing = set(val_s) - set(le.classes_)
                if missing:
                    le.classes_ = np.append(le.classes_, list(missing))
                X_va[col] = le.transform(val_s)

            if model_family == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("lgb_n_estimators", 300, 1500, step=100),
                    "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.08, log=True),
                    "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 63),
                    "subsample": trial.suggest_float("lgb_subsample", 0.6, 0.95),
                    "colsample_bytree": trial.suggest_float("lgb_colsample", 0.6, 0.95),
                    "reg_alpha": trial.suggest_float("lgb_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("lgb_lambda", 1e-4, 10.0, log=True),
                    "random_state": random_state,
                    "verbose": -1,
                    "n_jobs": -1
                }
                model = LGBMClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(30, verbose=False)])

            elif model_family == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 1500, step=100),
                    "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.08, log=True),
                    "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
                    "subsample": trial.suggest_float("xgb_subsample", 0.6, 0.95),
                    "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 0.95),
                    "reg_alpha": trial.suggest_float("xgb_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("xgb_lambda", 1e-4, 10.0, log=True),
                    "tree_method": "hist",
                    "device": "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ or os.path.exists("/usr/local/cuda") else "cpu",
                    "random_state": random_state,
                    "early_stopping_rounds": 30
                }
                model = XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            else:
                params = {
                    "iterations": trial.suggest_int("cat_iterations", 300, 1500, step=100),
                    "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.08, log=True),
                    "depth": trial.suggest_int("cat_depth", 3, 8),
                    "l2_leaf_reg": trial.suggest_float("cat_l2", 1e-2, 10.0, log=True),
                    "task_type": "GPU" if "CUDA_VISIBLE_DEVICES" in os.environ or os.path.exists("/usr/local/cuda") else "CPU",
                    "random_state": random_state,
                    "early_stopping_rounds": 30,
                    "verbose": False
                }
                model = CatBoostClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

            preds = model.predict_proba(X_va)[:, 1]
            val_scores.append(roc_auc_score(y_va, preds))

        mean_auc = float(np.mean(val_scores))
        return mean_auc

    logger.info(f"🔥 Starting Bayesian optimization loop ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    logger.info("=" * 75)
    logger.info(f"🏆 BEST TRIAL #{study.best_trial.number} | OOF AUC: {study.best_value:.5f}")
    logger.info(f"• Best Parameters: {study.best_params}")
    logger.info("=" * 75)

    return study.best_params


if __name__ == "__main__":
    run_distributed_tuning()
