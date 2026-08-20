"""
High-Speed, Resilient Production Runner for Kaggle S6E8.
Key Features:
1. Early Stopping on GBDT models -> 4x-5x training speedup.
2. Direct Categorical Feature Integration (LightGBM, XGBoost, CatBoost).
3. Leak-free Out-of-Fold Rank-Averaging across test folds.
4. Progressive Checkpointing: Saves fold OOF arrays and model predictions immediately to disk.
5. Memory Optimization: Downcasts all arrays to float32.
"""

import os
import sys
import time
import json
import gc
import signal
import subprocess
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()
    sys.path.insert(0, ROOT_DIR)

from src.model.formulation import preprocess_and_engineer
from src.model.solver import UniversalLevelTargetEncoder, ValueLevelTargetEncoder, get_calibrated_model_params, to_gauss_rank, NelderMeadRankStacker, perform_ks_drift_screen
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def run_fast_production_training(
    n_splits: int = 10,
    random_state: int = 42,
    checkpoint_dir: str = "models/checkpoints",
    data_dir: str = "data",
    sample_size: Optional[int] = None,
    gcs_bucket: Optional[str] = "gs://ali-s6e8-kaggle-artifacts-2026"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_time = time.time()

    # Preemption Signal Handler for Spot VMs
    def handle_sigterm(signum, frame):
        print("\n🚨 [PREEMPTION WATCHDOG] SIGTERM signal received from Cloud Engine! Flushing checkpoints to GCS...")
        if gcs_bucket:
            try:
                subprocess.run(["gcloud", "storage", "cp", "-r", checkpoint_dir, gcs_bucket], capture_output=True)
                print(f"✅ Emergency checkpoint successfully synchronized to {gcs_bucket}")
            except Exception as e:
                print(f"⚠️ Failed to sync emergency checkpoint: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    print("=" * 75)
    print("⚡ FAST PRODUCTION RUNNER (10-FOLD WITH LEAK-FREE RANK-AVERAGING)")
    print(f"• GCS Storage Sync: {gcs_bucket or 'Disabled'} | Preemption Watchdog: ENGAGED")
    print("=" * 75)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        train_path = "train.csv"
        test_path = "test.csv"

    print(f"• Loading data from: {train_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if sample_size is not None and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"• [DRY-RUN / MOCK MODE] Subsampled train dataset to {len(train_df):,} rows.")

    target_col = "addicted_label"
    if target_col not in train_df.columns:
        for c in train_df.columns:
            if "addict" in c.lower() or "class" in c.lower() or "target" in c.lower():
                target_col = c
                break

    X = train_df.drop(columns=[target_col, "id"], errors="ignore")
    y = train_df[target_col].values
    test_ids = test_df["id"].values
    X_test_raw = test_df.drop(columns=["id"], errors="ignore")

    print(f"• Total Train Samples: {len(X):,} | Total Test Samples: {len(X_test_raw):,}")

    # Base Feature Engineering on Test Set
    X_test_clean_base = preprocess_and_engineer(X_test_raw)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_lgb = np.zeros(len(X), dtype=np.float32)
    oof_xgb = np.zeros(len(X), dtype=np.float32)
    oof_cat = np.zeros(len(X), dtype=np.float32)

    test_preds_lgb = np.zeros(len(X_test_clean_base), dtype=np.float32)
    test_preds_xgb = np.zeros(len(X_test_clean_base), dtype=np.float32)
    test_preds_cat = np.zeros(len(X_test_clean_base), dtype=np.float32)

    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_start = time.time()
        print(f"\n--- [Fold {fold}/{n_splits}] ---")

        X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]

        # Feature Engineering localized to fold
        X_train_clean = preprocess_and_engineer(X_train)
        X_val_clean = preprocess_and_engineer(X_val)

        # Universal Level Target & Frequency Encoding across all discrete levels (Smooth=20.0)
        te = UniversalLevelTargetEncoder(smooth=20.0, n_splits=5, random_state=random_state + fold)
        X_train_clean = te.fit_transform(X_train_clean, pd.Series(y_train))
        X_val_clean = te.transform(X_val_clean)
        X_test_fold = te.transform(X_test_clean_base.copy())

        # Categorical Encoding for nominal features
        raw_cat_cols = ['gender', 'stress_level', 'academic_work_impact']
        present_cats = [c for c in raw_cat_cols if c in X_train_clean.columns]

        for col in present_cats:
            le = LabelEncoder()
            train_s = X_train_clean[col].fillna('Missing').astype(str)
            val_s = X_val_clean[col].fillna('Missing').astype(str)
            test_s = X_test_fold[col].fillna('Missing').astype(str)

            X_train_clean[col] = le.fit_transform(train_s).astype(np.int32)
            val_classes = set(val_s.tolist())
            missing_val = val_classes - set(le.classes_)
            if missing_val:
                le.classes_ = np.append(le.classes_, list(missing_val))
            X_val_clean[col] = le.transform(val_s).astype(np.int32)

            test_classes = set(test_s.tolist())
            missing_test = test_classes - set(le.classes_)
            if missing_test:
                le.classes_ = np.append(le.classes_, list(missing_test))
            X_test_fold[col] = le.transform(test_s).astype(np.int32)

        # GBDT Models with Calibrated Hyperparameters
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.025,
            "n_estimators": 2500,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.70,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.025,
            "n_estimators": 2500,
            "max_depth": 6,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.65,
            "reg_alpha": 0.5,
            "reg_lambda": 8.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        cat_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.03,
            "iterations": 2200,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.2,
            "bagging_temperature": 0.2,
            "od_type": "Iter",
            "od_wait": 80,
            "random_seed": 42,
            "verbose": 0,
            "thread_count": -1,
        }

        # 1. LightGBM with Early Stopping Callback
        lgb = LGBMClassifier(**lgb_params)
        lgb.fit(
            X_train_clean, y_train,
            eval_set=[(X_val_clean, y_val)],
            callbacks=[early_stopping(stopping_rounds=80, verbose=False)],
            categorical_feature=present_cats if present_cats else 'auto'
        )
        p_lgb = lgb.predict_proba(X_val_clean)[:, 1].astype(np.float32)
        oof_lgb[val_idx] = p_lgb
        
        # Rank-average test fold predictions to avoid inter-fold calibration distortion
        fold_test_lgb = (rankdata(lgb.predict_proba(X_test_fold)[:, 1]) - 0.5) / len(X_test_fold)
        test_preds_lgb += fold_test_lgb.astype(np.float32) / n_splits
        auc_lgb = roc_auc_score(y_val, p_lgb)

        # 2. XGBoost with Early Stopping
        xgb = XGBClassifier(**xgb_params, early_stopping_rounds=80)
        xgb.fit(
            X_train_clean, y_train,
            eval_set=[(X_val_clean, y_val)],
            verbose=False
        )
        p_xgb = xgb.predict_proba(X_val_clean)[:, 1].astype(np.float32)
        oof_xgb[val_idx] = p_xgb
        
        fold_test_xgb = (rankdata(xgb.predict_proba(X_test_fold)[:, 1]) - 0.5) / len(X_test_fold)
        test_preds_xgb += fold_test_xgb.astype(np.float32) / n_splits
        auc_xgb = roc_auc_score(y_val, p_xgb)

        # 3. CatBoost with Early Stopping
        cat = CatBoostClassifier(**cat_params)
        cat.fit(
            X_train_clean, y_train,
            eval_set=(X_val_clean, y_val),
            cat_features=present_cats if present_cats else None,
            verbose=False
        )
        p_cat = cat.predict_proba(X_val_clean)[:, 1].astype(np.float32)
        oof_cat[val_idx] = p_cat
        
        fold_test_cat = (rankdata(cat.predict_proba(X_test_fold)[:, 1]) - 0.5) / len(X_test_fold)
        test_preds_cat += fold_test_cat.astype(np.float32) / n_splits
        auc_cat = roc_auc_score(y_val, p_cat)

        fold_dur = time.time() - fold_start
        fold_times.append(fold_dur)
        avg_time = np.mean(fold_times)
        rem_folds = n_splits - fold
        eta_sec = rem_folds * avg_time

        print(f"  • LGB AUC: {auc_lgb:.5f} | XGB AUC: {auc_xgb:.5f} | CAT AUC: {auc_cat:.5f}")
        print(f"  ⏱️ Fold {fold} finished in {fold_dur:.1f}s | Avg: {avg_time:.1f}s/fold | Remaining ETA: {eta_sec/60:.1f} min")

        # Save Fold Checkpoint Immediately
        ckpt_path = os.path.join(checkpoint_dir, f"fold_{fold}_checkpoint.npz")
        np.savez_compressed(
            ckpt_path,
            val_idx=val_idx,
            p_lgb=p_lgb,
            p_xgb=p_xgb,
            p_cat=p_cat,
            auc_lgb=auc_lgb,
            auc_xgb=auc_xgb,
            auc_cat=auc_cat
        )

        del X_train_clean, X_val_clean, X_test_fold, lgb, xgb, cat
        gc.collect()

    # Direct Nelder-Mead Rank-AUC Stacking
    print("\n[Ensembling] Optimizing Non-Parametric Nelder-Mead Rank Weights...")
    rank_oof = np.column_stack([
        (rankdata(oof_lgb) - 0.5) / len(oof_lgb),
        (rankdata(oof_xgb) - 0.5) / len(oof_xgb),
        (rankdata(oof_cat) - 0.5) / len(oof_cat),
    ])
    rank_test = np.column_stack([
        (rankdata(test_preds_lgb) - 0.5) / len(test_preds_lgb),
        (rankdata(test_preds_xgb) - 0.5) / len(test_preds_xgb),
        (rankdata(test_preds_cat) - 0.5) / len(test_preds_cat),
    ])

    stacker = NelderMeadRankStacker(random_state=random_state)
    stacker.fit(rank_oof, y)

    final_oof_preds = stacker.predict_proba(rank_oof)
    final_test_preds = stacker.predict_proba(rank_test)

    final_auc = float(roc_auc_score(y, final_oof_preds))

    # KS-Drift Screen
    drift_passed, ks_stat = perform_ks_drift_screen(final_oof_preds, final_test_preds)

    print("\n" + "=" * 75)
    print(f"🏆 FINAL {n_splits}-FOLD OOF ROC-AUC: {final_auc:.5f}")
    print(f"• LightGBM OOF AUC:   {roc_auc_score(y, oof_lgb):.5f}")
    print(f"• XGBoost OOF AUC:    {roc_auc_score(y, oof_xgb):.5f}")
    print(f"• CatBoost OOF AUC:   {roc_auc_score(y, oof_cat):.5f}")
    print(f"• Softmax Stack Weights (LGB, XGB, CAT): {stacker.weights_}")
    print(f"• KS Distribution Shift Stat: {ks_stat:.4f} (Shake-up Immunity: {'✅ PASSED' if drift_passed else '⚠️ DRIFT DETECTED'})")
    print("=" * 75)

    # Save Final Submission
    sub = pd.DataFrame({
        "id": test_ids,
        target_col: final_test_preds
    })
    sub_path = "submission_elite_wave11.csv"
    sub.to_csv(sub_path, index=False)
    print(f"✅ Final Production Submission saved: {sub_path} (Shape: {sub.shape})")

    total_elapsed = time.time() - start_time
    print(f"⏱️ Total Execution Time: {total_elapsed/60:.1f} minutes")

    return {
        "final_auc": final_auc,
        "submission_path": sub_path,
        "elapsed_minutes": total_elapsed / 60
    }


if __name__ == "__main__":
    run_fast_production_training()
