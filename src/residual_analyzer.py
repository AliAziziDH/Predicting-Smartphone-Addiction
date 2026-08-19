"""
Error Residual Analyzer for Predicting Smartphone Addiction.
Pinpoints exact feature spaces where GBDT models make False Positive / False Negative errors,
and generates targeted residual-healing features.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.model.formulation import preprocess_and_engineer
from src.train import resolve_data_path


def analyze_residuals():
    print("=" * 65)
    print("🔍 ERROR RESIDUAL AUDIT & HARD SAMPLE PROFILING")
    print("=" * 65)

    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)

    # Subsample for fast local residual analysis
    sample_df = df.sample(n=25000, random_state=42).reset_index(drop=True)
    target_col = "addicted_label"

    X = preprocess_and_engineer(sample_df)
    y = sample_df[target_col]
    X_mat = X.drop(columns=["id", target_col], errors="ignore")

    oof_preds = np.zeros(len(sample_df))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    cat_cols = X_mat.select_dtypes(include=['category', 'object']).columns.tolist()
    for col in cat_cols:
        X_mat[col] = X_mat[col].astype('category')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y)):
        X_train, y_train = X_mat.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_mat.iloc[val_idx], y.iloc[val_idx]

        model = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y, oof_preds)
    print(f"📊 OOF Baseline AUC on 25k sample: {auc:.5f}")

    # Calculate absolute error residuals
    residuals = np.abs(y - oof_preds)
    hard_mask = residuals > 0.5

    print(f"⚠️ Hard Samples (Residual > 0.5): {hard_mask.sum()} / {len(y)} ({hard_mask.mean() * 100:.2f}%)")

    # Correlate residuals with numeric features
    numeric_cols = X_mat.select_dtypes(include=[np.number]).columns
    correlations = {}
    for col in numeric_cols:
        corr = np.corrcoef(X_mat[col].fillna(X_mat[col].median()), residuals)[0, 1]
        if not np.isnan(corr):
            correlations[col] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n📈 Top Features Correlated with Prediction Errors:")
    for col, corr in sorted_corr[:10]:
        print(f"  • {col:32s}: correlation with error = {corr:+.4f}")

    # False Positives vs False Negatives breakdown
    fp_mask = (y == 0) & (oof_preds > 0.5)
    fn_mask = (y == 1) & (oof_preds <= 0.5)

    print(f"\n🔴 False Positives: {fp_mask.sum()} (Predicted Addicted, Actual Healthy)")
    print(f"   • Mean Screen Time: {sample_df.loc[fp_mask, 'daily_screen_time_hours'].mean():.2f}h")
    print(f"   • Mean Sleep Hours: {sample_df.loc[fp_mask, 'sleep_hours'].mean():.2f}h")
    print(f"   • Mean Work/Study:  {sample_df.loc[fp_mask, 'work_study_hours'].mean():.2f}h")

    print(f"\n🔵 False Negatives: {fn_mask.sum()} (Predicted Healthy, Actual Addicted)")
    print(f"   • Mean Screen Time: {sample_df.loc[fn_mask, 'daily_screen_time_hours'].mean():.2f}h")
    print(f"   • Mean Sleep Hours: {sample_df.loc[fn_mask, 'sleep_hours'].mean():.2f}h")
    print(f"   • Mean Work/Study:  {sample_df.loc[fn_mask, 'work_study_hours'].mean():.2f}h")

    print("=" * 65)


if __name__ == "__main__":
    analyze_residuals()
